import base64
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TARGET_COLUMN_CANDIDATES = [
    "Rating",
    "rating",
    "RatingGroup",
    "rating_group",
    "Credit_Risk",
    "credit_risk",
    "CreditRisk",
    "creditRisk",
    "Risk",
    "risk",
    "RiskClass",
    "risk_class",
    "Risk_Level",
    "risk_level",
    "Target",
    "target",
    "Label",
    "label",
    "Class",
    "class"
]


def fail(message, details=None, status=1):
    payload = {"error": message}
    if details is not None:
        payload["details"] = details
    print(json.dumps(payload), file=sys.stderr)
    raise SystemExit(status)


def group_rating(rating):
    if rating in ["AAA", "AA", "A"]:
        return "Investment-High"
    if rating == "BBB":
        return "Investment-Low"
    if rating in ["BB", "B"]:
        return "Speculative"
    if rating in ["CCC", "CC", "C", "D"]:
        return "Distressed"
    return "Unknown"


def humanize_feature_name(name):
    cleaned = str(name).replace("num__", "").replace("cat__", "")
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def parse_request_payload():
    raw_payload = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    if not raw_payload:
        fail("Missing JSON request payload.")

    try:
        return json.loads(raw_payload)
    except Exception as exc:
        fail("Invalid JSON request payload.", {"details": str(exc)})


def load_default_dataframe():
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "set A corporate_rating.csv",
        project_root / "data" / "set A corporate_rating.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    fail("No uploaded dataset was provided and the default dataset was not found.")


def read_uploaded_file(path, suffix):
    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".xlsx":
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            fail("XLSX uploads require the openpyxl package in the Python virtual environment.", {
                "details": str(exc),
                "fix": "Run: .\\.venv\\Scripts\\python.exe -m pip install openpyxl"
            })

    if suffix == ".xls":
        try:
            return pd.read_excel(path, engine="xlrd")
        except ImportError as exc:
            fail("XLS uploads require the xlrd package in the Python virtual environment.", {
                "details": str(exc),
                "fix": "Run: .\\.venv\\Scripts\\python.exe -m pip install xlrd"
            })

    fail("Unsupported dataset file type.", {
        "fileExtension": suffix or "(none)",
        "supportedExtensions": [".csv", ".xlsx", ".xls"]
    })


def load_uploaded_dataframe(payload):
    file_name = payload.get("fileName", "")
    file_data = payload.get("fileData")
    file_encoding = payload.get("fileEncoding", "utf8")

    if not file_data:
        return load_default_dataframe()

    suffix = Path(file_name).suffix.lower()
    if file_encoding == "base64":
        try:
            binary = base64.b64decode(file_data)
        except Exception as exc:
            fail("Uploaded file could not be decoded.", {"details": str(exc)})

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(binary)
            temp_path = tmp.name

        try:
            return read_uploaded_file(temp_path, suffix)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if suffix in [".xlsx", ".xls"]:
        fail("Excel files must be sent as base64 data.")
    if suffix != ".csv":
        fail("Unsupported dataset file type.", {
            "fileExtension": suffix or "(none)",
            "supportedExtensions": [".csv", ".xlsx", ".xls"]
        })

    return pd.read_csv(io.StringIO(file_data))


def find_target_column(df, requested_target=None):
    if requested_target:
        return requested_target

    normalized_columns = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): column
        for column in df.columns
    }

    for candidate in TARGET_COLUMN_CANDIDATES:
        normalized_candidate = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if normalized_candidate in normalized_columns:
            return normalized_columns[normalized_candidate]

    return None


def build_target(df, payload):
    target_column = find_target_column(df, payload.get("targetColumn"))
    if target_column not in df.columns:
        fail("The dataset must contain a target column.", {
            "expectedColumn": target_column or "one of the supported target names",
            "supportedTargetNames": TARGET_COLUMN_CANDIDATES,
            "availableColumns": list(df.columns)
        })

    target_values = df[target_column]
    if target_column == "Rating":
        y = target_values.apply(group_rating)
        valid_mask = y != "Unknown"
        return y[valid_mask], valid_mask, target_column

    valid_mask = target_values.notna() & (target_values.astype(str).str.strip() != "")
    return target_values[valid_mask].astype(str), valid_mask, target_column


def build_pipeline(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append((
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            numeric_features
        ))
    if categorical_features:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ))

    if not transformers:
        fail("The dataset does not contain usable feature columns.")

    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])


def choose_split_options(y):
    class_counts = y.value_counts()
    can_stratify = len(class_counts) > 1 and int(class_counts.min()) >= 2
    test_size = 0.2 if len(y) >= 10 else 0.3
    return test_size, y if can_stratify else None


def feature_importance(best_model, X_test, y_test):
    try:
        result = permutation_importance(
            best_model,
            X_test,
            y_test,
            n_repeats=8,
            random_state=RANDOM_STATE,
            scoring="f1_weighted"
        )
        items = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": result.importances_mean
        }).sort_values("Importance", ascending=False).head(15)

        max_value = float(items["Importance"].abs().max()) if len(items) else 0.0
        if max_value <= 0:
            max_value = 1.0

        return [
            {
                "feature": humanize_feature_name(row["Feature"]),
                "value": round(float(abs(row["Importance"]) / max_value * 100), 2),
                "effect": "Pushes toward predicted class" if row["Importance"] >= 0 else "Pushes away from predicted class"
            }
            for _, row in items.iterrows()
        ]
    except Exception:
        return []


def main():
    payload = parse_request_payload()
    df = load_uploaded_dataframe(payload)
    if df.empty:
        fail("The uploaded dataset is empty.")

    y, valid_mask, target_column = build_target(df, payload)
    df = df.loc[valid_mask].copy()

    drop_cols = [
        target_column,
        "RatingGroup",
        "Name",
        "Symbol",
        "Rating Agency Name",
        "Date"
    ]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    X = X.dropna(axis=1, how="all")

    if len(X) < 4:
        fail("The dataset needs at least 4 usable rows for Random Forest training.")
    if y.nunique() < 2:
        fail("The target column must contain at least two classes.")

    test_size, stratify = choose_split_options(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    model = build_pipeline(X)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = list(model.named_steps["model"].classes_)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=labels)

    accuracy = float(accuracy_score(y_test, predictions))
    weighted = report.get("weighted avg", {})
    precision = float(weighted.get("precision", 0))
    recall = float(weighted.get("recall", 0))
    f1 = float(weighted.get("f1-score", 0))

    predicted_class = str(predictions[0])
    importances = feature_importance(model, X_test, y_test)
    if not importances:
        importances = [
            {"feature": humanize_feature_name(col), "value": 0, "effect": "Pushes toward predicted class"}
            for col in list(X.columns[:10])
        ]

    positive = [item["feature"] for item in importances if item["effect"].startswith("Pushes")][:3]
    negative = [item["feature"] for item in importances if not item["effect"].startswith("Pushes")][:3]
    if not negative:
        negative = [item["feature"] for item in importances[3:6]]

    strength = "Trained dynamically from the uploaded dataset using inferred numeric and categorical features."
    weakness = "Performance depends on the uploaded dataset size, class balance, and target column quality."

    result = {
        "model": "Random Forest",
        "prediction": predicted_class,
        "predictions": [str(item) for item in predictions.tolist()],
        "featureCount": int(X.shape[1]),
        "rowCount": int(len(df)),
        "targetColumn": target_column,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "strength": strength,
            "weakness": weakness
        },
        "confusionMatrix": {
            "labels": labels,
            "values": cm.tolist()
        },
        "shap": {
            "selectedClass": predicted_class,
            "prediction": predicted_class,
            "topFeatures": importances,
            "story": {
                "positive": positive,
                "negative": negative
            }
        },
        "modelData": {
            "tag": "Random Forest",
            "labels": labels,
            "metrics": {
                "accuracy": f"{accuracy:.4f}",
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1": f"{f1:.4f}",
                "strength": strength,
                "weakness": weakness
            },
            "matrix": cm.tolist(),
            "shap": [
                [
                    item["feature"],
                    item["value"],
                    -1 if not item["effect"].startswith("Pushes") else 1
                ]
                for item in importances
            ],
            "shapStory": {
                "positive": positive,
                "negative": negative
            }
        }
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()

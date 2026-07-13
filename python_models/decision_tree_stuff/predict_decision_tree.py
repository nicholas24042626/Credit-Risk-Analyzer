import base64
import io
import json
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_baseline import extract_groups, make_split, run_fair_baseline  # noqa: E402

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

RANDOM_STATE = 143


def group_rating(rating):
    """Collapse granular credit ratings into four financial risk tiers.

    Investment_High : AAA, AA, A
    Investment_Low  : BBB
    Speculative     : BB, B
    Distressed      : CCC, CC, C, D
    """
    if rating in ["AAA", "AA", "A"]:
        return "Investment-High"
    elif rating == "BBB":
        return "Investment-Low"
    elif rating in ["BB", "B"]:
        return "Speculative"
    elif rating in ["CCC", "CC", "C", "D"]:
        return "Distressed"
    else:
        return "Unknown"


def humanize_feature_name(name):
    cleaned = name.replace("num__", "").replace("cat__", "")
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]


def load_default_dataframe():
    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / "data" / "set A corporate_rating.csv"
    if default_path.exists():
        return pd.read_csv(default_path)
    raise FileNotFoundError("Default dataset was not found at the notebook's CSV path.")


def load_uploaded_dataframe(payload):
    file_name = payload.get("fileName", "")
    file_data = payload.get("fileData")
    file_encoding = payload.get("fileEncoding", "utf8")

    if not file_data:
        return load_default_dataframe()

    suffix = Path(file_name).suffix.lower()
    if file_encoding == "base64":
        binary = base64.b64decode(file_data)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(binary)
            temp_path = tmp.name
        try:
            if suffix == ".xlsx":
                return pd.read_excel(temp_path, engine="openpyxl")
            if suffix == ".xls":
                return pd.read_excel(temp_path, engine="xlrd")
            return pd.read_csv(temp_path)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    text = file_data
    if suffix in [".xlsx", ".xls"]:
        raise ValueError(
            "Excel files must be sent as base64 data. Use base64 for .xlsx/.xls uploads."
        )
    return pd.read_csv(io.StringIO(text))


def build_pipeline(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def main():
    payload = {}
    raw_payload = ""
    if len(sys.argv) > 1 and sys.argv[1]:
        raw_payload = sys.argv[1]
    else:
        raw_payload = sys.stdin.read().strip()

    if raw_payload:
        payload = json.loads(raw_payload)

    df = load_uploaded_dataframe(payload)
    if "Rating" not in df.columns:
        raise ValueError("The dataset must contain a Rating column.")

    # Shared cross-model comparison tier: identical cleaning, split, features,
    # and untuned-default estimator as the other three models' own
    # fairBaseline (see shared_baseline.py) -- independent of, and computed
    # before, this file's own SMOTE + tuned pipeline below.
    fair_baseline = run_fair_baseline(
        DecisionTreeClassifier(random_state=RANDOM_STATE), df
    )

    df["RatingGroup"] = df["Rating"].apply(group_rating)
    df = df[df["RatingGroup"] != "Unknown"].copy()

    target_col = "RatingGroup"
    drop_cols = [
        "Rating",
        "RatingGroup",
        "Name",
        "Symbol",
        "Rating Agency Name",
        "Date"
    ]
    existing_drop_cols = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=existing_drop_cols)
    y = df[target_col]

    # Company-level grouped split (not plain stratify=y): a company's repeat
    # year-rows must land entirely on one side of the split, or accuracy is
    # inflated by the model training on near-duplicate rows of a company that
    # also appears in the test set. Same fix already applied via
    # shared_baseline's fairBaseline tier -- this brings this file's own
    # PRIMARY reported number in line with it, not just the comparison tier.
    groups = extract_groups(df)
    X_train, X_test, y_train, y_test, split_strategy = make_split(X, y, groups=groups)

    preprocessor = build_pipeline(X)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

    # True baseline: untuned library defaults, same SMOTE-balanced data.
    # Previously this "baseline" was just an alias for the tuned model's own
    # predictions (dead code -- no separate fit ever happened), so the
    # baseline/tuned comparison in the response was always 0.0000 apart.
    baseline_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    baseline_model.fit(X_train_balanced, y_train_balanced)
    baseline_pred = baseline_model.predict(X_test_processed)
    baseline_report = classification_report(y_test, baseline_pred, output_dict=True)
    baseline_accuracy = float(accuracy_score(y_test, baseline_pred))
    baseline_f1 = float(baseline_report["weighted avg"]["f1-score"])

    # Tuned: small grid search over the tree's own highest-impact
    # hyperparameters (previously a single hand-picked min_samples_leaf=5
    # value with no search at all, despite being labelled "tuned").
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10],
    }
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=1,
    )
    grid_search.fit(X_train_balanced, y_train_balanced)
    best_model = grid_search.best_estimator_

    tuned_pred = best_model.predict(X_test_processed)
    report = classification_report(y_test, tuned_pred, output_dict=True)
    labels = list(best_model.classes_)
    cm = confusion_matrix(y_test, tuned_pred, labels=labels)

    accuracy = float(accuracy_score(y_test, tuned_pred))
    precision = float(report["weighted avg"]["precision"])
    recall = float(report["weighted avg"]["recall"])
    f1 = float(report["weighted avg"]["f1-score"])

    strength = "Best at separating Investment-Low and Speculative classes in the balanced tree."
    weakness = "Distressed is still the hardest class because the dataset is small and imbalanced."
    if f1 < 0.5:
        strength = "The tuned tree still finds useful structure in the financial ratios."
        weakness = "Class imbalance is reducing performance on the smallest class."

    selected_class = labels[0]
    sample_company = X_test.iloc[[0]].copy()
    sample_company_processed = preprocessor.transform(sample_company)
    predicted_class = best_model.predict(sample_company_processed)[0]

    shap_story = {
        "positive": [],
        "negative": []
    }
    shap_features = []
    tree_model = best_model

    try:
        import shap  # noqa: F401

        if hasattr(X_test_processed, "toarray"):
            X_test_processed = X_test_processed.toarray()
        raw_feature_names = preprocessor.get_feature_names_out()
        shap_feature_names = [
            name.replace("num__", "").replace("cat__", "")
            for name in raw_feature_names
        ]
        X_test_shap = pd.DataFrame(X_test_processed, columns=shap_feature_names, index=X_test.index)

        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_test_shap, check_additivity=False)

        def get_class_shap_values(all_shap_values, class_index):
            if isinstance(all_shap_values, list):
                return all_shap_values[class_index]

            all_shap_values = np.array(all_shap_values)
            if all_shap_values.ndim == 3:
                return all_shap_values[:, :, class_index]
            return all_shap_values

        class_index = labels.index(predicted_class)
        class_shap_values = get_class_shap_values(shap_values, class_index)

        local_shap_df = pd.DataFrame({
            "Feature": shap_feature_names,
            "SHAP Value": class_shap_values[0]
        })
        local_shap_df["Abs SHAP Value"] = local_shap_df["SHAP Value"].abs()
        local_shap_df["Effect"] = np.where(
            local_shap_df["SHAP Value"] > 0,
            "Pushes toward predicted class",
            "Pushes away from predicted class"
        )

        top_local_shap = local_shap_df.sort_values("Abs SHAP Value", ascending=False).head(15)
        shap_features = [
            {
                "feature": humanize_feature_name(row["Feature"]),
                "value": float(row["Abs SHAP Value"]),
                "effect": row["Effect"]
            }
            for _, row in top_local_shap.iterrows()
        ]

        positive = (
            local_shap_df[local_shap_df["SHAP Value"] > 0]
            .sort_values("SHAP Value", ascending=False)
            .head(3)["Feature"]
            .tolist()
        )
        negative = (
            local_shap_df[local_shap_df["SHAP Value"] < 0]
            .sort_values("SHAP Value", ascending=True)
            .head(3)["Feature"]
            .tolist()
        )

        shap_story = {
            "positive": [humanize_feature_name(name) for name in positive],
            "negative": [humanize_feature_name(name) for name in negative]
        }
    except Exception:
        perm_result = permutation_importance(
            best_model,
            X_test_processed,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="f1_weighted"
        )
        perm_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": perm_result.importances_mean
        }).sort_values(by="Importance", ascending=False).head(15)

        shap_features = [
            {
                "feature": humanize_feature_name(row["Feature"]),
                "value": float(abs(row["Importance"])),
                "effect": "Pushes toward predicted class" if row["Importance"] >= 0 else "Pushes away from predicted class"
            }
            for _, row in perm_df.iterrows()
        ]

    result = {
        "prediction": predicted_class,
        "samplePrediction": {
            "class": predicted_class,
            "prediction": predicted_class,
            "sampleIndex": int(X_test.index[0]),
        },
        "classLabels": labels,
        "baseline": {
            "accuracy": round(baseline_accuracy, 4),
            "f1": round(baseline_f1, 4),
            "classificationReport": baseline_report
        },
        "tuned": {
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "classificationReport": report
        },
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
        "featureImportance": [
            {
                "feature": item["feature"],
                "importance": round(item["value"], 4),
                "effect": item["effect"]
            }
            for item in shap_features
        ],
        "shap": {
            "selectedClass": predicted_class,
            "prediction": predicted_class,
            "topFeatures": shap_features,
            "story": shap_story
        },
        "modelData": {
            "tag": "Decision Tree",
            "labels": labels,
            "metrics": {
                "accuracy": f"{accuracy:.4f}",
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1": f"{f1:.4f}",
                "strength": strength,
                "weakness": weakness
            },
            "baseline": {
                "accuracy": f"{baseline_accuracy:.4f}",
                "f1": f"{baseline_f1:.4f}"
            },
            "matrix": cm.tolist(),
            "featureImportance": [
                {
                    "feature": item["feature"],
                    "importance": round(item["value"], 4),
                    "effect": item["effect"]
                }
                for item in shap_features
            ],
            "shap": [
                [item["feature"], round(item["value"], 4)]
                for item in shap_features
            ],
            "shapStory": shap_story,
            "fairBaseline": fair_baseline["metrics"] if fair_baseline else None
        }
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()

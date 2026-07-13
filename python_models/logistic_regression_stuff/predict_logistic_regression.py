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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    # StandardScaler matters here specifically (unlike the tree-based models
    # in this project): unscaled financial ratios spanning wildly different
    # magnitudes both slow lbfgs's convergence and distort the L1/L2 penalty,
    # which penalizes large-magnitude coefficients more than the underlying
    # feature actually warrants.
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
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

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
    return model


def to_dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def coefficient_fallback_importance(lr_model, feature_names, class_index, sample_row):
    """Local, coefficient-based explanation used when SHAP is unavailable.

    Mirrors the SHAP local-explanation shape (feature / abs value / effect)
    by multiplying each fitted coefficient by the corresponding value of the
    sample being explained, which approximates each feature's contribution
    to that sample's predicted-class log-odds.
    """
    coefs = lr_model.coef_
    if coefs.shape[0] == 1:
        # Binary classification: sklearn only stores one coefficient row,
        # for the positive class. Flip the sign for the negative class.
        coef_row = coefs[0] if class_index == 1 else -coefs[0]
    else:
        coef_row = coefs[class_index]

    contributions = coef_row * sample_row

    local_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": contributions
    })
    local_df["Abs SHAP Value"] = local_df["SHAP Value"].abs()
    local_df["Effect"] = np.where(
        local_df["SHAP Value"] > 0,
        "Pushes toward predicted class",
        "Pushes away from predicted class"
    )

    top_local = local_df.sort_values("Abs SHAP Value", ascending=False).head(15)
    shap_features = [
        {
            "feature": humanize_feature_name(row["Feature"]),
            "value": float(row["Abs SHAP Value"]),
            "effect": row["Effect"]
        }
        for _, row in top_local.iterrows()
    ]

    positive = (
        local_df[local_df["SHAP Value"] > 0]
        .sort_values("SHAP Value", ascending=False)
        .head(3)["Feature"]
        .tolist()
    )
    negative = (
        local_df[local_df["SHAP Value"] < 0]
        .sort_values("SHAP Value", ascending=True)
        .head(3)["Feature"]
        .tolist()
    )

    shap_story = {
        "positive": [humanize_feature_name(name) for name in positive],
        "negative": [humanize_feature_name(name) for name in negative]
    }

    return shap_features, shap_story


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
    # before, this file's own GridSearchCV-tuned pipeline below. Note this
    # intentionally does NOT scale features (StandardScaler), even though
    # Logistic Regression benefits from it -- the point of this tier is what
    # each algorithm does with no model-specific assistance at all.
    fair_baseline = run_fair_baseline(
        LogisticRegression(random_state=RANDOM_STATE), df
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
    # also appears in the test set.
    groups = extract_groups(df)
    X_train, X_test, y_train, y_test, split_strategy = make_split(
        X, y, groups=groups, test_size=0.20, random_state=RANDOM_STATE
    )

    baseline_model = build_pipeline(X)
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_accuracy = float(accuracy_score(y_test, baseline_pred))
    baseline_f1 = float(f1_score(y_test, baseline_pred, average="weighted"))
    baseline_report = classification_report(y_test, baseline_pred, output_dict=True)

    # Small, fast grid limited to Logistic Regression's own hyperparameters.
    # "saga" supports both l1 and l2 penalties for multiclass problems, so
    # C, penalty and solver can all be tuned together in one grid.
    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2", "l1"],
        "model__solver": ["saga"],
        "model__class_weight": ["balanced"]
    }

    grid_search = GridSearchCV(
        estimator=baseline_model,
        param_grid=param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    tuned_pred = best_model.predict(X_test)

    report = classification_report(y_test, tuned_pred, output_dict=True)
    labels = list(best_model.classes_)
    cm = confusion_matrix(y_test, tuned_pred, labels=labels)

    accuracy = float(accuracy_score(y_test, tuned_pred))
    precision = float(report["weighted avg"]["precision"])
    recall = float(report["weighted avg"]["recall"])
    f1 = float(report["weighted avg"]["f1-score"])

    strength = "Coefficients give a directly interpretable, linear read on how each ratio moves the predicted rating."
    weakness = "Distressed is still the hardest class because the dataset is small and imbalanced, and a linear model can't capture non-linear ratio interactions."
    if f1 < 0.5:
        strength = "The tuned logistic model still finds a useful linear signal in the financial ratios."
        weakness = "Class imbalance and non-linear relationships between ratios are limiting performance on the smallest classes."

    sample_company = X_test.iloc[[0]].copy()
    predicted_class = best_model.predict(sample_company)[0]

    preprocessor = best_model.named_steps["preprocessor"]
    lr_model = best_model.named_steps["model"]

    raw_feature_names = preprocessor.get_feature_names_out()
    shap_feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in raw_feature_names
    ]

    X_test_processed = to_dense(preprocessor.transform(X_test))
    X_test_shap = pd.DataFrame(X_test_processed, columns=shap_feature_names, index=X_test.index)

    class_index = labels.index(predicted_class)

    shap_story = {"positive": [], "negative": []}
    shap_features = []

    try:
        import shap  # noqa: F401

        X_train_processed = to_dense(preprocessor.transform(X_train))
        background_size = min(100, X_train_processed.shape[0])
        rng = np.random.RandomState(RANDOM_STATE)
        background_idx = rng.choice(X_train_processed.shape[0], size=background_size, replace=False)
        background = X_train_processed[background_idx]

        # LinearExplainer is the SHAP method built for linear models like
        # Logistic Regression (TreeExplainer is only valid for tree models).
        explainer = shap.LinearExplainer(lr_model, background)
        shap_values = explainer.shap_values(X_test_shap)

        def get_class_shap_values(all_shap_values, class_idx, n_classes):
            if isinstance(all_shap_values, list):
                return all_shap_values[class_idx]

            all_shap_values = np.array(all_shap_values)
            if all_shap_values.ndim == 3:
                return all_shap_values[:, :, class_idx]
            # Binary logistic regression: LinearExplainer returns a single
            # array of shap values for the positive class only.
            if n_classes == 2 and class_idx == 0:
                return -all_shap_values
            return all_shap_values

        class_shap_values = get_class_shap_values(shap_values, class_index, len(labels))

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
        sample_row = X_test_shap.iloc[0].to_numpy()
        shap_features, shap_story = coefficient_fallback_importance(
            lr_model, shap_feature_names, class_index, sample_row
        )

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
            "tag": "Logistic Regression",
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
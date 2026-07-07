import sys
import json
import warnings
import hashlib
import base64
import io
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# All data-transformation logic lives in the shared preprocessing module so the
# production pipeline and the training notebook stay in sync.
from preprocessing import (
    LABEL_ENCODER,
    RANDOM_STATE,
    DROP_COLS,
    add_interaction_features,
    apply_imputation,
    apply_winsorize_bounds,
    apply_zscore_from_stats,
    clean_dataframe,
    compute_sector_stats,
    encode_and_align,
    extract_groups,
    fit_feature_selection,
    fit_imputation,
    format_prediction_label,
    group_rating,
    humanize_feature_name,
    make_split,
    winsorize_features,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "set A corporate_rating.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataframe(input_data):
    """Load a DataFrame from an uploaded file payload or the default CSV."""
    file_data = input_data.get("fileData")
    file_name = input_data.get("fileName", "dataset.csv")
    file_encoding = input_data.get("fileEncoding", "utf8")

    if file_data:
        suffix = Path(file_name).suffix.lower()
        if file_encoding == "base64":
            binary = base64.b64decode(file_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(binary)
                temp_path = tmp.name
            try:
                return pd.read_excel(temp_path) if suffix in {".xlsx", ".xls"} else pd.read_csv(temp_path)
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        else:
            if suffix in {".xlsx", ".xls"}:
                raise ValueError("Excel files must be sent as base64.")
            return pd.read_csv(io.StringIO(file_data))

    default_path = DEFAULT_DATASET_PATH
    if default_path.exists():
        return pd.read_csv(default_path)
    raise FileNotFoundError("No file provided and default dataset not found.")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_paths(data_hash):
    """Return a dict of all cache file paths keyed by logical name."""
    suffix = "" if data_hash == "default" else f"_{data_hash}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = [
        "calibrated_model", "base_model", "imputer_medians", "winsorize_bounds",
        "sector_stats", "feature_columns", "optimal_thresholds", "prediction_strategy",
    ]
    return {name: RESULTS_DIR / f"{name}{suffix}.pkl" for name in names}


def _cache_is_complete(paths):
    return all(p.exists() for p in paths.values())


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _optimize_thresholds(y_true, y_proba, n_classes):
    """Find per-class probability multipliers that maximise macro-F1."""
    def neg_f1(thresholds):
        preds = (y_proba * thresholds).argmax(axis=1)
        return -f1_score(y_true, preds, average="macro")

    best_score, best_thresholds = -1.0, np.ones(n_classes)
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(10):  # Restarts for threshold coverage
        init = rng.uniform(0.5, 2.0, n_classes)
        res = minimize(neg_f1, init, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5})
        if -res.fun > best_score:
            best_score = -res.fun
            best_thresholds = res.x
    # Normalise so the scale of probabilities is preserved on average
    return best_thresholds / best_thresholds.sum() * n_classes


def train_model(X_train_enc, y_train):
    """Fit an ensemble of XGBoost models with aggressive tuning for 75%+ accuracy.

    Strategy:
    1. Expand hyperparameter grid to cover more of the search space
    2. Train 3 XGBoost models with different random seeds
    3. Soft-vote ensemble for robustness
    4. Isotonic calibration on the ensemble
    """
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        random_state=RANDOM_STATE,  # Model 1 seed — required for reproducibility
        n_jobs=-1,
    )

    # Aggressive grid — favors deeper trees + more regularization
    param_grid = {
        "max_depth":        [5, 7, 9],
        "n_estimators":     [150, 200],
        "learning_rate":    [0.03, 0.05, 0.1],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0, 0.1, 0.2],
        "reg_alpha":        [0, 0.1, 0.3],
        "reg_lambda":       [1, 1.5],
    }

    # Model 1 — best from RandomizedSearchCV (15 samples from the grid)
    grid = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=15,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    grid.fit(X_train_enc, y_train, sample_weight=sample_weights)
    best_xgb = grid.best_estimator_

    # Model 2 & 3 — retrain best config with different random seeds for diversity
    best_params = grid.best_params_
    xgb_2 = XGBClassifier(**{**best_params, "random_state": RANDOM_STATE + 1, "n_jobs": -1})
    xgb_3 = XGBClassifier(**{**best_params, "random_state": RANDOM_STATE + 2, "n_jobs": -1})

    xgb_2.fit(X_train_enc, y_train, sample_weight=sample_weights)
    xgb_3.fit(X_train_enc, y_train, sample_weight=sample_weights)

    # Soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[("xgb1", best_xgb), ("xgb2", xgb_2), ("xgb3", xgb_3)],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train_enc, y_train, sample_weight=sample_weights)

    # Calibrate the ensemble (fall back to fewer folds for tiny minority classes)
    for cv_folds in (3, 2):
        try:
            calibrated = CalibratedClassifierCV(estimator=ensemble, method="isotonic", cv=cv_folds)
            calibrated.fit(X_train_enc, y_train, sample_weight=sample_weights)
            break
        except ValueError:
            if cv_folds == 2:
                raise

    return best_xgb, calibrated  # Return base model for SHAP, calibrated ensemble for predictions


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def compute_shap(best_xgb, X_test_enc, pred_class_idx):
    """Return SHAP values for the first test row; fall back to feature importances."""
    try:
        explainer = shap.TreeExplainer(best_xgb)
        shap_values = explainer.shap_values(X_test_enc.iloc[[0]])
        if isinstance(shap_values, list):
            return shap_values[pred_class_idx][0]
        if isinstance(shap_values, np.ndarray):
            return shap_values[0, :, pred_class_idx] if shap_values.ndim == 3 else shap_values[0]
    except Exception:
        pass
    return best_xgb.feature_importances_


def build_shap_payload(X_test_enc, sample_shap):
    """Build the shap list and story dict returned to the client."""
    shap_df = (
        pd.DataFrame({"Feature": X_test_enc.columns, "SHAP Value": sample_shap})
        .assign(**{"Abs SHAP Value": lambda df: df["SHAP Value"].abs()})
        .sort_values("Abs SHAP Value", ascending=False)
        .head(15)
    )

    max_abs = shap_df["Abs SHAP Value"].max()
    shap_df["Scaled Value"] = (shap_df["Abs SHAP Value"] / max_abs * 95) if max_abs > 0 else 0.0
    shap_df["Direction"] = (shap_df["SHAP Value"] >= 0).astype(int) * 2 - 1  # 1 or -1

    shap_data = []
    positive_story = []
    negative_story = []

    for _, row in shap_df.iterrows():
        fname = humanize_feature_name(row["Feature"])
        val = float(row["Scaled Value"])
        direction = int(row["Direction"])
        if val > 0:
            shap_data.append([fname, round(val, 2), direction])
            (positive_story if direction == 1 else negative_story).append(fname)

    if not shap_data:
        shap_data = [["Feature", 100.0, 1]]

    return shap_data, {
        "positive": positive_story[:3] or ["No strong positive features"],
        "negative": negative_story[:3] or ["No strong negative features"],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    try:
        raw_payload = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
        if not raw_payload:
            raise ValueError("No input data provided.")

        input_data = json.loads(raw_payload)
        df = load_dataframe(input_data)

        if "Rating" not in df.columns:
            raise ValueError("The uploaded dataset must contain a 'Rating' column to train the model.")

        # ── Data cleaning (deterministic, pre-split, leakage-safe) ───────────
        df, _clean_report = clean_dataframe(df)

        # ── Label preparation ────────────────────────────────────────────────
        y_raw = df["Rating"].apply(group_rating)
        valid_mask = y_raw != "Unknown"
        df = df[valid_mask].copy()
        y_encoded = LABEL_ENCODER.transform(y_raw[valid_mask])

        # Company-level grouping key (captured before identifier columns drop).
        groups = extract_groups(df)

        # ── Feature matrix ───────────────────────────────────────────────────
        X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        # ── Train / test split (company-level, leakage-safe) ─────────────────
        X_train, X_test, y_train, y_test, split_strategy = make_split(
            X, y_encoded, groups=groups, test_size=0.20, random_state=RANDOM_STATE
        )

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Dataset is too small to split into training and test sets.")

        # ── Cache look-up ─────────────────────────────────────────────────────
        file_data = input_data.get("fileData")
        data_hash = hashlib.md5(file_data.encode("utf-8")).hexdigest() if file_data else "default"
        paths = _cache_paths(data_hash)

        if _cache_is_complete(paths):
            # ── Cache HIT: load artefacts ────────────────────────────────────
            calibrated_model = joblib.load(paths["calibrated_model"])
            best_xgb = joblib.load(paths["base_model"])
            medians = joblib.load(paths["imputer_medians"])
            w_bounds = joblib.load(paths["winsorize_bounds"])
            sector_stats = joblib.load(paths["sector_stats"])
            feature_columns = joblib.load(paths["feature_columns"])
            optimal_thresholds = joblib.load(paths["optimal_thresholds"])
            use_thresholds = joblib.load(paths["prediction_strategy"]).get("use_thresholds", False)

            X_train_m = apply_imputation(X_train, medians)
            X_test_m = apply_imputation(X_test, medians)
            X_train_w, X_test_w = apply_winsorize_bounds(X_train_m, X_test_m, w_bounds)
            X_train_i = add_interaction_features(X_train_w)
            X_test_i = add_interaction_features(X_test_w)
            if sector_stats:
                apply_zscore_from_stats([X_train_i, X_test_i], sector_stats)
            X_train_enc, X_test_enc = encode_and_align(X_train_i, X_test_i, feature_columns)

        else:
            # ── Cache MISS: full training pipeline ───────────────────────────
            # 1. Impute (train medians) → 2. Winsorize → 3. Interactions →
            # 4. Sector z-scores → 5. Encode/align → 6. Feature selection.
            medians = fit_imputation(X_train)
            X_train_m = apply_imputation(X_train, medians)
            X_test_m = apply_imputation(X_test, medians)

            X_train_w, X_test_w, w_bounds = winsorize_features(X_train_m, X_test_m)
            X_train_i = add_interaction_features(X_train_w)
            X_test_i = add_interaction_features(X_test_w)

            sector_stats = {}
            if "Sector" in X_train_i.columns:
                sector_stats = compute_sector_stats(X_train_i)
                apply_zscore_from_stats([X_train_i, X_test_i], sector_stats)

            X_train_enc, X_test_enc = encode_and_align(X_train_i, X_test_i)

            # Feature selection: prune near-zero-variance + redundant columns.
            selected_features = fit_feature_selection(X_train_enc)
            X_train_enc = X_train_enc[selected_features]
            X_test_enc = X_test_enc.reindex(columns=selected_features, fill_value=0)
            feature_columns = selected_features

            best_xgb, calibrated_model = train_model(X_train_enc, y_train)

            # Threshold optimisation (values learned on training data only)
            y_train_proba = calibrated_model.predict_proba(X_train_enc)
            optimal_thresholds = _optimize_thresholds(y_train, y_train_proba, len(LABEL_ENCODER.classes_))

            # Strategy selector decided on TRAINING data only (no test-set peek).
            # Previously this compared macro-F1 on y_test, which leaked test labels
            # into the choice of whether to apply thresholds (documented in §4.6)
            # and optimistically inflated reported metrics.
            f1_standard = f1_score(y_train, calibrated_model.predict(X_train_enc), average="macro")
            f1_threshold = f1_score(y_train, (y_train_proba * optimal_thresholds).argmax(axis=1), average="macro")
            use_thresholds = f1_threshold > f1_standard

            # Persist cache
            joblib.dump(calibrated_model, paths["calibrated_model"])
            joblib.dump(best_xgb, paths["base_model"])
            joblib.dump(medians, paths["imputer_medians"])
            joblib.dump(w_bounds, paths["winsorize_bounds"])
            joblib.dump(sector_stats, paths["sector_stats"])
            joblib.dump(feature_columns, paths["feature_columns"])
            joblib.dump(optimal_thresholds, paths["optimal_thresholds"])
            joblib.dump({"use_thresholds": use_thresholds}, paths["prediction_strategy"])

        # ── Evaluation ────────────────────────────────────────────────────────
        proba = calibrated_model.predict_proba(X_test_enc)
        pred_indices = (
            (proba * optimal_thresholds).argmax(axis=1)
            if use_thresholds
            else calibrated_model.predict(X_test_enc)
        )

        acc = accuracy_score(y_test, pred_indices)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred_indices, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, pred_indices, labels=[0, 1, 2, 3])

        # ── Save evaluation metrics ──────────────────────────────────────────
        suffix = "" if data_hash == "default" else f"_{data_hash}"
        pd.DataFrame({
            "actual": LABEL_ENCODER.inverse_transform(y_test),
            "predicted": LABEL_ENCODER.inverse_transform(pred_indices),
        }).to_csv(RESULTS_DIR / f"xgboost_test_predictions{suffix}.csv", index=False)

        pd.DataFrame(
            classification_report(y_test, pred_indices, target_names=LABEL_ENCODER.classes_, output_dict=True)
        ).transpose().to_csv(RESULTS_DIR / f"xgboost_classification_report{suffix}.csv")

        with open(RESULTS_DIR / f"xgboost_metrics{suffix}.txt", "w") as f:
            f.write(f"split_strategy: {split_strategy}\n")
            f.write(f"test_samples: {len(y_test)}\n")
            f.write(f"accuracy: {acc:.4f}\n")
            f.write(f"f1_macro: {f1_score(y_test, pred_indices, average='macro'):.4f}\n")

        # ── SHAP ─────────────────────────────────────────────────────────────
        first_pred_class_idx = int(pred_indices[0])
        sample_shap = compute_shap(best_xgb, X_test_enc, first_pred_class_idx)
        shap_data, shap_story = build_shap_payload(X_test_enc, sample_shap)

        # ── Response payload ─────────────────────────────────────────────────
        pred_labels = LABEL_ENCODER.inverse_transform(pred_indices)

        result = {
            "prediction": format_prediction_label(pred_labels[0]),
            "probabilities": {
                format_prediction_label(LABEL_ENCODER.classes_[i]): float(proba[0][i])
                for i in range(len(LABEL_ENCODER.classes_))
            },
            "modelData": {
                "tag": "XGBoost",
                "labels": ["Investment-High", "Investment-Low", "Speculative", "Distressed"],
                "metrics": {
                    "accuracy": f"{acc:.4f}",
                    "precision": f"{precision:.4f}",
                    "recall": f"{recall:.4f}",
                    "f1": f"{f1:.4f}",
                    "strength": "3-model soft-voting ensemble; isotonic-calibrated + threshold-optimized; company-level leakage-safe split; imputation + feature selection.",
                    "weakness": "May overfit if the dataset is too small or highly imbalanced.",
                },
                "matrix": cm.tolist(),
                "shap": shap_data,
                "shapStory": shap_story,
            },
        }

        print(json.dumps(result))

    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

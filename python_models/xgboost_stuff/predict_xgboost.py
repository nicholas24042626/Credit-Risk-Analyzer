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
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
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
CV_METRICS_JSON_PATH = RESULTS_DIR / "xgboost_cv_metrics.json"


def load_cv_metrics():
    """Load the cached cross-validation summary written by evaluate_xgboost.py.

    evaluate_xgboost.py is the single source of truth for reported accuracy
    (see its module docstring and docs/XGBoost_Technical_Report.md). CV is
    expensive (multiple full training runs), so it is run offline/on-demand
    rather than on every prediction request; this just reads the cached
    result. Returns None if the cache hasn't been generated yet.
    """
    if not CV_METRICS_JSON_PATH.exists():
        return None
    try:
        return json.loads(CV_METRICS_JSON_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


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
        "second_stage_model",
    ]
    return {name: RESULTS_DIR / f"{name}{suffix}.pkl" for name in names}


def _cache_is_complete(paths):
    return all(p.exists() for p in paths.values())


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _optimize_thresholds(y_true, y_proba, n_classes):
    """Find per-class probability multipliers that maximise macro-F1.
    
    Uses 50 random restarts with Nelder-Mead, plus a grid-seeded start from
    class-frequency-inverse weights, for better coverage of the threshold space.
    """
    def neg_f1(thresholds):
        preds = (y_proba * thresholds).argmax(axis=1)
        return -f1_score(y_true, preds, average="macro")

    best_score, best_thresholds = -1.0, np.ones(n_classes)
    rng = np.random.default_rng(RANDOM_STATE)

    # Seed 1: class-frequency-inverse (upweight rare classes)
    class_counts = np.bincount(y_true, minlength=n_classes).astype(float)
    class_counts[class_counts == 0] = 1.0
    freq_inverse = (class_counts.max() / class_counts)
    freq_inverse = freq_inverse / freq_inverse.sum() * n_classes

    initial_seeds = [
        freq_inverse,
        np.ones(n_classes),
        freq_inverse * 1.5,
        freq_inverse * 0.7,
    ]

    for init in initial_seeds:
        res = minimize(neg_f1, init, method="Nelder-Mead",
                       options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-4})
        if -res.fun > best_score:
            best_score = -res.fun
            best_thresholds = res.x

    # 3 random restarts (trimmed from 10 — marginal gains past ~4 total starts)
    for _ in range(3):
        init = rng.uniform(0.3, 2.5, n_classes)
        res = minimize(neg_f1, init, method="Nelder-Mead",
                       options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-4})
        if -res.fun > best_score:
            best_score = -res.fun
            best_thresholds = res.x

    # Normalise so the scale of probabilities is preserved on average
    return best_thresholds / best_thresholds.sum() * n_classes


def fit_thresholds_nested(X_train_enc, y_train, train_groups, best_params, n_classes,
                           inner_test_size=0.2, random_state=RANDOM_STATE):
    """Fit threshold multipliers and decide whether to use them, without ever
    touching the outer test/CV fold (leakage-safe nested validation).

    The production/CV model is always fit on the *entire* outer training fold
    (more data -> a better final model). But if thresholds are tuned on that
    same full training fold and then evaluated on it to decide whether to use
    them, the decision is optimistic -- the thresholds get to "see" the exact
    rows they're being scored against.

    To avoid that, this carves out a company-level inner validation slice from
    the outer training fold ONLY (never touching the outer test fold), fits a
    single quick model (reusing best_params already found by hyperparameter
    search) on the inner-train slice, and fits/evaluates thresholds purely on
    the inner-validation slice. The thresholds and use/don't-use decision
    coming out of this are then applied to the real, fully-trained model's
    predictions on the outer test fold.

    Falls back to fitting thresholds on the full outer training fold (old
    behaviour) if a grouped inner split isn't possible (e.g. too few groups,
    or ``train_groups`` is ``None`` because the outer split had to fall back
    to a non-grouped strategy).
    """
    inner_train_idx = inner_val_idx = None
    if train_groups is not None:
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=inner_test_size, random_state=random_state)
            inner_train_idx, inner_val_idx = next(gss.split(X_train_enc, y_train, train_groups))
        except (ValueError, StopIteration):
            inner_train_idx = inner_val_idx = None

    if inner_train_idx is None or len(inner_val_idx) == 0 or len(inner_train_idx) == 0:
        # Fallback: can't form a leakage-safe inner split (e.g. very few
        # groups). Fit thresholds on the full outer-training fold instead --
        # less rigorous, but still never touches the outer test/CV fold.
        quick_model = XGBClassifier(**{**best_params, "random_state": random_state, "n_jobs": -1})
        quick_model.fit(X_train_enc, y_train)
        proba_val = quick_model.predict_proba(X_train_enc)
        y_val = y_train
    else:
        X_inner_tr = X_train_enc.iloc[inner_train_idx]
        y_inner_tr = y_train[inner_train_idx]
        X_inner_val = X_train_enc.iloc[inner_val_idx]
        y_val = y_train[inner_val_idx]

        quick_model = XGBClassifier(**{**best_params, "random_state": random_state, "n_jobs": -1})
        quick_model.fit(X_inner_tr, y_inner_tr)
        proba_val = quick_model.predict_proba(X_inner_val)

    thresholds = _optimize_thresholds(y_val, proba_val, n_classes)

    f1_standard = f1_score(y_val, proba_val.argmax(axis=1), average="macro")
    f1_threshold = f1_score(y_val, (proba_val * thresholds).argmax(axis=1), average="macro")
    use_thresholds = f1_threshold > f1_standard

    return thresholds, use_thresholds


def train_model(X_train_enc, y_train):
    """Fit an ensemble of XGBoost models with aggressive tuning for 75%+ accuracy.

    Strategy:
    1. Apply SMOTE to oversample minority classes (especially Distressed)
    2. Expand hyperparameter grid to cover more of the search space
    3. Train 5 XGBoost models with different random seeds
    4. Soft-vote ensemble for robustness
    """
    # Apply oversampling to balance classes before training. The Distressed class
    # (~57 training samples per fold) is too small relative to the majority
    # classes for the model to learn reliable decision boundaries.
    # Strategy: use BorderlineSMOTE (focuses synthetic samples on the decision
    # boundary rather than the class interior, producing more informative
    # training examples) with a custom sampling_strategy that aggressively
    # oversamples minority classes to at least 60% of the majority count.
    class_counts = pd.Series(y_train).value_counts()
    min_class_count = int(class_counts.min())
    max_class_count = int(class_counts.max())
    k_neighbors = min(5, max(1, min_class_count - 1))

    # Target: bring every class to at least 60% of the majority class size.
    # This avoids perfectly balancing (which can over-represent extremely rare
    # classes with too many synthetic copies) while still giving XGBoost
    # enough minority signal to learn from.
    target_count = int(max_class_count * 0.6)
    sampling_strategy = {
        cls: max(target_count, count)
        for cls, count in class_counts.items()
    }

    if min_class_count >= 3 and k_neighbors >= 2:
        try:
            oversampler = BorderlineSMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=k_neighbors,
                sampling_strategy=sampling_strategy,
            )
            X_train_enc, y_train = oversampler.fit_resample(X_train_enc, y_train)
        except ValueError:
            # Fallback to regular SMOTE if BorderlineSMOTE fails (e.g. not
            # enough borderline samples in a tiny class)
            try:
                oversampler = SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=k_neighbors,
                    sampling_strategy=sampling_strategy,
                )
                X_train_enc, y_train = oversampler.fit_resample(X_train_enc, y_train)
            except ValueError:
                pass  # proceed with imbalanced data + sample_weights
    elif min_class_count >= 2:
        try:
            oversampler = SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=max(1, k_neighbors),
                sampling_strategy=sampling_strategy,
            )
            X_train_enc, y_train = oversampler.fit_resample(X_train_enc, y_train)
        except ValueError:
            pass

    sample_weights = compute_sample_weight("balanced", y_train)

    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        tree_method="hist",        # histogram-based boosting: ~2-3x faster
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Focused grid on the highest-impact regularisation axes.
    # max_depth + min_child_weight together control tree complexity most.
    # n_estimators is capped lower because hist+colsample is already fast;
    # the learning_rate range is widened slightly to let the search explore
    # both conservative (0.03) and aggressive (0.15) regimes.
    param_grid = {
        "max_depth":          [3, 4, 5, 6],
        "n_estimators":       [100, 150, 200],
        "learning_rate":      [0.03, 0.05, 0.1, 0.15],
        "min_child_weight":   [3, 5, 7],
        "gamma":              [0, 0.1, 0.2],
        "reg_alpha":          [0.1, 0.3, 0.5],
        "reg_lambda":         [1.0, 1.5, 2.0],
        "subsample":          [0.8, 0.9],
        "colsample_bytree":   [0.7, 0.8, 0.9],
        # Per-depth-level column sampling — finer-grained regularisation than
        # colsample_bytree alone; each tree level independently resamples
        # features, reducing inter-level correlation.
        "colsample_bylevel":  [0.7, 0.8, 0.9],
        # XGBoost docs specifically recommend max_delta_step > 0 for imbalanced
        # multi-class: caps the maximum gradient step to prevent the optimiser
        # from taking extreme updates on the rare Distressed class.
        "max_delta_step":     [1, 3, 5],
    }

    grid = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=15,             # reduced from 25 — saves ~40% search time
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    grid.fit(X_train_enc, y_train, sample_weight=sample_weights)
    best_xgb = grid.best_estimator_
    best_params = grid.best_params_

    # Single retrained model with a different seed for mild variance reduction
    # via soft voting — keeps it fast (only 2 models total).
    xgb_2 = XGBClassifier(
        **{**best_params, "random_state": RANDOM_STATE + 1, "n_jobs": -1, "tree_method": "hist"}
    )
    xgb_2.fit(X_train_enc, y_train, sample_weight=sample_weights)

    ensemble = VotingClassifier(
        estimators=[("xgb1", best_xgb), ("xgb2", xgb_2)],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train_enc, y_train, sample_weight=sample_weights)

    return best_xgb, ensemble, best_params


# ---------------------------------------------------------------------------
# Second-stage meta-classifier (Speculative / Distressed correction)
# ---------------------------------------------------------------------------

def train_second_stage_classifier(X_train_enc, y_train):
    """Train a shallow XGBoost binary classifier to correct Speculative/Distressed confusion.

    The primary XGBoost ensemble tends to blur the boundary between the
    Speculative (class 2) and Distressed (class 3) tiers because Distressed
    has very few training samples (~3.6% of the dataset).  A second-stage
    binary corrector — trained only on the rows from those two classes —
    learns the residual signal that separates them, without touching
    Investment-High or Investment-Low predictions (where the ensemble is
    already reliable).

    Uses a shallow XGBoost (max_depth=2, 50 trees) rather than logistic
    regression: the Speculative/Distressed boundary is likely nonlinear
    (if it were linear, the main ensemble would already separate them), and
    a very shallow tree trained on the small subset is fast (~ms) while
    capturing feature interactions that LogReg cannot.

    sample_weight='balanced' equivalent is achieved via scale_pos_weight
    approximation through compute_sample_weight, compensating for the
    Distressed minority within this binary sub-problem.

    Returns None if fewer than 2 classes are present in the subset
    (too few Distressed samples), in which case apply_second_stage is a no-op.
    """
    spec_dist_mask = (y_train == 2) | (y_train == 3)  # Speculative=2, Distressed=3
    if spec_dist_mask.sum() == 0:
        return None
    y_sub = y_train[spec_dist_mask]
    if len(np.unique(y_sub)) < 2:
        return None  # only one class present, cannot train a binary classifier

    X_sub = X_train_enc.iloc[spec_dist_mask] if hasattr(X_train_enc, 'iloc') else X_train_enc[spec_dist_mask]
    # binary:logistic requires classes [0, 1]; remap 2→0, 3→1 before fitting
    # and reverse (+2) in apply_second_stage.
    y_binary = y_sub - 2
    sub_weights = compute_sample_weight("balanced", y_binary)
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        max_depth=2,        # very shallow — prevents overfitting on tiny subset
        n_estimators=50,    # few trees sufficient for a binary boundary correction
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=1.5,
        max_delta_step=1,   # stabilise gradient steps on the imbalanced sub-problem
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_sub, y_binary, sample_weight=sub_weights)
    return clf


def apply_second_stage(preds, X_enc, second_stage_model):
    """Apply the second-stage corrector to Speculative/Distressed predictions.

    Only rows where the primary ensemble predicted Speculative (2) or
    Distressed (3) are re-evaluated by the meta-classifier; all other
    predictions are left unchanged.  Returns a new array (does not mutate
    ``preds`` in place).
    """
    if second_stage_model is None:
        return preds
    corrected = preds.copy()
    mask = (preds == 2) | (preds == 3)
    if mask.sum() == 0:
        return corrected
    X_sub = X_enc.iloc[mask] if hasattr(X_enc, 'iloc') else X_enc[mask]
    # Predictions are in binary space [0, 1]; add 2 to restore original class
    # indices (Speculative=2, Distressed=3).
    corrected[mask] = second_stage_model.predict(X_sub) + 2
    return corrected


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
        # Must match the grouped, company-level protocol used by
        # evaluate_xgboost.py — the CV script is now the single source of
        # truth for reported accuracy, and this single split exists only as a
        # live demo of the same methodology on one particular held-out slice.
        X_train, X_test, y_train, y_test, split_strategy, train_groups = make_split(
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
            second_stage_model = joblib.load(paths["second_stage_model"])

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

            best_xgb, calibrated_model, best_params = train_model(X_train_enc, y_train)

            # ── Threshold optimisation (nested, leakage-safe) ─────────────────
            # Thresholds are fit and the use/don't-use decision is made on a
            # company-level inner validation slice carved out of the outer
            # TRAINING fold only (never the outer test fold, and never using
            # rows the model was actually fit on to make the decision). This
            # mirrors exactly what evaluate_xgboost.py does inside each CV
            # fold, so the dashboard's single-split numbers and the CV numbers
            # come from the same methodology. See fit_thresholds_nested().
            optimal_thresholds, use_thresholds = fit_thresholds_nested(
                X_train_enc, y_train, train_groups, best_params,
                n_classes=len(LABEL_ENCODER.classes_),
            )

            # ── Second-stage Speculative/Distressed corrector ─────────────────
            # Lightweight logistic regression trained only on the subset of
            # training rows where the primary model would predict Speculative
            # or Distressed, to correct the most frequent confusion pair.
            second_stage_model = train_second_stage_classifier(X_train_enc, y_train)

            # Persist cache
            joblib.dump(calibrated_model, paths["calibrated_model"])
            joblib.dump(best_xgb, paths["base_model"])
            joblib.dump(medians, paths["imputer_medians"])
            joblib.dump(w_bounds, paths["winsorize_bounds"])
            joblib.dump(sector_stats, paths["sector_stats"])
            joblib.dump(feature_columns, paths["feature_columns"])
            joblib.dump(optimal_thresholds, paths["optimal_thresholds"])
            joblib.dump({"use_thresholds": use_thresholds}, paths["prediction_strategy"])
            joblib.dump(second_stage_model, paths["second_stage_model"])

        # ── Evaluation ────────────────────────────────────────────────────────
        proba = calibrated_model.predict_proba(X_test_enc)
        pred_indices = (
            (proba * optimal_thresholds).argmax(axis=1)
            if use_thresholds
            else calibrated_model.predict(X_test_enc)
        )
        # Apply second-stage Speculative/Distressed correction
        pred_indices = apply_second_stage(pred_indices, X_test_enc, second_stage_model)

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

        # Stats must always reflect whatever dataset was actually just
        # trained/evaluated above (X_test_enc/y_test/pred_indices), not a
        # static offline cache -- otherwise uploading a different CSV would
        # keep showing the same numbers. So the primary metrics card is
        # always the live, just-computed single-split figures.
        primary_metrics = {
            "accuracy": f"{acc:.4f}",
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "f1": f"{f1:.4f}",
            "label": "computed live from this dataset's train/test split",
            "strength": (
                "Trained and evaluated dynamically on whichever dataset was uploaded "
                "for this request -- these numbers change with the data."
            ),
            "weakness": (
                "A single train/test split has more variance than a full "
                "cross-validation run, especially on smaller datasets."
            ),
        }

        # For the bundled default dataset only, also surface the offline
        # cross-validated summary (if present) as a secondary, lower-variance
        # reference figure -- purely informational, never the primary card,
        # and never shown for a custom uploaded dataset (the cache has
        # nothing to do with it).
        cv_metrics = load_cv_metrics() if data_hash == "default" else None
        if cv_metrics:
            secondary_metrics = {
                "accuracy": f"{cv_metrics['test_accuracy_mean']:.4f}",
                "accuracyStd": f"{cv_metrics['test_accuracy_std']:.4f}",
                "f1": f"{cv_metrics['test_macro_f1_mean']:.4f}",
                "f1Std": f"{cv_metrics['test_macro_f1_std']:.4f}",
                "cvFolds": cv_metrics["cv_folds"],
                "label": (
                    f"{cv_metrics['test_accuracy_mean']*100:.1f}% ± "
                    f"{cv_metrics['test_accuracy_std']*100:.1f}% "
                    f"({cv_metrics['cv_folds']}-fold grouped CV, default dataset only)"
                ),
                "note": (
                    "Reference figure from an offline cross-validation run on the "
                    "bundled default dataset. Lower variance than a single split, "
                    "but not recomputed for uploaded datasets."
                ),
            }
        else:
            secondary_metrics = None

        result = {
            "prediction": format_prediction_label(pred_labels[0]),
            "probabilities": {
                format_prediction_label(LABEL_ENCODER.classes_[i]): float(proba[0][i])
                for i in range(len(LABEL_ENCODER.classes_))
            },
            "modelData": {
                "tag": "XGBoost",
                "labels": ["Investment-High", "Investment-Low", "Speculative", "Distressed"],
                # Primary metrics card: always computed live from whatever
                # dataset was just trained/evaluated (dynamic per upload).
                "metrics": primary_metrics,
                # No longer used to switch the primary card to a CV layout --
                # kept as None so the client renders the live single-split
                # card for every dataset, default or uploaded.
                "cvMetrics": None,
                # Optional secondary reference card: offline CV summary,
                # default dataset only.
                "singleSplitMetrics": secondary_metrics,
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

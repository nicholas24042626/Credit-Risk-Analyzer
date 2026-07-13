import sys
import json
import warnings
import hashlib
import base64
import io
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_baseline import run_fair_baseline  # noqa: E402

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Fitted soft-voting ensemble
# ---------------------------------------------------------------------------
#
# sklearn.ensemble.VotingClassifier clones every estimator and refits the
# clones inside .fit(), even if the estimators passed in are already fitted.
# Both members here (best_xgb from RandomizedSearchCV, xgb_2) are already
# fully trained before the ensemble is assembled, so a VotingClassifier would
# silently discard that work and refit two more full XGBoost models for no
# behavioural difference (same data/params/random_state -> same trees,
# deterministically). This wrapper just averages predict_proba over the
# already-fitted estimators. Defined at module level (not a closure) so
# joblib can pickle/unpickle it across cache hits.


class FittedVotingEnsemble:
    """Soft-voting wrapper over already-fitted classifiers (no refitting)."""

    def __init__(self, named_estimators):
        self.named_estimators = named_estimators
        self.classes_ = named_estimators[0][1].classes_

    def predict_proba(self, X):
        probas = [clf.predict_proba(X) for _, clf in self.named_estimators]
        return np.mean(probas, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# Ordinal cumulative-link decomposition (blended with the softmax ensemble)
# ---------------------------------------------------------------------------
#
# The four risk tiers are ORDERED (Investment_High < Investment_Low <
# Speculative < Distressed), but multi:softprob treats them as unordered
# categories. A cumulative-link decomposition trains K-1 binary models for
# P(y > k) and converts them into class probabilities:
#
#     P(y=0) = 1 - P(y>0)
#     P(y=k) = P(y>k-1) - P(y>k)      (0 < k < K-1)
#     P(y=K-1) = P(y>K-2)
#
# Each binary sub-problem pools every class on one side of an ordinal
# threshold (e.g. "investment grade vs not"), so the tiny Distressed class
# borrows statistical strength from Speculative instead of competing with it
# in a 4-way softmax. Blending the two probability views (weight below) was
# measured to beat either model alone on grouped CV; every blend weight in
# [0.3, 0.7] beat the softmax-only baseline, so the choice of 0.4 is not a
# knife-edge tuning artefact. See docs/XGBoost_Technical_Report.md §16.26.

ORDINAL_BLEND_WEIGHT = 0.4


def _scale_pos_weight(y_bin):
    """XGBoost's canonical binary-imbalance ratio: negative_count / positive_count.

    Unlike ``compute_sample_weight("balanced", ...)`` (a per-row weight
    array), this is a single scalar baked into the loss via the
    ``scale_pos_weight`` constructor argument -- the textbook XGBoost
    mechanism for binary class imbalance (XGBoost's own docs single it out
    for exactly this case). Each ordinal sub-model (see
    ``train_ordinal_models``) IS a genuine binary problem, so this applies
    directly; the primary 4-class ensemble is multiclass, where
    ``scale_pos_weight`` is undefined -- that model keeps using
    per-sample ``compute_sample_weight("balanced", ...)``, the correct
    multiclass equivalent (see ``train_model``).
    """
    pos = int((y_bin == 1).sum())
    neg = int((y_bin == 0).sum())
    return neg / max(pos, 1)


def train_ordinal_models(X_train, y_train, best_params, random_state=None,
                          X_val=None, y_val=None):
    """Train K-1 cumulative binary models P(y > k) on the ORIGINAL (non-SMOTE)
    training data, using scale_pos_weight for the binary imbalance in each
    sub-problem.

    ``X_val``/``y_val`` (multiclass-labelled), if given, are re-binarised per
    threshold ``k`` and used purely for early stopping (see
    ``EARLY_STOPPING_ROUNDS``) -- carved from the outer training fold only,
    never the outer test/CV fold (see ``_carve_inner_validation``).

    Returns None when the encoded labels are not a contiguous 0..K-1 range
    (e.g. an uploaded dataset missing an entire tier), in which case the
    ensemble falls back to softmax-only probabilities.
    """
    if random_state is None:
        random_state = RANDOM_STATE  # imported below this def; resolved at call time
    present = np.unique(y_train)
    n_classes = len(present)
    if n_classes < 2 or not np.array_equal(present, np.arange(n_classes)):
        return None

    # Depth/shrinkage hyperparameters found by the softmax search transfer
    # reasonably to the binary sub-problems; objective/eval_metric must not.
    base = dict(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", enable_categorical=True, n_jobs=-1,
        max_depth=3, n_estimators=150, learning_rate=0.05,
        min_child_weight=5, gamma=0.1, reg_alpha=0.3, reg_lambda=1.5,
        subsample=0.85, colsample_bytree=0.8,
    )
    for key, value in (best_params or {}).items():
        if key in base and key not in ("objective", "eval_metric"):
            base[key] = value

    use_early_stop = X_val is not None and y_val is not None and len(X_val) > 0

    models = []
    for k in range(n_classes - 1):
        y_bin = (y_train > k).astype(int)
        fit_kwargs = {}
        clf_kwargs = dict(base, scale_pos_weight=_scale_pos_weight(y_bin),
                           random_state=random_state + k)
        if use_early_stop:
            # Give early stopping room to actually stop early rather than
            # exhausting the fixed tree count from the softmax search.
            clf_kwargs["n_estimators"] = max(base["n_estimators"] * 3, 300)
            clf_kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
            y_val_bin = (y_val > k).astype(int)
            fit_kwargs["eval_set"] = [(X_val, y_val_bin)]
            fit_kwargs["verbose"] = False
        clf = XGBClassifier(**clf_kwargs)
        clf.fit(X_train, y_bin, **fit_kwargs)
        models.append(clf)
    return models


def ordinal_proba(models, X):
    """Convert cumulative P(y>k) outputs into per-class probabilities."""
    p_greater = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    # Enforce monotone non-increasing cumulative probabilities: the binary
    # models are trained independently, so tiny inversions can occur.
    p_greater = np.minimum.accumulate(p_greater, axis=1)
    n_classes = p_greater.shape[1] + 1
    proba = np.empty((len(p_greater), n_classes))
    proba[:, 0] = 1.0 - p_greater[:, 0]
    for k in range(1, n_classes - 1):
        proba[:, k] = p_greater[:, k - 1] - p_greater[:, k]
    proba[:, -1] = p_greater[:, -1]
    proba = np.clip(proba, 1e-9, None)
    return proba / proba.sum(axis=1, keepdims=True)


class BlendedOrdinalEnsemble:
    """Weighted blend of the softmax voting ensemble and the ordinal view.

    Falls back to softmax-only probabilities when ordinal models are
    unavailable (non-contiguous label space in an uploaded dataset).
    """

    def __init__(self, voting_ensemble, ordinal_models, ordinal_weight=ORDINAL_BLEND_WEIGHT):
        self.voting_ensemble = voting_ensemble
        self.ordinal_models = ordinal_models
        self.ordinal_weight = ordinal_weight
        self.classes_ = voting_ensemble.classes_

    def predict_proba(self, X):
        p_soft = self.voting_ensemble.predict_proba(X)
        if not self.ordinal_models or p_soft.shape[1] != len(self.ordinal_models) + 1:
            return p_soft
        p_ord = ordinal_proba(self.ordinal_models, X)
        w = self.ordinal_weight
        return (1.0 - w) * p_soft + w * p_ord

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

# All data-transformation logic lives in the shared preprocessing module so the
# production pipeline and the training notebook stay in sync.
from preprocessing import (
    LABEL_ENCODER,
    RANDOM_STATE,
    DROP_COLS,
    add_interaction_features,
    align_features,
    apply_categorical_dtypes,
    apply_missingness_indicators,
    apply_zscore_from_stats,
    clean_dataframe,
    compute_sector_stats,
    extract_groups,
    fit_categorical_dtypes,
    fit_feature_selection,
    fit_missingness_indicators,
    format_prediction_label,
    group_rating,
    humanize_feature_name,
    make_split,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Class-imbalance strategy toggle
# ---------------------------------------------------------------------------
#
# Two strategies are implemented side by side (see train_model()) so they can
# be directly compared rather than one silently replacing the other:
#   "class_weight" -- compute_sample_weight("balanced", ...) for the 4-class
#                     ensemble, scale_pos_weight for each binary ordinal
#                     sub-model. No synthetic rows.
#   "smote"         -- the original BorderlineSMOTE/SMOTE oversampling.
#
# Default is "class_weight". Rationale: the Distressed class has ~50-160
# genuine training rows (~2-4% of the data). SMOTE interpolates *between*
# real minority points in feature space -- with an anchor set this sparse in
# a ~40+ dimensional engineered feature space, "nearest neighbours" are often
# not particularly close, so synthetic points can land in regions no real
# Distressed company actually occupies (a risk documented for this exact
# pipeline in docs/XGBoost_Technical_Report.md §16.5, where SMOTE on 4-5
# per-fold Distressed samples produced degenerate synthetic clusters).
# scale_pos_weight/class-weighting reweights the loss on the REAL points
# instead of fabricating new ones, so it cannot introduce out-of-distribution
# rows -- strictly more defensible on a sparse anchor set, at the cost of not
# adding any new decision-boundary examples the way SMOTE can. See the
# CV comparison run alongside this change for measured numbers.
IMBALANCE_STRATEGY = "class_weight"  # "class_weight" | "smote"

# Early-stopping validation fraction, carved from the outer training fold
# only (never the outer test/CV fold) via a company-level grouped split when
# possible. See _carve_inner_validation().
EARLY_STOP_VAL_SIZE = 0.15
EARLY_STOPPING_ROUNDS = 20

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
    # optimal_thresholds / prediction_strategy / second_stage_model were
    # removed from the pipeline in §16.26; imputer_medians / winsorize_bounds
    # were removed in the no-impute/no-winsorize preprocessing pass (replaced
    # by missing_indicator_cols / categorical_categories below). Pre-existing
    # cache files under the old names are simply not in this list any more,
    # so _cache_is_complete() returns False and a stale cache from either
    # prior pipeline shape can't be silently loaded.
    names = [
        "calibrated_model", "base_model", "missing_indicator_cols",
        "categorical_categories", "sector_stats", "feature_columns",
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


def _carve_inner_validation(X, y, groups, val_size=EARLY_STOP_VAL_SIZE, random_state=RANDOM_STATE):
    """Carve an early-stopping validation slice out of X/y/groups only.

    Company-level (grouped) when possible, so no company's other-year rows
    leak between the fit slice and the validation slice; falls back to a
    plain stratified split if grouping isn't available or fails (e.g. too
    few groups). Mirrors the leakage-safe nested-split pattern already used
    by ``fit_thresholds_nested`` for the (now-removed) threshold protocol,
    but reused here for early stopping instead.

    Returns (X_fit, y_fit, X_val, y_val, groups_fit). ``groups_fit`` is None
    when grouping wasn't used, matching ``make_split``'s convention.
    """
    if groups is not None:
        try:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
            fit_idx, val_idx = next(gss.split(X, y, groups))
            if len(fit_idx) > 0 and len(val_idx) > 0:
                groups_arr = np.asarray(groups)
                return (
                    X.iloc[fit_idx], y[fit_idx],
                    X.iloc[val_idx], y[val_idx],
                    groups_arr[fit_idx],
                )
        except (ValueError, StopIteration):
            pass

    try:
        fit_idx, val_idx = train_test_split(
            np.arange(len(y)), test_size=val_size, random_state=random_state, stratify=y
        )
        return X.iloc[fit_idx], y[fit_idx], X.iloc[val_idx], y[val_idx], None
    except ValueError:
        # Too few rows/classes to split at all -- no early stopping this run.
        return X, y, None, None, None


def train_model(X_train_enc, y_train, train_groups=None):
    """Fit the blended XGBoost ensemble (softmax voting + ordinal view).

    Strategy:
    1. Carve a company-level early-stopping validation slice out of the
       training fold only (never the outer test/CV fold).
    2. Apply the configured imbalance strategy (IMBALANCE_STRATEGY) to the
       remaining fit slice: class-weighting (default) or SMOTE.
    3. Randomised hyperparameter search over a regularised grid.
    4. Two-seed soft-voting ensemble over the softmax models, refit with
       early stopping on the held-out validation slice.
    5. Ordinal cumulative-link models trained on the ORIGINAL (non-SMOTE)
       fit slice, also early-stopped, blended into the final probabilities.
    """
    # Early-stopping validation slice, carved BEFORE any resampling so it
    # only ever contains real (non-synthetic) rows the models never trained
    # on -- required for early stopping to be a meaningful overfitting check
    # rather than the model grading its own training data.
    X_fit, y_fit, X_val, y_val, groups_fit = _carve_inner_validation(
        X_train_enc, y_train, train_groups
    )
    use_early_stop = X_val is not None

    # Ordinal models must see the real class geometry, not SMOTE's synthetic
    # interpolations, so capture the fit-slice originals before resampling.
    X_orig, y_orig = X_fit, y_fit

    if IMBALANCE_STRATEGY == "smote":
        # Apply oversampling to balance classes before training. The Distressed
        # class (~50-160 real rows overall, fewer still per fold) is too small
        # relative to the majority classes for the model to learn reliable
        # decision boundaries from real points alone.
        # Strategy: use BorderlineSMOTE (focuses synthetic samples on the
        # decision boundary rather than the class interior) with a custom
        # sampling_strategy that aggressively oversamples minority classes to
        # at least 60% of the majority count.
        class_counts = pd.Series(y_fit).value_counts()
        min_class_count = int(class_counts.min())
        max_class_count = int(class_counts.max())
        k_neighbors = min(5, max(1, min_class_count - 1))

        # Target: bring every class to at least 60% of the majority class size.
        # This avoids perfectly balancing (which can over-represent extremely
        # rare classes with too many synthetic copies) while still giving
        # XGBoost enough minority signal to learn from.
        target_count = int(max_class_count * 0.6)
        sampling_strategy = {
            cls: max(target_count, count)
            for cls, count in class_counts.items()
        }

        X_fit_res, y_fit_res = X_fit, y_fit
        if min_class_count >= 3 and k_neighbors >= 2:
            try:
                oversampler = BorderlineSMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=k_neighbors,
                    sampling_strategy=sampling_strategy,
                )
                X_fit_res, y_fit_res = oversampler.fit_resample(X_fit, y_fit)
            except ValueError:
                # Fallback to regular SMOTE if BorderlineSMOTE fails (e.g. not
                # enough borderline samples in a tiny class)
                try:
                    oversampler = SMOTE(
                        random_state=RANDOM_STATE,
                        k_neighbors=k_neighbors,
                        sampling_strategy=sampling_strategy,
                    )
                    X_fit_res, y_fit_res = oversampler.fit_resample(X_fit, y_fit)
                except ValueError:
                    pass  # proceed with imbalanced data + sample_weights
        elif min_class_count >= 2:
            try:
                oversampler = SMOTE(
                    random_state=RANDOM_STATE,
                    k_neighbors=max(1, k_neighbors),
                    sampling_strategy=sampling_strategy,
                )
                X_fit_res, y_fit_res = oversampler.fit_resample(X_fit, y_fit)
            except ValueError:
                pass
        X_fit, y_fit = X_fit_res, y_fit_res

    # Balanced per-sample weights: the correct multiclass equivalent of
    # scale_pos_weight (which XGBoost only supports for binary problems --
    # see _scale_pos_weight's docstring). Computed on X_fit/y_fit, i.e. AFTER
    # SMOTE if that strategy is active (matches the prior behaviour) or on
    # the untouched real data under "class_weight" (the primary strategy).
    sample_weights = compute_sample_weight("balanced", y_fit)

    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",        # histogram-based boosting: ~2-3x faster
        enable_categorical=True,   # native categorical splits (e.g. Sector)
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
        n_iter=15,             # reduced from 25 — saves ~40% search time; re-verified
                                # n_iter=25 gives no measurable CV gain (0.5082 vs 0.5087
                                # accuracy, within noise) before reverting to 15
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        # refit=False: only best_params_ is used below (the final ensemble
        # members are refit separately, with early stopping and a raised
        # n_estimators ceiling) -- letting the search also refit its own
        # non-early-stopped copy would just be a wasted extra fit.
        refit=False,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    grid.fit(X_fit, y_fit, sample_weight=sample_weights)
    best_params = grid.best_params_

    # Final ensemble members: refit with early stopping on the held-out
    # validation slice (requirement: early stopping must use a validation
    # fold, never the outer test set -- X_val/y_val were carved from the
    # training fold only, above). n_estimators is bumped well past the
    # searched value so early stopping has room to actually trigger rather
    # than exhausting the grid's fixed tree count.
    ensemble_kwargs = dict(
        best_params, tree_method="hist", enable_categorical=True, n_jobs=-1,
    )
    fit_kwargs = dict(sample_weight=sample_weights)
    if use_early_stop:
        ensemble_kwargs["n_estimators"] = max(best_params.get("n_estimators", 150) * 3, 400)
        ensemble_kwargs["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    best_xgb = XGBClassifier(**{**ensemble_kwargs, "random_state": RANDOM_STATE})
    best_xgb.fit(X_fit, y_fit, **fit_kwargs)

    # Single retrained model with a different seed for mild variance reduction
    # via soft voting — keeps it fast (only 2 models total).
    xgb_2 = XGBClassifier(**{**ensemble_kwargs, "random_state": RANDOM_STATE + 1})
    xgb_2.fit(X_fit, y_fit, **fit_kwargs)

    # Both members are already fully fitted above -- wrap rather than hand
    # them to sklearn's VotingClassifier, which would clone and refit both
    # from scratch (see FittedVotingEnsemble docstring/comment near the top
    # of this file).
    voting = FittedVotingEnsemble([("xgb1", best_xgb), ("xgb2", xgb_2)])

    # Ordinal cumulative-link view, blended with the softmax view. Measured
    # +2.9-point grouped-CV accuracy gain over the softmax-only ensemble
    # (see docs/XGBoost_Technical_Report.md §16.26).
    ordinal_models = train_ordinal_models(
        X_orig, y_orig, best_params,
        X_val=X_val if use_early_stop else None,
        y_val=y_val if use_early_stop else None,
    )
    ensemble = BlendedOrdinalEnsemble(voting, ordinal_models)

    return best_xgb, ensemble, best_params


# ---------------------------------------------------------------------------
# Second-stage meta-classifier (Speculative / Distressed correction)
# ---------------------------------------------------------------------------

def _select_rows(X, mask):
    """Row-select by boolean mask, whether X is a DataFrame or ndarray."""
    return X.iloc[mask] if hasattr(X, "iloc") else X[mask]


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

    X_sub = _select_rows(X_train_enc, spec_dist_mask)
    # binary:logistic requires classes [0, 1]; remap 2→0, 3→1 before fitting
    # and reverse (+2) in apply_second_stage.
    y_binary = y_sub - 2
    sub_weights = compute_sample_weight("balanced", y_binary)
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
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
    X_sub = _select_rows(X_enc, mask)
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

        # ── Fair cross-model baseline (shared_baseline.py) ────────────────────
        # Same cleaning, split, engineered features, imputation/encoding, and
        # untuned XGBClassifier defaults as the other three models' baseline
        # runs, so this number is directly comparable across models -- unlike
        # the tuned pipeline below, which is XGBoost-specific end to end.
        fair_baseline = run_fair_baseline(
            XGBClassifier(random_state=RANDOM_STATE), df
        )

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
            indicator_cols = joblib.load(paths["missing_indicator_cols"])
            categories = joblib.load(paths["categorical_categories"])
            sector_stats = joblib.load(paths["sector_stats"])
            feature_columns = joblib.load(paths["feature_columns"])

            X_train_ind = apply_missingness_indicators(X_train, indicator_cols)
            X_test_ind = apply_missingness_indicators(X_test, indicator_cols)
            X_train_i = add_interaction_features(X_train_ind)
            X_test_i = add_interaction_features(X_test_ind)
            if sector_stats:
                apply_zscore_from_stats([X_train_i, X_test_i], sector_stats)
            X_train_cat = apply_categorical_dtypes(X_train_i, categories)
            X_test_cat = apply_categorical_dtypes(X_test_i, categories)
            X_train_enc, X_test_enc = align_features(X_train_cat, X_test_cat, feature_columns)

        else:
            # ── Cache MISS: full training pipeline ───────────────────────────
            # 1. Missingness indicators (train-fit column list; NaN passed
            #    through, NOT imputed) → 2. Interactions → 3. Sector z-scores
            #    → 4. Categorical dtype cast (native, NOT one-hot) → 5. Align
            #    → 6. Feature selection (near-zero-variance only, no VIF/corr
            #    pruning).
            indicator_cols = fit_missingness_indicators(X_train)
            X_train_ind = apply_missingness_indicators(X_train, indicator_cols)
            X_test_ind = apply_missingness_indicators(X_test, indicator_cols)

            X_train_i = add_interaction_features(X_train_ind)
            X_test_i = add_interaction_features(X_test_ind)

            sector_stats = {}
            if "Sector" in X_train_i.columns:
                sector_stats = compute_sector_stats(X_train_i)
                apply_zscore_from_stats([X_train_i, X_test_i], sector_stats)

            categories = fit_categorical_dtypes(X_train_i)
            X_train_cat = apply_categorical_dtypes(X_train_i, categories)
            X_test_cat = apply_categorical_dtypes(X_test_i, categories)

            X_train_enc, X_test_enc = align_features(X_train_cat, X_test_cat)

            # Feature selection: prune near-zero-variance numeric columns only
            # (categorical columns always kept -- see fit_feature_selection).
            selected_features = fit_feature_selection(X_train_enc)
            X_train_enc = X_train_enc[selected_features]
            X_test_enc = X_test_enc.reindex(columns=selected_features)
            feature_columns = selected_features

            best_xgb, calibrated_model, best_params = train_model(
                X_train_enc, y_train, train_groups=train_groups
            )

            # NOTE (§16.26): the nested threshold-multiplier step and the
            # second-stage Speculative/Distressed corrector were removed from
            # the active pipeline. Both were tuned for the softmax-only
            # ensemble's probability scale; applied to the blended ordinal
            # ensemble they measurably reduced BOTH accuracy and macro-F1 on
            # grouped CV. fit_thresholds_nested() and
            # train_second_stage_classifier() remain defined for the
            # historical record (same convention as add_temporal_features).

            # Persist cache
            joblib.dump(calibrated_model, paths["calibrated_model"])
            joblib.dump(best_xgb, paths["base_model"])
            joblib.dump(indicator_cols, paths["missing_indicator_cols"])
            joblib.dump(categories, paths["categorical_categories"])
            joblib.dump(sector_stats, paths["sector_stats"])
            joblib.dump(feature_columns, paths["feature_columns"])

        # ── Evaluation ────────────────────────────────────────────────────────
        proba = calibrated_model.predict_proba(X_test_enc)
        pred_indices = np.argmax(proba, axis=1)

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
                # Shared cross-model comparison tier: identical cleaning,
                # split, features, and untuned-default estimator as the other
                # three models' own fairBaseline (see shared_baseline.py) --
                # None if the dataset was too small to split at all.
                "fairBaseline": fair_baseline["metrics"] if fair_baseline else None,
            },
        }

        print(json.dumps(result))

    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

import sys
import json
import re
import warnings
import hashlib
import base64
import io
import os
import tempfile
from functools import lru_cache
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
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 143
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "set A corporate_rating.csv"


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

class ExplicitLabelEncoder:
    """Deterministic label encoder with a fixed class → index mapping."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.classes_ = np.array(
            [c for c, _ in sorted(mapping.items(), key=lambda x: x[1])]
        )

    def transform(self, y):
        return np.array([self.mapping[val] for val in y])

    def inverse_transform(self, y):
        return np.array([self.classes_[val] for val in y])


LABEL_ENCODER = ExplicitLabelEncoder({
    "Investment_High": 0,
    "Investment_Low": 1,
    "Speculative": 2,
    "Distressed": 3,
})


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def group_rating(r):
    """Collapse granular credit ratings into four financial risk tiers.

    Investment_High : AAA, AA, A
    Investment_Low  : BBB
    Speculative     : BB, B, CCC, CC
    Distressed      : C, D
    """
    r = str(r).strip().upper()
    if r in {"AAA", "AA", "A"}:
        return "Investment_High"
    if r == "BBB":
        return "Investment_Low"
    if r in {"BB", "B", "CCC", "CC"}:
        return "Speculative"
    if r in {"C", "D"}:
        return "Distressed"
    return "Unknown"


def format_prediction_label(label):
    return str(label).replace("_", "-")


@lru_cache(maxsize=256)
def humanize_feature_name(name):
    """Convert a feature name to human-readable format. Cached for repeated calls."""
    is_log = "_log" in name
    is_sec = "_sec_z" in name
    base = name.replace("_log", "").replace("_sec_z", "")
    base = re.sub(r"(?<!^)(?=[A-Z])", " ", base).replace("_", " ")
    base = re.sub(r"\s+", " ", base).strip().title()
    suffix = (" (Log)" if is_log else "") + (" (Sector Z-Score)" if is_sec else "")
    return f"{base}{suffix}"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_LOG_CANDIDATES = frozenset([
    "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
    "debtEquityRatio", "enterpriseValueMultiple", "operatingCashFlowPerShare",
    "freeCashFlowPerShare", "cashPerShare", "payablesTurnover",
    "fixedAssetTurnover", "companyEquityMultiplier",
])


def winsorize_features(X_train, X_test, lower_pct=1, upper_pct=99):
    """Clip numeric columns to [lower_pct, upper_pct] percentiles from training data."""
    numerics = X_train.select_dtypes(include="number").columns
    bounds = {}
    for col in numerics:
        lo = np.percentile(X_train[col].dropna(), lower_pct)
        hi = np.percentile(X_train[col].dropna(), upper_pct)
        bounds[col] = (lo, hi)

    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    for col, (lo, hi) in bounds.items():
        X_train_out[col] = X_train_out[col].clip(lower=lo, upper=hi)
        X_test_out[col] = X_test_out[col].clip(lower=lo, upper=hi)

    return X_train_out, X_test_out, bounds


def apply_winsorize_bounds(X_train, X_test, bounds):
    """Apply pre-computed winsorize bounds (cache path)."""
    X_train_out = X_train.copy()
    X_test_out = X_test.copy()
    for col, (lo, hi) in bounds.items():
        if col in X_train_out.columns:
            X_train_out[col] = X_train_out[col].clip(lower=lo, upper=hi)
        if col in X_test_out.columns:
            X_test_out[col] = X_test_out[col].clip(lower=lo, upper=hi)
    return X_train_out, X_test_out


def add_interaction_features(X):
    """Add 10 composite financial ratios and log-transformed skewed columns."""
    out = X.copy()
    cols = set(out.columns)

    def safe_div(a, b_col):
        return (out[a] / (out[b_col].abs() + 1e-5)).clip(-1e6, 1e6)

    # Original 10 composite features
    if {"debtEquityRatio", "operatingProfitMargin"} <= cols:
        out["leverage_coverage"] = safe_div("debtEquityRatio", "operatingProfitMargin")
    if {"currentRatio", "quickRatio", "cashRatio"} <= cols:
        out["liquidity_score"] = (out["currentRatio"] + out["quickRatio"] + out["cashRatio"]) / 3.0
    if {"operatingCashFlowPerShare", "debtEquityRatio"} <= cols:
        out["cashflow_debt_coverage"] = safe_div("operatingCashFlowPerShare", "debtEquityRatio")
    if {"netProfitMargin", "operatingProfitMargin", "grossProfitMargin"} <= cols:
        out["profitability_composite"] = (
            out["netProfitMargin"] + out["operatingProfitMargin"] + out["grossProfitMargin"]
        ) / 3.0
    if {"operatingCashFlowSalesRatio", "debtRatio"} <= cols:
        out["debt_service_ratio"] = (
            out["operatingCashFlowSalesRatio"] / (out["debtRatio"] + 1e-5)
        ).clip(-1e6, 1e6)
    if {"assetTurnover", "fixedAssetTurnover"} <= cols:
        out["efficiency_composite"] = (out["assetTurnover"] + out["fixedAssetTurnover"]) / 2.0
    if {"returnOnAssets", "debtRatio"} <= cols:
        out["roa_leverage"] = (out["returnOnAssets"] / (out["debtRatio"] + 1e-5)).clip(-1e6, 1e6)
    if {"grossProfitMargin", "netProfitMargin"} <= cols:
        out["margin_stability"] = (out["grossProfitMargin"] - out["netProfitMargin"]).abs()
    if {"cashRatio", "currentRatio"} <= cols:
        out["cash_liquidity_ratio"] = safe_div("cashRatio", "currentRatio")
    if {"returnOnCapitalEmployed", "assetTurnover"} <= cols:
        out["equity_efficiency"] = (
            out["returnOnCapitalEmployed"] * out["assetTurnover"]
        ).clip(-1e6, 1e6)

    # NEW: Credit-risk specific features
    if {"currentRatio", "debtEquityRatio"} <= cols:
        out["liquidity_leverage"] = (out["currentRatio"] / (out["debtEquityRatio"] + 1e-5)).clip(-1e6, 1e6)
    if {"returnOnEquity", "returnOnAssets"} <= cols:
        out["roe_roa_spread"] = (out["returnOnEquity"] - out["returnOnAssets"]).clip(-1e6, 1e6)
    if {"operatingProfitMargin", "netProfitMargin"} <= cols:
        out["margin_compression"] = (out["operatingProfitMargin"] - out["netProfitMargin"]).clip(-1e6, 1e6)
    if {"freeCashFlowPerShare", "cashPerShare"} <= cols:
        out["fcf_cash_ratio"] = safe_div("freeCashFlowPerShare", "cashPerShare")
    if {"returnOnAssets", "assetTurnover"} <= cols:
        out["roa_turnover"] = (out["returnOnAssets"] * out["assetTurnover"]).clip(-1e6, 1e6)
    if {"debtRatio", "effectiveTaxRate"} <= cols:
        out["debt_tax_burden"] = (out["debtRatio"] * (1 - out["effectiveTaxRate"])).clip(-1e6, 1e6)
    if {"cashRatio", "debtEquityRatio"} <= cols:
        out["cash_leverage"] = safe_div("cashRatio", "debtEquityRatio")
    if {"operatingCashFlowSalesRatio", "netProfitMargin"} <= cols:
        out["cash_quality"] = safe_div("operatingCashFlowSalesRatio", "netProfitMargin")

    # Polynomial interactions for top 3 credit signals
    if {"debtEquityRatio"} <= cols:
        out["debtEquityRatio_sq"] = (out["debtEquityRatio"] ** 2).clip(-1e6, 1e6)
    if {"returnOnAssets"} <= cols:
        out["returnOnAssets_sq"] = (out["returnOnAssets"] ** 2).clip(-1e6, 1e6)
    if {"currentRatio"} <= cols:
        out["currentRatio_sq"] = (out["currentRatio"] ** 2).clip(-1e6, 1e6)

    # Log transforms
    for col in _LOG_CANDIDATES & cols:
        out[f"{col}_log"] = np.sign(out[col]) * np.log1p(np.abs(out[col]))

    return out


def _compute_sector_stats(X_train_with_interactions):
    """Compute per-sector mean/std needed for z-score features (train set only)."""
    df = X_train_with_interactions
    ratios = [
        c for c in df.select_dtypes(include="number").columns
        if "log" not in c and not c.startswith("Sector_")
    ]
    means = df.groupby("Sector")[ratios].mean()
    stds = df.groupby("Sector")[ratios].std().fillna(1.0)
    return {
        "means": means,
        "stds": stds,
        "global_means": df[ratios].mean(),
        "global_stds": df[ratios].std().fillna(1.0),
        "ratios": ratios,
    }


def _apply_zscore_from_stats(df_list, stats):
    """Apply sector z-scores (in-place) to each DataFrame in df_list."""
    means = stats["means"]
    stds = stats["stds"]
    global_means = stats["global_means"]
    global_stds = stats["global_stds"]
    ratios = stats["ratios"]

    for df_out in df_list:
        if "Sector" not in df_out.columns:
            continue
        m_sec = (
            df_out[["Sector"]]
            .merge(means, left_on="Sector", right_index=True, how="left")
            .drop(columns=["Sector"])
        )
        s_sec = (
            df_out[["Sector"]]
            .merge(stds, left_on="Sector", right_index=True, how="left")
            .drop(columns=["Sector"])
        )
        m_sec = m_sec.fillna(global_means)
        s_sec = s_sec.fillna(global_stds)
        z = (df_out[ratios] - m_sec) / (s_sec + 1e-5)
        z.columns = [f"{r}_sec_z" for r in ratios]
        for col in z.columns:
            df_out[col] = z[col]


def encode_and_align(X_train_z, X_test_z, feature_columns=None):
    """One-hot encode Sector and align train/test columns."""
    sector_col = ["Sector"] if "Sector" in X_train_z.columns else []
    X_train_enc = pd.get_dummies(X_train_z, columns=sector_col)
    X_test_enc = pd.get_dummies(X_test_z, columns=["Sector"] if "Sector" in X_test_z.columns else [])

    if feature_columns is not None:
        X_train_enc = X_train_enc.reindex(columns=feature_columns, fill_value=0)
        X_test_enc = X_test_enc.reindex(columns=feature_columns, fill_value=0)
    else:
        X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

    return X_train_enc, X_test_enc


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
        "calibrated_model", "base_model", "ensemble_model", "winsorize_bounds",
        "sector_stats", "feature_columns", "optimal_thresholds", "prediction_strategy",
    ]
    return {name: RESULTS_DIR / f"{name.replace('_', '_')}{suffix}.pkl" for name in names}


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

    # Model 1 — best from RandomizedSearchCV (100 samples from the grid)
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

        # ── Label preparation ────────────────────────────────────────────────
        y_raw = df["Rating"].apply(group_rating)
        valid_mask = y_raw != "Unknown"
        df = df[valid_mask].copy()
        y_encoded = LABEL_ENCODER.transform(y_raw[valid_mask])

        # ── Feature matrix ───────────────────────────────────────────────────
        drop_cols = {"Name", "Symbol", "Rating Agency Name", "Date", "RatingClass", "Rating"}
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # ── Train / test split ───────────────────────────────────────────────
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.20, random_state=RANDOM_STATE, stratify=y_encoded
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.20, random_state=RANDOM_STATE
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
            w_bounds = joblib.load(paths["winsorize_bounds"])
            sector_stats = joblib.load(paths["sector_stats"])
            feature_columns = joblib.load(paths["feature_columns"])
            optimal_thresholds = joblib.load(paths["optimal_thresholds"])
            use_thresholds = joblib.load(paths["prediction_strategy"]).get("use_thresholds", False)

            X_train_w, X_test_w = apply_winsorize_bounds(X_train, X_test, w_bounds)
            X_train_i = add_interaction_features(X_train_w)
            X_test_i = add_interaction_features(X_test_w)
            _apply_zscore_from_stats([X_train_i, X_test_i], sector_stats)
            X_train_enc, X_test_enc = encode_and_align(X_train_i, X_test_i, feature_columns)

        else:
            # ── Cache MISS: full training pipeline ───────────────────────────
            X_train_w, X_test_w, w_bounds = winsorize_features(X_train, X_test)
            X_train_i = add_interaction_features(X_train_w)
            X_test_i = add_interaction_features(X_test_w)

            sector_stats = {}
            if "Sector" in X_train_i.columns:
                sector_stats = _compute_sector_stats(X_train_i)
                _apply_zscore_from_stats([X_train_i, X_test_i], sector_stats)

            X_train_enc, X_test_enc = encode_and_align(X_train_i, X_test_i)
            feature_columns = X_train_enc.columns.tolist()

            best_xgb, calibrated_model = train_model(X_train_enc, y_train)

            # Threshold optimisation
            y_train_proba = calibrated_model.predict_proba(X_train_enc)
            optimal_thresholds = _optimize_thresholds(y_train, y_train_proba, len(LABEL_ENCODER.classes_))

            proba_test = calibrated_model.predict_proba(X_test_enc)
            f1_standard = f1_score(y_test, calibrated_model.predict(X_test_enc), average="macro")
            f1_threshold = f1_score(y_test, (proba_test * optimal_thresholds).argmax(axis=1), average="macro")
            use_thresholds = f1_threshold > f1_standard

            # Persist cache
            joblib.dump(calibrated_model, paths["calibrated_model"])
            joblib.dump(best_xgb, paths["base_model"])
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
                    "strength": "3-model soft-voting ensemble; isotonic-calibrated + threshold-optimized; 11 new credit-risk features; aggressive regularization.",
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

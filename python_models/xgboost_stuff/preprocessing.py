"""Shared preprocessing for the XGBoost credit-risk pipeline.

This module centralises every data-transformation step so the production
inference script (``predict_xgboost.py``) and the training notebook can import
the exact same logic instead of duplicating it (which is how the two drift out
of sync).

Pipeline order — MUST be preserved:

    grouped stratified split          (leakage-safe, company-level)
        -> median imputation          (fit on train only)
        -> winsorization (P1-P99)     (fit on train only)
        -> interaction / log features (deterministic formulas)
        -> sector z-scores            (fit on train only)
        -> one-hot encode + align     (all categoricals, not just Sector)
        -> feature selection          (fit on train only)

Every fitted parameter (medians, winsorize bounds, sector stats, feature
columns, selected features) is computed exclusively on the training fold and
then applied identically to the test fold and to inference inputs. This is what
keeps the pipeline free of preprocessing/feature-engineering leakage.
"""

import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# ---------------------------------------------------------------------------
# Named constants (previously scattered magic numbers)
# ---------------------------------------------------------------------------

RANDOM_STATE = 143

# Small additive constant that prevents division-by-zero while remaining
# negligible relative to financial-ratio scales.
EPS = 1e-5

# Symmetric clip applied to engineered ratios to prevent numeric overflow from
# division by near-zero denominators.
CLIP_BOUND = 1e6

# Winsorization percentiles.
WINSOR_LOWER_PCT = 1
WINSOR_UPPER_PCT = 99

# Feature-selection thresholds.
NEAR_ZERO_VAR_THRESHOLD = 1e-8   # drop columns with essentially no variance
HIGH_CORR_THRESHOLD = 0.95       # drop one of any pair correlated above this

# Number of folds used to carve out the held-out test set (1/N ~= test size).
SPLIT_N_FOLDS = 5

# Columns never used as predictors.
DROP_COLS = frozenset(
    {"Name", "Symbol", "Rating Agency Name", "Date", "RatingClass", "Rating"}
)

# Candidate identifier columns for company-level grouping, in priority order.
GROUP_ID_CANDIDATES = ("Symbol", "Name")

_LOG_CANDIDATES = frozenset([
    "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
    "debtEquityRatio", "enterpriseValueMultiple", "operatingCashFlowPerShare",
    "freeCashFlowPerShare", "cashPerShare", "payablesTurnover",
    "fixedAssetTurnover", "companyEquityMultiplier",
])


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

class ExplicitLabelEncoder:
    """Deterministic label encoder with a fixed class -> index mapping."""

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
# Label / name helpers
# ---------------------------------------------------------------------------

def group_rating(r):
    """Collapse granular credit ratings into four financial risk tiers.

    Investment_High : AAA, AA, A
    Investment_Low  : BBB
    Speculative     : BB, B
    Distressed      : CCC, CC, C, D
    """
    r = str(r).strip().upper()
    if r in {"AAA", "AA", "A"}:
        return "Investment_High"
    if r == "BBB":
        return "Investment_Low"
    if r in {"BB", "B"}:
        return "Speculative"
    if r in {"CCC", "CC", "C", "D"}:
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
# Train / test split (company-level, leakage-safe)
# ---------------------------------------------------------------------------

def make_split(X, y, groups=None, test_size=0.20, random_state=RANDOM_STATE):
    """Split into train/test, preferring a group-aware stratified split.

    When ``groups`` is provided (e.g. company symbol), all records for a given
    company are kept entirely within one side of the split. This prevents
    company-level leakage, where the model would otherwise see other year
    records of a company that also appears in the test set.

    Falls back to stratified ``train_test_split`` when groups are unavailable
    or produce too few members to split, and finally to a plain split.
    """
    if groups is not None:
        try:
            n_folds = max(2, min(SPLIT_N_FOLDS, round(1.0 / test_size)))
            sgkf = StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
            train_idx, test_idx = next(sgkf.split(X, y, groups))
            return (
                X.iloc[train_idx], X.iloc[test_idx],
                y[train_idx], y[test_idx],
                "grouped_stratified",
            )
        except Exception:
            pass

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, "stratified"
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, "random"


# ---------------------------------------------------------------------------
# Missing-value imputation (fit on train only)
# ---------------------------------------------------------------------------

def fit_imputation(X_train):
    """Return per-column medians for numeric features (training set only)."""
    numerics = X_train.select_dtypes(include="number").columns
    medians = X_train[numerics].median()
    # Columns that are entirely NaN have no median; default them to 0.0 so the
    # downstream transforms remain well-defined.
    return medians.fillna(0.0).to_dict()


def apply_imputation(X, medians):
    """Fill numeric NaNs using pre-computed training medians."""
    X_out = X.copy()
    for col, value in medians.items():
        if col in X_out.columns:
            X_out[col] = X_out[col].fillna(value)
    return X_out


# ---------------------------------------------------------------------------
# Winsorization (outlier capping)
# ---------------------------------------------------------------------------

def winsorize_features(X_train, X_test, lower_pct=WINSOR_LOWER_PCT, upper_pct=WINSOR_UPPER_PCT):
    """Clip numeric columns to [lower_pct, upper_pct] percentiles from training data.

    Columns that are entirely NaN (no finite values to compute a percentile
    from) are skipped rather than raising.
    """
    numerics = X_train.select_dtypes(include="number").columns
    bounds = {}
    for col in numerics:
        finite = X_train[col].dropna()
        if finite.empty:
            # All-NaN column: nothing to clip against, skip safely.
            continue
        lo = np.percentile(finite, lower_pct)
        hi = np.percentile(finite, upper_pct)
        bounds[col] = (lo, hi)

    X_train_out, X_test_out = apply_winsorize_bounds(X_train, X_test, bounds)
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


# ---------------------------------------------------------------------------
# Interaction / log features
# ---------------------------------------------------------------------------

def add_interaction_features(X):
    """Add composite financial ratios, polynomial terms, and log transforms."""
    out = X.copy()
    cols = set(out.columns)

    def safe_div(a, b_col):
        return (out[a] / (out[b_col].abs() + EPS)).clip(-CLIP_BOUND, CLIP_BOUND)

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
            out["operatingCashFlowSalesRatio"] / (out["debtRatio"] + EPS)
        ).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"assetTurnover", "fixedAssetTurnover"} <= cols:
        out["efficiency_composite"] = (out["assetTurnover"] + out["fixedAssetTurnover"]) / 2.0
    if {"returnOnAssets", "debtRatio"} <= cols:
        out["roa_leverage"] = (out["returnOnAssets"] / (out["debtRatio"] + EPS)).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"grossProfitMargin", "netProfitMargin"} <= cols:
        out["margin_stability"] = (out["grossProfitMargin"] - out["netProfitMargin"]).abs()
    if {"cashRatio", "currentRatio"} <= cols:
        out["cash_liquidity_ratio"] = safe_div("cashRatio", "currentRatio")
    if {"returnOnCapitalEmployed", "assetTurnover"} <= cols:
        out["equity_efficiency"] = (
            out["returnOnCapitalEmployed"] * out["assetTurnover"]
        ).clip(-CLIP_BOUND, CLIP_BOUND)

    # Credit-risk specific features
    if {"currentRatio", "debtEquityRatio"} <= cols:
        out["liquidity_leverage"] = (out["currentRatio"] / (out["debtEquityRatio"] + EPS)).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"returnOnEquity", "returnOnAssets"} <= cols:
        out["roe_roa_spread"] = (out["returnOnEquity"] - out["returnOnAssets"]).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"operatingProfitMargin", "netProfitMargin"} <= cols:
        out["margin_compression"] = (out["operatingProfitMargin"] - out["netProfitMargin"]).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"freeCashFlowPerShare", "cashPerShare"} <= cols:
        out["fcf_cash_ratio"] = safe_div("freeCashFlowPerShare", "cashPerShare")
    if {"returnOnAssets", "assetTurnover"} <= cols:
        out["roa_turnover"] = (out["returnOnAssets"] * out["assetTurnover"]).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"debtRatio", "effectiveTaxRate"} <= cols:
        out["debt_tax_burden"] = (out["debtRatio"] * (1 - out["effectiveTaxRate"])).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"cashRatio", "debtEquityRatio"} <= cols:
        out["cash_leverage"] = safe_div("cashRatio", "debtEquityRatio")
    if {"operatingCashFlowSalesRatio", "netProfitMargin"} <= cols:
        out["cash_quality"] = safe_div("operatingCashFlowSalesRatio", "netProfitMargin")

    # Polynomial interactions for top 3 credit signals
    if {"debtEquityRatio"} <= cols:
        out["debtEquityRatio_sq"] = (out["debtEquityRatio"] ** 2).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"returnOnAssets"} <= cols:
        out["returnOnAssets_sq"] = (out["returnOnAssets"] ** 2).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"currentRatio"} <= cols:
        out["currentRatio_sq"] = (out["currentRatio"] ** 2).clip(-CLIP_BOUND, CLIP_BOUND)

    # Log transforms (sign-preserving)
    for col in _LOG_CANDIDATES & cols:
        out[f"{col}_log"] = np.sign(out[col]) * np.log1p(np.abs(out[col]))

    return out


# ---------------------------------------------------------------------------
# Sector-relative z-scores (fit on train only)
# ---------------------------------------------------------------------------

def compute_sector_stats(X_train_with_interactions):
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


def apply_zscore_from_stats(df_list, stats):
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
        z = (df_out[ratios] - m_sec) / (s_sec + EPS)
        z.columns = [f"{r}_sec_z" for r in ratios]
        for col in z.columns:
            df_out[col] = z[col]


# ---------------------------------------------------------------------------
# Encoding + alignment (all categoricals, not just Sector)
# ---------------------------------------------------------------------------

def encode_and_align(X_train_z, X_test_z, feature_columns=None):
    """One-hot encode every categorical column and align train/test columns.

    Previously only ``Sector`` was encoded, so any other string/object column
    that survived the drop step would leak into the model matrix as a raw
    object dtype and break XGBoost. This detects all object/category columns
    dynamically and encodes them generically.
    """
    train_cats = X_train_z.select_dtypes(include=["object", "category"]).columns.tolist()
    test_cats = X_test_z.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train_enc = pd.get_dummies(X_train_z, columns=train_cats)
    X_test_enc = pd.get_dummies(X_test_z, columns=test_cats)

    if feature_columns is not None:
        X_train_enc = X_train_enc.reindex(columns=feature_columns, fill_value=0)
        X_test_enc = X_test_enc.reindex(columns=feature_columns, fill_value=0)
    else:
        X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

    return X_train_enc, X_test_enc


# ---------------------------------------------------------------------------
# Feature selection (fit on train only)
# ---------------------------------------------------------------------------

def fit_feature_selection(
    X_train_enc,
    var_threshold=NEAR_ZERO_VAR_THRESHOLD,
    corr_threshold=HIGH_CORR_THRESHOLD,
):
    """Return the list of columns to keep after pruning the feature space.

    1. Drop near-zero-variance columns (no discriminative signal).
    2. Drop one column from every highly-correlated pair (|r| > corr_threshold)
       to reduce redundancy and overfitting risk on the ~130-feature space.

    Selection statistics are computed on the training matrix only. Falls back
    to the full column set if pruning would remove everything.
    """
    columns = list(X_train_enc.columns)
    if not columns:
        return columns

    # 1. Near-zero variance
    variances = X_train_enc.var(axis=0, numeric_only=True)
    keep = [c for c in columns if float(variances.get(c, 0.0)) > var_threshold]
    if not keep:
        return columns

    # 2. High correlation — drop the later column of each correlated pair
    corr = X_train_enc[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = {col for col in upper.columns if (upper[col] > corr_threshold).any()}
    selected = [c for c in keep if c not in to_drop]

    return selected or keep


def extract_groups(df):
    """Return a company-level grouping array from the first available ID column."""
    for id_col in GROUP_ID_CANDIDATES:
        if id_col in df.columns:
            return df[id_col].astype(str).values
    return None

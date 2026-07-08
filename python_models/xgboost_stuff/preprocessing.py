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
HIGH_CORR_THRESHOLD = 0.98       # drop one of any pair correlated above this

# Number of folds used to carve out the held-out test set (1/N ~= test size).
SPLIT_N_FOLDS = 5

# Minimum training-fold row count a sector needs before its own mean/std are
# trusted for z-score normalisation; smaller sectors fall back to global
# stats (see compute_sector_stats) to avoid noisy per-sector estimates.
MIN_SECTOR_GROUP_SIZE = 20

# Columns never used as predictors.
DROP_COLS = frozenset(
    {"Name", "Symbol", "Rating Agency Name", "Date", "RatingClass", "Rating"}
)

# Candidate identifier columns for company-level grouping, in priority order.
GROUP_ID_CANDIDATES = ("Symbol", "Name")

# Columns that are mathematically non-negative (ratios or counts of non-negative
# quantities). A negative value in any of these is a data-entry error rather than
# signal, so it is converted to NaN during cleaning and later filled by the
# train-only median imputer. Margins, returns, cash-flow-per-share, the effective
# tax rate, and the equity/leverage multipliers are deliberately EXCLUDED because
# they can be legitimately negative (losses, tax benefits, negative equity).
NON_NEGATIVE_COLS = frozenset({
    "currentRatio", "quickRatio", "cashRatio", "cashPerShare",
    "daysOfSalesOutstanding", "debtRatio",
    "assetTurnover", "fixedAssetTurnover", "payablesTurnover",
})

# Columns that are known to be non-numeric identifiers/categoricals and must
# never be coerced to numeric during cleaning.
KNOWN_NON_NUMERIC_COLS = frozenset({
    "Name", "Symbol", "Rating Agency Name", "Date", "Rating", "RatingClass", "Sector",
})

# Minimum fraction of non-null values in an *unknown* object column that must
# parse as numbers before the whole column is coerced to numeric. Below this it
# is treated as genuinely categorical and left untouched. Known feature columns
# (below) are always coerced regardless of this ratio.
NUMERIC_COERCE_MIN_PARSE_RATIO = 0.9

# Financial-ratio feature columns of the corporate-rating schema. These are
# numeric by definition, so they are coerced unconditionally during cleaning
# even if an upload pollutes them with text tokens beyond the ratio threshold.
KNOWN_NUMERIC_FEATURE_COLS = frozenset({
    "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
    "netProfitMargin", "pretaxProfitMargin", "grossProfitMargin",
    "operatingProfitMargin", "returnOnAssets", "returnOnCapitalEmployed",
    "returnOnEquity", "assetTurnover", "fixedAssetTurnover", "debtEquityRatio",
    "debtRatio", "effectiveTaxRate", "freeCashFlowOperatingCashFlowRatio",
    "freeCashFlowPerShare", "cashPerShare", "companyEquityMultiplier",
    "ebitPerRevenue", "enterpriseValueMultiple", "operatingCashFlowPerShare",
    "operatingCashFlowSalesRatio", "payablesTurnover",
})

# Ordinal severity of raw letter grades (higher == worse credit quality). Used to
# resolve conflicting multi-agency ratings by keeping the most conservative
# (worst) grade for a given company-date. Unknown grades map to -1 so any known
# grade in the same group wins.
RATING_SEVERITY = {
    "AAA": 0, "AA": 1, "A": 2, "BBB": 3, "BB": 4, "B": 5,
    "CCC": 6, "CC": 7, "C": 8, "D": 9,
}

_LOG_CANDIDATES = frozenset([
    "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
    "debtEquityRatio", "enterpriseValueMultiple", "operatingCashFlowPerShare",
    "freeCashFlowPerShare", "cashPerShare", "payablesTurnover",
    "fixedAssetTurnover", "companyEquityMultiplier",
])

# Ratios for which a per-company trailing trend is computed (see
# add_temporal_features). Chosen because each is a well-known credit-quality
# signal where the *direction of travel* (improving vs deteriorating) carries
# information beyond a single snapshot value: leverage build-up, margin
# compression, liquidity erosion, and declining asset returns are classic
# early-warning signs that a static ratio value alone does not capture.
TEMPORAL_TREND_COLS = frozenset({
    "debtEquityRatio", "netProfitMargin", "currentRatio", "returnOnAssets",
    "operatingProfitMargin", "grossProfitMargin", "cashRatio", "debtRatio",
})


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

    This must stay in sync with the other four group_rating()/ratingRuleEngine
    implementations: predict_decision_tree.py, predict_random_forest.py,
    predict_logistic_regression.py, and client.js. See
    docs/XGBoost_Technical_Report.md §2.3 for the justification (CCC/CC are
    grouped with Distressed, not Speculative, so the Distressed tier has
    enough support to be measurable).

    A 3-class variant (Distressed merged into Speculative) was tested as a
    diagnostic and did raise CV accuracy (~46% -> ~56%), but was reverted to
    keep this model's class definition consistent with the rest of the
    dashboard.
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
# Data cleaning (deterministic, per-cell — safe to run before the split)
# ---------------------------------------------------------------------------

def _coerce_numeric_like(out, report):
    """Coerce object columns that are 'mostly numeric' to numeric dtype.

    Uploaded CSVs frequently encode missing or errored cells as text tokens such
    as "N/A", "-", "" or "#DIV/0!". Pandas then reads the entire column as object
    dtype, which silently bypasses every numeric transform downstream
    (imputation, winsorization, z-scoring all select numeric dtypes only) — the
    column would either be dropped or crash XGBoost. Coercing such columns turns
    the garbage tokens into NaN so they are imputed normally. Genuinely
    categorical columns (numeric-parse ratio below the threshold) are left as-is.
    """
    coerced = []
    for col in out.columns:
        if col in KNOWN_NON_NUMERIC_COLS or out[col].dtype != object:
            continue
        parsed = pd.to_numeric(out[col], errors="coerce")
        non_null = int(out[col].notna().sum())
        if non_null == 0:
            continue
        parse_ratio = parsed.notna().sum() / non_null
        # Known ratio columns are numeric by definition; coerce them regardless of
        # how much junk an upload introduced. Unknown columns must clear the ratio
        # threshold so genuinely categorical columns are preserved.
        if col in KNOWN_NUMERIC_FEATURE_COLS or parse_ratio >= NUMERIC_COERCE_MIN_PARSE_RATIO:
            out[col] = parsed
            coerced.append(col)
    report["numeric_like_columns_coerced"] = coerced
    return out


def _resolve_rating_conflicts(out, report):
    """Collapse conflicting / duplicate multi-agency ratings per company-date.

    When the same entity on the same date carries more than one rating (e.g. two
    agencies disagree, or the same rating appears twice), the rows share
    identical features but may carry contradictory targets — training noise. We
    keep a single row per ``(id, Date)`` group, choosing the most conservative
    (worst) grade, which is the safe default for credit-risk screening.

    Uses the label column but no test information or fitted statistics, so it is
    a deterministic data-quality step and remains leakage-safe (analogous to the
    pre-split exclusion of Unknown-rated rows). Skipped when an identifier or the
    Date column is absent, since collapsing a company's entire history would be
    wrong.
    """
    report["rating_conflict_groups"] = 0
    report["rating_conflict_rows_dropped"] = 0

    id_col = next((c for c in GROUP_ID_CANDIDATES if c in out.columns), None)
    if id_col is None or "Date" not in out.columns or "Rating" not in out.columns:
        return out

    key = [id_col, "Date"]
    group_size = out.groupby(key)["Rating"].transform("size")
    if not (group_size > 1).any():
        return out  # every company-date is already unique (the default dataset)

    distinct = out.groupby(key)["Rating"].transform("nunique")
    report["rating_conflict_groups"] = int(out.loc[distinct > 1, key].drop_duplicates().shape[0])

    severity = out["Rating"].astype(str).str.strip().str.upper().map(RATING_SEVERITY).fillna(-1)
    n_before = len(out)
    out = (
        out.assign(_sev=severity)
        .sort_values("_sev", ascending=False, kind="stable")
        .drop_duplicates(subset=key, keep="first")
        .drop(columns="_sev")
        .sort_index()
        .reset_index(drop=True)
    )
    report["rating_conflict_rows_dropped"] = n_before - len(out)
    return out


def clean_dataframe(df):
    """Deterministic data cleaning applied *before* the train/test split.

    Every operation here is per-row or per-cell and uses NO aggregate statistics,
    so performing it before the split introduces no leakage (unlike imputation,
    winsorization, and z-scoring, which are fit on the training fold only). The
    steps are ordered so that later steps see the cleaned output of earlier ones.

      1. Coerce "mostly numeric" text columns to numeric (handles "N/A",
         "#DIV/0!", "" etc. in uploaded files) -> garbage becomes NaN.
      2. Drop exact duplicate rows.
      3. Normalise the Sector column's surrounding whitespace so "Energy" and
         "Energy " collapse to one category before one-hot encoding.
      4. Resolve conflicting/duplicate multi-agency ratings: keep one worst-case
         row per (company, Date).
      5. Replace +/-inf with NaN.
      6. Null impossible negatives in the mathematically non-negative columns.
      7. Drop rows whose entire numeric feature vector is NaN (no signal).

    On the default dataset steps 1, 2, 4 and 7 are no-ops (it is already clean on
    those axes), so the cached artifacts and reported metrics are unaffected; the
    steps exist to protect arbitrary user-uploaded datasets. Returns
    ``(cleaned_df, report)`` where ``report`` is a dict of counts for auditing.
    """
    out = df.copy()
    report = {}

    # 1. Coerce numeric-like text columns
    out = _coerce_numeric_like(out, report)

    # 2. Exact duplicate rows
    n_before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    report["duplicate_rows_dropped"] = n_before - len(out)

    # 3. Sector whitespace normalisation
    if "Sector" in out.columns:
        out["Sector"] = out["Sector"].astype(str).str.strip()

    # 4. Conflicting / duplicate multi-agency ratings
    out = _resolve_rating_conflicts(out, report)

    # 5. Infinities -> NaN
    numeric_cols = out.select_dtypes(include="number").columns
    if len(numeric_cols):
        inf_count = int(np.isinf(out[numeric_cols].to_numpy(dtype="float64", na_value=np.nan)).sum())
        report["inf_values_nulled"] = inf_count
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    else:
        report["inf_values_nulled"] = 0

    # 6. Impossible negatives -> NaN (defensive: coerce any column still object,
    #    e.g. an unknown-schema non-negative column that dodged step 1).
    neg_count = 0
    for col in NON_NEGATIVE_COLS & set(out.columns):
        if out[col].dtype == object:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        mask = out[col] < 0
        neg_count += int(mask.sum())
        out.loc[mask, col] = np.nan
    report["impossible_negatives_nulled"] = neg_count

    # 7. Drop rows with an entirely empty numeric feature vector
    numeric_cols = out.select_dtypes(include="number").columns
    if len(numeric_cols):
        all_nan = out[numeric_cols].isna().all(axis=1)
        report["empty_feature_rows_dropped"] = int(all_nan.sum())
        if all_nan.any():
            out = out.loc[~all_nan].reset_index(drop=True)
    else:
        report["empty_feature_rows_dropped"] = 0

    return out, report


# ---------------------------------------------------------------------------
# Temporal (per-company trend) features — deterministic, pre-split, safe
# ---------------------------------------------------------------------------

def add_temporal_features(df, trend_cols=TEMPORAL_TREND_COLS):
    """Add per-company trailing-trend features from each company's own history.

    The dataset averages ~3.4 records per company, but every existing feature
    treats each company-year row as an independent snapshot -- the model never
    sees whether a company's leverage is climbing or falling, or whether
    margins are compressing over time, even though that history is sitting
    unused in the data. This adds, for each column in ``trend_cols``, a
    ``{col}_trend`` feature: the change from that company's own PREVIOUS
    chronological record to the current one (NaN/first record -> 0.0, flagged
    via ``has_prior_record``).

    Why this is leakage-safe to compute *before* the train/test split (like
    ``clean_dataframe``, unlike imputation/winsorisation/z-scoring which fit
    parameters and must run train-only):
    - No aggregate statistic is fit across companies. Each row's trend value
      depends ONLY on that same company's own prior row(s).
    - The train/test split is company-level (``StratifiedGroupKFold`` /
      ``GroupShuffleSplit`` on Symbol/Name, see ``make_split``): every record
      for a given company lands entirely on one side of the split. A trend
      feature computed from company X's own history therefore never uses
      information from any row that could end up in a *different* split
      partition -- it is exactly as leakage-safe as looking at a single row's
      raw ratio value.
    - No test *labels* are read; only feature columns and the Date column are
      used to order and difference each company's own rows.

    Degrades gracefully to a no-op (returns ``df`` unchanged) if there is no
    identifier column (Symbol/Name) or no Date column -- e.g. a user-uploaded
    dataset that lacks temporal structure -- exactly like the Sector-dependent
    steps elsewhere in this module degrade when ``Sector`` is absent.

    Returns ``(df_with_trends, report)`` where ``report`` is a small dict for
    auditing (mirrors the ``clean_dataframe`` convention).
    """
    id_col = next((c for c in GROUP_ID_CANDIDATES if c in df.columns), None)
    if id_col is None or "Date" not in df.columns:
        return df, {"temporal_features_added": False, "reason": "missing identifier or Date column"}

    parsed_dates = pd.to_datetime(df["Date"], errors="coerce")
    if parsed_dates.isna().all():
        return df, {"temporal_features_added": False, "reason": "Date column could not be parsed"}

    available_cols = [c for c in trend_cols if c in df.columns]
    if not available_cols:
        return df, {"temporal_features_added": False, "reason": "none of TEMPORAL_TREND_COLS present"}

    out = df.copy()
    out["_temporal_sort_date"] = parsed_dates
    # Stable sort: within each company, chronological order; unparsable dates
    # (NaT) sort last so they never displace a real prior record incorrectly.
    out = out.sort_values([id_col, "_temporal_sort_date"], kind="stable", na_position="last")

    company_group = out.groupby(id_col)
    for col in available_cols:
        prev_val = company_group[col].shift(1)
        trend = (out[col] - prev_val).clip(-CLIP_BOUND, CLIP_BOUND)
        out[f"{col}_trend"] = trend.fillna(0.0)

    # 0 for a company's first available record (no prior year to compare
    # against), 1 otherwise -- lets the model distinguish "genuinely flat
    # trend" from "no trend information available".
    out["has_prior_record"] = (out.groupby(id_col).cumcount() > 0).astype(int)

    out = out.drop(columns=["_temporal_sort_date"]).sort_index()

    report = {
        "temporal_features_added": True,
        "trend_columns": [f"{c}_trend" for c in available_cols],
        "rows_with_prior_record": int(out["has_prior_record"].sum()),
        "rows_total": int(len(out)),
    }
    return out, report


# ---------------------------------------------------------------------------
# Train / test split (company-level, leakage-safe)
# ---------------------------------------------------------------------------

def make_split(X, y, groups=None, test_size=0.30, random_state=RANDOM_STATE):
    """Split into train/test, preferring a group-aware stratified split.

    When ``groups`` is provided (e.g. company symbol), all records for a given
    company are kept entirely within one side of the split. This prevents
    company-level leakage, where the model would otherwise see other year
    records of a company that also appears in the test set.

    Falls back to stratified ``train_test_split`` when groups are unavailable
    or produce too few members to split, and finally to a plain split.

    Returns ``(X_train, X_test, y_train, y_test, split_strategy, train_groups)``.
    ``train_groups`` is the ``groups`` array sliced to the training rows (or
    ``None`` if no groups were provided/used), so callers can perform a further
    leakage-safe, company-level *nested* split within the training fold (e.g.
    for threshold tuning) without re-deriving the grouping key.
    """
    if groups is not None:
        try:
            n_folds = max(2, min(SPLIT_N_FOLDS, round(1.0 / test_size)))
            sgkf = StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            )
            train_idx, test_idx = next(sgkf.split(X, y, groups))
            groups_arr = np.asarray(groups)
            return (
                X.iloc[train_idx], X.iloc[test_idx],
                y[train_idx], y[test_idx],
                "grouped_stratified",
                groups_arr[train_idx],
            )
        except Exception:
            pass

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, "stratified", None
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test, "random", None


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

    # Altman Z-Score proxies and basic additions
    if "currentRatio" in cols:
        out["working_capital_proxy"] = out["currentRatio"] - 1.0

    if {"netProfitMargin", "assetTurnover"} <= cols:
        out["net_income_to_assets"] = (out["netProfitMargin"] * out["assetTurnover"]).clip(-CLIP_BOUND, CLIP_BOUND)
        
    if {"ebitPerRevenue", "assetTurnover"} <= cols:
        out["ebit_to_assets"] = (out["ebitPerRevenue"] * out["assetTurnover"]).clip(-CLIP_BOUND, CLIP_BOUND)
        
    if {"ebitPerRevenue", "netProfitMargin"} <= cols:
        out["interest_tax_burden_proxy"] = (out["ebitPerRevenue"] - out["netProfitMargin"]).clip(-CLIP_BOUND, CLIP_BOUND)

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

    # --- Additional credit-risk discriminators ---

    # DuPont decomposition components (ROE = margin * turnover * leverage)
    if {"netProfitMargin", "assetTurnover", "companyEquityMultiplier"} <= cols:
        out["dupont_roe"] = (out["netProfitMargin"] * out["assetTurnover"] * out["companyEquityMultiplier"]).clip(-CLIP_BOUND, CLIP_BOUND)

    # Interest coverage proxy: EBIT relative to debt burden
    if {"ebitPerRevenue", "debtEquityRatio"} <= cols:
        out["interest_coverage_proxy"] = safe_div("ebitPerRevenue", "debtEquityRatio")

    # Cash-flow adequacy: can the company service debt from operations?
    if {"operatingCashFlowPerShare", "debtRatio"} <= cols:
        out["ocf_debt_adequacy"] = (out["operatingCashFlowPerShare"] / (out["debtRatio"] + EPS)).clip(-CLIP_BOUND, CLIP_BOUND)

    # Defensive interval proxy: liquidity relative to burn rate
    if {"cashRatio", "operatingCashFlowSalesRatio"} <= cols:
        out["defensive_interval"] = (out["cashRatio"] / (out["operatingCashFlowSalesRatio"].abs() + EPS)).clip(-CLIP_BOUND, CLIP_BOUND)

    # Earnings quality: gap between operating cash flow and reported profit
    if {"operatingCashFlowSalesRatio", "operatingProfitMargin"} <= cols:
        out["earnings_quality"] = (out["operatingCashFlowSalesRatio"] - out["operatingProfitMargin"]).clip(-CLIP_BOUND, CLIP_BOUND)

    # FCF yield proxy (FCF per share relative to enterprise value)
    if {"freeCashFlowPerShare", "enterpriseValueMultiple"} <= cols:
        out["fcf_yield_proxy"] = safe_div("freeCashFlowPerShare", "enterpriseValueMultiple")

    # Leverage intensity: debt ratio * equity multiplier (compounds leverage signal)
    if {"debtRatio", "companyEquityMultiplier"} <= cols:
        out["leverage_intensity"] = (out["debtRatio"] * out["companyEquityMultiplier"]).clip(-CLIP_BOUND, CLIP_BOUND)

    # Gross-to-net margin conversion efficiency (how much of gross profit survives)
    if {"netProfitMargin", "grossProfitMargin"} <= cols:
        out["margin_conversion"] = safe_div("netProfitMargin", "grossProfitMargin")

    # Quick ratio minus cash ratio: non-cash current asset component
    if {"quickRatio", "cashRatio"} <= cols:
        out["receivables_liquidity"] = (out["quickRatio"] - out["cashRatio"]).clip(-CLIP_BOUND, CLIP_BOUND)

    # Return on debt: how productive is borrowed capital?
    if {"returnOnAssets", "debtEquityRatio"} <= cols:
        out["return_on_debt"] = (out["returnOnAssets"] * out["debtEquityRatio"]).clip(-CLIP_BOUND, CLIP_BOUND)

    # Operating efficiency gap: asset turnover vs fixed asset turnover ratio
    if {"assetTurnover", "fixedAssetTurnover"} <= cols:
        out["asset_composition_efficiency"] = safe_div("assetTurnover", "fixedAssetTurnover")

    # Altman Z-score proxy (simplified: combines profitability, leverage, liquidity)
    if {"currentRatio", "returnOnAssets", "debtRatio", "assetTurnover", "ebitPerRevenue"} <= cols:
        out["altman_z_proxy"] = (
            1.2 * (out["currentRatio"] - 1.0) +
            1.4 * out["returnOnAssets"] +
            3.3 * out["ebitPerRevenue"] +
            0.6 * (1.0 / (out["debtRatio"] + EPS)).clip(-10, 10) +
            1.0 * out["assetTurnover"]
        ).clip(-CLIP_BOUND, CLIP_BOUND)

    # Piotroski-style binary signals (score 0-1 each, sum = composite health)
    piotroski_components = []
    if "returnOnAssets" in cols:
        out["_pio_roa"] = (out["returnOnAssets"] > 0).astype(float)
        piotroski_components.append("_pio_roa")
    if "operatingCashFlowSalesRatio" in cols:
        out["_pio_ocf"] = (out["operatingCashFlowSalesRatio"] > 0).astype(float)
        piotroski_components.append("_pio_ocf")
    if {"operatingCashFlowSalesRatio", "returnOnAssets"} <= cols:
        out["_pio_accrual"] = (out["operatingCashFlowSalesRatio"] > out["returnOnAssets"]).astype(float)
        piotroski_components.append("_pio_accrual")
    if "currentRatio" in cols:
        out["_pio_liquidity"] = (out["currentRatio"] > 1.0).astype(float)
        piotroski_components.append("_pio_liquidity")
    if piotroski_components:
        out["piotroski_score"] = out[piotroski_components].sum(axis=1)
        out = out.drop(columns=piotroski_components)

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

def compute_sector_stats(X_train_with_interactions, min_group_size=MIN_SECTOR_GROUP_SIZE):
    """Compute per-sector mean/std needed for z-score features (train set only).

    Sectors with fewer than ``min_group_size`` training rows produce noisy
    mean/std estimates (a handful of companies is not enough to characterise
    a sector's "typical" ratio range), and those noisy per-sector statistics
    were one contributor to the ~39-point train/test accuracy gap found by
    ``evaluate_xgboost.py``. Such sectors fall back to the global mean/std
    instead of their own, so the z-score feature reduces to a globally
    normalised value rather than a spuriously precise sector-relative one.
    """
    df = X_train_with_interactions
    ratios = [
        c for c in df.select_dtypes(include="number").columns
        if "log" not in c and not c.startswith("Sector_")
    ]
    sector_sizes = df.groupby("Sector").size()
    reliable_sectors = sector_sizes[sector_sizes >= min_group_size].index

    means = df.groupby("Sector")[ratios].mean()
    stds = df.groupby("Sector")[ratios].std().fillna(1.0)

    global_means = df[ratios].mean()
    global_stds = df[ratios].std().fillna(1.0)

    # Overwrite small-sector rows with global stats (broadcast row-wise).
    unreliable_mask = ~means.index.isin(reliable_sectors)
    if unreliable_mask.any():
        means.loc[unreliable_mask, :] = global_means.values
        stds.loc[unreliable_mask, :] = global_stds.values

    return {
        "means": means,
        "stds": stds,
        "global_means": global_means,
        "global_stds": global_stds,
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

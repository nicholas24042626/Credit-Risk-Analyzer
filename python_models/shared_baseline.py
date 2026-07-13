"""Shared "fair baseline" pipeline used by all four models for comparison.

Every model-specific script (predict_xgboost.py, predict_decision_tree.py,
predict_random_forest.py, predict_logistic_regression.py) keeps its own
"tuned" pipeline (hyperparameter search, model-specific feature handling,
etc.) untouched. This module exists ONLY to give all four a second, identical
run -- same cleaning, same target, same company-level split, same engineered
features, same imputation/encoding, same untuned estimator defaults, same
metrics -- so cross-model accuracy comparisons measure the algorithm, not
four different data pipelines.

Design choices (see conversation history / docs/XGBoost_Technical_Report.md
for the reasoning, not repeated here in full):

- Company-level grouped split, NOT plain ``stratify=y``: a company's repeat
  year-rows must land entirely on one side of the split, or accuracy is
  inflated by the model seeing near-duplicate rows of a test-set company
  during training. This is a leakage fix, not a style preference.
- Median imputation + one-hot encoding for ALL FOUR models here, including
  XGBoost, even though XGBoost's own tuned pipeline uses native missing-value
  handling and native categorical splits instead. The point of this baseline
  is to isolate "is the algorithm better", so no model gets a friendlier data
  representation than another in this tier.
- Interaction/log features and sector z-scores ARE included for all four
  (not just raw ratios): they are plain pandas arithmetic, not XGBoost
  internals, and a linear model benefits from them more than a tree does
  (it cannot discover a ratio-of-ratios on its own).
- True library defaults for the estimator (only ``random_state`` set by the
  caller) -- no grid/randomized search, no hand-picked non-default
  hyperparameters.
- Class balancing via ``sample_weight`` at fit time (computed once, here,
  with ``compute_sample_weight("balanced", ...)``), not each estimator's own
  ``class_weight`` constructor argument. XGBoost's sklearn wrapper has no
  ``class_weight`` parameter for multiclass problems, so ``sample_weight`` is
  the one balancing mechanism all four estimators' ``.fit()`` accepts --
  using it uniformly means every model gets the identical per-row weight
  vector instead of four different "balanced" implementations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 143

EPS = 1e-5
CLIP_BOUND = 1e6

NEAR_ZERO_VAR_THRESHOLD = 1e-8
SPLIT_N_FOLDS = 5
BASELINE_TEST_SIZE = 0.30

DROP_COLS = frozenset(
    {"Name", "Symbol", "Rating Agency Name", "Date", "RatingClass", "Rating"}
)
GROUP_ID_CANDIDATES = ("Symbol", "Name")

NON_NEGATIVE_COLS = frozenset({
    "currentRatio", "quickRatio", "cashRatio", "cashPerShare",
    "daysOfSalesOutstanding", "debtRatio",
    "assetTurnover", "fixedAssetTurnover", "payablesTurnover",
})

KNOWN_NON_NUMERIC_COLS = frozenset({
    "Name", "Symbol", "Rating Agency Name", "Date", "Rating", "RatingClass", "Sector",
})

NUMERIC_COERCE_MIN_PARSE_RATIO = 0.9

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

MIN_SECTOR_GROUP_SIZE = 20


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

def group_rating(r):
    """Collapse granular credit ratings into four financial risk tiers.

    Canonical, shared implementation -- every model calls this one, not a
    per-model copy. Hyphenated string labels ("Investment-High") are used
    directly as the class labels for all four models (no int-encoding step),
    since sklearn/XGBoost estimators all accept string class labels natively
    and this keeps confusion-matrix output human-readable without an
    inverse-transform step.
    """
    r = str(r).strip().upper()
    if r in {"AAA", "AA", "A"}:
        return "Investment-High"
    if r == "BBB":
        return "Investment-Low"
    if r in {"BB", "B"}:
        return "Speculative"
    if r in {"CCC", "CC", "C", "D"}:
        return "Distressed"
    return "Unknown"


# ---------------------------------------------------------------------------
# Data cleaning (deterministic, per-cell -- safe to run before the split)
# ---------------------------------------------------------------------------

def _coerce_numeric_like(out, report):
    coerced = []
    for col in out.columns:
        if col in KNOWN_NON_NUMERIC_COLS or out[col].dtype != object:
            continue
        parsed = pd.to_numeric(out[col], errors="coerce")
        non_null = int(out[col].notna().sum())
        if non_null == 0:
            continue
        parse_ratio = parsed.notna().sum() / non_null
        if col in KNOWN_NUMERIC_FEATURE_COLS or parse_ratio >= NUMERIC_COERCE_MIN_PARSE_RATIO:
            out[col] = parsed
            coerced.append(col)
    report["numeric_like_columns_coerced"] = coerced
    return out


def _resolve_rating_conflicts(out, report):
    report["rating_conflict_groups"] = 0
    report["rating_conflict_rows_dropped"] = 0

    id_col = next((c for c in GROUP_ID_CANDIDATES if c in out.columns), None)
    if id_col is None or "Date" not in out.columns or "Rating" not in out.columns:
        return out

    key = [id_col, "Date"]
    group_size = out.groupby(key)["Rating"].transform("size")
    if not (group_size > 1).any():
        return out

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

    Identical steps/order to xgboost_stuff/preprocessing.py's clean_dataframe
    (this is the one piece that module already documented as model-agnostic;
    duplicated here rather than imported cross-package so this module has no
    dependency on any one model's own package).
    """
    out = df.copy()
    report = {}

    out = _coerce_numeric_like(out, report)

    n_before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    report["duplicate_rows_dropped"] = n_before - len(out)

    if "Sector" in out.columns:
        out["Sector"] = out["Sector"].astype(str).str.strip()

    out = _resolve_rating_conflicts(out, report)

    numeric_cols = out.select_dtypes(include="number").columns
    if len(numeric_cols):
        inf_count = int(np.isinf(out[numeric_cols].to_numpy(dtype="float64", na_value=np.nan)).sum())
        report["inf_values_nulled"] = inf_count
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    else:
        report["inf_values_nulled"] = 0

    neg_count = 0
    for col in NON_NEGATIVE_COLS & set(out.columns):
        if out[col].dtype == object:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        mask = out[col] < 0
        neg_count += int(mask.sum())
        out.loc[mask, col] = np.nan
    report["impossible_negatives_nulled"] = neg_count

    numeric_cols = out.select_dtypes(include="number").columns
    if len(numeric_cols):
        all_nan = out[numeric_cols].isna().all(axis=1)
        report["empty_feature_rows_dropped"] = int(all_nan.sum())
        if all_nan.any():
            out = out.loc[~all_nan].reset_index(drop=True)
    else:
        report["empty_feature_rows_dropped"] = 0

    return out, report


def extract_groups(df):
    for id_col in GROUP_ID_CANDIDATES:
        if id_col in df.columns:
            return df[id_col].astype(str).values
    return None


# ---------------------------------------------------------------------------
# Train / test split (company-level, leakage-safe) -- shared by all 4
# ---------------------------------------------------------------------------

def make_split(X, y, groups=None, test_size=BASELINE_TEST_SIZE, random_state=RANDOM_STATE):
    if groups is not None:
        try:
            n_folds = max(2, min(SPLIT_N_FOLDS, round(1.0 / test_size)))
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            train_idx, test_idx = next(sgkf.split(X, y, groups))
            groups_arr = np.asarray(groups)
            return (
                X.iloc[train_idx], X.iloc[test_idx],
                y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx],
                y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx],
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
# Feature engineering -- same interaction/log features and sector z-scores
# for all four models (see module docstring for why these are included).
# ---------------------------------------------------------------------------

def add_interaction_features(X):
    out = X.copy()
    cols = set(out.columns)

    def safe_div(a, b_col):
        return (out[a] / (out[b_col].abs() + EPS)).clip(-CLIP_BOUND, CLIP_BOUND)

    if "currentRatio" in cols:
        out["working_capital_proxy"] = out["currentRatio"] - 1.0
    if {"netProfitMargin", "assetTurnover"} <= cols:
        out["net_income_to_assets"] = (out["netProfitMargin"] * out["assetTurnover"]).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"ebitPerRevenue", "assetTurnover"} <= cols:
        out["ebit_to_assets"] = (out["ebitPerRevenue"] * out["assetTurnover"]).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"ebitPerRevenue", "netProfitMargin"} <= cols:
        out["interest_tax_burden_proxy"] = (out["ebitPerRevenue"] - out["netProfitMargin"]).clip(-CLIP_BOUND, CLIP_BOUND)
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
        out["equity_efficiency"] = (out["returnOnCapitalEmployed"] * out["assetTurnover"]).clip(-CLIP_BOUND, CLIP_BOUND)
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
    if {"netProfitMargin", "assetTurnover", "companyEquityMultiplier"} <= cols:
        out["dupont_roe"] = (out["netProfitMargin"] * out["assetTurnover"] * out["companyEquityMultiplier"]).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"ebitPerRevenue", "debtEquityRatio"} <= cols:
        out["interest_coverage_proxy"] = safe_div("ebitPerRevenue", "debtEquityRatio")
    if {"operatingCashFlowPerShare", "debtRatio"} <= cols:
        out["ocf_debt_adequacy"] = (out["operatingCashFlowPerShare"] / (out["debtRatio"] + EPS)).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"cashRatio", "operatingCashFlowSalesRatio"} <= cols:
        out["defensive_interval"] = (out["cashRatio"] / (out["operatingCashFlowSalesRatio"].abs() + EPS)).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"operatingCashFlowSalesRatio", "operatingProfitMargin"} <= cols:
        out["earnings_quality"] = (out["operatingCashFlowSalesRatio"] - out["operatingProfitMargin"]).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"freeCashFlowPerShare", "enterpriseValueMultiple"} <= cols:
        out["fcf_yield_proxy"] = safe_div("freeCashFlowPerShare", "enterpriseValueMultiple")
    if {"debtRatio", "companyEquityMultiplier"} <= cols:
        out["leverage_intensity"] = (out["debtRatio"] * out["companyEquityMultiplier"]).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"netProfitMargin", "grossProfitMargin"} <= cols:
        out["margin_conversion"] = safe_div("netProfitMargin", "grossProfitMargin")
    if {"quickRatio", "cashRatio"} <= cols:
        out["receivables_liquidity"] = (out["quickRatio"] - out["cashRatio"]).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"returnOnAssets", "debtEquityRatio"} <= cols:
        out["return_on_debt"] = (out["returnOnAssets"] * out["debtEquityRatio"]).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"assetTurnover", "fixedAssetTurnover"} <= cols:
        out["asset_composition_efficiency"] = safe_div("assetTurnover", "fixedAssetTurnover")
    if {"currentRatio", "returnOnAssets", "debtRatio", "assetTurnover", "ebitPerRevenue"} <= cols:
        out["altman_z_proxy"] = (
            1.2 * (out["currentRatio"] - 1.0) +
            1.4 * out["returnOnAssets"] +
            3.3 * out["ebitPerRevenue"] +
            0.6 * (1.0 / (out["debtRatio"] + EPS)).clip(-10, 10) +
            1.0 * out["assetTurnover"]
        ).clip(-CLIP_BOUND, CLIP_BOUND)

    piotroski_components = []
    if "returnOnAssets" in cols:
        out["_pio_roa"] = (out["returnOnAssets"] > 0).astype(float).where(out["returnOnAssets"].notna())
        piotroski_components.append("_pio_roa")
    if "operatingCashFlowSalesRatio" in cols:
        out["_pio_ocf"] = (out["operatingCashFlowSalesRatio"] > 0).astype(float).where(out["operatingCashFlowSalesRatio"].notna())
        piotroski_components.append("_pio_ocf")
    if {"operatingCashFlowSalesRatio", "returnOnAssets"} <= cols:
        both_present = out["operatingCashFlowSalesRatio"].notna() & out["returnOnAssets"].notna()
        out["_pio_accrual"] = (out["operatingCashFlowSalesRatio"] > out["returnOnAssets"]).astype(float).where(both_present)
        piotroski_components.append("_pio_accrual")
    if "currentRatio" in cols:
        out["_pio_liquidity"] = (out["currentRatio"] > 1.0).astype(float).where(out["currentRatio"].notna())
        piotroski_components.append("_pio_liquidity")
    if piotroski_components:
        out["piotroski_score"] = out[piotroski_components].sum(axis=1)
        out = out.drop(columns=piotroski_components)

    if {"debtEquityRatio"} <= cols:
        out["debtEquityRatio_sq"] = (out["debtEquityRatio"] ** 2).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"returnOnAssets"} <= cols:
        out["returnOnAssets_sq"] = (out["returnOnAssets"] ** 2).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"currentRatio"} <= cols:
        out["currentRatio_sq"] = (out["currentRatio"] ** 2).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"debtEquityRatio", "currentRatio"} <= cols:
        out["leverage_liquidity_stress"] = (
            out["debtEquityRatio"] * (1.0 / (out["currentRatio"] + EPS))
        ).clip(-CLIP_BOUND, CLIP_BOUND)
    if {"debtRatio", "ebitPerRevenue"} <= cols:
        out["debt_to_earnings_power"] = safe_div("debtRatio", "ebitPerRevenue")

    for col in _LOG_CANDIDATES & cols:
        out[f"{col}_log"] = np.sign(out[col]) * np.log1p(np.abs(out[col]))

    return out


def compute_sector_stats(X_train_with_interactions, min_group_size=MIN_SECTOR_GROUP_SIZE):
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

    unreliable_mask = ~means.index.isin(reliable_sectors)
    if unreliable_mask.any():
        means.loc[unreliable_mask, :] = global_means.values
        stds.loc[unreliable_mask, :] = global_stds.values

    return {"means": means, "stds": stds, "global_means": global_means, "global_stds": global_stds, "ratios": ratios}


def apply_zscore_from_stats(df_list, stats):
    means, stds = stats["means"], stats["stds"]
    global_means, global_stds = stats["global_means"], stats["global_stds"]
    ratios = stats["ratios"]

    for df_out in df_list:
        if "Sector" not in df_out.columns:
            continue
        m_sec = df_out[["Sector"]].merge(means, left_on="Sector", right_index=True, how="left").drop(columns=["Sector"])
        s_sec = df_out[["Sector"]].merge(stds, left_on="Sector", right_index=True, how="left").drop(columns=["Sector"])
        m_sec = m_sec.fillna(global_means)
        s_sec = s_sec.fillna(global_stds)
        z = (df_out[ratios] - m_sec) / (s_sec + EPS)
        z.columns = [f"{r}_sec_z" for r in ratios]
        df_out[z.columns] = z


# ---------------------------------------------------------------------------
# Imputation + one-hot encoding -- used for ALL FOUR models in this baseline
# tier (see module docstring: deliberately not native-NaN/native-categorical,
# so no model gets a friendlier data representation than another here).
# ---------------------------------------------------------------------------

def fit_imputation(X_train):
    numerics = X_train.select_dtypes(include="number").columns
    medians = X_train[numerics].median()
    return medians.fillna(0.0).to_dict()


def apply_imputation(X, medians):
    X_out = X.copy()
    for col, value in medians.items():
        if col in X_out.columns:
            X_out[col] = X_out[col].fillna(value)
    return X_out


def one_hot_encode_and_align(X_train, X_test):
    """One-hot encode every categorical column and align train/test columns.

    Fit-on-train discipline: dummy columns are derived from ``pd.get_dummies``
    on each side independently, then aligned so a category seen only in test
    doesn't create a train-side column mismatch (matches the old
    ``encode_and_align`` this replaces, still used by DT/RF/LogReg's own
    pipelines -- this is that same idea, centralised).
    """
    train_cats = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    test_cats = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
    X_train_enc = pd.get_dummies(X_train, columns=train_cats)
    X_test_enc = pd.get_dummies(X_test, columns=test_cats)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    return X_train_enc, X_test_enc


def fit_feature_selection(X_train_enc, var_threshold=NEAR_ZERO_VAR_THRESHOLD):
    """Drop near-zero-variance columns, fit on the training matrix only."""
    columns = list(X_train_enc.columns)
    if not columns:
        return columns
    variances = X_train_enc.var(axis=0, numeric_only=True)
    keep = [c for c in columns if float(variances.get(c, 0.0)) > var_threshold]
    return keep or columns


# ---------------------------------------------------------------------------
# Shared evaluation -- identical metric set/shape for all four models
# ---------------------------------------------------------------------------

def evaluate_predictions(y_test, y_pred, labels):
    """Return accuracy, weighted + macro precision/recall/f1, per-class
    recall, and a confusion matrix -- the same shape for every model.
    """
    accuracy = float(accuracy_score(y_test, y_pred))
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", labels=labels, zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", labels=labels, zero_division=0
    )
    per_class_recall = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return {
        "accuracy": round(accuracy, 4),
        "precisionWeighted": round(float(p_w), 4),
        "recallWeighted": round(float(r_w), 4),
        "f1Weighted": round(float(f1_w), 4),
        "precisionMacro": round(float(p_m), 4),
        "recallMacro": round(float(r_m), 4),
        "f1Macro": round(float(f1_m), 4),
        "recallPerClass": {
            label: round(float(recall), 4) for label, recall in zip(labels, per_class_recall)
        },
        "confusionMatrix": {"labels": list(labels), "values": cm.tolist()},
    }


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------

CANONICAL_LABELS = ["Investment-High", "Investment-Low", "Speculative", "Distressed"]


def run_fair_baseline(estimator, df):
    """Run the full shared pipeline on a cleaned-or-raw dataframe and fit/
    evaluate ``estimator`` (expected to already have ``random_state`` set,
    with every other hyperparameter left at the library default -- do NOT
    set ``class_weight`` on the estimator; balancing is applied uniformly
    here via ``sample_weight`` instead, see module docstring).

    ``df`` must contain a ``Rating`` column. Returns a result dict with the
    fitted pipeline pieces (for a caller that wants to also predict a single
    sample) plus the shared evaluation metrics -- or ``None`` if the dataset
    is too small to split.
    """
    df = df.copy()
    df, _clean_report = clean_dataframe(df)

    y_raw = df["Rating"].apply(group_rating)
    valid_mask = y_raw != "Unknown"
    df = df[valid_mask].copy()
    y = y_raw[valid_mask].reset_index(drop=True)
    df = df.reset_index(drop=True)

    groups = extract_groups(df)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    X_train, X_test, y_train, y_test, split_strategy = make_split(X, y, groups=groups)
    if len(X_train) == 0 or len(X_test) == 0:
        return None

    medians = fit_imputation(X_train)
    X_train_m = apply_imputation(X_train, medians)
    X_test_m = apply_imputation(X_test, medians)

    X_train_i = add_interaction_features(X_train_m)
    X_test_i = add_interaction_features(X_test_m)

    if "Sector" in X_train_i.columns:
        sector_stats = compute_sector_stats(X_train_i)
        apply_zscore_from_stats([X_train_i, X_test_i], sector_stats)

    X_train_enc, X_test_enc = one_hot_encode_and_align(X_train_i, X_test_i)

    selected = fit_feature_selection(X_train_enc)
    X_train_enc = X_train_enc[selected]
    X_test_enc = X_test_enc.reindex(columns=selected, fill_value=0)

    # Any residual NaN (e.g. an engineered ratio from a column with no
    # training-fold median) is filled with 0 -- sklearn/XGBoost estimators
    # under default settings cannot accept NaN.
    X_train_enc = X_train_enc.fillna(0)
    X_test_enc = X_test_enc.fillna(0)

    # Encode labels to contiguous integers 0..n-1 before fitting. Not every
    # estimator accepts raw string class labels the same way -- XGBoost's
    # sklearn wrapper in particular requires y to already be encoded this way
    # (no automatic LabelEncoder as in older versions) -- so this is applied
    # uniformly rather than special-cased per model. Predictions are mapped
    # straight back to the string labels below, so callers/metrics never see
    # the encoding.
    labels = [l for l in CANONICAL_LABELS if l in set(y_train) | set(y_test)]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y_train_idx = y_train.map(label_to_idx)

    sample_weight = compute_sample_weight("balanced", y_train_idx)
    estimator.fit(X_train_enc, y_train_idx, sample_weight=sample_weight)
    y_pred = pd.Series(estimator.predict(X_test_enc)).map(idx_to_label).to_numpy()

    metrics = evaluate_predictions(y_test, y_pred, labels)
    metrics["splitStrategy"] = split_strategy
    metrics["testSamples"] = int(len(y_test))
    metrics["trainSamples"] = int(len(y_train))

    return {
        "metrics": metrics,
        "estimator": estimator,
        "feature_columns": selected,
        "medians": medians,
    }

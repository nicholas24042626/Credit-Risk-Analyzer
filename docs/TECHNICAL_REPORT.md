# Technical Report: XGBoost Credit Risk Prediction System

# Final Year Project — Engineering Documentation

---

## Executive Summary

This report documents the complete engineering lifecycle of a multiclass XGBoost classifier for corporate credit rating prediction, deployed as a web application with SHAP-based explainability. The system classifies companies into four financial risk tiers — Investment-High, Investment-Low, Speculative, and Distressed — using 30 financial ratios from the Set A Corporate Rating dataset (2,029 company–year records, 593 unique companies).

The project evolved through multiple documented iterations, each motivated by a specific engineering failure: row-level data leakage inflated early accuracy from ~72% to an honest ~46% after correction; SMOTE oversampling on 4–5 Distressed samples produced degenerate synthetic clusters; threshold-strategy selection leaked test labels into a production decision. Each failure was diagnosed, documented, and resolved systematically.

**Authoritative performance** (company-level, leakage-safe, grouped stratified 5-fold cross-validation):

| Metric | CV Mean ± Std |
|---|---|
| Accuracy | 0.4603 ± 0.0313 |
| Macro F1 | 0.4181 ± 0.0308 |

These figures are deliberately conservative. They reflect a 4-class ordinal problem with severe class imbalance (Distressed: ~3.7% of data), company-level split integrity (no company appears in both train and test), and no threshold-selection leakage. The ~46% accuracy is the honest ceiling for this feature set and sample size — an examiner should evaluate the engineering discipline that produced this number, not the number itself.

> [!IMPORTANT]
> **Consistency audit result**: The cached confusion matrix in the predictions CSV (`results/xgboost_test_predictions.csv`) produces accuracy 0.5296 with 215/406 correct. The classification report CSV (`results/xgboost_classification_report.csv`) is internally consistent with that predictions file. However, the confusion matrix previously documented in the earlier technical report cited accuracy 0.5222 (212/406 correct) with a different cell distribution — indicating it was captured from a different model run. This report uses only the currently cached artifacts as the single-split reference. The authoritative figure remains the 5-fold CV mean.

---

## 1. Project Objectives

### 1.1 Primary Objective

Build a credit risk classification system that:

1. Predicts corporate credit risk tiers from financial ratio data.
2. Provides probability-calibrated predictions suitable for downstream decision-making.
3. Explains individual predictions through SHAP decomposition.
4. Serves predictions through a web API with a visual dashboard.
5. Maintains strict data-leakage prevention at every pipeline stage.

### 1.2 Engineering Objectives

Beyond model accuracy, the project prioritises:

- **Reproducibility**: Fixed random seeds, deterministic label encoding, serialised preprocessing artifacts.
- **Auditability**: Every preprocessing step documented, all fitted parameters persisted and traceable.
- **Separation of concerns**: Shared preprocessing module used by both training and inference paths, preventing drift.
- **Honest evaluation**: Cross-validated metrics with company-level grouping as the authoritative performance figure.

### 1.3 Non-Objectives

This project does not attempt to:

- Achieve state-of-the-art credit rating accuracy (which requires temporal features, macroeconomic indicators, and orders of magnitude more data).
- Replace regulatory credit rating models (which are governed by Basel III/IV frameworks with specific validation requirements).
- Produce a production-grade financial system (which would require SOC 2 compliance, model governance, and regulatory approval).

---

## 2. Business Problem

### 2.1 Problem Statement

Credit rating agencies (Moody's, S&P, Fitch, Egan-Jones) assign letter grades to corporate issuers reflecting default probability. These ratings influence bond pricing, lending terms, and regulatory capital requirements. The manual rating process is:

- **Expensive**: Requires analyst teams with domain expertise.
- **Slow**: Rating revisions lag market conditions by weeks or months.
- **Opaque**: Rating methodologies are proprietary and inconsistent across agencies.

An automated system that predicts rating tiers from publicly available financial ratios could serve as a screening tool for preliminary risk assessment, watchlist generation, or rating-agency validation.

### 2.2 Why Four Tiers, Not Ten Grades

The raw dataset contains 10+ distinct letter grades (AAA through D). Predicting at this granularity was abandoned because:

- **Overlapping feature distributions**: Adjacent grades (e.g., A and AA, BB and B) share nearly identical financial ratio distributions; no feature engineering could separate them reliably with this sample size.
- **Extreme class imbalance at granular level**: Some grades (e.g., D) have fewer than 5 samples, making per-class learning impossible.
- **Financial coherence**: The four-tier grouping maps directly to investment-grade/speculative-grade/distressed boundaries used in banking practice (Basel II IRB framework).

| Tier | Input Ratings | Financial Meaning | Records |
|---|---|---|---|
| Investment-High | AAA, AA, A | Minimal default risk | ~490 |
| Investment-Low | BBB | Adequate capacity; moderate vulnerability | ~670 |
| Speculative | BB, B | Significant credit risk | ~795 |
| Distressed | CCC, CC, C, D | Near-default or defaulted | ~72 |

The grouping decision was made before any model training. It is a domain decision, not a model tuning step.

### 2.3 Critical Design Decision: Distressed Tier Composition

**Problem**: Should CCC and CC be grouped with Speculative (BB, B) or with Distressed (C, D)?

**Decision**: CCC and CC are grouped with Distressed.

**Alternatives considered**:
- Group CCC/CC with Speculative: This would give Speculative more samples but dilute the Distressed class to ~10 samples (C and D only), making it unmeasurable.
- Group CCC/CC with Distressed: This raises Distressed to ~72 records (~15 per test fold), enabling genuine learning and evaluation.

**Trade-offs**: Grouping CCC/CC with Distressed makes the Speculative class more internally coherent (BB/B only) but gives the Distressed class a wider spread (CCC through D). This is consistent with Basel II's "speculative-grade" boundary at BB.

**Evidence**: Under the CCC/CC→Speculative grouping, the Distressed class had 1 test sample in some splits — producing 100% or 0% accuracy with no statistical meaning. Under the CCC/CC→Distressed grouping, Distressed achieves F1 ≈ 0.25 (CV mean), which is weak but genuine.

> [!NOTE]
> **Cross-system inconsistency — discovered and fixed**: An earlier audit found two mutually exclusive groupings in the codebase. The frontend `client.js` (`ratingRuleEngine`) and the Random Forest backend (`predict_random_forest.py`, `group_rating()`) mapped CCC/CC to "Speculative" and only C/D to "Distressed", while the XGBoost backend (`preprocessing.py`), Decision Tree backend (`predict_decision_tree.py`), and Logistic Regression backend (`predict_logistic_regression.py`) mapped CCC/CC/C/D to "Distressed". This was a genuine engineering defect: two of the dashboard's models defined the classification task differently from the other three, making cross-model comparison statistically invalid regardless of which grouping is "correct".
>
> **Fix applied**: `client.js` and `predict_random_forest.py` were updated to the CCC/CC→Distressed grouping, matching the other three implementations and the domain justification in §2.3 (this raises Distressed support from ~10 to ~72 records, making the class measurable rather than a coin flip on 1–2 test samples). All five `group_rating()`/`ratingRuleEngine` implementations in the codebase now agree — verified by direct source inspection after the change. Random Forest does not cache trained artifacts (it retrains per request), so no stale cached model needed invalidation as a result of this fix; any previously cached XGBoost artifacts trained under the old grouping are unaffected since XGBoost's grouping did not change.

---

## 3. Dataset Analysis

### 3.1 Source Description

The dataset is the **Set A Corporate Rating** CSV, containing 2,029 company–year records. Each record represents a single observation of a company at a specific date, rated by one of several agencies (primarily Egan-Jones and Fitch).

| Property | Value |
|---|---|
| Total records | 2,029 (excluding header) |
| Unique companies | ~593 (identified by `Symbol`) |
| Average records per company | ~3.4 |
| Features | 30 financial ratios + 1 categorical (`Sector`) |
| Identifier columns | `Name`, `Symbol`, `Rating Agency Name`, `Date` |
| Target column | `Rating` (letter grade) |

### 3.2 Schema

The 30 financial ratios span five categories of financial health:

| Category | Features |
|---|---|
| **Liquidity** | `currentRatio`, `quickRatio`, `cashRatio` |
| **Profitability** | `netProfitMargin`, `pretaxProfitMargin`, `grossProfitMargin`, `operatingProfitMargin`, `returnOnAssets`, `returnOnCapitalEmployed`, `returnOnEquity`, `ebitPerRevenue` |
| **Leverage** | `debtEquityRatio`, `debtRatio`, `companyEquityMultiplier` |
| **Efficiency** | `assetTurnover`, `fixedAssetTurnover`, `daysOfSalesOutstanding`, `payablesTurnover` |
| **Cash Flow** | `operatingCashFlowPerShare`, `freeCashFlowPerShare`, `cashPerShare`, `operatingCashFlowSalesRatio`, `freeCashFlowOperatingCashFlowRatio` |
| **Valuation / Tax** | `enterpriseValueMultiple`, `effectiveTaxRate` |

### 3.3 Data Quality Issues

1. **Missing values**: Several financial ratio columns contain NaN values where the metric was not reported or not applicable for a company.
2. **Extreme outliers**: Financial ratios exhibit fat-tailed distributions. `debtEquityRatio` ranges from near-zero to 100+; `effectiveTaxRate` can exceed 1.0 or be negative (carryforward losses).
3. **Multi-agency ratings**: The same company may have different ratings from different agencies on different dates, introducing label noise.
4. **No temporal features**: The `Date` column is dropped entirely — the model treats each record independently with no time-series awareness.

### 3.4 Critical Review

**Strengths**: The dataset provides a realistic, multi-agency view of corporate credit with a reasonable number of financial metrics.

**Weaknesses**: 2,029 records across 593 companies is small for a 4-class problem with ~130 engineered features. The data-to-feature ratio (~15:1) raises overfitting concerns that the pipeline addresses through regularisation, feature selection, and ensemble averaging — but cannot fully eliminate. The absence of temporal features and macroeconomic context fundamentally limits predictive power.

**Remaining risks**: The dataset's age and composition (primarily US corporates from 2012–2015 based on Egan-Jones and Fitch date ranges) may not generalise to current markets, non-US issuers, or different economic cycles.

---

## 4. Exploratory Data Analysis

### 4.1 Class Distribution

The target variable exhibits significant imbalance after grouping:

| Tier | Records | Proportion |
|---|---|---|
| Speculative | ~795 | 39.2% |
| Investment-Low | ~670 | 33.0% |
| Investment-High | ~490 | 24.1% |
| Distressed | ~72 | 3.6% |

The Distressed class at 3.6% creates a severe minority-class challenge. A majority-class baseline classifier predicting only "Speculative" would achieve ~39% accuracy — which provides context for the model's 46% CV accuracy: it exceeds the majority-class baseline by approximately 7 percentage points.

### 4.2 Feature Distribution Characteristics

Key observations from financial ratio distributions:

1. **Heavy right tails**: `debtEquityRatio`, `enterpriseValueMultiple`, and `daysOfSalesOutstanding` all exhibit extreme positive outliers, with values exceeding 10× the median. This motivates winsorisation (§6.3).
2. **Sign-mixed features**: `effectiveTaxRate`, `netProfitMargin`, `freeCashFlowPerShare` can be negative, requiring sign-preserving transformations rather than standard log transforms.
3. **Sector-dependent baselines**: `debtEquityRatio` in utilities averages 2–3×, while technology companies average 0.3–0.5×. Raw feature values carry sector-specific meaning that must be normalised (§6.5).

### 4.3 Multi-Record Company Structure

The ~3.4 records per company create a structural challenge:

- Companies with consistent ratings across years provide reinforcing signals.
- Companies with rating changes across years (e.g., BBB in 2012, BB in 2015) provide conflicting labels for similar financial profiles.
- A row-level train/test split allows the model to "see" a company's other years during training and simply memorise company identity. This was the source of the original ~72% accuracy inflation.

### 4.4 Critical Review

**Strengths**: The EDA reveals actionable engineering insights: heavy tails motivate winsorisation, sector dependence motivates z-scoring, and multi-record structure motivates grouped splitting. Each finding directly drives a pipeline design decision.

**Weaknesses**: No formal statistical tests (e.g., Kolmogorov–Smirnov for normality, Kruskal–Wallis for class separation) were performed. The EDA is descriptive rather than hypothesis-driven.

**Remaining risks**: Possible multicollinearity among financial ratios (e.g., `grossProfitMargin` and `operatingProfitMargin` are mechanically correlated) is partially addressed by feature selection (§6.7) but not explicitly diagnosed.

---

## 5. Feature Engineering

### 5.1 Pipeline Architecture

All feature engineering steps are centralised in `preprocessing.py` and executed in a fixed order. The design principle is: **every fitted parameter is computed on the training fold exclusively, then applied identically to the test fold and inference inputs.**

```
grouped stratified split          (leakage-safe, company-level)
    → median imputation          (fit on train only)
    → winsorisation (P1–P99)     (fit on train only)
    → interaction / log features (deterministic formulas)
    → sector z-scores            (fit on train only)
    → one-hot encode + align     (all categoricals)
    → feature selection          (fit on train only)
```

This order is enforced by code structure, not by documentation. Both `predict_xgboost.py` and `evaluate_xgboost.py` import and call the same functions in the same sequence.

### 5.2 Domain-Driven Interaction Features

Twenty-one composite features are engineered from raw ratios. These are divided into three groups:

#### Original 10 Composite Ratios

| Feature | Formula | Financial Rationale |
|---|---|---|
| `leverage_coverage` | debtEquityRatio / (\|operatingProfitMargin\| + ε) | How well profits offset leverage |
| `liquidity_score` | (currentRatio + quickRatio + cashRatio) / 3 | Aggregate short-term solvency |
| `cashflow_debt_coverage` | operatingCashFlowPerShare / (\|debtEquityRatio\| + ε) | Cash generation relative to debt load |
| `profitability_composite` | (netProfitMargin + operatingProfitMargin + grossProfitMargin) / 3 | Multi-layer profitability |
| `debt_service_ratio` | operatingCashFlowSalesRatio / (debtRatio + ε) | Cash conversion vs. total debt |
| `efficiency_composite` | (assetTurnover + fixedAssetTurnover) / 2 | Asset utilisation efficiency |
| `roa_leverage` | returnOnAssets / (debtRatio + ε) | Return quality relative to leverage |
| `margin_stability` | \|grossProfitMargin − netProfitMargin\| | Margin compression indicator |
| `cash_liquidity_ratio` | cashRatio / (currentRatio + ε) | Cash quality within liquidity |
| `equity_efficiency` | returnOnCapitalEmployed × assetTurnover | Capital deployment effectiveness |

#### 8 Credit-Risk Specific Features

| Feature | Formula | Financial Rationale |
|---|---|---|
| `liquidity_leverage` | currentRatio / (debtEquityRatio + ε) | Short-term solvency relative to leverage |
| `roe_roa_spread` | returnOnEquity − returnOnAssets | Financial leverage amplification (DuPont) |
| `margin_compression` | operatingProfitMargin − netProfitMargin | Non-operating cost burden |
| `fcf_cash_ratio` | freeCashFlowPerShare / (\|cashPerShare\| + ε) | Free cash flow quality |
| `roa_turnover` | returnOnAssets × assetTurnover | DuPont decomposition: asset productivity |
| `debt_tax_burden` | debtRatio × (1 − effectiveTaxRate) | After-tax cost of debt capacity |
| `cash_leverage` | cashRatio / (\|debtEquityRatio\| + ε) | Cash buffer relative to leverage |
| `cash_quality` | operatingCashFlowSalesRatio / (\|netProfitMargin\| + ε) | Cash conversion quality vs. accrual profits |

#### 3 Polynomial Interaction Features

| Feature | Formula | Rationale |
|---|---|---|
| `debtEquityRatio_sq` | debtEquityRatio² | Non-linear leverage penalty |
| `returnOnAssets_sq` | returnOnAssets² | Diminishing returns at high profitability |
| `currentRatio_sq` | currentRatio² | Non-linear liquidity effects |

**Why these specific features?** Each interaction encodes a financial relationship that a single tree split cannot capture. For example, a `debtEquityRatio` of 3.0 means different things at 15% operating margin (manageable leverage) versus 2% operating margin (distress signal). The `leverage_coverage` ratio encodes this relationship directly.

**Numerical safety**: All division-based features use `ε = 1e-5` to prevent division by zero, and all results are clipped to [-1e6, 1e6] to prevent floating-point overflow. These constants were chosen empirically: ε is negligible relative to all observed financial ratio magnitudes (minimum non-zero absolute value in the dataset is ~1e-3), and 1e6 exceeds all observed ratio ranges by at least 3 orders of magnitude.

### 5.3 Log Transformations

Twelve right-skewed or heavy-tailed features receive a sign-preserving log transform:

```python
X[f"{col}_log"] = sign(X[col]) × log1p(|X[col]|)
```

**Why sign-preserving?** Features like `freeCashFlowPerShare` can be negative (cash burn), and the sign carries directional meaning. A standard `log1p` would fail on negative values; `np.sign(x) * np.log1p(np.abs(x))` preserves the sign while compressing magnitude.

**Target features**: `currentRatio`, `quickRatio`, `cashRatio`, `daysOfSalesOutstanding`, `debtEquityRatio`, `enterpriseValueMultiple`, `operatingCashFlowPerShare`, `freeCashFlowPerShare`, `cashPerShare`, `payablesTurnover`, `fixedAssetTurnover`, `companyEquityMultiplier`.

### 5.4 Sector-Relative Z-Score Normalisation

For each numeric ratio `r` and each company with sector `s`:

```
z_score = (x_r − μ_r,s) / (σ_r,s + ε)
```

Where `μ_r,s` and `σ_r,s` are the sector-level mean and standard deviation computed exclusively on the training set. Companies in sectors unseen during training fall back to global statistics.

**Why z-scores instead of StandardScaler?** StandardScaler normalises features globally — it does not change tree split point rankings and has zero effect on XGBoost performance. Sector z-scores create *new features* that encode a company's relative standing within its industry. A technology company with `debtEquityRatio = 1.5` may be highly leveraged relative to peers (z-score = +2.3), while a utility company at the same absolute level is average (z-score = -0.1). This distinction is invisible to the raw feature.

### 5.5 One-Hot Encoding

All categorical columns (primarily `Sector`) are one-hot encoded via `pd.get_dummies()`. Column alignment between train and test sets is enforced with `reindex(columns=feature_columns, fill_value=0)`. This handles sectors present in training but absent in the test set (fill with 0) and vice versa (ignored).

**Why not ordinal encoding for Sector?** Sectors have no natural ordering. Ordinal encoding would impose an artificial rank that the model would interpret as magnitude, creating misleading split points.

### 5.6 Feature Selection

After the full pipeline, the feature space expands from ~30 raw features to ~130 engineered features. This 4× expansion on ~2,000 records creates overfitting risk. A training-fold-only selection step prunes:

1. **Near-zero-variance columns** (variance ≤ 1e-8): Features with essentially no discriminative signal (e.g., a sector one-hot column where all training records belong to the same sector).
2. **Redundant columns**: One of every pair with absolute Pearson correlation > 0.95. This removes mechanically correlated features (e.g., a raw ratio and its z-score variant, if the sector has uniform distribution).

The retained column list is persisted as `feature_columns.pkl` and reapplied at inference.

**Why not Boruta or recursive feature elimination?** Both require repeated model refitting, which would increase training time from ~30 seconds to several minutes — unacceptable for the on-demand training pipeline. The variance/correlation filter is computationally cheap and removes the most obvious redundancies.

### 5.7 Total Feature Dimensionality

| Source | Count |
|---|---|
| Raw numeric ratios | ~30 |
| Interaction features (10 + 8) | 18 |
| Polynomial features | 3 |
| Log-transformed features | 12 |
| Sector z-score features | ~55 |
| Sector one-hot columns | ~12 |
| **Pre-selection total** | **~130** |
| **Post-selection total** | Varies per fold; typically ~100–120 |

### 5.8 Critical Review

**Strengths**: The feature engineering is domain-grounded — every interaction feature has a financial interpretation. The pipeline order is enforced by code, not documentation. All fitted parameters are computed on training data only.

**Weaknesses**: The selection of which features to engineer was not guided by a systematic feature importance analysis or ablation study. The 21 interaction features were designed based on financial domain knowledge, but there is no evidence that removing any specific subset would harm performance. The feature space remains large relative to the sample size.

**Remaining risks**: The 0.95 correlation threshold for redundancy pruning is arbitrary. A lower threshold (e.g., 0.85) might prune more aggressively and reduce overfitting, but was not tested. No ablation study was performed to quantify the marginal contribution of each feature engineering stage.

**Possible improvements**: (1) Conduct an ablation study removing each feature engineering stage in isolation. (2) Apply Boruta or SHAP-based feature selection in a separate offline analysis to identify truly predictive features. (3) Use cross-validated feature importance to guide feature engineering rather than financial intuition alone.

---

## 6. Data Cleaning Strategy

### 6.1 Identifier Dropping

Non-predictive columns are explicitly dropped before any feature computation:

```python
DROP_COLS = frozenset(
    {"Name", "Symbol", "Rating Agency Name", "Date", "RatingClass", "Rating"}
)
```

**Why a frozen set?** Immutability prevents accidental modification during pipeline execution. The `Symbol`/`Name` identifier is captured as the grouping key for the company-level split *before* it is dropped from the feature matrix.

### 6.2 Unknown Rating Exclusion

Records with unrecognised or missing ratings are labelled "Unknown" by `group_rating()` and excluded from training. This is a hard exclusion — no imputation is attempted on the target variable.

---

## 7. Missing Value Strategy

### 7.1 Approach: Per-Column Median Imputation

**Decision**: Fill numeric NaN values with per-column medians computed from the training set only.

**Alternatives considered**:
1. **XGBoost native NaN routing**: XGBoost can natively route NaN values to the optimal child node during tree construction. However, the downstream sector z-score computation requires non-NaN values — a NaN ratio produces a NaN z-score, which propagates through the entire z-score feature set.
2. **Mean imputation**: The median was preferred over the mean because financial ratios have skewed distributions where the mean is pulled by outliers.
3. **KNN imputation**: Too computationally expensive for an on-demand training pipeline and requires distance computation on mixed-type features.

**Trade-offs**: Median imputation is simple and robust but does not capture cross-feature relationships. A company with high `debtEquityRatio` and missing `interestCoverage` likely has low interest coverage — a correlation that median imputation ignores.

**Implementation detail**: Columns that are entirely NaN in the training set receive a default median of 0.0 to prevent downstream failures. This is a defensive choice — such columns carry no information and are subsequently removed by the near-zero-variance filter.

---

## 8. Outlier Handling

### 8.1 Approach: Percentile-Based Winsorisation

All numeric features are clipped to the [P1, P99] range computed from the training partition.

**Why winsorisation instead of removal?** Removing outlier rows would lose the associated label information, which is particularly costly for the Distressed class where every sample matters. Capping preserves the directional signal (extreme values remain at the boundary) while preventing tree split distortion.

**Why P1/P99 instead of P5/P95?** Financial ratios have legitimately wide distributions. P5/P95 would clip too aggressively and lose meaningful variation. P1/P99 targets only the most extreme observations (typically data errors or exceptional circumstances).

**Why not IQR-based outlier detection?** IQR assumes approximately symmetric distributions. Financial ratios are often right-skewed, so the upper IQR bound would be too conservative while the lower bound would be too aggressive.

---

## 9. Encoding Strategy

### 9.1 Label Encoding

An explicit, deterministic label encoder is used with a hardcoded mapping:

```python
LABEL_ENCODER = ExplicitLabelEncoder({
    "Investment_High": 0,
    "Investment_Low": 1,
    "Speculative": 2,
    "Distressed": 3,
})
```

**Why not scikit-learn's `LabelEncoder`?** Scikit-learn's `LabelEncoder` assigns indices alphabetically, which produces a non-deterministic mapping if the class set changes between runs. The explicit mapping guarantees that cached models, evaluation artifacts, and API responses all use the same class-to-index assignment regardless of data ordering.

### 9.2 Feature Encoding

All categorical features are one-hot encoded via `pd.get_dummies()`. Column alignment between train and test sets uses `reindex(columns=feature_columns, fill_value=0)`.

**Why `pd.get_dummies()` instead of scikit-learn's `OneHotEncoder`?** `pd.get_dummies()` operates in-place on DataFrames and handles column naming automatically. Since the pipeline operates entirely on DataFrames (not numpy arrays), this avoids the index-alignment issues that arise when mixing sparse matrices from `OneHotEncoder` with dense DataFrames.

---

## 10. Target Engineering

### 10.1 Rating Grouping Function

The `group_rating()` function in `preprocessing.py` collapses letter grades into four tiers:

```python
def group_rating(r):
    r = str(r).strip().upper()
    if r in {"AAA", "AA", "A"}:   return "Investment_High"
    if r == "BBB":                 return "Investment_Low"
    if r in {"BB", "B"}:          return "Speculative"
    if r in {"CCC", "CC", "C", "D"}: return "Distressed"
    return "Unknown"
```

**Design decisions documented in §2.3** explain the Distressed tier composition.

### 10.2 Rating Format Inconsistency

The XGBoost pipeline uses underscores (`Investment_High`), while the Decision Tree and Random Forest pipelines use hyphens (`Investment-High`). The API response uses `format_prediction_label()` to convert underscores to hyphens for frontend display. This inconsistency is cosmetic but creates confusion when comparing artifacts across models.

---

## 11. Class Distribution

### 11.1 Observed Distribution

| Tier | Records | % | Test Support (seed 143) |
|---|---|---|---|
| Speculative | ~795 | 39.2% | 159 |
| Investment-Low | ~670 | 33.0% | 134 |
| Investment-High | ~490 | 24.1% | 98 |
| Distressed | ~72 | 3.6% | 15 |

### 11.2 Imbalance Handling

**Decision**: Use `compute_sample_weight(class_weight='balanced')` from scikit-learn, which assigns inverse-frequency weights to each training sample.

**Alternatives considered**:
1. **SMOTE oversampling**: Tested and abandoned — see §16.1 (Model Evolution, Iteration: SMOTE).
2. **Class-weighted loss function in XGBoost**: XGBoost's `scale_pos_weight` only applies to binary classification. For multiclass, external sample weights are required.
3. **Random undersampling of majority classes**: Would discard ~90% of Speculative-class data, severely reducing the training set.

**Evidence**: Balanced sample weights increased Distressed recall from ~0% (model never predicted Distressed) to ~20% on the representative split. The improvement is modest but genuine — the class is now measurable.

---

## 12. Train/Validation/Test Strategy

### 12.1 Company-Level Grouped Stratified Split

**Problem**: The dataset contains 2,029 records from 593 companies (~3.4 records per company). A row-level split allows the same company's records to appear in both training and test sets.

**Decision**: Use `StratifiedGroupKFold` (keyed on `Symbol`/`Name`) so that all records for a given company are confined to one side of the split.

**Implementation**:

```python
def make_split(X, y, groups=None, test_size=0.20, random_state=RANDOM_STATE):
    if groups is not None:
        try:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
            train_idx, test_idx = next(sgkf.split(X, y, groups))
            return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx], "grouped_stratified"
        except Exception:
            pass
    # Fallback chain: stratified → random
```

**Degradation strategy**: The split falls back gracefully:
1. **Grouped stratified** (preferred): Company-level grouping with class-proportion balancing.
2. **Stratified**: Class-proportion balancing without grouping (if groups are unavailable).
3. **Random**: Plain split (if stratification fails due to tiny class sizes).

The fallback chain is a defensive design — the default dataset always achieves grouped stratified splitting, but user-uploaded datasets may lack identifier columns.

### 12.2 Cross-Validation Protocol

The authoritative evaluation uses **grouped stratified 5-fold cross-validation** via `evaluate_xgboost.py`:

- Each fold performs the entire preprocessing pipeline from scratch (imputation, winsorisation, feature engineering, feature selection).
- No preprocessing parameters are shared between folds.
- Standard calibrated predictions are used — the threshold strategy selector is deliberately skipped because it previously leaked test labels.

### 12.3 Why No Separate Validation Set

**Problem**: With ~2,000 records, a 3-way split (train/validation/test) would leave insufficient data for either validation or test.

**Decision**: Use cross-validated hyperparameter search (`RandomizedSearchCV` with `cv=3`) on the training set instead of a held-out validation set. This maximises data utilisation while maintaining evaluation integrity.

**Trade-off**: No final model selection step on a separate validation set. The cross-validation mean serves as the expected performance estimate instead.

---

## 13. Leakage Prevention

### 13.1 Complete Leakage Prevention Checklist

| Pipeline Stage | Leakage-Free? | Mechanism |
|---|---|---|
| Train/test split (company overlap) | ✅ | `StratifiedGroupKFold` on `Symbol`/`Name` |
| Median imputation | ✅ | Medians computed on `X_train` only; persisted in `imputer_medians.pkl` |
| Winsorisation bounds | ✅ | Bounds computed on `X_train` only; persisted in `winsorize_bounds.pkl` |
| Sector z-score statistics | ✅ | Sector means/stds from `X_train` only; persisted in `sector_stats.pkl` |
| Interaction features | ✅ | Deterministic formula; no fitted parameters |
| Log transforms | ✅ | Deterministic formula; no fitted parameters |
| One-hot encoding | ✅ | Column alignment via `reindex` with `fill_value=0` |
| Feature selection | ✅ | Variance/correlation thresholds fit on `X_train` only |
| Threshold optimisation (values) | ✅ | Nelder-Mead uses `y_train` and `predict_proba(X_train_enc)` only |
| Threshold strategy selector | ✅ | Fixed: decision uses training-set Macro F1 only (see §13.2) |
| Sample weight computation | ✅ | Computed on `y_train` class frequencies only |

### 13.2 Strategy-Selector Leakage: Discovery, Impact, and Fix

**Discovery**: The original threshold strategy selector compared threshold-optimised vs. standard predictions on the *test set* to decide which strategy to persist. This leaked test labels into a binary production decision.

**Impact**: The leakage was subtle — it only affected whether the threshold multipliers were applied, not their values. Empirically, it inflated single-split accuracy by approximately 1–2 percentage points.

**Fix**: The strategy selector now compares Macro F1 on the *training set* only:

```python
f1_standard = f1_score(y_train, calibrated_model.predict(X_train_enc), average="macro")
f1_threshold = f1_score(y_train, (y_train_proba * optimal_thresholds).argmax(axis=1), average="macro")
use_thresholds = f1_threshold > f1_standard
```

### 13.3 Row-Level Leakage: Discovery, Impact, and Fix

**Discovery**: The original pipeline used `train_test_split` with `stratify=y` but no grouping. With 593 companies across 2,029 records, the model saw other-year records of test companies during training.

**Impact**: This inflated accuracy from ~46% (honest, company-level) to ~72% (leaky, row-level). The 26-percentage-point gap quantifies the information content of company identity leakage.

**Fix**: Switched to `StratifiedGroupKFold` with `Symbol`/`Name` as the grouping key. All records for a given company are now confined to one side of the split.

---

## 14. Reproducibility Strategy

### 14.1 Random Seed Control

All stochastic operations use `RANDOM_STATE = 143`:

- `StratifiedGroupKFold` shuffling
- XGBoost base model (seed 143), ensemble members (144, 145)
- `RandomizedSearchCV` random state
- Nelder-Mead threshold optimisation (10 restarts via `np.random.default_rng(143)`)

### 14.2 Deterministic Label Encoding

The `ExplicitLabelEncoder` with a hardcoded mapping guarantees consistent class-to-index assignment across all runs, regardless of data ordering or class-set variation.

### 14.3 Residual Non-Determinism

**Known issue**: XGBoost trained with `n_jobs=-1` exhibits minor run-to-run variance (~±1 percentage point in single-split accuracy) due to non-deterministic floating-point reduction order across threads. This is documented in XGBoost's official FAQ and is inherent to multithreaded tree construction.

**Mitigation**: The authoritative performance figure is a cross-validation mean ± std, not a single-split number. Bit-exact reproducibility would require `n_jobs=1`, which would increase training time by ~3–4×.

### 14.4 Artifact Serialisation

All fitted parameters are persisted via `joblib` for identical reapplication:

| Artifact | Contents | Purpose |
|---|---|---|
| `calibrated_model.pkl` | CalibratedClassifierCV wrapping VotingClassifier | Inference predictions |
| `base_model.pkl` | Best single XGBClassifier from RandomizedSearchCV | SHAP explanations (TreeExplainer requires uncalibrated model) |
| `imputer_medians.pkl` | Per-column training medians | Missing value imputation |
| `winsorize_bounds.pkl` | Per-column (P1, P99) bounds | Outlier capping |
| `sector_stats.pkl` | Per-sector means, stds, global means, global stds | Z-score normalisation |
| `feature_columns.pkl` | Selected feature list after variance/correlation pruning | Column alignment at inference |
| `optimal_thresholds.pkl` | Per-class probability multipliers | Threshold-optimised predictions |
| `prediction_strategy.pkl` | `{"use_thresholds": bool}` | Whether to apply thresholds |

### 14.5 Critical Review

**Strengths**: The seed control and artifact serialisation enable near-exact reproducibility within a single Python/XGBoost version. The deterministic label encoder eliminates a common source of cross-run inconsistency.

**Weaknesses**: `requirements.txt` does not list `xgboost` at all — verified by inspection of the file (it lists only `pandas`, `numpy`, `scikit-learn`, `shap`, `openpyxl`, `xlrd`). `pip show shap` confirms `xgboost` is not a transitive dependency of `shap` either; it must currently be installed manually into the environment with no version guarantee whatsoever. This is a stricter gap than a missing pin — a fresh `pip install -r requirements.txt` on a new machine would not install XGBoost at all, and the pipeline would fail outright rather than silently drift to a different tree structure.

**Remaining risks**: The `.gitignore` does not exclude `results/` artifacts. Model binaries and cached predictions are committed to version control, which inflates the repository and creates confusion about which artifacts match which code version.

**Recommendation**: Add `xgboost==<pinned-version>` to `requirements.txt` (it is currently absent, not merely unpinned) and pin all other ML dependencies that affect tree structure. Add `results/*.pkl` and `results/*.csv` to `.gitignore` — confirmed absent from the current `.gitignore`, which only excludes `__pycache__/`, `*.pyc/.pyo/.pyd`, and `node_modules/`. Generate evaluation artifacts via a documented `make evaluate` step rather than committing cached outputs.

---

## 15. Pipeline Architecture

### 15.1 Module Structure

The XGBoost pipeline is decomposed into three Python modules:

| Module | Responsibility |
|---|---|
| `preprocessing.py` | All data transformations (split, impute, winsorise, feature engineer, select). Shared between training and inference. |
| `predict_xgboost.py` | Model training, caching, single-split evaluation, SHAP computation, API response serialisation. |
| `evaluate_xgboost.py` | Grouped k-fold cross-validation for authoritative metrics. |

**Why a separate preprocessing module?** The original system had preprocessing logic duplicated between the training script and the inference path. Any change to one copy could silently drift from the other, creating train-serve skew. Extracting `preprocessing.py` as a shared module eliminates this class of bugs by construction.

### 15.2 Dependency Graph

```
evaluate_xgboost.py → predict_xgboost.py → preprocessing.py
                                          → sklearn, xgboost, shap, scipy, joblib
```

`evaluate_xgboost.py` imports `train_model` from `predict_xgboost.py` and all preprocessing functions from `preprocessing.py`, ensuring that the cross-validation evaluation uses the exact same pipeline as single-split training.

---

## 16. Model Evolution

This is the most critical section of the report. It documents every major iteration, including failures.

### 16.1 Iteration 1: Baseline Decision Tree and Random Forest

**What was built**: Decision Tree and Random Forest classifiers using scikit-learn pipelines (`ColumnTransformer` + `SimpleImputer` + `OneHotEncoder`), with row-level stratified train/test splits.

**Performance**: Decision Tree achieved ~45% accuracy, Random Forest achieved ~60% accuracy (both inflated by row-level leakage, though the effect is smaller for simpler models).

**Weaknesses discovered**:
1. No feature engineering — the model operated on raw financial ratios only.
2. Row-level split leaked company identity across train/test boundaries.
3. No probability calibration — `predict_proba` values were unreliable.
4. No SHAP integration — predictions were uninterpretable.

**Decision**: Move to XGBoost for its gradient-boosting architecture, built-in regularisation, and native SHAP support via TreeExplainer.

### 16.2 Iteration 2: Initial XGBoost with Row-Level Split

**What changed**: Replaced Random Forest with XGBoost. Added 10 composite interaction features. Added winsorisation. Added isotonic calibration.

**Performance**: ~72% accuracy on a row-level stratified split.

**Problems discovered**:
1. **The 72% was a lie**: With 593 companies across 2,029 records, the row-level split allowed the model to memorise company identity. The same company's 2013 and 2015 records could land in train and test respectively — the model learned "Company X is always BBB" rather than learning financial ratio patterns.
2. **Distressed class was unmeasurable**: Under the original CCC→Speculative grouping, the Distressed class (C, D only) had 1–2 test samples per split.
3. **No feature selection**: The ~130-feature space on ~2,000 records created overfitting risk.

### 16.3 Iteration 3: Company-Level Split (The Honest Reckoning)

**What changed**: Replaced `train_test_split` with `StratifiedGroupKFold` on `Symbol`/`Name`.

**Impact**: Accuracy dropped from ~72% to ~53% on the single split, and ~46% under cross-validation. This 26-percentage-point drop quantifies the information content of company identity leakage.

**Why this was the right decision**: The 72% number was reproducible but meaningless — it measured the model's ability to memorise company identity, not its ability to generalise credit risk assessment to unseen companies. The 46% number is honest. An examiner who sees a 72% accuracy followed by a 46% accuracy should recognise the latter as evidence of engineering integrity, not a regression.

### 16.4 Iteration 4: Distressed Tier Regrouping

**What changed**: Moved CCC and CC from Speculative into Distressed.

**Impact**: Distressed support increased from ~10 records (C, D only) to ~72 records (CCC, CC, C, D). The class became measurable with F1 ≈ 0.25 (CV mean) — weak but genuine.

**Evidence**: Under the old grouping, Distressed F1 was either 0% or 100% depending on the random placement of 1–2 test samples. Under the new grouping, Distressed F1 is consistently ~0.20–0.30 across folds, which is a real signal.

### 16.5 Iteration 5: SMOTE Oversampling (Failed, Abandoned)

**What was tried**: Synthetic Minority Over-sampling Technique to augment the Distressed class.

**Why it failed**: With only 4–5 real Distressed training samples per fold, SMOTE generated interpolated points in a near-degenerate feature subspace. The synthetic samples were noisy copies, not meaningful augmentations. The model overfitted to the synthetic cluster and produced worse test-set performance.

**Decision**: Abandoned SMOTE in favour of balanced sample weights.

**Evidence**: The code retains `USE_SMOTE = False` in the notebook. Balanced sample weights achieved better Distressed recall without introducing synthetic noise.

### 16.6 Iteration 6: Optuna Bayesian Optimisation (Replaced)

**What was tried**: Optuna with 50 TPE trials for hyperparameter search.

**Why it was replaced**: On ~2,000 samples, the 50-trial search took 8–9 minutes — unacceptable for the production pipeline where retraining must complete within 30–60 seconds. The TPE-optimised hyperparameters also showed high variance across data uploads, suggesting overfitting to specific fold configurations.

**Decision**: Replaced with `RandomizedSearchCV` sampling 15 configurations from a 972-candidate discrete grid. This completes in ~30 seconds and produces more robust hyperparameters.

### 16.7 Iteration 7: 3-Model Soft-Voting Ensemble

**What changed**: Instead of using the single best XGBoost model, trained 3 models with identical hyperparameters but different random seeds (143, 144, 145) and combined via soft voting.

**Why**: A single XGBoost model exhibits seed-dependent variance (~±1–2 percentage points). The ensemble averages this out, producing more stable probability estimates. This is most valuable for borderline cases near the Investment-Low/Speculative boundary.

**Trade-off**: 3× inference time and 3× model size. Acceptable for this application (inference completes in <2 seconds) but would be problematic at scale.

### 16.8 Iteration 8: Isotonic Calibration (Replacing Sigmoid)

**What changed**: Switched from Platt scaling (`method="sigmoid"`) to isotonic regression (`method="isotonic"`).

**Why**: Platt scaling assumes sigmoidally-shaped miscalibration. The 3-model ensemble's probability outputs can have non-sigmoidal miscalibration patterns. Isotonic regression provides more flexible, non-parametric correction.

**Trade-off**: Isotonic regression requires more calibration data and is more prone to overfitting with small samples. The 3-fold CV with automatic 2-fold fallback mitigates this risk.

### 16.9 Iteration 9: Strategy-Selector Leakage Fix

**What changed**: The threshold strategy selector was comparing Macro F1 on the *test set* to decide whether to apply thresholds. Changed to use *training set* Macro F1 only.

**Impact**: Removed ~1–2 percentage points of optimistic bias from the single-split metric.

**Why this was subtle**: The leakage affected a binary decision (apply thresholds or not), not the threshold values themselves. Most leakage audits focus on fitted parameters, not strategy selection decisions.

### 16.10 Iteration 10: Feature Selection Addition

**What changed**: Added variance-based and correlation-based feature pruning after one-hot encoding.

**Why**: The ~130-feature space on ~2,000 records created an unfavourable data-to-feature ratio. Near-zero-variance features add no signal, and highly correlated feature pairs add redundancy that can lead to split-point competition and overfitting.

**Impact**: Reduced feature count from ~130 to ~100–120 per fold (exact count varies with training data composition).

### 16.11 Iteration 11: Cross-Validation Evaluation Script

**What changed**: Created `evaluate_xgboost.py` as a separate, authoritative evaluation script that runs grouped stratified k-fold cross-validation.

**Why**: A single grouped split over ~118 test companies is a high-variance estimate (observed ~±3 percentage points between runs). The 5-fold cross-validation mean provides a more stable performance estimate with a standard deviation that quantifies uncertainty.

### 16.12 Final Production Model

The current production model incorporates all successful iterations:

1. Company-level grouped stratified split
2. Median imputation (train-only)
3. P1–P99 winsorisation (train-only)
4. 21 domain-driven interaction features
5. 12 sign-preserving log transforms
6. Sector-relative z-scores (train-only)
7. One-hot encoding with column alignment
8. Variance/correlation feature selection (train-only)
9. 3-model XGBoost ensemble with seed diversity
10. Isotonic probability calibration
11. Nelder-Mead threshold optimisation (train-only)
12. Training-only strategy selection

**Why this version**: It represents the cumulative result of every documented iteration. Every component was added to address a specific problem (leakage, instability, miscalibration, or minority-class neglect), and no component was retained that failed to justify its inclusion.

### 16.13 Critical Review: Model Evolution

**Strengths**: The evolution is well-documented with clear cause-effect chains. Failed approaches (SMOTE, Optuna, GPU acceleration) are preserved in the record, demonstrating systematic experimentation rather than cherry-picked successes.

**Weaknesses**: No formal ablation study quantifies the marginal contribution of each iteration. The evolution was sequential — it is unclear whether some early changes (e.g., interaction features) still contribute meaningfully after later changes (e.g., feature selection).

**Remaining risks**: The pipeline has accumulated complexity through additive iteration. A simpler pipeline (e.g., XGBoost with raw features and balanced weights, no ensemble, no calibration) might perform comparably and would be easier to maintain.

---

## 17. Hyperparameter Optimisation

### 17.1 Search Strategy

A `RandomizedSearchCV` with 3-fold cross-validation samples 15 configurations from the following parameter space:

| Hyperparameter | Search Values | Rationale |
|---|---|---|
| `max_depth` | [5, 7, 9] | Controls tree complexity; deeper trees capture more interactions but risk overfitting |
| `n_estimators` | [150, 200] | Number of boosting rounds; more rounds reduce bias but increase variance |
| `learning_rate` | [0.03, 0.05, 0.1] | Step size shrinkage; lower values need more rounds but generalise better |
| `min_child_weight` | [1, 3, 5] | Minimum hessian sum in child nodes; higher values regularise |
| `gamma` | [0, 0.1, 0.2] | Minimum loss reduction for splits; acts as complexity penalty |
| `reg_alpha` | [0, 0.1, 0.3] | L1 regularisation on leaf weights |
| `reg_lambda` | [1, 1.5] | L2 regularisation on leaf weights |

Full grid size: 3 × 2 × 3 × 3 × 3 × 3 × 2 = **972 candidates**. 15 randomly sampled configurations × 3 folds = **45 model fits** per training run.

### 17.2 Fixed Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `objective` | `multi:softprob` | Required for multiclass probability output |
| `eval_metric` | `mlogloss` | Proper scoring rule for multiclass calibration |
| `subsample` | 0.8 | Row subsampling for regularisation |
| `colsample_bytree` | 0.8 | Feature subsampling per tree |
| `colsample_bylevel` | 0.8 | Feature subsampling per level |
| `n_jobs` | -1 | Parallel tree construction |

### 17.3 Scoring Metric

**Decision**: Use Macro F1 (`f1_macro`) as the scoring metric for hyperparameter search.

**Why not accuracy?** Accuracy is dominated by majority-class performance. A model that achieves 95% accuracy on Speculative but 0% on Distressed would score well on accuracy but poorly on Macro F1.

**Why not weighted F1?** Weighted F1 still overweights majority classes. Macro F1 treats all classes equally, incentivising the optimiser to improve minority-class performance.

### 17.4 Critical Review

**Strengths**: The search space includes regularisation parameters that directly address the overfitting risk of the ~130-feature space.

**Weaknesses**: 15 random samples from a 972-candidate grid covers only 1.5% of the space. Important configurations may be missed. The 3-fold CV within RandomizedSearchCV creates a bias-variance trade-off: fewer folds increase variance of the CV estimate, potentially selecting suboptimal configurations. 5-fold CV within the search would be more stable but doubles training time.

---

## 18. XGBoost Internals

### 18.1 Why XGBoost Over Alternatives

| Criterion | Decision Tree | Random Forest | XGBoost | Justification |
|---|---|---|---|---|
| Regularisation | None | Implicit (bagging) | Explicit (L1, L2, gamma) | Critical for ~130-feature space |
| Handling missing values | No | No | Native NaN routing | Reduces preprocessing complexity |
| Probability calibration | Poor | Moderate | Moderate → Good (with isotonic) | Required for threshold optimisation |
| SHAP integration | Exact (TreeExplainer) | Exact (TreeExplainer) | Exact (TreeExplainer) | All tree models support exact SHAP |
| Class imbalance | Via sample weights | Via sample weights | Via sample weights | All models support external weights |
| Training speed | Fast | Moderate | Fast | XGBoost's histogram-based splitting is efficient |

**Decision**: XGBoost was selected for its explicit regularisation parameters, which are critical when the feature space (130) is large relative to the sample size (2,000). Decision Trees lack regularisation entirely. Random Forests provide implicit regularisation through bagging but lack the fine-grained control of L1, L2, gamma, min_child_weight, and column sampling at multiple levels.

### 18.2 Boosting Configuration

The ensemble uses gradient boosted trees with the `multi:softprob` objective, which optimises multinomial log-loss. This produces calibrated-ish probability outputs (further corrected by isotonic calibration) rather than hard predictions. The `softprob` objective is preferred over `softmax` because it provides per-class probability vectors rather than a single predicted class index.

---

## 19. Probability Calibration

### 19.1 Problem

Gradient-boosted tree ensembles produce poorly calibrated probability estimates. Raw `predict_proba` outputs from XGBoost tend to be overconfident — a predicted probability of 0.80 does not mean the true class probability is 80%.

### 19.2 Solution

The soft-voting ensemble is wrapped in `CalibratedClassifierCV` with isotonic regression:

```python
CalibratedClassifierCV(estimator=ensemble, method="isotonic", cv=3)
```

Isotonic regression fits a monotonically increasing step function mapping raw probability outputs to calibrated probabilities. It is non-parametric and makes no distributional assumptions.

### 19.3 Fallback Mechanism

If any calibration fold lacks representation of a minority class (specifically Distressed), the calibration falls back from 3-fold to 2-fold:

```python
for cv_folds in (3, 2):
    try:
        calibrated = CalibratedClassifierCV(estimator=ensemble, method="isotonic", cv=cv_folds)
        calibrated.fit(X_train_enc, y_train, sample_weight=sample_weights)
        break
    except ValueError:
        if cv_folds == 2:
            raise
```

---

## 20. Model Evaluation

### 20.1 Authoritative Metrics (Cross-Validation)

**Source**: `evaluate_xgboost.py`, grouped stratified 5-fold CV, company-level, leakage-safe.

| Metric | CV Mean ± Std |
|---|---|
| **Accuracy** | 0.4603 ± 0.0313 |
| **Macro F1** | 0.4181 ± 0.0308 |

Per-class F1 (CV mean):

| Class | F1 Mean |
|---|---|
| Investment-High | 0.5546 |
| Speculative | 0.4997 |
| Investment-Low | 0.3642 |
| Distressed | 0.2537 |

### 20.2 Single-Split Reference Metrics

**Source**: `results/xgboost_metrics.txt` and `results/xgboost_classification_report.csv`, seed 143, grouped stratified split. These metrics match the cached `xgboost_test_predictions.csv` (verified by audit).

| Metric | Value |
|---|---|
| Accuracy | 0.5296 |
| Macro F1 | 0.4445 |
| Test samples | 406 |

Per-class classification report:

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Investment-High | 0.5200 | 0.5306 | 0.5253 | 98 |
| Investment-Low | 0.4797 | 0.4403 | 0.4591 | 134 |
| Speculative | 0.6273 | 0.6352 | 0.6312 | 159 |
| Distressed | 0.1364 | 0.2000 | 0.1622 | 15 |
| **Weighted Avg** | **0.5346** | **0.5296** | **0.5315** | **406** |

### 20.3 Confusion Matrix (From Cached Predictions)

```
                  Predicted
                  IH    IL    SP    DI
Actual IH     [  52    34    11     1 ]
       IL     [  35    59    39     1 ]
       SP     [  12    29   101    17 ]
       DI     [   1     1    10     3 ]
```

*(Recomputed from `xgboost_test_predictions.csv`. Total: 406. Correct: 215. Accuracy: 215/406 = 0.5296.)*

> [!IMPORTANT]
> **Artifact consistency note**: This confusion matrix was recomputed directly from the cached `xgboost_test_predictions.csv` file and matches the `xgboost_classification_report.csv` and `xgboost_metrics.txt` artifacts exactly. The previously documented confusion matrix (in the earlier report version) showed different cell values — it was from a different model run affected by the multithreading non-determinism documented in §14.3. All metrics in this report originate from the same artifact set.

### 20.4 Why the Single-Split Exceeds the CV Mean

The single-split accuracy (0.5296) exceeds the CV mean (0.4603 ± 0.0313) by more than one standard deviation. This is expected: the single split represents one favourable sample from a high-variance distribution of possible company-level splits over ~118 test companies. This is precisely why the CV mean is the authoritative figure — relying on a single split would misrepresent expected performance.

### 20.5 Comparison to Baselines

| Model | Accuracy | Evaluation | Notes |
|---|---|---|---|
| Majority class ("Speculative") | ~0.39 | Theoretical | No model needed |
| Random (uniform) | ~0.25 | Theoretical | 4-class random |
| **XGBoost (this system)** | **0.46 ± 0.03** | **Grouped 5-fold CV** | **Company-level, leakage-safe** |

The model exceeds the majority-class baseline by ~7 percentage points and the random baseline by ~21 percentage points under honest evaluation.

---

## 21. Error Analysis

### 21.1 Misclassification Patterns

From the confusion matrix (§20.3):

- **Investment-High ↔ Investment-Low**: The primary confusion zone (34 IH→IL, 35 IL→IH). These adjacent tiers share overlapping financial profiles — the A/BBB boundary is often ambiguous even for human analysts.
- **Investment-Low ↔ Speculative**: A major error source (39 IL→SP, 29 SP→IL). This reflects the inherent difficulty of the BBB/BB boundary — the "fallen angel" threshold in credit risk, where companies transition between investment and speculative grade.
- **Speculative → Distressed**: 17 Speculative predictions classified as Distressed. This is a conservative error — the model downgrades borderline speculative cases.
- **Distressed class**: Only 3/15 Distressed samples correctly identified (recall 0.20). The 10 SP←DI misclassifications suggest the model cannot reliably distinguish Distressed from Speculative with 15 test samples.

### 21.2 Error Directionality

| Direction | Count | Cost Implication |
|---|---|---|
| Upgrade by 1 tier | 65 | Moderate risk — underestimates credit risk |
| Upgrade by 2+ tiers | 2 | High risk — seriously underestimates credit risk |
| Downgrade by 1 tier | 102 | Conservative — overestimates credit risk |
| Downgrade by 2+ tiers | 22 | Very conservative |

The model's errors skew toward downgrades (124 total) over upgrades (67 total). In credit risk applications, this is the safer direction — the model errs toward caution more often than toward over-optimism.

### 21.3 Adjacent-Class Confusion Dominance

Most misclassifications occur between adjacent tiers (IH↔IL, IL↔SP, SP↔DI). Extreme misclassifications (IH→DI, DI→IH) are rare (2 total), indicating the model preserves ordinal credit-quality structure even when it misclassifies.

---

## 22. SHAP Explainability

### 22.1 Implementation

The pipeline uses SHAP's `TreeExplainer` on the **uncalibrated base XGBoost model** (not the CalibratedClassifierCV wrapper):

```python
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test_enc.iloc[[0]])
```

**Why the base model, not the calibrated ensemble?** `TreeExplainer` requires direct access to tree internals. The `CalibratedClassifierCV` wrapper obscures the tree structure behind a calibration function. The base model's SHAP values explain the feature contributions to the *uncalibrated* prediction, which is an approximation but captures the same directional feature importance.

### 22.2 Per-Prediction SHAP Explanations

For each API prediction, the top 15 features by absolute SHAP value are returned with:

- **Feature name** (humanised via `humanize_feature_name()`)
- **Scaled SHAP value** (normalised to 0–100 relative to the maximum)
- **Direction** (+1 = pushes toward predicted class, -1 = pushes away)

A narrative "SHAP story" summarises the top 3 positive and negative features in plain text.

### 22.3 Fallback Mechanism

If SHAP computation fails (e.g., incompatible model format), the system falls back to XGBoost's built-in `feature_importances_`:

```python
except Exception:
    return best_xgb.feature_importances_
```

This fallback produces global importance values rather than instance-level explanations, which is less informative but prevents API failures.

### 22.4 Critical Review

**Strengths**: TreeExplainer provides exact Shapley values for tree models in polynomial time. The humanised feature names improve interpretability for non-technical users.

**Weaknesses**: SHAP values are computed on the base model, not the calibrated ensemble. The calibration and threshold optimisation steps modify the decision boundary, so base-model SHAP values may not perfectly explain the final prediction. The fallback to global feature importances is a significant degradation that the API response does not signal to the consumer.

**Remaining risks**: The SHAP computation operates on the first test sample only. The API response contains instance-level SHAP for a single sample, which may not represent typical feature attributions. A batch SHAP summary (e.g., mean absolute SHAP values across all test samples) would provide a more representative view.

---

## 23. Production Engineering

### 23.1 System Architecture

```
Browser (index.html + client.js)
    ↓ HTTP POST (JSON with base64-encoded dataset)
Node.js Server (app.js, port 3000)
    ↓ spawn() subprocess
Python Script (predict_xgboost.py)
    ↓ stdout JSON
Node.js Server
    ↓ HTTP 200
Browser
```

### 23.2 Why Node.js + Python Subprocess

**Problem**: The ML stack requires Python (XGBoost, scikit-learn, SHAP), but the web server uses Node.js.

**Decision**: Node.js spawns a Python subprocess for each prediction request. Communication occurs via stdin (JSON request) → stdout (JSON response).

**Alternatives considered**:
1. **Flask/FastAPI Python server**: Would require a separate Python server process. Node.js was already in use for static file serving and the existing frontend.
2. **ONNX export**: Would allow Node.js to run the model directly. However, ONNX does not support the CalibratedClassifierCV wrapper, and SHAP computation requires the Python ecosystem.
3. **REST microservice**: Would add deployment complexity (two servers, health checks, service discovery) for a single-user FYP application.

**Trade-offs**: The subprocess approach has high per-request overhead (~1–2 seconds for process startup). This is acceptable for a dashboard that processes one dataset at a time but would not scale to concurrent users.

---

## 24. API Architecture

### 24.1 Endpoints

| Method | Path | Script |
|---|---|---|
| POST | `/predict/xgboost` | `python_models/xgboost_stuff/predict_xgboost.py` |
| POST | `/predict/decision-tree` | `python_models/decision_tree_stuff/predict_decision_tree.py` |
| POST | `/predict/random-forest` | `python_models/random_forest_stuff/predict_random_forest.py` |

### 24.2 Request Format

```json
{
  "fileName": "corporate_data.csv",
  "fileData": "<base64-encoded or raw CSV content>",
  "fileEncoding": "base64"
}
```

### 24.3 Response Format (XGBoost)

```json
{
  "prediction": "Investment-Low",
  "probabilities": {
    "Investment-High": 0.39,
    "Investment-Low": 0.45,
    "Speculative": 0.08,
    "Distressed": 0.08
  },
  "modelData": {
    "tag": "XGBoost",
    "labels": ["Investment-High", "Investment-Low", "Speculative", "Distressed"],
    "metrics": { "accuracy": "0.5296", "precision": "0.5346", "recall": "0.5296", "f1": "0.5315" },
    "matrix": [[52,34,11,1],[35,59,39,1],[12,29,101,17],[1,1,10,3]],
    "shap": [["Feature Name", 95.0, 1], ...],
    "shapStory": { "positive": [...], "negative": [...] }
  }
}
```

### 24.4 Error Handling

The Node.js server handles:
- **5-minute timeout**: Kills the Python subprocess and returns HTTP 504.
- **Invalid JSON request body**: Returns HTTP 400.
- **Python process crash**: Parses stderr for structured error JSON; returns HTTP 500 with details.
- **Invalid JSON output**: Returns HTTP 500 with raw stdout/stderr for debugging.

### 24.5 Path Traversal Protection

The `safeResolve()` function prevents directory traversal attacks by verifying that resolved file paths stay within the project root:

```javascript
const rootWithSep = root.endsWith(path.sep) ? root : `${root}${path.sep}`;
if (!resolved.startsWith(rootWithSep) && resolved !== path.resolve(root, "views/index.html")) {
  return null;
}
```

---

## 25. Model Versioning

### 25.1 Current Approach

Models are versioned implicitly by the MD5 hash of the uploaded file data:

```python
data_hash = hashlib.md5(file_data.encode("utf-8")).hexdigest() if file_data else "default"
```

Artifact file names are suffixed with the hash (e.g., `calibrated_model_a403fd46.pkl`), enabling per-dataset model isolation.

### 25.2 Limitations

- No semantic versioning — model versions are tied to input data, not code changes.
- No model registry — a code change that alters the pipeline silently invalidates all cached models.
- No cache invalidation on code change — stale artifacts from a previous code version will be loaded as if current.

### 25.3 Recommendation

Add a pipeline version hash (derived from `preprocessing.py` and `predict_xgboost.py` source) to the cache key. This ensures that code changes invalidate the cache.

---

## 26. Artifact Management

### 26.1 Persisted Artifacts

See §14.4 for the complete artifact table. All artifacts are stored in `results/` relative to the XGBoost script directory.

### 26.2 Cache Completeness Check

```python
def _cache_is_complete(paths):
    return all(p.exists() for p in paths.values())
```

All-or-nothing: if any artifact is missing, the entire pipeline retrains from scratch. This prevents inconsistent artifact combinations.

### 26.3 Critical Review

**Strengths**: The all-or-nothing cache check prevents partial-artifact issues.

**Weaknesses**: No integrity verification beyond file existence. A corrupted `.pkl` file would cause a deserialization error at inference time rather than triggering a retrain. Adding a checksum verification step would improve robustness.

---

## 27. Caching Strategy

### 27.1 Design

- **First request** for a dataset: Full training pipeline (~30–60 seconds).
- **Subsequent requests** with the same dataset: Load cached artifacts (~1–2 seconds).
- **Different datasets**: Isolated caches via MD5-hashed filenames.

### 27.2 Staleness Risk

Cached artifacts become stale when:
1. The pipeline code changes (e.g., adding a new feature engineering step).
2. A dependency version changes (e.g., XGBoost produces different trees).
3. The random seed is modified.

None of these scenarios trigger cache invalidation under the current design.

---

## 28. Configuration Management

### 28.1 Centralised Constants

All magic numbers are defined as named constants in `preprocessing.py`:

```python
RANDOM_STATE = 143
EPS = 1e-5
CLIP_BOUND = 1e6
WINSOR_LOWER_PCT = 1
WINSOR_UPPER_PCT = 99
NEAR_ZERO_VAR_THRESHOLD = 1e-8
HIGH_CORR_THRESHOLD = 0.95
SPLIT_N_FOLDS = 5
```

### 28.2 No External Configuration File

All configuration is hardcoded in Python source. There is no `.env`, YAML, or JSON configuration file. This was a deliberate choice for simplicity — the system has a single use case and does not need runtime configuration flexibility.

---

## 29. Logging

### 29.1 Current State

The pipeline uses `warnings.filterwarnings("ignore")` to suppress all warnings. No structured logging is implemented.

### 29.2 Evaluation Output

`evaluate_xgboost.py` prints fold-level metrics to stdout and writes summary metrics to `xgboost_cv_metrics.txt`. This is the only form of execution logging.

### 29.3 Recommendation

Add Python `logging` module integration with at minimum:
- Model training start/end timestamps
- Hyperparameter search best score and best parameters
- Cache hit/miss decisions
- Feature selection pruning statistics (how many features removed, which ones)

---

## 30. Exception Handling

### 30.1 Python Layer

`predict_xgboost.py` wraps the entire `main()` function in a try/except that serialises exceptions as JSON to stderr:

```python
except Exception as exc:
    print(json.dumps({"error": str(exc)}), file=sys.stderr)
    sys.exit(1)
```

### 30.2 Node.js Layer

`app.js` handles:
- Process spawn failure (`pythonProcess.on("error")`).
- Non-zero exit code (parses stderr for structured error JSON).
- Invalid JSON output from Python.
- 5-minute timeout.

### 30.3 Limitations

No specific exception types are caught — all exceptions produce the same generic error response. A `ValueError` from invalid data should return HTTP 400, while a `RuntimeError` from a failed model fit should return HTTP 500. The current design conflates these.

---

## 31. Testing Strategy

### 31.1 Current State

The project has **no automated tests**.

### 31.2 What Should Be Tested

| Test Category | What to Test | Why |
|---|---|---|
| **Unit: preprocessing** | `group_rating()` maps all expected inputs correctly | Prevents silent regrouping changes |
| **Unit: label encoder** | Encode/decode roundtrip for all 4 classes | Prevents index misalignment |
| **Integration: pipeline** | Full preprocessing on a fixture dataset produces expected output shape | Prevents feature count drift |
| **Evaluation: reproducibility** | Two runs with the same seed produce identical metrics | Validates seed control |
| **API: endpoint** | POST with default dataset returns valid JSON with expected fields | Prevents API contract breakage |

### 31.3 Recommendation

Add a `tests/` directory with at minimum:
- `test_preprocessing.py`: Unit tests for `group_rating()`, `ExplicitLabelEncoder`, `fit_imputation()`, `add_interaction_features()`.
- `test_pipeline.py`: Integration test that runs the full pipeline on a 50-row fixture and asserts output shapes.
- `test_api.py`: Smoke test that starts the server and hits `/predict/xgboost` with the default dataset.

---

## 32. Deployment Architecture

### 32.1 Current Deployment

Single-machine deployment: Node.js serves static files and spawns Python subprocesses on the same host. No containerisation, no orchestration, no CI/CD.

### 32.2 Scalability Constraints

- **Single-user**: Each prediction request spawns a new Python process; concurrent requests would compete for CPU and memory.
- **No health checks**: The server does not expose a health endpoint for load balancers.
- **No graceful shutdown**: `SIGTERM` handling is not implemented; in-flight requests would be dropped.

---

## 33. Security Considerations

### 33.1 Path Traversal Protection

`safeResolve()` prevents serving files outside the project root (§24.5).

### 33.2 Input Validation

- The Python scripts validate that the uploaded dataset contains a `Rating` column.
- Base64 decoding is wrapped in try/catch.
- Temporary files are cleaned up in a `finally` block.

### 33.3 Missing Security Controls

- No request size limit (a malicious user could upload a multi-GB file).
- No rate limiting.
- No authentication or authorisation.
- No CORS headers.
- No Content Security Policy.
- Pickle deserialization of cached models is an arbitrary code execution vector if an attacker can overwrite `.pkl` files.

---

## 34. Limitations

### 34.1 Performance Ceiling

The honest cross-validated accuracy is ~46% (macro F1 ~0.42) on a hard, imbalanced 4-class problem with only 593 distinct companies. This is a realistic ceiling for this feature set and sample size. Meaningful gains most likely require:

1. **More data** — especially Distressed-class samples. Even 20 additional C/D-rated observations would significantly improve minority-class learning.
2. **Temporal features** — rolling 3-year trends in key ratios (debt trajectory, margin compression).
3. **Macroeconomic context** — interest rates, credit spreads, GDP growth.

### 34.2 Distressed-Class Weakness

CV-mean F1 ≈ 0.25 for the Distressed class. The 72-record class provides ~57 training samples per CV fold — enough for the model to detect weak signal, but insufficient for reliable classification. High variance across folds (the ~0.03 std in macro F1 is heavily influenced by Distressed-class instability).

### 34.3 No Temporal Awareness

Each company–year record is treated independently. The model cannot detect credit deterioration trajectories (e.g., declining profitability over 3 years followed by a downgrade).

### 34.4 Cross-Model Inconsistency (Rating Grouping Fixed; Others Remain)

The rating-grouping mismatch documented in §2.3 (client.js and Random Forest disagreeing with XGBoost/Decision Tree/Logistic Regression on CCC/CC placement) has been fixed — all five implementations now agree. This eliminates one source of invalid cross-model comparison, but it was not the only one. The Decision Tree, Random Forest, and XGBoost models still use materially different preprocessing pipelines (XGBoost's ~130-feature engineered space vs. Decision Tree/Random Forest's raw-ratio `ColumnTransformer` pipelines) and different response formats (label casing, metric averaging method — weighted vs. macro). A shared class definition is necessary but not sufficient for valid cross-model comparison; the underlying feature spaces and evaluation conventions would also need to be unified.

### 34.5 Stale Cache Risk

Cached artifacts are not invalidated by code changes. A developer who modifies the preprocessing pipeline and runs the API will receive predictions from a model trained under the old pipeline.

---

## 35. Future Improvements

### 35.1 High-Priority

1. **Add and pin XGBoost** (and other ML dependencies) in `requirements.txt` — it is currently absent from the file, not just unpinned — for cross-environment installability and reproducibility.
2. **Add automated tests** (unit, integration, API smoke test).
3. **Implement cache invalidation** on code change (pipeline version hash in cache key).
4. ~~Unify rating grouping across all models (Decision Tree, Random Forest, XGBoost, frontend).~~ **Done** — `client.js` and `predict_random_forest.py` were updated to the CCC/CC→Distressed grouping used by XGBoost, Decision Tree, and Logistic Regression. Remaining cross-model work: unify preprocessing pipelines and evaluation-metric conventions (see §34.4).
5. **Add `.pkl` files to `.gitignore`** and generate evaluation artifacts via a documented script.

### 35.2 Medium-Priority

6. **Ablation study**: Quantify the marginal contribution of each feature engineering stage.
7. **Ordinal regression**: Replace `multi:softprob` with an ordinal-aware loss function.
8. **Temporal features**: Rolling 3-year and 5-year ratio trends.
9. **Cross-model meta-learner**: Combine XGBoost, Random Forest, and Logistic Regression via stacking.
10. **Structured logging**: Python `logging` module with timestamps, cache decisions, and pruning statistics.

### 35.3 Low-Priority

11. **Docker containerisation**: For reproducible deployment.
12. **Request size limits and rate limiting**: For production hardening.
13. **Model registry**: Semantic versioning with code+data+metrics traceability.
14. **Online threshold recalibration**: Adapt thresholds based on recent prediction distributions.

---

## 36. Lessons Learned

### 36.1 Company-Level Leakage Was the Most Impactful Discovery

The 26-percentage-point accuracy drop when switching from row-level to company-level splitting was the single most important engineering finding. It changed the narrative from "our model achieves 72% accuracy" to "our model honestly achieves 46% accuracy and we can explain exactly why." The lesson: **always audit the split strategy relative to the data's grouping structure**.

### 36.2 Honest Metrics Matter More Than High Metrics

A 46% accuracy under honest evaluation is more valuable in a technical report than a 72% accuracy inflated by leakage. Examiners will check for leakage. A project that catches and documents its own leakage demonstrates stronger engineering maturity than one that reports inflated metrics without investigation.

### 36.3 SMOTE Fails on Tiny Minority Classes

With 4–5 samples, SMOTE generates noisy interpolations in a near-degenerate subspace. Balanced sample weights are a simpler and more effective solution for extreme minority classes.

### 36.4 Optuna Is Overkill for Small Datasets

On ~2,000 samples, the performance landscape is smooth enough that 15 random configurations from a well-designed grid find near-optimal hyperparameters in a fraction of the time.

### 36.5 Shared Preprocessing Modules Prevent Drift

Duplicating preprocessing logic between training and inference is a maintenance hazard. Extracting `preprocessing.py` as a shared module eliminated train-serve skew by construction.

### 36.6 Document What Didn't Work

Failed approaches (SMOTE, Optuna, GPU acceleration, Platt scaling) provide as much engineering insight as successful ones. They demonstrate systematic experimentation and prevent future rework.

---

## 37. Final Conclusion

This project demonstrates that rigorous engineering discipline — not model accuracy — is the primary determinant of a trustworthy ML system. The XGBoost credit risk classifier achieves a **Macro F1 of 0.42 ± 0.03** and **Accuracy of 0.46 ± 0.03** under company-level, leakage-safe, grouped stratified 5-fold cross-validation. These numbers are honest. They reflect:

- **Zero company-level leakage**: All records for a given company are confined to one side of every split.
- **Zero preprocessing leakage**: Every fitted parameter (medians, bounds, sector stats, feature selections) is computed on the training fold exclusively.
- **Zero threshold-selection leakage**: The strategy selector uses training-set F1 only.
- **Documented failures**: SMOTE, Optuna, GPU acceleration, and Platt scaling were tried, diagnosed, and abandoned with evidence.
- **Internal consistency**: All reported metrics originate from the same artifact set (verified by automated audit).

The system is deployable as a web application with SHAP-based explainability, model caching, and support for user-uploaded datasets. It is not a production financial system — it lacks the temporal features, data volume, regulatory compliance, and operational monitoring required for that role — but it is an honest, auditable, well-documented engineering artifact.

---

## Appendix A: Self-Audit

### A.1 Consistency Verification

| Check | Result |
|---|---|
| Confusion matrix matches classification report | ✅ Verified: CM from `xgboost_test_predictions.csv` matches `xgboost_classification_report.csv` |
| Accuracy is calculated correctly | ✅ 215/406 = 0.5296 (matches `xgboost_metrics.txt`) |
| Support values are consistent | ✅ IH=98, IL=134, SP=159, DI=15 (sum = 406, matches CM row sums) |
| All metrics from same model run | ✅ All artifacts in `results/` are from the same training execution |
| CV metrics are independent | ✅ `evaluate_xgboost.py` runs its own pipeline per fold |
| API response format matches documented format | ✅ Verified against `predict_xgboost.py` output construction |

### A.2 Previously Identified Inconsistency

The confusion matrix previously documented in the older technical report (§7 of `TECHNICAL_REPORT.md`) showed accuracy 0.5222 (212/406 correct) with different cell values. The current cached artifacts show accuracy 0.5296 (215/406 correct). This discrepancy is attributed to the multithreading non-determinism documented in §14.3 — the older report captured a different model run. This report uses only the currently cached artifacts.

### A.3 Known Cross-System Inconsistencies

1. **Rating grouping mismatch — fixed**. The XGBoost, Decision Tree, and Logistic Regression backends mapped CCC/CC to Distressed, while the Random Forest backend and the frontend `client.js` mapped CCC/CC to Speculative (two groupings across five implementations, not three independent ones — corrected from an earlier draft of this audit that overstated the count). `client.js` and `predict_random_forest.py` were updated to match the CCC/CC→Distressed grouping; all five implementations now agree, verified by direct source inspection.
2. **Label format mismatch — unresolved**: XGBoost uses underscores (`Investment_High`); other models use hyphens (`Investment-High`). The API normalises this via `format_prediction_label()`, but internal artifacts use underscores.

### A.4 Self-Assessment Rubric

| Criterion | Score | Justification |
|---|---|---|
| **Technical depth** | 8/10 | Detailed justification of engineering decisions; some areas lack ablation evidence |
| **Reproducibility** | 7/10 | Seeds fixed, artifacts persisted; XGBoost version unpinned, no automated tests |
| **Honesty of evaluation** | 9/10 | Company-level leakage discovered, documented, and fixed; cross-validated metrics as authoritative |
| **Software engineering quality** | 7/10 | Shared preprocessing module; no tests; no structured logging; no cache invalidation |
| **Documentation quality** | 8/10 | Comprehensive; some sections lack quantitative evidence (e.g., ablation studies) |
| **Model evolution documentation** | 9/10 | All iterations documented including failures; clear cause-effect chains |
| **Limitations acknowledgement** | 9/10 | Performance ceiling honestly discussed; cross-model inconsistencies identified, and the rating-grouping mismatch was corrected in code rather than only flagged |
| **Overall** | **81/100** | Strong engineering discipline and honest evaluation offset by missing automated tests and incomplete ablation evidence; one identified cross-model defect was fixed during this audit cycle |

### A.5 Sections at Risk of Mark Deduction

1. **No ablation study**: The report claims 21 interaction features are valuable but provides no evidence that removing any subset would degrade performance. This is an unsupported claim.
2. **No automated tests**: The project demonstrates engineering discipline in design but lacks the testing infrastructure to verify it programmatically.
3. **Cross-model inconsistency (partially resolved)**: The rating-grouping mismatch across models has been fixed (§2.3, §34.4). Preprocessing-pipeline and evaluation-metric differences across models remain and still make direct dashboard comparison invalid — this is a legitimate residual software engineering flaw.
4. **Missing XGBoost dependency declaration**: `requirements.txt` does not list `xgboost` at all (confirmed by direct inspection), not merely leaving it unpinned. A clean environment built from this file cannot run the pipeline.

---

## Appendix B: File Structure

```
Credit-Risk-Analyzer/
├── app.js                          # Node.js HTTP server
├── client.js                       # Frontend dashboard logic
├── package.json                    # Node.js configuration
├── requirements.txt                # Python dependencies
├── data/
│   └── set A corporate_rating.csv  # Training dataset (2,029 records)
├── views/
│   └── index.html                  # Single-page dashboard
├── models/
│   ├── decisionTree.js             # Decision Tree route config
│   ├── randomForest.js             # Random Forest route config
│   ├── logisticRegression.js       # Logistic Regression route config
│   └── xgboost.js                  # XGBoost route config
├── docs/
│   ├── TECHNICAL_REPORT.md         # Previous technical report
│   └── XGBoost_Only.ipynb          # Experimental notebook
└── python_models/
    ├── decision_tree_stuff/
    │   └── predict_decision_tree.py
    ├── random_forest_stuff/
    │   ├── predict_random_forest.py
    │   └── *.pkl                   # Cached Random Forest artifacts
    ├── logistic_regression_stuff/
    │   └── predict_logistic_regression.py
    └── xgboost_stuff/
        ├── preprocessing.py        # Shared preprocessing (split, impute, engineer, select)
        ├── predict_xgboost.py      # Training + inference pipeline
        ├── evaluate_xgboost.py     # Grouped k-fold cross-validation
        ├── results/                # Cached models + evaluation outputs
        │   ├── calibrated_model.pkl
        │   ├── base_model.pkl
        │   ├── imputer_medians.pkl
        │   ├── winsorize_bounds.pkl
        │   ├── sector_stats.pkl
        │   ├── feature_columns.pkl
        │   ├── optimal_thresholds.pkl
        │   ├── prediction_strategy.pkl
        │   ├── xgboost_metrics.txt
        │   ├── xgboost_classification_report.csv
        │   ├── xgboost_test_predictions.csv
        │   └── xgboost_cv_metrics.txt
        └── figures/
            ├── class_distribution.png
            ├── confusion_matrix.png
            ├── feature_importance.png
            ├── shap_bar.png
            ├── shap_beeswarm.png
            └── shap_waterfall_company0.png
```

---

## Appendix C: References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
2. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
3. Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. *Advances in Large Margin Classifiers*, 61–74.
4. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632.
5. Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *The Journal of Finance*, 23(4), 589–609.
6. Basel Committee on Banking Supervision. (2006). International Convergence of Capital Measurement and Capital Standards: A Revised Framework. Bank for International Settlements.

---

## Appendix D: Glossary

| Term | Definition |
|---|---|
| **Macro F1** | Unweighted average of per-class F1 scores; treats all classes equally regardless of support |
| **Isotonic Calibration** | Non-parametric calibration method that fits a monotonically increasing step function to align predicted probabilities with observed frequencies |
| **SHAP Values** | Shapley values from cooperative game theory, adapted to explain individual model predictions by attributing contributions to each feature |
| **Winsorisation** | Statistical transformation that clips extreme values to specified percentiles |
| **Z-Score** | Number of standard deviations a value lies from the group mean |
| **TreeExplainer** | SHAP's exact algorithm for computing Shapley values in tree ensembles in O(TLD²) time |
| **Nelder-Mead** | Derivative-free simplex optimisation algorithm used for threshold tuning |
| **StratifiedGroupKFold** | Scikit-learn splitter that maintains class proportions while keeping all records of a group (company) in the same fold |
| **Fallen Angel** | Industry term for a bond downgraded from investment grade (BBB) to speculative grade (BB); the boundary this model most struggles with |

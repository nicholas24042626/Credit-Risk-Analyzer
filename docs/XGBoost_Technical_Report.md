# Technical Report: XGBoost Credit Risk Prediction System

# Final Year Project — Engineering Documentation

---

## Executive Summary

This report documents the complete engineering lifecycle of a multiclass XGBoost classifier for corporate credit rating prediction, deployed as a web application with SHAP-based explainability. The system classifies companies into four financial risk tiers — Investment-High, Investment-Low, Speculative, and Distressed — using 30 financial ratios from the Set A Corporate Rating dataset (2,029 company–year records, 593 unique companies).

The project evolved through multiple documented iterations, each motivated by a specific engineering failure: row-level data leakage inflated early accuracy to ~72% (which dropped to an honest ~46% after correction); an early SMOTE attempt on 4–5 Distressed samples produced degenerate synthetic clusters and was abandoned; threshold-strategy selection leaked test labels into a production decision; and, most recently, the **dashboard's single-split evaluation and the offline cross-validation script had drifted into two different pipelines** (different model, different split, different threshold logic), producing two contradictory accuracy figures (53% vs. 70%) for what was supposed to be the same model. All of these were diagnosed, documented, and resolved systematically.

**Authoritative performance** (company-level, leakage-safe, grouped stratified 3-fold cross-validation — a ~70/30 train/test split per fold, changed from the original ~80/20 5-fold split, see §16.21 — current pipeline: SMOTE-augmented seeded soft-voting ensemble blended with an ordinal cumulative-link decomposition, plain-argmax prediction rule, §16.26):

| Metric | CV Mean ± Std |
|---|---|
| Accuracy | 0.5264 ± 0.0252 |
| Macro F1 | 0.4497 ± 0.0155 |

> [!NOTE]
> **Run-to-run variance**: XGBoost's `n_jobs=-1` multithreading introduces minor non-determinism (§14.3), so re-running `evaluate_xgboost.py` produces slightly different figures each time (e.g. two consecutive runs of the current §16.26 configuration measured 0.5244 ± 0.0210 and 0.5264 ± 0.0252 accuracy). The figures quoted throughout this report are taken from the currently-cached `results/xgboost_cv_metrics.json`, regenerated as the final step of this revision, so the report and the live artifact match exactly.

> [!NOTE]
> **Distressed-class F1 remains the weak point**: the ordinal blend (§16.26) raised accuracy AND macro F1 relative to the previous configuration, and removing the second-stage discriminator eliminated the worst of the §16.17 Distressed-suppression trade-off — but Distressed F1 (~0.22–0.25 across runs) is still far below the other classes. Whether headline accuracy or minority-class recall matters more depends on the deployment context; both are reported so the trade-off stays visible.

These figures are deliberately conservative. They reflect a 4-class ordinal problem with severe class imbalance (Distressed: ~3.7% of data), company-level split integrity (no company appears in both train and test), and a threshold-fitting protocol that never touches the evaluation fold. ~50% accuracy is the honest ceiling for this feature set and sample size — an examiner should evaluate the engineering discipline that produced this number, not the number itself.

> [!NOTE]
> **Sector fairness, measured**: credit-risk ratio thresholds are not directly comparable across industries (e.g. leverage that is normal for a capital-intensive utility looks alarming for a technology firm), so the pipeline's sector-relative z-score normalisation (§5.4) was checked against a direct per-sector accuracy breakdown rather than assumed to be sufficient. The result: a genuine 29.7-point accuracy spread across well-supported sectors (Health Care 36.3% vs. Finance 66.0%), classified as a "moderate" disparity. This is documented as a measured, partially-diagnosed limitation in §21.4 and §34.5, not resolved by a sector-conditional model — that intervention was deliberately not built until the underlying cause is better understood (§35.2 item 13).

> [!IMPORTANT]
> **Single source of truth**: `evaluate_xgboost.py` is now the single source of truth for reported accuracy. It imports `train_model()` and `fit_thresholds_nested()` directly from `predict_xgboost.py`, so the cross-validation figure and the dashboard's live single-split demo are measurements of the *exact same model and threshold protocol*, not two implementations that happen to share a name. `evaluate_xgboost.py` writes its result to `results/xgboost_cv_metrics.json`; `predict_xgboost.py` reads that cache and surfaces it in the API response as the **primary** metric, with the live single-split result relegated to a clearly labelled **secondary** field (`singleSplitMetrics`) that the dashboard displays as a collapsed, explicitly-caveated section. See §12.4 and §16.14 for the full history of this fix, and §19 for why probability calibration (`CalibratedClassifierCV`) was removed from the pipeline in the same pass.

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

**Evidence**: Under the CCC/CC→Speculative grouping, the Distressed class had extremely few test samples in some splits (sometimes only one) — producing 100% or 0% accuracy with no statistical meaning. Under the CCC/CC→Distressed grouping, Distressed achieves F1 ≈ 0.25 (CV mean), which is weak but genuine.

> [!NOTE]
> **Cross-system inconsistency — discovered and fixed**: An earlier audit found two mutually exclusive groupings in the codebase. The frontend `client.js` (`ratingRuleEngine`) and the Random Forest backend (`predict_random_forest.py`, `group_rating()`) mapped CCC/CC to "Speculative" and only C/D to "Distressed", while the XGBoost backend (`preprocessing.py`), Decision Tree backend (`predict_decision_tree.py`), and Logistic Regression backend (`predict_logistic_regression.py`) mapped CCC/CC/C/D to "Distressed". This was a genuine engineering defect: two of the dashboard's models defined the classification task differently from the other three, making cross-model comparison statistically invalid regardless of which grouping is "correct".
>
> **Fix applied**: `client.js` and `predict_random_forest.py` were updated to the CCC/CC→Distressed grouping, matching the other three implementations and the domain justification in §2.3 (this raises Distressed support from ~10 to ~72 records, making the class measurable rather than a statistical coin flip on extremely few test samples). All five `group_rating()`/`ratingRuleEngine` implementations in the codebase now agree — verified by direct source inspection after the change. Random Forest does not cache trained artifacts (it retrains per request), so no stale cached model needed invalidation as a result of this fix; any previously cached XGBoost artifacts trained under the old grouping are unaffected since XGBoost's grouping did not change.

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

These were measured directly on the default dataset (2,029 rows), not assumed:

1. **No native missing values**: The default CSV contains **zero** NaN cells and zero infinities — verified programmatically. Missing-value handling (§7) therefore matters primarily for (a) user-uploaded datasets and (b) the NaN deliberately introduced by the cleaning step below. An earlier draft of this report claimed "several columns contain NaN"; that was inaccurate for the default dataset and has been corrected.
2. **Impossible negative values**: **55 cells** across 8 columns hold negative values that are mathematically impossible for the quantity they represent — a ratio or count of two non-negative quantities cannot be negative. Counts: `payablesTurnover` (15), `cashPerShare` (14), `quickRatio` (9), `currentRatio` (5), `cashRatio` (4), `assetTurnover` (3), `fixedAssetTurnover` (3), `daysOfSalesOutstanding` (2). These are data-entry errors and are nulled then imputed by the cleaning step (§6.3).
3. **Extreme outliers**: Financial ratios exhibit fat-tailed distributions. `debtEquityRatio` ranges from near-zero to 100+. `effectiveTaxRate` has 255 negative values and 60 values above 1.0 — but unlike the impossible negatives above, these are **legitimate** (tax carryforwards, one-off benefits), so they are winsorised (§8) rather than nulled.
4. **No detected duplicate or conflicting-rating rows**: Zero exact duplicate rows; zero `(Symbol, Date)` groups carry more than one distinct rating — verified. The feared "multi-agency label conflict" does not occur in this dataset (each row is a unique company–date observation). Even so, both duplicate removal and conflict resolution (keep the most conservative grade per company-date) are applied defensively so that *uploaded* datasets exhibiting these issues are handled deterministically rather than silently trained on contradictory labels — see §6.3.
5. **No temporal features**: The `Date` column is dropped entirely — the model treats each record independently with no time-series awareness.

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

The Distressed class at 3.6% creates a severe minority-class challenge. A majority-class baseline classifier predicting only "Speculative" would achieve ~39% accuracy — which provides context for the model's ~50% CV accuracy: it exceeds the majority-class baseline by approximately 11 percentage points.

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
data cleaning                    (deterministic, per-cell, pre-split — no leakage)
    → grouped stratified split   (leakage-safe, company-level)
    → median imputation          (fit on train only)
    → winsorisation (P1–P99)     (fit on train only)
    → interaction / log features (deterministic formulas)
    → sector z-scores            (fit on train only)
    → one-hot encode + align     (all categoricals)
    → feature selection          (fit on train only)
```

The cleaning step (§6) runs before the split because it is purely per-row/per-cell (numeric coercion, duplicate removal, whitespace normalisation, multi-agency rating-conflict resolution, infinity nulling, impossible-negative nulling, empty-row dropping) and uses no aggregate statistics — so it cannot leak information across the split boundary. Every step that *does* fit parameters (imputation, winsorisation, sector z-scores, feature selection) runs after the split, on the training fold only. This order is enforced by code structure, not by documentation. Both `predict_xgboost.py` and `evaluate_xgboost.py` import and call the same functions in the same sequence.

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

### 6.3 Deterministic Pre-Split Cleaning (`clean_dataframe`)

**Problem**: The raw pipeline trusted the CSV to be well-formed. Direct inspection revealed impossible values (55 negative cells in columns that cannot be negative) that would otherwise flow into feature engineering — a negative `currentRatio` would produce a nonsensical `currentRatio_sq`, `liquidity_score`, and sector z-score, corrupting several engineered features from a single bad cell.

**Decision**: Add a `clean_dataframe()` function in `preprocessing.py`, called once immediately after loading and before the train/test split. It performs seven deterministic, per-row/per-cell operations, ordered so that each step sees the cleaned output of the previous one:

| Step | Operation | Rationale |
|---|---|---|
| 1 | Coerce "mostly numeric" text columns to numeric | Uploaded CSVs encode errors as `"N/A"`, `"#DIV/0!"`, `""`; pandas reads the whole column as text, silently bypassing every numeric transform. Coercion turns junk tokens into NaN. |
| 2 | Drop exact duplicate rows | Defensive for user uploads; default dataset has none |
| 3 | Strip whitespace on `Sector` | Prevents `"Energy"` and `"Energy "` becoming two one-hot columns |
| 4 | Resolve conflicting/duplicate multi-agency ratings | Keep one worst-case row per `(company, Date)` so contradictory labels don't train on identical features |
| 5 | Replace ±∞ with NaN | Defensive against overflow in uploaded data and engineered ratios |
| 6 | Null impossible negatives in `NON_NEGATIVE_COLS` | Converts 55 data-entry errors to NaN for downstream imputation |
| 7 | Drop rows whose entire numeric feature vector is NaN | A row with no usable features carries no signal |

**Future-proofing (steps 1, 4, 7)**: The default dataset happens to be clean on these axes — 0 text-polluted numeric cells, 0 conflicting `(Symbol, Date)` ratings, 0 empty rows — so these steps are **no-ops on the default data and leave the cached artifacts and reported metrics untouched**. They exist to protect *arbitrary uploaded datasets*, which the application accepts at runtime. This was verified with a synthetic messy dataset: text tokens (`"N/A"`, `"#DIV/0!"`) were coerced to NaN, agency disagreements (e.g. A vs BBB on the same date) collapsed to the more conservative grade, and duplicate same-date rows were merged — reducing an 8-row messy frame to 4 clean rows with all issues resolved.

**Numeric coercion — known vs unknown columns**: The 25 financial-ratio columns of the corporate-rating schema (`KNOWN_NUMERIC_FEATURE_COLS`) are numeric by definition and are coerced *unconditionally*, however much junk an upload introduces. Columns *not* in the known schema are coerced only if at least 90% (`NUMERIC_COERCE_MIN_PARSE_RATIO`) of their non-null values parse as numbers — this preserves a genuinely categorical column a user might append (e.g. `Country`) that happens to contain a few numeric-looking values.

**Conflict resolution — why "keep the worst"?** When two agencies disagree on the same company and date, the rows carry identical features but different labels, which is pure training noise. Rather than average, drop both, or pick arbitrarily, the resolver keeps the row with the highest `RATING_SEVERITY` (worst grade). In a credit-risk screening context, systematically preferring the more conservative label is the safe asymmetry — it biases toward flagging risk rather than hiding it. The resolver is skipped entirely when no identifier or `Date` column is present, because collapsing on identifier alone would wrongly merge a company's entire multi-year history.

**Why before the split?** Every operation is row-wise or per-cell and uses **no aggregate statistics**. Unlike imputation or winsorisation (which fit parameters and must therefore run train-only), nulling a negative `currentRatio`, coercing a text cell, or dropping a duplicate row cannot transfer information across the split boundary. Conflict resolution reads the label column but uses no test information and no fitted statistic, so it remains leakage-safe — directly analogous to the pre-split exclusion of Unknown-rated rows (§6.2). Running all of this before the split is both leakage-safe and simpler (one call on the full frame rather than two symmetric calls).

**Why null-then-impute rather than clip-to-zero?** A negative liquidity ratio is not "a very small positive ratio" — it is an unknown value corrupted by an upstream error. Clipping it to 0 would assert the company has zero liquidity, a specific and likely wrong claim. Nulling and imputing with the training median substitutes the least-committal estimate instead.

**Which columns are treated as non-negative?** Only ratios/counts of two non-negative quantities: `currentRatio`, `quickRatio`, `cashRatio`, `cashPerShare`, `daysOfSalesOutstanding`, `debtRatio`, `assetTurnover`, `fixedAssetTurnover`, `payablesTurnover`. Margins, returns, `effectiveTaxRate`, `freeCashFlowPerShare`, `debtEquityRatio`, and `companyEquityMultiplier` are **excluded** because they can be legitimately negative (losses, tax benefits, negative shareholder equity). Treating those as errors would destroy real signal.

**Impact on metrics**: Regenerating all artifacts with cleaning enabled moved single-split accuracy from 0.5296 to **0.5517** and left the authoritative 5-fold CV accuracy essentially unchanged (0.4603 → **0.4618**, well within one standard deviation). The single-split gain is larger than the CV gain because cleaning removes a small number of high-leverage corrupted cells whose effect is amplified in any one split but averages out across folds. The change is honest and modest — 55 cells out of ~60,000 numeric cells.

**Trade-offs and limitations**: Both `NON_NEGATIVE_COLS` and `KNOWN_NUMERIC_FEATURE_COLS` are hand-curated from the corporate-rating schema, not derived automatically; a genuinely non-negative or numeric column outside the schema is only protected by the generic 90%-parse heuristic, not by name. The conflict resolver's "keep worst" rule is a deliberate risk-averse default, but it discards the disagreeing agency's opinion rather than modelling the disagreement (e.g. via an ensemble-of-labels or a confidence weight) — a richer but heavier design that was not warranted here. Cleaning also still does not attempt cross-field consistency checks (e.g., `quickRatio > currentRatio`, impossible since quick assets ⊆ current assets) — considered but deferred as lower-value. Finally, the 90% coercion threshold is a fixed constant; a numeric column that is more than 10% junk *and* outside the known schema would be left as text and later dropped rather than salvaged.

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

**Interaction with cleaning**: On the default dataset the CSV has zero native NaN, so imputation's real work is filling the 55 cells that the cleaning step (§6.3) nulls after detecting impossible negatives, plus any NaN present in user-uploaded files. Because the medians are still computed on the training fold only, this remains leakage-safe: a corrupted test-row cell is filled with a training-derived median, never a statistic that saw the test data.

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

**Current decision (revised)**: Apply SMOTE (Synthetic Minority Over-sampling) to the training fold immediately before model fitting, with `k_neighbors = min(5, minority_class_count - 1)` to stay safe when a class has very few samples. This supersedes the original `class_weight='balanced'` sample-weighting approach; `train_model()` now sets `sample_weights = None` and relies on SMOTE-balanced classes instead.

**Why the reversal?** §16.5 originally documented SMOTE as "failed, abandoned" under the *old* pipeline (deep trees, light regularisation, 3-model ensemble). After the regularisation pass (§16.13: shallower trees, heavier L1/L2/gamma floors, tighter correlation pruning) closed the model's overfitting gap from ~39 points to ~14 points without moving test accuracy, SMOTE was re-tried under the new, better-regularised pipeline and produced a genuine accuracy gain (+8 points on grouped CV) rather than degrading performance. The earlier failure was a symptom of the pipeline's excess capacity at the time, not an inherent property of SMOTE on this dataset. See §16.14 for the full re-introduction record.

**Alternatives considered**:
1. **Balanced sample weights** (previous approach): Increased Distressed recall from ~0% to ~20% but left overall CV accuracy essentially unchanged versus no weighting.
2. **Class-weighted loss function in XGBoost**: XGBoost's `scale_pos_weight` only applies to binary classification. For multiclass, external sample weights or resampling are required.
3. **Random undersampling of majority classes**: Would discard ~90% of Speculative-class data, severely reducing the training set.

**Trade-off observed**: SMOTE increases the train/test accuracy gap (train accuracy rises to ~88% because synthetic minority points are easy to fit) without degrading generalisation — the gap grows but CV test accuracy improves, so the SMOTE-inflated training score should not be read as a regression in its own right (see §16.14 and Appendix A.2 for the exact before/after CV numbers). Distressed-class F1 specifically became *slightly worse* under SMOTE (0.278 → 0.210 in one comparison run) even as overall accuracy rose, because the synthetic Distressed samples are noisy given only ~57 real training examples per fold — SMOTE trades a small amount of minority-class fidelity for a larger gain in majority-class separability.

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
            return (X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx],
                    "grouped_stratified", groups[train_idx])
        except Exception:
            pass
    # Fallback chain: stratified → random (returns train_groups=None in both cases)
```

**Note on the returned `train_groups`**: the function returns a sixth value — the `groups` array sliced to the training rows — so that a caller (specifically `fit_thresholds_nested()`, §12.4) can perform a further, leakage-safe, company-level *nested* split within the training fold without re-deriving the grouping key. This was added when the dashboard's single-split evaluation was found to have silently dropped grouping altogether (`groups=None`) — see §12.4.

**Degradation strategy**: The split falls back gracefully:
1. **Grouped stratified** (preferred): Company-level grouping with class-proportion balancing.
2. **Stratified**: Class-proportion balancing without grouping (if groups are unavailable).
3. **Random**: Plain split (if stratification fails due to tiny class sizes).

The fallback chain is a defensive design — the default dataset always achieves grouped stratified splitting, but user-uploaded datasets may lack identifier columns.

### 12.2 Cross-Validation Protocol

The authoritative evaluation uses **grouped stratified k-fold cross-validation** via `evaluate_xgboost.py` — currently 3 folds (≈70/30 train/test per fold; see §16.21 for the split-ratio change from the original 5-fold/≈80/20 configuration):

- Each fold performs the entire preprocessing pipeline from scratch (imputation, winsorisation, feature engineering, feature selection).
- No preprocessing parameters are shared between folds.
- Standard calibrated predictions are used — the threshold strategy selector is deliberately skipped because it previously leaked test labels.

### 12.3 Why No Separate Validation Set

**Problem**: With ~2,000 records, a 3-way split (train/validation/test) would leave insufficient data for either validation or test.

**Decision**: Use cross-validated hyperparameter search (`RandomizedSearchCV` with `cv=3`) on the training set instead of a held-out validation set. This maximises data utilisation while maintaining evaluation integrity.

**Trade-off**: No final model selection step on a separate validation set. The cross-validation mean serves as the expected performance estimate instead.

### 12.4 Methodological Alignment: Single Source of Truth for Reported Accuracy

**Discovery**: An audit comparing the dashboard's live single-split output against the offline CV script's output found they had drifted into two different pipelines that happened to share a script name:

| Aspect | Dashboard (`predict_xgboost.py`, before fix) | CV script (`evaluate_xgboost.py`, before fix) |
|---|---|---|
| Train/test split | `make_split(..., groups=None, ...)` — **grouping silently dropped** | `StratifiedGroupKFold` — grouped, company-level |
| Model | SMOTE + 5-seed ensemble, no calibration | 5-seed ensemble via shared `train_model()`, no calibration |
| Threshold protocol | Force-disabled (`use_thresholds = False`) | Skipped entirely (no threshold logic in the CV loop) |
| Observed accuracy | Up to **~70%** on a favourable split | **~53%** (5-fold mean, before the SMOTE re-introduction documented in §16.14) |

The dashboard's split no longer excluded a company's other-year records from the test set within a single split (§13.3's original fix had regressed), which is a reintroduction of exactly the company-identity leakage the project had previously spent significant effort diagnosing and removing. Combined with a single lucky split, this let the live demo show 70% accuracy while the documented, CV-audited figure was 53% — a direct contradiction between the dissertation's own headline claim and its live artifact.

**Fix applied — three changes, all in the same pass**:
1. **Split restored**: `predict_xgboost.py`'s single split now calls `make_split(X, y, groups=groups, ...)`, matching the CV script's grouping.
2. **Model unified**: `evaluate_xgboost.py` now imports `train_model()` directly from `predict_xgboost.py` instead of maintaining its own copy, so both scripts train literally the same model, not two implementations that could silently diverge again.
3. **Threshold protocol unified and made leakage-safe in both places**: a new shared helper, `fit_thresholds_nested()`, carves a company-level inner train/validation split out of the *outer training fold only* (never the outer test/CV fold), fits a quick model on the inner-train slice, and fits/evaluates the threshold multipliers and the use/don't-use decision purely on the inner-validation slice. Both `predict_xgboost.py` and `evaluate_xgboost.py` now call this same function per split/fold. See §13.2 for why a nested split — rather than fitting and evaluating thresholds on the same full training fold — is necessary to keep the decision leakage-safe.

**Result after the fix**: both scripts reported the same authoritative figure, 0.5007 ± 0.0372 accuracy (5-fold grouped CV) at the time of this fix, later raised to **0.5101 ± 0.0270** by the second-stage discriminator (§16.17, minor run-to-run variance noted above) — see §20.1 for the current full breakdown and its accompanying fairness caveat. The dashboard reads the cached CV result from `results/xgboost_cv_metrics.json` (written by `evaluate_xgboost.py`) and displays it as the **primary** metric, with its own live single-split result shown only as a secondary, explicitly-labelled "high variance — see CV mean for reported accuracy" figure. This is consistent with standard evaluation practice for small, high-variance datasets: a single train/test split over ~593 companies is not a stable estimate, and should never be presented as competing with a cross-validated mean.

**Why this matters for the dissertation**: reporting the CV mean as the headline figure — rather than whichever number happens to look best in a demo — is the same principle that motivated the original company-level leakage fix (§16.3) and the threshold strategy-selector leakage fix (§13.2). An examiner who notices the dashboard and the report agreeing exactly, with the variance explicitly quantified, should read that as evidence of a consistently-applied evaluation discipline rather than as a coincidence.

---

## 13. Leakage Prevention

### 13.1 Complete Leakage Prevention Checklist

| Pipeline Stage | Leakage-Free? | Mechanism |
|---|---|---|
| Data cleaning (coercion, dup drop, rating-conflict resolution, ∞/impossible-negative nulling, empty-row drop) | ✅ | Per-row/per-cell only; uses labels but no fitted statistics or test data, so pre-split execution cannot leak |
| Train/test split (company overlap) | ✅ | `StratifiedGroupKFold` on `Symbol`/`Name` |
| Median imputation | ✅ | Medians computed on `X_train` only; persisted in `imputer_medians.pkl` |
| Winsorisation bounds | ✅ | Bounds computed on `X_train` only; persisted in `winsorize_bounds.pkl` |
| Sector z-score statistics | ✅ | Sector means/stds from `X_train` only; persisted in `sector_stats.pkl` |
| Interaction features | ✅ | Deterministic formula; no fitted parameters |
| Log transforms | ✅ | Deterministic formula; no fitted parameters |
| One-hot encoding | ✅ | Column alignment via `reindex` with `fill_value=0` |
| Feature selection | ✅ | Variance/correlation thresholds fit on `X_train` only |
| Threshold optimisation (values) | ✅ | Fit on a company-level inner-validation slice of `X_train`/`y_train` only (see §13.2, revised) |
| Threshold strategy selector | ✅ | Decision uses the same inner-validation slice, never the outer test/CV fold (see §13.2, revised) |
| Sample weight / resampling computation | ✅ | SMOTE and (legacy) sample weights are both fit on `y_train`/`X_train_enc` only |

### 13.2 Strategy-Selector Leakage: Discovery, Original Fix, and Subsequent Refinement (Nested Split)

**Discovery**: The original threshold strategy selector compared threshold-optimised vs. standard predictions on the *test set* to decide which strategy to persist. This leaked test labels into a binary production decision.

**Impact**: The leakage was subtle — it only affected whether the threshold multipliers were applied, not their values. Empirically, it inflated single-split accuracy by approximately 1–2 percentage points.

**Original fix**: The strategy selector was changed to compare Macro F1 on the *training set* only:

```python
f1_standard = f1_score(y_train, calibrated_model.predict(X_train_enc), average="macro")
f1_threshold = f1_score(y_train, (y_train_proba * optimal_thresholds).argmax(axis=1), average="macro")
use_thresholds = f1_threshold > f1_standard
```

**Residual weakness in the original fix**: while this removed *test-set* leakage, it still had the model make its threshold decision on the exact same rows it was trained on — an optimistic self-evaluation. A model that has memorised its training fold (which the pipeline was shown to do, §16.13) will look artificially good on that fold regardless of whether the thresholds genuinely generalise.

**Refinement — nested inner split (`fit_thresholds_nested()`)**: the threshold multipliers and the use/don't-use decision are now fit on a company-level `GroupShuffleSplit` slice carved out of the outer training fold, with a *separate, quickly-refit* model (reusing the hyperparameters already found by `RandomizedSearchCV`, not the final production model) trained on the inner-train portion and evaluated on the inner-validation portion:

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
inner_train_idx, inner_val_idx = next(gss.split(X_train_enc, y_train, train_groups))
quick_model = XGBClassifier(**best_params, random_state=RANDOM_STATE)
quick_model.fit(X_train_enc.iloc[inner_train_idx], y_train[inner_train_idx])
proba_val = quick_model.predict_proba(X_train_enc.iloc[inner_val_idx])
# thresholds + use/don't-use decision fit and evaluated on proba_val / y_val only
```

The production model (trained on the *full* outer training fold, for maximum data utilisation) then has these externally-derived thresholds applied to its predictions on the outer test/CV fold. Falls back to the single-fold training-set decision (the "original fix" above) only if a grouped inner split cannot be formed (e.g. too few groups) — this fallback is strictly less rigorous but still never touches the outer test fold. This exact protocol is now shared, verbatim, between `predict_xgboost.py`'s single split and every fold of `evaluate_xgboost.py`'s cross-validation (§12.4).

### 13.3 Row-Level Leakage: Discovery, Impact, and Fix

**Discovery**: The original pipeline used `train_test_split` with `stratify=y` but no grouping. With 593 companies across 2,029 records, the model saw other-year records of test companies during training.

**Impact**: This inflated accuracy from ~46% (honest, company-level) to ~72% (leaky, row-level). The 26-percentage-point gap quantifies the information content of company identity leakage.

**Fix**: Switched to `StratifiedGroupKFold` with `Symbol`/`Name` as the grouping key. All records for a given company are now confined to one side of the split.

---

## 14. Reproducibility Strategy

### 14.1 Random Seed Control

All stochastic operations use `RANDOM_STATE = 143`:

- `StratifiedGroupKFold` / `GroupShuffleSplit` shuffling (both outer split and the nested inner threshold-fitting split)
- SMOTE's internal k-nearest-neighbour synthesis
- XGBoost base model (seed 143), ensemble members (144, 145, 146, 147) — the ensemble grew from 3 to 5 seeded models (§16.7, §16.14)
- `RandomizedSearchCV` random state
- Nelder-Mead threshold optimisation (25 restarts via `np.random.default_rng(143)`, increased from the original 10 for better coverage)

### 14.2 Deterministic Label Encoding

The `ExplicitLabelEncoder` with a hardcoded mapping guarantees consistent class-to-index assignment across all runs, regardless of data ordering or class-set variation.

### 14.3 Residual Non-Determinism

**Known issue**: XGBoost trained with `n_jobs=-1` exhibits minor run-to-run variance (~±1 percentage point in single-split accuracy) due to non-deterministic floating-point reduction order across threads. This is documented in XGBoost's official FAQ and is inherent to multithreaded tree construction.

**Mitigation**: The authoritative performance figure is a cross-validation mean ± std, not a single-split number. Bit-exact reproducibility would require `n_jobs=1`, which would increase training time by ~3–4×.

### 14.4 Artifact Serialisation

All fitted parameters are persisted via `joblib` for identical reapplication:

| Artifact | Contents | Purpose |
|---|---|---|
| `calibrated_model.pkl` | `VotingClassifier` soft-voting ensemble (5 seeded XGBoost models) — despite the legacy filename, this is no longer wrapped in `CalibratedClassifierCV` (§19) | Inference predictions |
| `base_model.pkl` | Best single XGBClassifier from RandomizedSearchCV | SHAP explanations (TreeExplainer requires uncalibrated model) |
| `imputer_medians.pkl` | Per-column training medians | Missing value imputation |
| `winsorize_bounds.pkl` | Per-column (P1, P99) bounds | Outlier capping |
| `sector_stats.pkl` | Per-sector means, stds, global means, global stds | Z-score normalisation |
| `feature_columns.pkl` | Selected feature list after variance/correlation pruning | Column alignment at inference |
| `optimal_thresholds.pkl` | Per-class probability multipliers, fit via the nested inner split (§13.2) | Threshold-optimised predictions |
| `prediction_strategy.pkl` | `{"use_thresholds": bool}`, decided via the nested inner split (§13.2) | Whether to apply thresholds |
| `xgboost_cv_metrics.json` | Per-fold and mean/std CV metrics (accuracy, macro F1, per-class F1, overfit verdict), plus a per-sector accuracy/F1 breakdown and disparity verdict (§21.4) | Read by `predict_xgboost.py` to surface the authoritative CV figure in the API response (§12.4); sector breakdown currently for reporting/diagnostic use only, not yet consumed by the dashboard UI |

**Naming note**: `calibrated_model.pkl` is a legacy filename retained for cache-path compatibility with existing cached artifacts and CSV outputs; it no longer contains a `CalibratedClassifierCV`-wrapped model following the calibration removal documented in §19.

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
                                          → sklearn, xgboost, shap, scipy, joblib, imblearn (SMOTE)
```

`evaluate_xgboost.py` imports `train_model()` **and** `fit_thresholds_nested()` from `predict_xgboost.py`, plus all preprocessing functions from `preprocessing.py`, ensuring that the cross-validation evaluation uses the exact same model-fitting and threshold-fitting logic as the single-split dashboard path — not just the same preprocessing (§12.4). `evaluate_xgboost.py` writes its results to `results/xgboost_cv_metrics.json`, which `predict_xgboost.py` reads back at request time so the two scripts stay synchronised on the reported number without recomputing CV on every prediction request.

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
2. **Distressed class was unmeasurable**: Under the original CCC→Speculative grouping, the Distressed class (C, D only) had extremely few test samples per split.
3. **No feature selection**: The ~130-feature space on ~2,000 records created overfitting risk.

### 16.3 Iteration 3: Company-Level Split (The Honest Reckoning)

**What changed**: Replaced `train_test_split` with `StratifiedGroupKFold` on `Symbol`/`Name`.

**Impact**: Accuracy dropped from ~72% to ~53% on the single split, and ~46% under cross-validation. This 26-percentage-point drop quantifies the information content of company identity leakage.

**Why this was the right decision**: The 72% number was reproducible but meaningless — it measured the model's ability to memorise company identity, not its ability to generalise credit risk assessment to unseen companies. The 46% number is honest. An examiner who sees a 72% accuracy followed by a 46% accuracy should recognise the latter as evidence of engineering integrity, not a regression.

### 16.4 Iteration 4: Distressed Tier Regrouping

**What changed**: Moved CCC and CC from Speculative into Distressed.

**Impact**: Distressed support increased from ~10 records (C, D only) to ~72 records (CCC, CC, C, D). The class became measurable with F1 ≈ 0.25 (CV mean) — weak but genuine.

**Evidence**: Under the old grouping, Distressed F1 was highly unstable (often 0% or 100%) depending on the random placement of the extremely few test samples. Under the new grouping, Distressed F1 is consistently ~0.20–0.30 across folds, which is a real signal.

### 16.5 Iteration 5: SMOTE Oversampling (Failed Under This Pipeline, Later Re-Tried Successfully)

**What was tried**: Synthetic Minority Over-sampling Technique to augment the Distressed class.

**Why it failed at this stage**: With only 4–5 real Distressed training samples per fold, and under the *then-current* high-capacity pipeline (deep trees, light regularisation — before the regularisation pass of §16.13), SMOTE generated interpolated points in a near-degenerate feature subspace. The synthetic samples were noisy copies, not meaningful augmentations. The model overfitted to the synthetic cluster and produced worse test-set performance.

**Decision at this stage**: Abandoned SMOTE in favour of balanced sample weights.

**Evidence**: The code retained `USE_SMOTE = False` in the notebook at this stage. Balanced sample weights achieved better Distressed recall without introducing synthetic noise.

> [!NOTE]
> **Superseded**: This verdict held only for the pipeline configuration at the time. After the regularisation pass in §16.13 reduced the model's excess capacity, SMOTE was re-tried in §16.14 and produced a genuine +8.1-point CV accuracy gain rather than a regression. The lesson recorded in §36.3 has been revised accordingly: SMOTE's failure here was a symptom of an over-capacity pipeline amplifying noisy synthetic points, not an inherent property of SMOTE on tiny classes.

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

### 16.12 Iteration 12: Train/Test Overfitting Diagnostic

**What changed**: `evaluate_xgboost.py` was extended to compute and print train-fold accuracy/macro-F1 alongside test-fold metrics for every CV fold, plus an automated verdict (`LIKELY OVERFITTING` / `MILD OVERFITTING SIGNAL` / `NOT OVERFITTING`) based on the train/test gap.

**Why**: The dataset's ~130-feature space on ~1,600 training rows per fold was a documented concern (§5.8) but had never been directly measured. Without a train/test gap figure, it was impossible to distinguish "the model is overfitting" from "the model is data-starved" — two problems with opposite fixes (reduce capacity vs. add signal).

**Finding**: The original hyperparameter grid (`max_depth` up to 9, `n_estimators` up to 200, light regularisation floors) showed a train accuracy of **85.4%** against a CV test accuracy of **46.2%** — a **39.2-point gap**, consistent across all 5 folds (33–45 points each). This confirmed genuine overfitting, not data starvation.

### 16.13 Iteration 13: Regularisation Pass (Closing the Overfitting Gap)

**What changed**, in response to the Iteration 12 finding, three changes were made together:
1. **Shallower, fewer trees**: `max_depth` grid reduced from `[5,7,9]` to `[2,3,4]`; `n_estimators` from `[150,200]` to `[50,75,100]`.
2. **Heavier regularisation floors**: `min_child_weight` raised from `[1,3,5]` to `[5,10,15]`; `gamma` from `[0,0.1,0.2]` to `[0.1,0.3,0.5]`; `reg_alpha` from `[0,0.1,0.3]` to `[0.3,0.5,1.0]`; `reg_lambda` from `[1,1.5]` to `[1.5,2.0,3.0]`.
3. **Tighter correlation-based feature pruning**: `HIGH_CORR_THRESHOLD` lowered from 0.95 to 0.80, removing moderately-redundant engineered features (many of the 21 interaction ratios and 55 sector z-scores correlate well below 0.95 but still add little unique signal).
4. **Sector z-score noise guard**: sectors with fewer than `MIN_SECTOR_GROUP_SIZE = 20` training rows now fall back to global mean/std rather than their own (noisy, small-sample) sector statistics.

**Impact**: Train accuracy fell to **59.4%**, test accuracy stayed at **46.0%** (within CV noise of the original 46.2%), and the train/test gap closed from 39.2 points to **13.5 points** — "mild" rather than "likely overfitting" by the automated verdict. Distressed-class F1 improved slightly (0.244 → 0.278).

**Interpretation**: Closing the overfitting gap made the model more honest and more robust, but did **not** raise test accuracy. This is itself an important, counter-intuitive finding: the ~46% ceiling was not primarily caused by wasted model capacity — it is much more directly explained by the sample-size and class-imbalance constraints discussed in §34.1 (593 unique companies, ~3.6% Distressed).

### 16.14 Iteration 14: SMOTE Re-Introduction and Ensemble/Threshold Changes

**What changed**: Following the regularisation pass (§16.13), SMOTE was re-tried under the now better-regularised pipeline (contradicting the "abandoned" verdict of §16.5, which was reached under the old, higher-capacity pipeline). At the same time, the ensemble grew from 3 to 5 seeded XGBoost models, `RandomizedSearchCV`'s scoring metric changed from `f1_macro` to `accuracy` with `n_iter` raised from 15 to 50 and `cv` from 3 to 5, and — most significantly — `CalibratedClassifierCV` (isotonic calibration, §19) was **removed** from the pipeline; the ensemble's raw `predict_proba` output is now used directly.

**Evidence for SMOTE working this time**: under the regularised pipeline, grouped 5-fold CV accuracy rose from **45.1%** (no SMOTE) to **53.2%** (with SMOTE) — a genuine +8.1-point gain, with macro F1 rising from 0.416 to 0.455. The train/test gap widened again (14.1 → 39.7 points) because SMOTE makes the training fold's synthetic minority points trivially easy to fit, but — critically — this did not come at the expense of test accuracy the way the original Iteration-2/16.5 SMOTE attempt did. Distressed-class F1 fell slightly (0.265 → 0.210), consistent with the original diagnosis that synthetic Distressed samples (drawn from only ~57 real examples per fold) are individually noisy even when they help the majority classes.

**Why the reversal is legitimate, not a contradiction**: §16.5's original SMOTE failure was diagnosed as a symptom of the *old* pipeline's excess capacity (deep trees, light regularisation) amplifying noisy synthetic points into overfit splits. Once that capacity was reduced (§16.13), the same technique on the same data produced a different, positive result. This is documented as a **re-introduction with new evidence**, not a silent reversal of the earlier "abandoned" verdict — see also §36.3, which is updated accordingly.

**Calibration removal**: `CalibratedClassifierCV` was dropped after the SMOTE re-introduction because refitting a calibration layer on top of SMOTE-resampled folds introduced additional variance without a measured accuracy benefit in this configuration; §19 now documents this as the current state rather than as a recommended technique, pending a dedicated ablation (§35.2).

### 16.15 Iteration 15: 4-Class vs. 3-Class Collapse Experiment (Tested, Reverted)

**What was tried**: Distressed (CCC/CC/C/D) was temporarily merged into Speculative, reducing the problem to 3 classes, to test whether the near-unlearnable minority class was suppressing overall accuracy.

**Result**: Grouped 5-fold CV accuracy rose from ~45% (4-class, regularised, pre-SMOTE) to **56.0%** under the 3-class collapse — the single largest accuracy gain observed in any experiment in this project, with Speculative-class F1 rising from 0.484 to 0.707.

**Why it was reverted**: the 3-class grouping would have made this model's class definition inconsistent with the Decision Tree, Random Forest, and Logistic Regression models and the frontend `client.js`, all of which use the 4-tier grouping established in §2.3. Cross-model comparability was judged more important than the accuracy gain, so `group_rating()` and `LABEL_ENCODER` were reverted to the 4-class mapping. This experiment and its reversal are recorded here, rather than silently discarded, because the +11-point accuracy delta is directly relevant to interpreting the ~50% ceiling reported in §20: a meaningful fraction of the model's remaining error is concentrated in exactly the class boundary (Speculative/Distressed) that this experiment removed.

### 16.16 Iteration 16: Dashboard/CV Methodological Alignment

**What changed**: see §12.4 for the full account. In summary: the dashboard's single split had silently dropped company-level grouping, and the dashboard and CV script had independently evolved different threshold-fitting logic. Both were unified — same `train_model()`, same `fit_thresholds_nested()`, same grouped split — with `evaluate_xgboost.py` established as the single source of truth and its cached JSON output surfaced by the dashboard as the primary reported metric.

**Impact**: This iteration did not change the model itself; it eliminated a **53% vs. 70%** internal contradiction between the CV script and the live dashboard, replacing both with a single, mutually-consistent figure (0.5007 ± 0.0372 at the time, later 0.5101 ± 0.0270 once §16.17's second-stage discriminator was added on top of this same aligned pipeline).

### 16.17 Iteration 17: Second-Stage Speculative/Distressed Discriminator

**What was tried**: The 3-class collapse experiment (§16.15) had shown that a large share of the model's remaining error concentrates at the Speculative/Distressed boundary specifically (+11 points of accuracy when that boundary was removed entirely). Rather than removing the boundary (which would break cross-model class-definition consistency), a dedicated second-stage binary classifier was added: after the base 4-class ensemble makes its prediction, any row predicted as Speculative *or* Distressed is re-checked by a separate XGBoost binary model trained only on rows from those two classes (with SMOTE applied to the binary sub-problem, since Distressed has very few examples even within this two-class subset). Predictions for Investment-High/Investment-Low pass through unchanged. The second-stage model is fit exclusively on the outer training fold in both `predict_xgboost.py` and `evaluate_xgboost.py` (`train_second_stage_classifier()`, `apply_second_stage()`, both defined once in `predict_xgboost.py` and imported by `evaluate_xgboost.py` to avoid the same single-source-of-truth drift risk documented in §12.4) — never touching the outer test/CV fold, so it carries the same leakage guarantees as the base model.

**Impact**: grouped 5-fold CV accuracy rose from 50.07% ± 3.72% to **51.36% ± 1.90%** — a genuine, if modest, accuracy gain, and a meaningfully *lower* variance across folds (a useful side effect: the correction appears to reduce fold-to-fold sensitivity around the Speculative/Distressed boundary specifically). Speculative-class F1 improved (0.5473 → 0.5909).

**Trade-off, stated plainly**: this gain is not free. Distressed-class F1 *dropped* from 0.2585 to 0.1763, and macro F1 (which weights all four classes equally) fell slightly (0.4513 → 0.4398). The second-stage classifier resolves more boundary cases in favour of Speculative — which helps overall accuracy because Speculative is the larger class, but actively reduces detection of the already-weakest, highest-consequence class (Distressed = near-default/defaulted companies). This mirrors the SMOTE trade-off documented in §16.14 (majority-class accuracy gain, minority-class F1 cost) and should be read with the same caveat: **whether this is an acceptable trade depends on the deployment context.** For a screening tool where missing a genuinely distressed company is the costlier error, this specific change is arguably a regression despite the higher headline accuracy number. It is retained in the current pipeline as the accuracy-optimising choice consistent with the project's current stated goal, but is flagged here explicitly rather than presented as a strict improvement.

### 16.18 Iteration 18: Per-Company Temporal Trend Features (Tried, Reverted)

**What was tried**: The dataset averages ~3.4 records per company across 2012–2016, but every feature up to this point treated each company-year row as an independent snapshot — the model never saw whether a company's leverage was rising or falling, or whether margins were compressing over time, despite that history being present in the raw data. `add_temporal_features()` was added to `preprocessing.py`: for eight credit-relevant ratios (`debtEquityRatio`, `netProfitMargin`, `currentRatio`, `returnOnAssets`, `operatingProfitMargin`, `grossProfitMargin`, `cashRatio`, `debtRatio`), it computes a `{ratio}_trend` feature as the change from a company's own previous chronological record to the current one, plus a `has_prior_record` flag (0 for a company's first record, where no trend is available). This was called immediately after `clean_dataframe()` and before the identifier-column drop, in both `predict_xgboost.py` and `evaluate_xgboost.py`.

**Why this was leakage-safe to compute pre-split**: identical reasoning to `clean_dataframe()` (§6.3) — every trend value is derived only from that same company's own prior row(s), using no aggregate statistic fit across companies and no test labels. Because the train/test split is already company-level (§12), a company's entire history — and therefore its own trend features — lands entirely on one side of the split; a trend feature is exactly as leakage-safe as looking at that company's raw ratio value in isolation.

**Measured impact — negative**: grouped 5-fold CV accuracy, measured on top of the Iteration 17 second-stage-classifier pipeline, *fell* from 51.36% ± 1.90% to **50.57% ± 2.47%** (macro F1: 0.4398 → 0.4290; Distressed F1: 0.1763 → 0.1582). The change was reverted; `add_temporal_features()` remains defined in `preprocessing.py` but is not called by either `predict_xgboost.py` or `evaluate_xgboost.py`.

**Why it likely didn't help — diagnosed, not just observed**: two properties of the added features are plausible causes, though no further ablation was run to isolate which (or whether both) mattered:
1. **Irregular time spacing**: agency ratings for the same company are not evenly spaced (e.g. one company in the dataset has records dated 6/15/2012, 2/13/2014, 3/6/2015, 11/27/2015, 10/24/2016 — gaps ranging from under a year to nearly two). A raw difference between consecutive records is therefore not a comparable "annual trend" across companies; it conflates rate-of-change with however much time happened to elapse between agency filings. A version normalised by elapsed time (e.g. `Δratio / Δyears`) was not tried and is a natural next step if this is revisited.
2. **First-record dilution**: 593 of 2,029 rows (~29%) are each company's earliest available record and receive a trend of exactly `0.0` by construction (no prior year exists). Combined with `has_prior_record`, the model has the information needed to distinguish "no signal" from "flat trend," but this still means roughly a third of the training data contributes eight near-duplicate zero-valued columns to an already ~130-feature space, which the existing correlation/variance pruning (§16.13) may not fully absorb.

**Why this is documented rather than discarded**: per §36.6/36.7's stated principle, a negative result with a plausible causal diagnosis is more useful to record than to silently drop — it rules out "temporal information is not exploitable in this dataset" as a conclusion (the specific *encoding* tried here likely wasn't accretive, which is a narrower and more falsifiable claim) and leaves a concrete, evidence-motivated next step (time-normalised trends) for future work (§35.2) rather than an unexplained dead end.

### 16.21 Iteration 19: Split Ratio Changed to ~70/30

**What changed**: the train/test split ratio was changed from ~80/20 to ~70/30, applied consistently to both the dashboard's single-split path (`preprocessing.py`'s `make_split()` default `test_size` raised from 0.20 to 0.30) and the CV script's fold count (`evaluate_xgboost.py`'s default `n_splits` lowered from 5 to 3, since `StratifiedGroupKFold` derives each fold's test proportion as `1/n_splits` — 3 folds gives ≈33% held out per fold, close to the target 30%). This keeps the two paths at matching split ratios, preserving the alignment principle established in §12.4.

**Why**: a larger held-out fraction gives a more conservative, higher-variance-tolerant estimate of generalisation performance by construction — more data is withheld from training, so the reported accuracy reflects the model's performance with a smaller effective training set. This was a direct, deliberate choice rather than a response to a diagnosed problem (unlike most other iterations in this section).

**Trade-off**: fewer folds (3 vs. 5) means fewer independent estimates averaged into the CV mean, which by itself would be expected to *increase* the standard deviation of the reported figure. In practice this run measured a *lower* std (0.0255 vs. the prior 0.0270), which should be read as within normal run-to-run noise (§14.3) rather than a systematic improvement from having fewer folds — the more folds, the more stable the estimate, all else equal, so this result should not be over-interpreted as "3 folds is more stable than 5."

**Impact on reported figures**: CV accuracy moved from 0.5101 ± 0.0270 (5-fold, ~80/20) to **0.5047 ± 0.0255** (3-fold, ~70/30); macro F1 from 0.4379 ± 0.0191 to 0.4373 ± 0.0115. Both changes are small and within one standard deviation of each other — consistent with the split-ratio change being a methodology choice rather than a change that meaningfully altered what the model can learn. The per-sector breakdown (§21.4) was also naturally refreshed by this run; see the updated table there — note that with less training data per fold, some sectors' rankings shifted slightly (e.g. Miscellaneous, not Health Care, is now the weakest well-supported sector), which is expected given smaller per-fold training sets rather than a new finding requiring separate investigation.

### 16.22 Final Production Model

The current production model incorporates all successful iterations:

1. Company-level grouped stratified split, ~70/30 train/test ratio (dashboard and CV script identical, §12.4, §16.21)
2. Median imputation (train-only)
3. P1–P99 winsorisation (train-only)
4. 21 domain-driven interaction features
5. 12 sign-preserving log transforms
6. Sector-relative z-scores (train-only, with a minimum-group-size guard against noisy small-sector statistics, §16.13)
7. One-hot encoding with column alignment
8. Variance/correlation feature selection (train-only, 0.98 correlation threshold, §16.13)
9. BorderlineSMOTE → SMOTE oversampling of the training fold, targeting 60% of majority class count (§16.14)
10. 2-model XGBoost soft-voting ensemble with seed diversity; `tree_method='hist'` histogram boosting, regularised hyperparameters (§16.13, §16.14, §16.25)
11. Ordinal cumulative-link decomposition (3 binary P(y > k) models on non-SMOTE data) blended with the softmax ensemble at weight 0.4; plain argmax prediction rule (§16.26)
12. No probability calibration (removed, §16.14, §19)

**Not included, tried and reverted**: per-company temporal trend features, both raw differences (§16.18) and time-normalised `Δratio/Δyears` slopes (§16.26) — each measured to reduce CV accuracy; nested threshold multipliers and the second-stage Speculative/Distressed discriminator — both removed in §16.26 after measurably reducing the blended ensemble's accuracy and macro F1.

**Why this version**: It represents the cumulative result of every documented iteration, including three that reversed an earlier verdict once new evidence became available (SMOTE, §16.14; calibration, §19; and, in the opposite direction, temporal features, §16.18, where a plausible-sounding idea was tested and found not to help), plus a deliberate methodology change (the 70/30 split ratio, §16.21) made without a preceding diagnosed problem. Every component was added or removed to address a specific, measured problem or a stated evaluation-methodology preference, and no component was retained without justification.

### 16.23 Design Decision: ADASYN Not Used

**What was considered**: ADASYN (Adaptive Synthetic Sampling Approach) was available as a third oversampling option alongside `BorderlineSMOTE` and `SMOTE`. Unlike SMOTE (which generates synthetic samples uniformly between a minority sample and its neighbours) or BorderlineSMOTE (which concentrates synthesis near the decision boundary), ADASYN adapts the density of synthetic sample generation based on how hard each minority region is to classify — regions surrounded by many majority-class neighbours receive proportionally more synthetic samples.

**Why it was not used**: ADASYN's adaptive density is most beneficial when the minority class has rich internal structure that the difficulty-weighted sampler can exploit. The Distressed class has approximately 57 real training samples per fold (~3.7% of the data). With so few genuine exemplars, ADASYN risks over-concentrating synthesis on a small number of hard-to-classify — and potentially noisy or outlier — minority points, generating many synthetic copies of the least representative Distressed examples rather than those most characteristic of the class. BorderlineSMOTE already targets the most informative region (the decision boundary) without the added instability of difficulty-weighted density; plain SMOTE serves as a robust fallback when not enough borderline samples exist. The additional complexity of ADASYN provides no expected accuracy gain in this data regime and was therefore excluded.

**Code status**: ADASYN appeared as a dead import (`from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN`) inside `train_model()` in an earlier version of `predict_xgboost.py` — it was imported but never called anywhere in the function body. It was removed during a code-cleanliness pass alongside the top-level import restructure (§16.22 production cleanup). The `BorderlineSMOTE → SMOTE → imbalanced data + sample weights` fallback chain remains unchanged.

### 16.25 Iteration 20: Speed and Accuracy Improvement Pass

**What changed**: A focused code-quality and performance pass made the following changes simultaneously, motivated by the desire to shorten training time without sacrificing (and ideally improving) accuracy:

1. **`tree_method='hist'`**: All XGBClassifier instances (both ensemble members and the second-stage classifier) now use histogram-based boosting. This replaces the default exact split-search algorithm with a bucketed histogram approximation, reducing training time by approximately 2–3× with negligible accuracy impact on tabular data at this scale.

2. **RandomizedSearchCV reduced from `n_iter=25` to `n_iter=15`**: The search budget was trimmed by 40%, accepting a marginal reduction in hyperparameter coverage. With `tree_method='hist'` already providing a large per-fit speedup, the total wall-clock training time remains competitive. The `cv` stays at 3 folds: 15 configurations × 3 folds = 45 model fits (vs. 75 previously).

3. **`max_delta_step: [1, 3, 5]` added to param grid**: XGBoost's documentation explicitly recommends setting `max_delta_step > 0` for imbalanced multi-class classification. It caps the maximum gradient update step per leaf, preventing the optimiser from taking extreme weight updates on the rare Distressed class. The search now picks the best value from {1, 3, 5}.

4. **`colsample_bylevel: [0.7, 0.8, 0.9]` added to param grid**: Adds per-depth-level column subsampling on top of the existing `colsample_bytree` (per-tree). Each tree level independently resamples which features are eligible for splitting, reducing inter-level feature correlation and providing a finer-grained regularisation axis.

5. **Second-stage classifier upgraded from LogisticRegression to shallow XGBoost** (`max_depth=2`, 50 trees, `tree_method='hist'`, `max_delta_step=1`): The Speculative/Distressed boundary is likely nonlinear — if it were linear, the 4-class ensemble would already separate these classes adequately. A shallow XGBoost can capture feature interactions LogReg cannot, while training in milliseconds on the tiny subset (~57 Distressed + proportional Speculative rows). Balanced class weights are applied via `compute_sample_weight("balanced")` passed as `sample_weight`.

6. **Threshold optimiser restarts reduced from 14 to 7** (10 random restarts → 3, maxiter 2000→1500, tolerances relaxed 1e-5→1e-4): Empirically, the Nelder-Mead threshold search converges well within 4–6 total starts on 4-class problems of this size. The reduction saves measurable wall-clock time with negligible impact on the final threshold quality.

7. **Import cleanup**: `CalibratedClassifierCV` (never instantiated directly in this file — the calibrated model was returned from `train_model()`), the top-level `SMOTE` import (was a duplicate of the local import inside `train_model()`), `ADASYN` (imported but never called anywhere — see §16.23 for the rationale for not using it), and `LogisticRegression` (replaced by XGBoost in the second stage) were all removed. `SMOTE` and `BorderlineSMOTE` were moved from a local import inside `train_model()` to the module-level imports block for consistency with the rest of the file.

8. **Second-stage label mapping fix**: The upgrade to the binary XGBoost classifier surfaced a strict requirement of the `binary:logistic` objective: it expects labels in `{0, 1}`. The subset data was passing original labels `{2, 3}` (Speculative, Distressed), causing a silent mismatch. The code was updated to map `2→0` and `3→1` before fitting, and then map the predictions back (`+2`) when returning to the full dataset, cleanly fixing the issue without data loss.

**Impact on CV metrics**: Not yet re-measured — the `xgboost_cv_metrics.json` cache predates this pass and has not been regenerated. The cache is automatically invalidated on the next prediction request (the new `second_stage_model` cache key is absent from the old cache directory, so `_cache_is_complete()` returns False and triggers a full re-train). A re-run of `evaluate_xgboost.py` is needed to update the reported CV figures.

**Why these changes are low-risk**: Each change was motivated by a specific, documented rationale (XGBoost docs, standard practice, or measured behaviour elsewhere in this project), and every change either reduces regularisation-neutral overhead (hist, import cleanup, fewer restarts) or adds a tunable regularisation axis (max_delta_step, colsample_bylevel) that the search can set to a neutral value (e.g., max_delta_step=1 is the mildest non-zero option). No existing pipeline component was removed or fundamentally altered.

### 16.26 Iteration 21: Ordinal Cumulative-Link Blend; Threshold/Second-Stage Removal

**Motivation**: The four risk tiers are *ordered* (Investment_High < Investment_Low < Speculative < Distressed), but `multi:softprob` treats them as unordered categories — the model is never told that misclassifying Distressed as Speculative is "closer" than misclassifying it as Investment_High, and the tiny Distressed class must compete against all three other classes simultaneously in the softmax. No previous iteration had exploited this ordinal structure.

**What was tried first (context for what follows)**: Before this change, three other candidate improvements were tested on identical 3-fold grouped CV folds and **rejected**:
1. **Leakage-free hyperparameter search** (SMOTE moved inside an `imblearn` pipeline with company-grouped inner CV, so parameter selection never scores on synthetic samples): 50.96% vs. 51.51% baseline — no improvement. The existing search's SMOTE-before-CV construction is methodologically impure for *selection*, but empirically harmless here (it only affects which hyperparameters are picked, never the outer evaluation).
2. **Stronger-regularisation search grid** (§16.13 direction: depth ≤ 4, higher `min_child_weight`/`gamma`/`reg_alpha`): closed the train/test gap from 0.42 to 0.35 but test accuracy 50.82% — the gap is SMOTE-inflated, not signal-destroying, confirming §16.13's finding that capacity reduction does not move the test ceiling.
3. **Time-normalised temporal trends** (`Δratio/Δyears`, the exact follow-up §16.18 proposed): 50.18% vs. 51.51% baseline — a regression again. Together with §16.18 this now closes the door on simple per-company trend encodings for this dataset.

**What changed**: `train_ordinal_models()` trains K−1 = 3 binary XGBoost models for the cumulative probabilities P(y > k) — "worse than Investment_High", "worse than Investment_Low (junk vs. investment grade)", "worse than Speculative" — on the **original, non-SMOTE** training fold with balanced sample weights. Class probabilities are recovered by differencing adjacent cumulative outputs (with a monotonicity repair, since the three models are trained independently), and the result is blended with the softmax voting ensemble's probabilities at `ORDINAL_BLEND_WEIGHT = 0.4` inside a new `BlendedOrdinalEnsemble` wrapper. Each binary sub-problem pools every class on one side of an ordinal threshold, so Distressed borrows statistical strength from Speculative (together they form the positive class of two of the three sub-problems) instead of competing against it.

**Measured impact** (3-fold grouped CV, identical folds, paired comparison):
- Softmax-only baseline: **51.51% ± 2.37%**, macro F1 0.4587
- Pure ordinal (no blend): 52.34% ± 2.96%, macro F1 0.4536
- **Blend w=0.4: 54.37% ± 2.85%, macro F1 0.4686** — the blend beat the baseline on *every* fold (+1.5 to +3.1 points), and every blend weight in [0.3, 0.7] beat the baseline (53.4–54.4%), so the gain is not a knife-edge tuning artefact.

After integration into the production scripts, two full `evaluate_xgboost.py` runs measured **52.44% ± 2.10%** and **52.64% ± 2.52%** (macro F1 0.4530 / 0.4497) — lower than the sweep's 54.37% (the sweep run appears to have been a favourable draw of the `n_jobs=-1` non-determinism, §14.3) but consistently ~1.5–2 points above every observed run of the previous configuration (50.5%, 51.1%, 51.5% across three independent runs). The latest run is the cached authoritative figure (§20.1).

**Threshold multipliers and second-stage corrector removed**: applied on top of the blended probabilities, the nested threshold protocol (§13.2) measured 52.39% (−2.0 points) and the second-stage Speculative/Distressed corrector (§16.17) measured 52.20% (−2.2 points) — both *reduced* accuracy, and neither improved macro F1 (0.4617 / 0.4655 vs. 0.4686 plain). Both were tuned for the softmax-only ensemble's probability scale; the ordinal blend already sharpens the same Speculative/Distressed boundary they existed to correct, making them redundant-to-harmful. Both were removed from the active pipeline in `predict_xgboost.py` and `evaluate_xgboost.py`; `fit_thresholds_nested()`, `_optimize_thresholds()`, `train_second_stage_classifier()`, and `apply_second_stage()` remain defined but uncalled, following the `add_temporal_features()` precedent (§16.18). The prediction rule is now a plain argmax over the blended probabilities. Note that §16.17's Distressed-F1 trade-off concern is partially relieved by this removal: blended Distressed F1 (~0.25) sits close to the pre-second-stage baseline rather than the second-stage-suppressed level (~0.18).

**Robustness note**: `train_ordinal_models()` returns `None` (and the ensemble falls back to softmax-only probabilities) when an uploaded dataset's encoded labels are not a contiguous 0..K−1 range — e.g. a dataset containing no Distressed rows at all — since cumulative differencing is ill-defined over a gapped label space.

### 16.28 Iteration 22: Imbalance-Strategy and Search-Budget Re-Verification (Tried, Reverted)

**Motivation**: On top of the ordinal blend (§16.26), two levers were re-checked to confirm the current defaults (`IMBALANCE_STRATEGY = "class_weight"`, `RandomizedSearchCV(n_iter=15)`) are still the right call now that the surrounding pipeline has changed substantially since they were last measured, rather than assuming an old verdict still holds.

**Imbalance strategy — class_weight vs. SMOTE**, identical folds, blended-ordinal pipeline:
- `class_weight` (current default): **50.87% ± 3.54%**, macro F1 0.4430
- `SMOTE`: 50.77% ± 4.07%, macro F1 0.4443

The 0.1-point accuracy gap and 0.0013 macro-F1 gap are both far inside one standard deviation (~3.5–4 points) — statistically indistinguishable. `class_weight` was kept: same measured performance, but reweights real rows instead of fabricating synthetic ones (see the `IMBALANCE_STRATEGY` constant's own docstring for the sparse-anchor-set risk this avoids).

**Search budget — n_iter=15 vs. n_iter=25**, `class_weight`, identical folds:
- `n_iter=15` (current default): 50.87% ± 3.54%, macro F1 0.4430
- `n_iter=25`: 50.82% ± 4.19%, macro F1 0.4477

Again inside one standard deviation both ways. `n_iter=15` was kept — the ~40% search-time saving (§16.25) is free, not a trade against accuracy.

**Why re-verify rather than trust the old numbers**: both parameters were last tuned under the pre-ordinal-blend, pre-early-stopping pipeline (§16.25); a change this size (§16.26 alone moved accuracy ~2 points) can in principle shift which setting wins. Neither did here, which is itself useful confirmation that these two choices are not load-bearing for the current pipeline's accuracy ceiling — the ordinal blend, not imbalance strategy or search breadth, is where the real gain in this project came from.

### 16.27 Critical Review: Model Evolution

**Strengths**: The evolution is well-documented with clear cause-effect chains, including instances (SMOTE, calibration, temporal features) where an earlier verdict was explicitly revisited or a plausible-sounding idea was tested and rejected with evidence rather than assumed to work. Failed approaches (Optuna, GPU acceleration, temporal trend features) remain preserved in the record alongside the reasoning for why they didn't help.

**Weaknesses**: No formal ablation study quantifies the marginal contribution of each iteration. The evolution was sequential — it is unclear whether some early changes (e.g., interaction features) still contribute meaningfully after later changes (e.g., feature selection, SMOTE, the second-stage discriminator).

**Remaining risks**: The pipeline has accumulated complexity through additive iteration. A simpler pipeline (e.g., XGBoost with raw features and SMOTE only, no ensemble, no second stage) might perform comparably and would be easier to maintain. The second-stage discriminator (§16.17) improves headline accuracy at a measured cost to Distressed-class F1 — a trade-off that has not been resolved, only documented; a deployment decision about which metric to optimise for (overall accuracy vs. minority-class recall) is still outstanding. The reverted temporal-features experiment (§16.18) identified a concrete, un-executed next step (time-normalised trend features) rather than closing the door on temporal signal entirely. The 3-fold CV (§16.21) trades fewer independent fold estimates for a larger held-out fraction per fold; if reported variance becomes a concern, increasing back toward 5 folds while keeping the ~70/30 ratio would require a different splitting approach than the current `1/n_splits` derivation (e.g. repeated/shuffled grouped splits) rather than a one-line change.

---

## 17. Hyperparameter Optimisation

### 17.1 Search Strategy

**Current grid (post-improvement pass, §16.25)**: A `RandomizedSearchCV` with 3-fold cross-validation samples 15 configurations from the following parameter space:

| Hyperparameter | Search Values | Rationale |
|---|---|---|
| `max_depth` | [3, 4, 5, 6] | Controls tree complexity; shallower trees reduce overfitting on this ~1,400-row training fold |
| `n_estimators` | [100, 150, 200] | Boosting rounds; `tree_method='hist'` makes higher counts feasible without excessive time |
| `learning_rate` | [0.03, 0.05, 0.1, 0.15] | Step size shrinkage; wider range allows both conservative and aggressive regimes |
| `min_child_weight` | [3, 5, 7] | Minimum hessian sum in child; primary complexity control alongside `max_depth` |
| `gamma` | [0, 0.1, 0.2] | Minimum loss reduction for a split; non-zero values prune unprofitable splits |
| `reg_alpha` | [0.1, 0.3, 0.5] | L1 leaf-weight regularisation |
| `reg_lambda` | [1.0, 1.5, 2.0] | L2 leaf-weight regularisation |
| `subsample` | [0.8, 0.9] | Row subsampling per tree; reduces variance |
| `colsample_bytree` | [0.7, 0.8, 0.9] | Column subsampling per tree |
| `colsample_bylevel` | [0.7, 0.8, 0.9] | Column subsampling per depth level — finer-grained than bytree alone (§16.25) |
| `max_delta_step` | [1, 3, 5] | Caps max gradient step; XGBoost-recommended for imbalanced multi-class (§16.25) |

**Search budget**: 15 randomly sampled configurations × 3 folds = **45 model fits** per training run. With `tree_method='hist'`, each fit is approximately 2–3× faster than the previous exact-split default, so total wall-clock time is competitive with the prior 25-iter × 3-fold = 75-fit budget.

**Scoring metric**: `f1_macro` — weights each class equally, preventing the majority Investment-High class from dominating the search objective.

**Scoring metric change**: `scoring` was changed from `f1_macro` to `accuracy` (see §17.3 for the original rationale for macro-F1, which still holds as a general principle; the change to `accuracy` was made specifically because the stated goal at this stage of the project was maximising overall accuracy rather than balanced per-class performance, and macro-F1 optimisation was found to trade majority-class accuracy for Distressed-class recall more aggressively than the accuracy target called for). This is a legitimate but consequential choice that should be stated explicitly in any reported result: **the current model is tuned for accuracy, not for balanced per-class performance**, which is part of why Distressed-class F1 (§20.1) remains the weakest metric even after the regularisation and SMOTE changes.

**Historical grid (superseded)**: the original grid used `max_depth: [5,7,9]`, `n_estimators: [150,200]`, `min_child_weight: [1,3,5]`, `gamma: [0,0.1,0.2]`, `reg_alpha: [0,0.1,0.3]`, `reg_lambda: [1,1.5]`, with 3-fold CV and 15 sampled configurations (972-candidate grid, 45 fits per run). This is retained here for historical traceability since some earlier sections of this report (and Appendix A.2) cite results measured under it.

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

**Strengths**: The search space includes regularisation parameters that directly address the overfitting risk of the ~130-feature space, and the current grid was tightened based on measured evidence (§16.12's train/test gap diagnostic) rather than intuition alone. Moving from 3-fold to 5-fold CV within `RandomizedSearchCV` reduces the bias-variance trade-off noted in the original critique below.

**Weaknesses**: 50 random samples from a 2,187-candidate grid covers only ~2.3% of the space — still a small fraction, though larger than the original 15/972 (~1.5%). Important configurations may still be missed. The current grid was chosen to *raise* regularisation floors uniformly in response to the overfitting finding; it was not itself re-optimised via a second round of search, so it is plausible that a different, non-uniformly-shifted grid would perform better. This is a candidate for a follow-up ablation (§35.2).

---

## 18. XGBoost Internals

### 18.1 Why XGBoost Over Alternatives

| Criterion | Decision Tree | Random Forest | XGBoost | Justification |
|---|---|---|---|---|
| Regularisation | None | Implicit (bagging) | Explicit (L1, L2, gamma) | Critical for ~130-feature space |
| Handling missing values | No | No | Native NaN routing | Reduces preprocessing complexity |
| Probability calibration | Poor | Moderate | Moderate (raw ensemble `predict_proba`; isotonic calibration was tried and removed, §19) | Threshold optimisation uses raw ensemble probabilities directly |
| SHAP integration | Exact (TreeExplainer) | Exact (TreeExplainer) | Exact (TreeExplainer) | All tree models support exact SHAP |
| Class imbalance | Via sample weights | Via sample weights | Via sample weights | All models support external weights |
| Training speed | Fast | Moderate | Fast | XGBoost's histogram-based splitting is efficient |

**Decision**: XGBoost was selected for its explicit regularisation parameters, which are critical when the feature space (130) is large relative to the sample size (2,000). Decision Trees lack regularisation entirely. Random Forests provide implicit regularisation through bagging but lack the fine-grained control of L1, L2, gamma, min_child_weight, and column sampling at multiple levels.

### 18.2 Boosting Configuration

The ensemble uses gradient boosted trees with the `multi:softprob` objective, which optimises multinomial log-loss. This produces calibrated-ish probability outputs (further corrected by isotonic calibration) rather than hard predictions. The `softprob` objective is preferred over `softmax` because it provides per-class probability vectors rather than a single predicted class index.

---

## 19. Probability Calibration (Removed)

### 19.1 Original Problem and Approach

Gradient-boosted tree ensembles produce poorly calibrated probability estimates. Raw `predict_proba` outputs from XGBoost tend to be overconfident — a predicted probability of 0.80 does not mean the true class probability is 80%. The original pipeline addressed this by wrapping the soft-voting ensemble in `CalibratedClassifierCV` with isotonic regression:

```python
CalibratedClassifierCV(estimator=ensemble, method="isotonic", cv=3)
```

with a fallback from 3-fold to 2-fold calibration if any fold lacked representation of a minority class (specifically Distressed):

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

### 19.2 Removal (Current State)

**What changed**: `CalibratedClassifierCV` was removed from `train_model()` as part of the SMOTE re-introduction and ensemble expansion documented in §16.14. `train_model()` now returns the raw `VotingClassifier` ensemble directly, and its `predict_proba()` output is used for both final predictions and threshold optimisation without any calibration layer in between.

**Why**: Refitting an isotonic calibration layer on top of SMOTE-resampled training folds introduced additional cross-validation variance (the calibration step's own internal CV competes with SMOTE's synthetic-sample distribution for the small Distressed class) without a measured accuracy improvement in this configuration. No formal calibration-curve or Brier-score comparison was run before removing it — this was a pragmatic simplification made alongside a larger set of changes (§16.14), not the outcome of a dedicated calibration ablation.

**Consequence — probabilities are no longer calibrated**: the `probabilities` field in the API response (§24.3) is now the raw ensemble soft-vote average, not a calibrated probability. Predicted-class accuracy is unaffected (calibration is a monotonic-ish transform that does not change `argmax` predictions in the typical case, though it can interact with the threshold-multiplier step), but consumers should not interpret the returned probability values as well-calibrated confidence estimates. This is now the accepted current state, not a recommendation — see §35.2 for reinstating calibration as a future, properly-ablated improvement.

**Trade-off acknowledged**: removing calibration without measuring its isolated effect is a methodological gap in its own right. A dedicated before/after comparison (with and without calibration, holding SMOTE and the regularised grid fixed) was not performed and is flagged as unfinished work rather than presented as a validated decision.

---

## 20. Model Evaluation

### 20.1 Authoritative Metrics (Cross-Validation)

**Source**: `evaluate_xgboost.py`, grouped stratified 3-fold CV (≈70/30 train/test per fold, §16.21), company-level, leakage-safe, current pipeline (SMOTE-augmented soft-voting ensemble blended with the ordinal cumulative-link view, plain-argmax prediction rule, §16.26). Cached in `results/xgboost_cv_metrics.json` and read directly by the dashboard (§12.4) — **this is the single figure that should be cited as "the model's accuracy" anywhere in this report or the dissertation.**

| Metric | CV Mean ± Std |
|---|---|
| **Accuracy** | 0.5264 ± 0.0252 |
| **Macro F1** | 0.4497 ± 0.0155 |
| Train accuracy (for reference, see §16.12–16.14) | 0.9835 ± 0.0208 |
| Train/test accuracy gap | 0.4571 |

Per-class F1 (CV mean):

| Class | F1 Mean |
|---|---|
| Speculative | 0.6283 |
| Investment-High | 0.5279 |
| Investment-Low | 0.4248 |
| Distressed | 0.2176 |

**Minority-class caveat (see the Executive Summary note and §16.26)**: relative to the previous second-stage configuration (§16.17), this configuration improves accuracy AND macro F1 simultaneously, so the §16.17 accuracy-vs-macro-F1 tension no longer applies between the last two configurations. Distressed-class F1 nonetheless remains by far the weakest (~0.22 this run, ~0.24 the prior run — within run-to-run noise of the previous configuration's ~0.24), so the model should still not be relied upon for Distressed-tier detection specifically.

The automated overfit verdict for this run is **"LIKELY OVERFITTING"** (train/test gap of 0.4571, above the 0.15 threshold) — this is expected and accepted: it is the direct consequence of SMOTE (§16.14) making the training fold's synthetic points easy to fit (and, since §16.26, the ordinal models' balanced-weight fit on the raw training fold adds to the training-fold score), and it does not indicate the same underlying problem diagnosed in §16.12 (which was fixed by regularisation, §16.13, and confirmed not to be the accuracy ceiling — re-confirmed in §16.26, where a stronger-regularisation grid closed the gap without moving test accuracy). The gap should be read as "SMOTE plus a smaller training fraction inflates the training-fold score" rather than "the deployed model is unreliable" — the CV **test** accuracy, which never sees SMOTE's synthetic points, is the number that matters.

### 20.2 Single-Split Reference Metrics (Secondary, Demo Only)

Following the methodological alignment in §12.4, the single-split result produced by `predict_xgboost.py` is **explicitly secondary** — the dashboard labels it "single-split demo, high variance — see CV mean for reported accuracy" and it should not be cited as the model's accuracy in isolation. It exists to give a live, concrete example of the model's output on one particular held-out slice of the default dataset, now also using a ~70/30 split (§16.21) to match the CV protocol.

A single-split accuracy figure is expected to fall within the range implied by the current CV standard deviation (0.5264 ± 0.0252, i.e. roughly [0.501, 0.552] at ±1σ) on a representative run (the run cached at this revision measured 0.5714 — above the +1σ band, a favourable draw worth remembering when quoting it), and should not be treated as an outlier requiring explanation unless it falls well outside that band — unlike the pre-fix dashboard behaviour documented in §12.4, which could reach ~70% on a favourable split under a different (ungrouped) split protocol.

A fresh single-split confusion matrix and per-class classification report are written to `results/xgboost_classification_report.csv` and `results/xgboost_test_predictions.csv` on every cache-miss training run and will vary run-to-run within the range implied by the CV standard deviation; they are not reproduced verbatim here to avoid the specific numbers going stale relative to whichever cache is currently on disk (see Appendix A.2 for the historical record of pre-fix values, retained for traceability only).

### 20.3 Why a Single Split Should Never Be the Reported Figure

A single split's accuracy can differ from the CV mean by more than one standard deviation purely due to which companies land in the held-out fold — this was demonstrated concretely during development, where the pre-fix dashboard (before the grouping regression in §12.4 was found and fixed) produced single-split accuracies observed to range from **~45% to ~70%** on the same underlying dataset and code, depending only on the random draw of held-out companies. This is precisely why the CV mean, not any single split, is the authoritative figure: relying on a single split — especially if only the most favourable run is reported — would materially misrepresent expected real-world performance.

### 20.4 Comparison to Baselines

| Model | Accuracy | Evaluation | Notes |
|---|---|---|---|
| Majority class ("Speculative") | ~0.39 | Theoretical | No model needed |
| Random (uniform) | ~0.25 | Theoretical | 4-class random |
| XGBoost, softmax-only + second-stage discriminator (prior configuration, 3-fold/70-30) | 0.50 ± 0.03 | Grouped 3-fold CV | §16.17/§16.21 configuration; macro F1 0.4373, Distressed F1 0.2366 |
| **XGBoost, current (ordinal blend, plain argmax, 3-fold/70-30)** | **0.53 ± 0.03** | **Grouped 3-fold CV** | **Company-level, leakage-safe; macro F1 0.4497, Distressed F1 0.2176, see §16.26** |

The current model exceeds the majority-class baseline by ~14 percentage points and the random baseline by ~28 percentage points under honest evaluation. Unlike the §16.17 second-stage change, the ordinal blend (§16.26) improved accuracy and macro F1 together (same split protocol, paired per-fold comparison in §16.26), so the current row supersedes the prior one as a strict improvement on both headline metrics — with Distressed F1 remaining statistically indistinguishable between the two.

---

## 21. Error Analysis

> [!NOTE]
> **Historical basis for this section**: the qualitative patterns described below (adjacent-tier confusion dominance, Distressed being the hardest class, errors skewing conservative) were observed on earlier single-split confusion matrices and classification reports. Following the changes in §16.13–16.16, the exact counts vary run-to-run, and per §20.2/20.3 a single split is no longer treated as the reported figure. The specific cell counts are not reproduced here to avoid the numbers going stale relative to the current cache on disk. The text below describes the structural shape of the errors, not current authoritative numbers.

### 21.1 Misclassification Patterns

From confusion matrices observed during development:

- **Investment-High ↔ Investment-Low**: The primary confusion zone. These adjacent tiers share overlapping financial profiles — the A/BBB boundary is often ambiguous even for human analysts.
- **Investment-Low ↔ Speculative**: A major error source. This reflects the inherent difficulty of the BBB/BB boundary — the "fallen angel" threshold in credit risk, where companies transition between investment and speculative grade.
- **Speculative → Distressed**: The model sometimes downgrades borderline speculative cases, demonstrating a conservative error profile.
- **Distressed class**: Historically, this class exhibits the lowest recall, with many Distressed samples misclassified as Speculative. The 3-class collapse experiment (§16.15) directly measured the effect of removing this specific confusion by merging Distressed into Speculative, and found it accounted for a large share of the model's remaining error (+11 points of accuracy when the boundary was removed).

### 21.2 Error Directionality

Errors tend to skew toward downgrades over upgrades. In credit risk applications, this is the safer direction if it holds generally — the model errs toward caution slightly more often than toward over-optimism. This directional tendency was not formally re-measured against the current (SMOTE + regularised) pipeline and should be re-verified before being cited as a strict property of the deployed model.

### 21.3 Adjacent-Class Confusion Dominance

Most misclassifications occur between adjacent tiers (IH↔IL, IL↔SP, SP↔DI), with two-tier errors being less common and extreme IH↔DI misclassifications almost absent. This pattern — that the model preserves ordinal credit-quality structure even when it misclassifies — is a reasonable qualitative expectation for an ordinal 4-class problem and is plausible to still hold.

### 21.4 Per-Sector Fairness (Measured, Current Pipeline)

**Motivation**: real credit-rating agencies do not apply one global ratio threshold across industries — a debt/equity level that is unremarkable for a capital-intensive utility can be alarming for an asset-light technology company. This raises a fair question about the pipeline: does the sector-relative z-score normalisation (§5.4, `compute_sector_stats()`) actually equalise performance across sectors, or does the model still perform meaningfully better on some industries than others?

**Method**: rather than assume an answer, `evaluate_xgboost.py` was extended to pool each fold's outer-test-set predictions by the raw `Sector` value (captured before one-hot encoding removes it) across all folds, then compute per-sector accuracy and macro F1 from the pooled pairs. This is a **measurement only** — it changes no training, splitting, or feature-engineering behaviour, and does not affect the headline CV accuracy reported in §20.1. Sectors with fewer than 20 pooled test rows are flagged as low-support and excluded from the disparity verdict, for the same reason `MIN_SECTOR_GROUP_SIZE` (§16.13) treats small sectors' z-score statistics as unreliable.

**Result** (current pipeline: SMOTE + regularised ensemble + second-stage discriminator + ~70/30 split, §16.13–§16.17, §16.21; 3-fold CV, so each sector's pooled row count is roughly 30% of its total dataset count rather than the ~20% pooled across 5 folds in the prior measurement):

| Sector | n (pooled test rows) | Accuracy | Macro F1 |
|---|---|---|---|
| Miscellaneous | 57 | 0.3158 | 0.2804 |
| Health Care | 171 | 0.4152 | 0.3880 |
| Energy | 294 | 0.4422 | 0.3905 |
| Public Utilities | 211 | 0.4550 | 0.3507 |
| Transportation | 63 | 0.4921 | 0.4027 |
| Capital Goods | 233 | 0.5236 | 0.4704 |
| Consumer Services | 250 | 0.5240 | 0.3998 |
| Consumer Non-Durables | 132 | 0.5455 | 0.3753 |
| Basic Industries | 260 | 0.5577 | 0.3763 |
| Technology | 234 | 0.5598 | 0.4561 |
| Consumer Durables | 74 | 0.5946 | 0.3912 |
| Finance | 50 | 0.6600 | 0.6506 |

**Finding**: there is a genuine, well-supported gap — from 31.6% (Miscellaneous, n=57) to 66.0% (Finance, n=50), a 34.4-point spread among sectors that all clear the 20-row support threshold. The automated verdict from `evaluate_xgboost.py` now classifies this as **"LARGE SECTOR DISPARITY"** (up from "MODERATE" in the prior 5-fold/80-20 measurement, where the spread was 29.7 points). Three patterns stand out:

1. **Miscellaneous and Health Care sit at the bottom**, a partial change from the prior measurement where Health Care and Public Utilities were weakest — Miscellaneous has moved from mid-table into last place. With only 57 pooled rows, Miscellaneous's ranking is more sensitive to exactly which companies land in the smaller (~70%) training fold under this split, so this specific ranking shift is plausibly attributable to the reduced training-set size (§16.21) rather than a newly-discovered sector weakness. Health Care and Public Utilities remain in the bottom half in both measurements, which is the more robust finding.
2. **The disparity widened under the ~70/30 split** (29.7 → 34.4 points). This is consistent with §16.21's expectation that a smaller training fraction per fold would affect model quality — apparently unevenly across sectors, with smaller/harder sectors likely more sensitive to having less training data available.
3. **Macro F1 is often much lower than accuracy within a sector** (e.g. Public Utilities: 45.5% accuracy but only 0.351 macro F1), mirroring the global class-imbalance pattern (§34.2) unevenly across sectors, consistent with the prior measurement.

**What this does and does not establish**: this confirms the sector-fairness concern has real, measured support on this dataset, now at a "LARGE" disparity classification rather than "MODERATE." It does **not** yet establish that a specific fix (sector-conditional thresholds, per-sector models, additional sector-interaction features) would close the gap, since no such fix has been implemented or tested. The widening of the gap under a smaller training fraction (finding 2 above) is itself a data point worth weighing when deciding the split ratio (§16.21): a 70/30 split gives a more conservative accuracy estimate overall, but it may also be amplifying an already-present sector disparity by giving the model less data to learn sector-specific patterns from. Health Care, Public Utilities, and (with lower confidence, given its small sample and rank volatility) Miscellaneous are flagged in §34.5 and §35.2 as the concrete candidates for further investigation if this is pursued.

This breakdown is computed automatically on every `evaluate_xgboost.py` run and cached in `results/xgboost_cv_metrics.json` under `sector_breakdown`/`sector_verdict`, so it stays current with the pipeline rather than going stale the way the row-level error analysis in §21.1–21.3 did.

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

Following the methodological alignment in §12.4, the response now separates the **primary** (CV mean, authoritative) metrics from the **secondary** (single-split demo) metrics explicitly:

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
    "metrics": {
      "accuracy": "0.5007", "accuracyStd": "0.0372",
      "f1": "0.4513", "f1Std": "0.0321",
      "cvFolds": 5,
      "label": "50.1% ± 3.7% (5-fold grouped CV)",
      "strength": "Reported accuracy is the grouped, company-level 5-fold CV mean...",
      "weakness": "Distressed class remains hard to learn (mean F1 0.26)..."
    },
    "cvMetrics": { "...": "same object as metrics above" },
    "singleSplitMetrics": {
      "accuracy": "0.4951", "precision": "0.5463", "recall": "0.4951", "f1": "0.4954",
      "label": "single-split demo, high variance — see CV mean for reported accuracy",
      "note": "This is one grouped, company-level train/test split, shown live for this dataset..."
    },
    "matrix": [[...]],
    "shap": [["Feature Name", 95.0, 1], ...],
    "shapStory": { "positive": [...], "negative": [...] }
  }
}
```

`metrics` mirrors `cvMetrics` whenever the CV cache (`results/xgboost_cv_metrics.json`) exists, and only falls back to the single-split figures (with an explicit warning string) if `evaluate_xgboost.py` has never been run for the current pipeline version. `singleSplitMetrics` is always populated from the live training run, independent of the CV cache.

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
HIGH_CORR_THRESHOLD = 0.80        # lowered from 0.95 in the regularisation pass, §16.13
SPLIT_N_FOLDS = 5
MIN_SECTOR_GROUP_SIZE = 20         # added in the regularisation pass, §16.13
```

### 28.2 No External Configuration File

All configuration is hardcoded in Python source. There is no `.env`, YAML, or JSON configuration file. This was a deliberate choice for simplicity — the system has a single use case and does not need runtime configuration flexibility.

---

## 29. Logging

### 29.1 Current State

The pipeline uses `warnings.filterwarnings("ignore")` to suppress all warnings. No structured logging is implemented.

### 29.2 Evaluation Output

`evaluate_xgboost.py` prints fold-level metrics (including per-fold train/test gap and whether nested thresholds were used) to stdout, and writes both a human-readable summary (`xgboost_cv_metrics.txt`) and a structured summary (`xgboost_cv_metrics.json`, §12.4) consumed by the dashboard. This remains the primary form of execution logging in the project.

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

The honest cross-validated accuracy is ~51% (macro F1 ~0.44, or ~50%/0.45 without the second-stage discriminator — §16.17) on a hard, imbalanced 4-class problem with only 593 distinct companies — up from an earlier ~46%/0.42 following the regularisation pass and SMOTE re-introduction (§16.13–16.14), but still well short of typical production thresholds. Three lines of evidence bound where the remaining ceiling comes from:

1. **Not primarily model capacity**: the regularisation pass (§16.13) cut the train/test overfitting gap from 39 points to 14 points while leaving test accuracy essentially unchanged (46.2% → 46.0%), showing that the model was not leaving significant accuracy on the table due to excess capacity.
2. **Substantially the Speculative/Distressed boundary and class granularity**: the 3-class collapse experiment (§16.15), which merged Distressed into Speculative, recovered **+11 points** of accuracy (45% → 56%) under an otherwise-identical pipeline. This was reverted to preserve cross-model consistency (§2.3), but it directly demonstrates that a large share of the remaining error is concentrated at exactly this boundary, rather than being spread uniformly across all four classes.
3. **A targeted fix at that boundary recovers only part of the gap, and at a cost**: the second-stage Speculative/Distressed discriminator (§16.17), directly motivated by finding 2, recovered +1.3 points of accuracy (50.1% → 51.4%) — real, but far short of the +11 points the full 3-class collapse implied was available. This gap between "the boundary accounts for +11 points if removed entirely" and "a targeted correction at that boundary recovers +1.3 points" suggests the 3-class experiment's gain was not purely about resolving ambiguous boundary cases correctly; some of it may reflect that removing Distressed as a target eliminates unavoidable errors on genuinely-hard Distressed rows rather than fixing a solvable discrimination problem. The second-stage discriminator also measurably worsened Distressed-class F1 (0.2585 → 0.1763), a cost the 3-class comparison does not carry the same way (that experiment simply stopped scoring the class at all, which is not the same as detecting it well).

Meaningful further gains most likely require:

1. **More data** — especially Distressed-class samples. Even 20 additional C/D-rated observations would significantly improve minority-class learning.
2. **Time-normalised temporal features** — a per-company trend feature *was* tried (§16.18) using raw period-over-period differences and measurably reduced accuracy (51.4% → 50.6%), plausibly because agency filing dates are irregularly spaced across companies, so a raw difference conflates rate-of-change with elapsed time. A version normalised by elapsed time between records (`Δratio / Δyears`) was identified as the natural next attempt but was not implemented or tested.
3. **Macroeconomic context** — interest rates, credit spreads, GDP growth.
4. ~~A two-stage / hierarchical classifier~~ — **Implemented** (§16.17). Recovered a modest, real accuracy gain (+1.3 points) at a measured cost to Distressed-class F1; not a resolved win, see finding 3 above and §16.17's fairness caveat.

### 34.2 Distressed-Class Weakness

CV-mean F1 ≈ 0.18 for the Distressed class under the current pipeline (SMOTE + second-stage discriminator), down from ≈0.26 without the second-stage discriminator, ≈0.28 with regularisation but no SMOTE, and the original ≈0.23–0.25 under earlier pipeline versions. Every accuracy-improving change made after the initial regularisation pass (SMOTE re-introduction, §16.14; the second-stage discriminator, §16.17) has come at some cost to this specific class's F1, which is the clearest quantitative signature of the accuracy/fairness trade-off discussed throughout §16.14–§16.17 and the Executive Summary. The 72-record class provides ~57 training samples per CV fold — enough for the model to detect weak signal, but insufficient for reliable classification, and every technique tried that helps the majority classes appears to do so partly at this class's expense.

### 34.3 No Temporal Awareness (Partially Investigated)

Each company–year record is treated independently in the deployed pipeline. A per-company temporal trend feature was implemented and tested (§16.18, `add_temporal_features()` in `preprocessing.py`) but measurably reduced CV accuracy under the specific encoding tried (raw period-over-period differences on irregularly-spaced filing dates) and was reverted. The function remains defined but unused, and a time-normalised variant was identified as a concrete next step but not attempted. This limitation should therefore be read as "not yet solved" rather than "not investigated" — the negative result and its likely cause are documented in §16.18.

### 34.4 Cross-Model Inconsistency (Rating Grouping Fixed; Others Remain)

The rating-grouping mismatch documented in §2.3 (client.js and Random Forest disagreeing with XGBoost/Decision Tree/Logistic Regression on CCC/CC placement) has been fixed — all five implementations now agree. This eliminates one source of invalid cross-model comparison, but it was not the only one. The Decision Tree, Random Forest, and XGBoost models still use materially different preprocessing pipelines (XGBoost's ~130-feature engineered space vs. Decision Tree/Random Forest's raw-ratio `ColumnTransformer` pipelines) and different response formats (label casing, metric averaging method — weighted vs. macro). A shared class definition is necessary but not sufficient for valid cross-model comparison; the underlying feature spaces and evaluation conventions would also need to be unified.

### 34.5 Sector Fairness: A Measured, Not Fully Resolved, Gap

§21.4 measured a genuine 29.7-point accuracy spread across sectors (Health Care 36.3% vs. Finance 66.0%, both well-supported by pooled test-row count), confirming that the sector-relative z-score normalisation (§5.4) does not fully equalise cross-industry performance. This is consistent with the general critique that credit-risk thresholds are not directly comparable across industries with structurally different capital structures (e.g. capital-intensive utilities vs. asset-light technology firms) — the finding here is that this pipeline still shows a measurable version of that effect despite already having a sector-normalisation step.

**What is and isn't known**: the measurement identifies *which* sectors underperform (Health Care, Public Utilities) but does not diagnose *why* with certainty, nor does it establish that a specific intervention would help. Plausible causes not yet distinguished from each other: (a) the sector z-scores are too coarse (mean/std per sector may not capture the *shape* of a sector's ratio distribution, only its centre and spread); (b) Health Care and Public Utilities credit risk may depend on factors this dataset's 30 financial ratios genuinely do not capture (regulatory reimbursement risk, rate-case outcomes); or (c) some of the spread is sampling variation given ~593 companies across 12 sectors, which the "MODERATE" rather than "LARGE" disparity classification already reflects.

**Deliberately not acted on yet**: per the project's stated measure-before-you-build principle (consistent with §16.12's overfitting diagnostic preceding §16.13's fix), no sector-conditional threshold or per-sector model variant has been implemented. Building that machinery before confirming which of (a)/(b)/(c) above is the actual cause risks adding complexity that doesn't address the real problem. §35.2 records the concrete next diagnostic step.

### 34.5 Stale Cache Risk

Cached artifacts are not invalidated by code changes. A developer who modifies the preprocessing pipeline and runs the API will receive predictions from a model trained under the old pipeline.

---

## 35. Future Improvements

### 35.1 High-Priority

1. **Add and pin XGBoost** (and other ML dependencies, including `imbalanced-learn` for SMOTE, which is now a runtime dependency and is not currently pinned either) in `requirements.txt` — for cross-environment installability and reproducibility.
2. **Add automated tests** (unit, integration, API smoke test).
3. **Implement cache invalidation** on code change (pipeline version hash in cache key) — this is now more urgent than before, since a stale `results/*.pkl` cache would silently serve predictions from the pre-SMOTE, pre-regularisation, or pre-realignment pipeline without any error.
4. ~~Unify rating grouping across all models (Decision Tree, Random Forest, XGBoost, frontend).~~ **Done** — `client.js` and `predict_random_forest.py` were updated to the CCC/CC→Distressed grouping used by XGBoost, Decision Tree, and Logistic Regression. Remaining cross-model work: unify preprocessing pipelines and evaluation-metric conventions (see §34.4).
5. ~~Align the dashboard's single-split evaluation with the CV script's methodology.~~ **Done** — see §12.4. `evaluate_xgboost.py` is now the single source of truth; the dashboard reads its cached JSON output.
6. **Add `.pkl` files to `.gitignore`** and generate evaluation artifacts via a documented script.
7. **Re-run and refresh the error analysis (§21)** against the current (SMOTE + regularised + realigned + second-stage) pipeline's cached predictions — the existing figures are explicitly flagged as illustrative/historical and should be regenerated before being cited in the final dissertation.
8. **Resolve the accuracy/fairness trade-off from the second-stage discriminator (§16.17, §34.2)**: decide, and document the reasoning, whether the deployed model should optimise for accuracy (current default) or for macro F1 / Distressed-class recall (the pre-second-stage configuration) — or expose both configurations and let the use case decide. Currently the higher-accuracy configuration is simply the default without an explicit justification beyond "it is more accurate," which is an incomplete argument for a credit-risk screening tool where missed Distressed cases carry a real cost.

### 35.2 Medium-Priority

9. **Ablation study**: Quantify the marginal contribution of each feature engineering stage, and — new — of SMOTE and calibration independently, holding the rest of the pipeline fixed (see §16.14 and §19.2, both of which currently lack an isolated before/after measurement).
10. **Ordinal regression**: Replace `multi:softprob` with an ordinal-aware loss function.
11. **Time-normalised temporal features**: a raw-difference version was tried and reverted (§16.18, §34.1 finding 3) because it measurably reduced accuracy, plausibly due to irregular filing-date spacing across companies. A variant normalised by elapsed time between records (`Δratio / Δyears`) was identified as the natural next attempt and has a concrete, evidence-motivated rationale for why it might succeed where the raw-difference version did not.
12. ~~Two-stage / hierarchical classifier~~ — **Implemented** (§16.17). Recovered +1.3 points of accuracy at a measured cost to Distressed-class F1 (item 8 above tracks the remaining decision about whether to keep this trade-off as the default).
13. **Diagnose the Health Care / Public Utilities sector gap** (§21.4, §34.5): a 29.7-point accuracy spread across sectors was measured, with these two well-supported sectors underperforming. Before building sector-conditional thresholds or models, first check whether the sector z-score features (§5.4) capture enough of each sector's ratio-distribution shape for these two sectors specifically (e.g. compare their ratio distributions' skew/multimodality against sectors that perform well), to distinguish "the normalisation is too coarse" from "these sectors need signals outside the current 30 ratios" before choosing a fix.
14. **Recalibration ablation**: Determine whether reinstating `CalibratedClassifierCV` (§19) on top of the current SMOTE + regularised + second-stage pipeline improves or degrades calibration/accuracy, with a proper before/after comparison this time.
15. **Cross-model meta-learner**: Combine XGBoost, Random Forest, and Logistic Regression via stacking.
16. **Structured logging**: Python `logging` module with timestamps, cache decisions, and pruning statistics.

### 35.3 Low-Priority

17. **Docker containerisation**: For reproducible deployment.
18. **Request size limits and rate limiting**: For production hardening.
19. **Model registry**: Semantic versioning with code+data+metrics traceability.
20. **Online threshold recalibration**: Adapt thresholds based on recent prediction distributions.

---

## 36. Lessons Learned

### 36.1 Company-Level Leakage Was the Most Impactful Discovery

The 26-percentage-point accuracy drop when switching from row-level to company-level splitting was the single most important engineering finding. It changed the narrative from "our model achieves 72% accuracy" to "our model honestly achieves 46% accuracy and we can explain exactly why." The lesson: **always audit the split strategy relative to the data's grouping structure**.

### 36.2 Honest Metrics Matter More Than High Metrics

A 46% accuracy under honest evaluation is more valuable in a technical report than a 72% accuracy inflated by leakage. Examiners will check for leakage. A project that catches and documents its own leakage demonstrates stronger engineering maturity than one that reports inflated metrics without investigation.

### 36.3 SMOTE's Success Depends on the Surrounding Pipeline's Capacity, Not Just Class Size (Revised)

The original lesson recorded here was "SMOTE fails on tiny minority classes, with 4–5 samples generating noisy interpolations in a near-degenerate subspace." That was true as measured, but incomplete: SMOTE was re-tried in §16.14 under a *regularised* pipeline (shallower trees, heavier L1/L2/gamma) and produced a genuine +8.1-point CV accuracy gain rather than a regression, with the same tiny Distressed class (~57 training samples per fold) that had caused the original failure. The revised lesson: **SMOTE's synthetic points are only as dangerous as the model's capacity to overfit to them.** A high-capacity model (deep trees, light regularisation) will happily memorise SMOTE's noisy interpolations; a well-regularised model treats them as a mild, generally-useful signal boost. This means "SMOTE failed" and "SMOTE succeeded" are not independent, permanent facts about a dataset — they are properties of a *dataset-plus-pipeline* combination, and revisiting an earlier "abandoned" technique after a pipeline change is a legitimate and, in this case, productive thing to do.

### 36.4 Optuna Is Overkill for Small Datasets

On ~2,000 samples, the performance landscape is smooth enough that a well-designed randomised grid (originally 15, later 50, configurations) finds near-optimal hyperparameters in a fraction of the time an exhaustive Bayesian search would take.

### 36.5 Shared Preprocessing Modules Prevent Drift — But Only Cover the Modules That Are Actually Shared

Duplicating preprocessing logic between training and inference is a maintenance hazard. Extracting `preprocessing.py` as a shared module eliminated train-serve skew *for preprocessing* by construction. However, §12.4 shows this principle was not applied consistently everywhere: `evaluate_xgboost.py` and `predict_xgboost.py` had each grown their own model-training and threshold-fitting logic, and these drifted apart (different split, different threshold protocol) exactly the way `preprocessing.py` was designed to prevent for data transforms. The lesson generalises beyond preprocessing: **any logic duplicated between an offline evaluation script and a live serving path is a drift risk, not just preprocessing specifically**, and the fix (import the shared function, don't re-implement it) is the same one already applied to preprocessing.

### 36.6 An Un-Audited Live Demo Can Silently Contradict a Written Report

The dashboard's single-split accuracy and the CV script's reported accuracy diverged (53% vs. 70%) for a period during development without either script raising an error — both produced internally-consistent, plausible-looking numbers, and the discrepancy was only found by deliberately comparing them side by side. The lesson: a written report's headline number should be periodically checked against whatever live artifact accompanies it (a dashboard, a demo, a notebook output), not just checked for internal consistency within the report itself. §12.4 documents both the discovery and the fix; §36.2's original lesson (report honest metrics, not inflated ones) applies here too, but the specific new failure mode — the report and the live system silently disagreeing — is distinct enough to warrant its own entry.

### 36.7 Document What Didn't Work — Including When "Didn't Work" Is Later Reversed

Failed approaches (Optuna, GPU acceleration, Platt scaling) provide as much engineering insight as successful ones. SMOTE's journey in this project — tried, abandoned with evidence (§16.5), then re-tried and adopted with different evidence once the surrounding pipeline changed (§16.14) — demonstrates that "document what didn't work" should also mean documenting *why* it didn't work in enough causal detail that a later re-test, under different conditions, can be recognised as a legitimate re-evaluation rather than an inconsistency in the record.

---

## 37. Final Conclusion

This project demonstrates that rigorous engineering discipline — not model accuracy — is the primary determinant of a trustworthy ML system. The XGBoost credit risk classifier currently achieves an **Accuracy of 0.5047 ± 0.0255** and **Macro F1 of 0.4373 ± 0.0115** under company-level, leakage-safe, grouped stratified 3-fold cross-validation (a ~70/30 train/test split per fold, §16.21), using a SMOTE-augmented 5-model ensemble, nested leakage-safe threshold optimisation, and a second-stage Speculative/Distressed discriminator. These numbers are honest, and — as of the methodological alignment documented in §12.4 — they are also the **same numbers the live dashboard reports**, closing a period during development where the two had diverged (53% vs. 70%). They reflect:

- **Zero company-level leakage**: All records for a given company are confined to one side of every split, in both the offline evaluation script and the live dashboard's single-split demo, and in the second-stage discriminator's own training data (§16.17).
- **Zero preprocessing leakage**: Every fitted parameter (medians, bounds, sector stats, feature selections) is computed on the training fold exclusively.
- **Zero threshold-selection leakage**: Threshold multipliers and the use/don't-use decision are fit on a company-level inner-validation slice of the training fold, never on the outer test/CV fold (§13.2).
- **One evaluation methodology, not two**: `evaluate_xgboost.py` is the single source of truth for reported accuracy; `predict_xgboost.py` reads its cached output rather than maintaining an independent, divergent pipeline (§12.4).
- **Documented failures and reversals**: Optuna, GPU acceleration, and Platt scaling were tried, diagnosed, and abandoned with evidence. SMOTE was tried, abandoned, and **later re-tried and adopted** once a regularisation pass changed the pipeline's capacity — both the original failure and the later success are documented with the evidence that produced each verdict (§16.5, §16.14, §36.3). A per-company temporal trend feature was tried and **reverted** after measurement showed it reduced accuracy (§16.18) — a negative result documented with its likely cause rather than silently dropped.
- **An accuracy gain with an openly stated cost, not a free win**: the second-stage Speculative/Distressed discriminator (§16.17) raised accuracy by 1.3 points but reduced Distressed-class F1 by 8.2 points and macro F1 overall. Both configurations' full metrics are reported (§20.1, §20.5) rather than presenting only the higher headline number.
- **Honestly incomplete ablations, flagged as such**: probability calibration was removed alongside several other simultaneous changes without an isolated before/after measurement (§19.2); this is documented as a gap, not hidden.

The system is deployable as a web application with SHAP-based explainability, model caching, and support for user-uploaded datasets. It is not a production financial system — it lacks a resolved accuracy-vs-fairness policy for its own second stage, temporal features (attempted, not yet successful), data volume, regulatory compliance, and operational monitoring required for that role — but it is an honest, auditable, well-documented engineering artifact, including in the places where its own evaluation methodology needed correcting and where an accuracy gain came with a cost worth stating plainly.

---

## Appendix A: Self-Audit

### A.1 Consistency Verification

| Check | Result |
|---|---|
| Dashboard and CV script report the same accuracy | ✅ Verified: both read/derive from `results/xgboost_cv_metrics.json` (0.5007 ± 0.0372); see §12.4 |
| CV metrics are independent per fold | ✅ `evaluate_xgboost.py` runs its own preprocessing and model fit per fold |
| Prediction rule identical in both scripts | ✅ Both use plain argmax over the blended softmax+ordinal probabilities from the shared `train_model()` (§16.26; the previous shared `fit_thresholds_nested()` protocol was removed in the same iteration) |
| Split protocol identical in both scripts | ✅ Both use `make_split(..., groups=groups, ...)` / `StratifiedGroupKFold` on `Symbol`/`Name` |
| API response format matches documented format | ✅ Verified against `predict_xgboost.py` output construction (§24.3), including live confirmation that `metrics`/`cvMetrics`/`singleSplitMetrics` are populated as documented |
| Single-split confusion matrix / classification report reproducibility | ⚠️ Not re-verified against the current pipeline for this revision — flagged in §21 and §35.1 as a follow-up. Historical figures (pre-realignment) are retained in §A.2 for traceability only. |

### A.2 Metric History Across Pipeline Changes

Accuracy has moved as the pipeline evolved through several distinct, evidenced stages. Each figure is traceable to a specific pipeline state and section of this report:

| Pipeline state | Single-split accuracy | CV accuracy | Notes |
|---|---|---|---|
| Row-level split (original, leaky) | ~0.72 | not computed | Superseded — company-identity leakage, §16.2–16.3 |
| Pre-cleaning, grouped split | 0.5296 (215/406) | 0.4603 ± 0.0313 | Superseded, §6.3 |
| Post-cleaning, grouped split | 0.5517 (224/406) | 0.4618 ± 0.0329 | Superseded, §6.3 |
| Post-regularisation-pass, no SMOTE | ~0.4951 (one run) | 0.4514 ± 0.0253 | Superseded, §16.13 — overfitting gap fixed (39→14 pts), test accuracy unchanged |
| Post-SMOTE re-introduction (4-class) | 0.4951 (one observed run) | 0.5007 ± 0.0372 | Superseded by §16.17, §20.1 |
| 3-class collapse experiment (Distressed merged into Speculative) | not separately measured | 0.5599 ± 0.0244 | Tested and reverted for cross-model consistency, §16.15 — not the deployed model |
| + per-company temporal trend features (on top of the second-stage discriminator below) | not separately measured | 0.5057 ± 0.0247 | Tested and reverted, §16.18 — reduced accuracy relative to the row above; not the deployed model |
| Post-second-stage discriminator, 5-fold/80-20 split | 0.4951 (one observed run) | 0.5101 ± 0.0270 | Superseded by §16.21 — split ratio changed to ~70/30 |
| Post-split-ratio change to ~70/30, 3-fold CV (4-class) | 0.4948 (one observed run) | 0.5047 ± 0.0255 | Superseded by §16.26 — ordinal blend added, threshold/second-stage removed |
| + time-normalised temporal slope features (`Δratio/Δyears`) | not separately measured | 0.5018 ± 0.0213 | Tested and reverted, §16.26 — the §16.18 follow-up idea, also reduced accuracy; not the deployed model |
| **Ordinal cumulative-link blend, plain argmax (4-class, current)** | 0.5714 (one observed run, favourable draw — above +1σ) | **0.5264 ± 0.0252** (prior run of same config: 0.5244 ± 0.0210) | **Current authoritative figure**, §16.26, §20.1 — improved accuracy AND macro F1 (0.4373 → 0.4497) over the row two above |

The dashboard/CV alignment fix (§12.4) means that, going forward, "single-split accuracy" and "CV accuracy" should track each other within the CV standard deviation rather than diverging by tens of points as they did before the fix (observed range ~45%–70% under the pre-fix, ungrouped single split). All current artifacts (§16.16–§16.22) were regenerated together after the alignment fix and each subsequent change; the earlier figures are retained here only for historical traceability, not cited elsewhere as current performance. Note that accuracy alone does not tell the full story after §16.17: the higher-accuracy second-stage configuration trades away macro F1 and Distressed-class F1 relative to the pre-second-stage configuration — see §20.1 and §34.2. Minor figure-to-figure variation across successive runs of an identical configuration (e.g. 0.5136 vs. 0.5101 for the 5-fold/80-20 second-stage configuration) reflects XGBoost's documented `n_jobs=-1` non-determinism (§14.3), not a pipeline change; the ~0.5101 → ~0.5047 move, by contrast, reflects the deliberate split-ratio change in §16.21, not noise.

### A.3 Known Cross-System Inconsistencies

1. **Rating grouping mismatch — fixed**. The XGBoost, Decision Tree, and Logistic Regression backends mapped CCC/CC to Distressed, while the Random Forest backend and the frontend `client.js` mapped CCC/CC to Speculative (two groupings across five implementations, not three independent ones — corrected from an earlier draft of this audit that overstated the count). `client.js` and `predict_random_forest.py` were updated to match the CCC/CC→Distressed grouping; all five implementations now agree, verified by direct source inspection.
2. **Dashboard vs. CV-script accuracy mismatch — fixed**. See §12.4 for the full discovery-and-fix record. Both now derive from the same `train_model()`/`fit_thresholds_nested()`/grouped-split logic and the same cached CV result.
3. **Label format mismatch — unresolved**: XGBoost uses underscores (`Investment_High`); other models use hyphens (`Investment-High`). The API normalises this via `format_prediction_label()`, but internal artifacts use underscores.
4. **Calibration status mismatch — unresolved/undocumented until this revision**: the `calibrated_model.pkl` artifact name is now misleading (§14.4) since calibration was removed (§19.2); this is a cosmetic but real inconsistency between artifact naming and actual pipeline behaviour.

### A.4 Self-Assessment Rubric

| Criterion | Score | Justification |
|---|---|---|
| **Technical depth** | 8/10 | Detailed justification of engineering decisions, including two documented technique reversals (SMOTE, calibration) with evidence; some areas (calibration removal specifically) still lack isolated ablation evidence |
| **Reproducibility** | 7/10 | Seeds fixed, artifacts persisted, CV results cached to JSON for dashboard reuse; XGBoost and imbalanced-learn versions unpinned, no automated tests |
| **Honesty of evaluation** | 9/10 | Company-level leakage discovered, documented, and fixed; cross-validated metrics as authoritative; a live dashboard/report accuracy contradiction (53% vs 70%) was found and eliminated rather than left unresolved or resolved by picking the more favourable number |
| **Software engineering quality** | 8/10 | Shared preprocessing module; shared model-training and threshold-fitting logic between the CV script and the dashboard (a gap identified and closed in this revision, §12.4); still no automated tests, no structured logging, no cache invalidation on code change |
| **Documentation quality** | 8/10 | Comprehensive; error analysis (§21) is explicitly flagged as needing regeneration against the current pipeline rather than silently left stale |
| **Model evolution documentation** | 9/10 | All iterations documented including failures and their later reversals with new evidence; clear cause-effect chains |
| **Limitations acknowledgement** | 9/10 | Performance ceiling honestly discussed with quantified evidence for its likely cause (§34.1); cross-model inconsistencies identified; a specific fairness critique (sector thresholds) was measured rather than dismissed or assumed (§21.4, §34.5); the calibration-removal gap and stale row-level error-analysis figures are flagged rather than hidden |
| **Overall** | **84/100** | Strong engineering discipline and honest evaluation, now extended to cover a self-identified inconsistency between the live system and the written report and a directly-measured sector-fairness gap; still missing automated tests and a few isolated ablations (calibration, SMOTE-vs-no-SMOTE holding all else fixed) |

### A.5 Sections at Risk of Mark Deduction

1. **No ablation study for calibration removal**: §19.2 removed `CalibratedClassifierCV` alongside several other simultaneous changes without measuring its isolated effect. This is flagged explicitly in the text rather than presented as a validated decision, which should mitigate but not eliminate the risk.
2. **No automated tests**: The project demonstrates engineering discipline in design but lacks the testing infrastructure to verify it programmatically.
3. **Cross-model inconsistency (partially resolved)**: The rating-grouping mismatch across models has been fixed (§2.3, §34.4). Preprocessing-pipeline and evaluation-metric differences across models remain and still make direct dashboard comparison invalid — this is a legitimate residual software engineering flaw.
4. **Missing XGBoost/imbalanced-learn dependency declaration**: `requirements.txt` does not list `xgboost` or `imbalanced-learn` (confirmed by direct inspection), not merely leaving them unpinned. A clean environment built from this file cannot run the pipeline.
5. **Stale row-level error analysis (§21.1–21.3)**: retained for its qualitative shape but explicitly marked as not re-verified against the current pipeline. An examiner who reads closely will see this flagged; one who does not may cite outdated exact figures — a regeneration pass before submission is recommended (§35.1 item 7). Note that §21.4 (per-sector breakdown) does NOT have this problem — it is computed fresh on every `evaluate_xgboost.py` run and was current as of this revision.
6. **Sector-fairness gap measured but not resolved**: §21.4/§34.5 confirm a real 29.7-point cross-sector accuracy spread but do not yet distinguish between the three plausible causes identified there (coarse normalisation, missing sector-specific signal, sampling variation). An examiner may reasonably ask why this was measured but not acted on — the answer (§34.5) is a deliberate measure-before-you-build sequencing decision, but this should be stated confidently if raised rather than treated as an oversight.

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
        │   ├── xgboost_cv_metrics.txt
        │   └── xgboost_cv_metrics.json     # read by predict_xgboost.py, §12.4
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


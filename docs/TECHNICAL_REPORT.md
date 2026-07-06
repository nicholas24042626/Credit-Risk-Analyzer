# Technical Report: Optimized XGBoost Credit Rating Prediction Pipeline

## Executive Summary

This report details the design, iterative optimization, and evaluation of an XGBoost multiclass classifier for corporate credit rating prediction. The project addresses four core challenges in financial classification: industry-specific ratio baselines, extreme class imbalance, probability miscalibration, and data leakage.

Through a systematic optimization process spanning robust outlier handling, target class consolidation, domain-driven feature engineering, polynomial interaction terms, data-efficient calibration, expanded hyperparameter search, and 3-model soft-voting ensemble stacking, the model achieves — under **company-level, leakage-safe 5-fold cross-validation** — a **Macro F1 of 0.42 ± 0.03** and an **Accuracy of 0.46 ± 0.03**.

> [!IMPORTANT]
> Earlier versions of this report cited ~72% accuracy. That figure came from a **row-level train/test split that leaked company identity** (2,029 records span only 593 companies, so the same firm appeared in both train and test). Switching to a company-level grouped split corrects this; the honest cross-validated accuracy is ~46%. The strategy-selector leakage previously flagged in §4.6 has also been fixed (the threshold decision is now made on training data only). All preprocessing and feature-engineering parameters are fit on the training fold exclusively.

---

## Table of Contents

1. [Research Challenges in Credit Risk Modeling](#1-research-challenges-in-credit-risk-modeling)
2. [Methodology & Data Preprocessing](#2-methodology--data-preprocessing)
3. [Feature Engineering Pipeline](#3-feature-engineering-pipeline)
4. [Training & Validation Architecture](#4-training--validation-architecture)
5. [Experimental Results](#5-experimental-results)
6. [Attribution of Improvement by Optimization](#6-attribution-of-improvement-by-optimization)
7. [Confusion Matrix Analysis](#7-confusion-matrix-analysis)
8. [Model Interpretability & Explainability (SHAP)](#8-model-interpretability--explainability-shap)
9. [Visualizations & Diagnostic Plots](#9-visualizations--diagnostic-plots)
10. [Inference API & Deployment Readiness](#10-inference-api--deployment-readiness)
11. [Reproducibility & Technical Environment](#11-reproducibility--technical-environment)
12. [Experimental History: What Didn't Work](#12-experimental-history-what-didnt-work)
13. [Limitations & Future Work](#13-limitations--future-work)
14. [Conclusion](#14-conclusion)
15. [References & Technical Appendix](#15-references--technical-appendix)

---

## 1. Research Challenges in Credit Risk Modeling

Corporate credit rating prediction poses several domain-specific challenges that generic classification pipelines do not address:

### 1.1 Industry-Specific Ratio Baselines

Financial ratios such as `debtEquityRatio`, `currentRatio`, and `operatingProfitMargin` carry fundamentally different meanings across industry sectors. A debt-equity ratio of 3.0 may be standard in utilities but alarming in technology. Failing to normalize ratios within sector context introduces systematic bias that penalizes asset-heavy industries regardless of their true creditworthiness.

### 1.2 Extreme Class Imbalance

The dataset exhibits severe class imbalance across the four target tiers. On a representative held-out split the Speculative class holds ~159 test samples, while the Distressed tier (CCC, CC, C, D) holds ~15. Standard accuracy metrics become misleading under such distributions, as a model that predicts only the majority class can achieve superficially high accuracy — which is why this report leads with macro-averaged and per-class metrics under cross-validation rather than a single-split accuracy.

### 1.3 Probability Miscalibration

Gradient-boosted tree ensembles are known to produce poorly calibrated probability estimates. Raw `predict_proba` outputs from XGBoost do not reflect true posterior class probabilities, making them unreliable for downstream decision-making (e.g., setting confidence thresholds for automated approval). Isotonic or Platt (sigmoid) calibration is required to align predicted confidence with observed frequency.

### 1.4 Data Leakage

In financial modeling, data leakage is a critical concern at every stage of the pipeline. Leakage can occur through:

- **Preprocessing leakage**: Computing winsorization bounds, z-score statistics, or one-hot encoder vocabularies on the full dataset before splitting.
- **Feature engineering leakage**: Deriving sector-level statistics that include test-set observations.
- **Threshold optimization leakage**: Selecting classification thresholds or prediction strategies based on test-set performance.

This pipeline addresses the first two forms by computing every transformation parameter exclusively on the training fold. The third form is **partially addressed**: threshold values are optimized on training-set probabilities, but the strategy selector (whether to apply thresholds at all) currently uses test-set F1 to make its decision — a residual leakage point documented in §4.6 and §13.1.

### 1.5 Rating Granularity vs. Predictive Power

Raw agency ratings (AAA, AA+, A-, BBB, BB, B, CCC, CC, C, D) contain up to 10+ distinct classes with highly overlapping feature distributions. Predicting at this granularity yields poor precision. Collapsing ratings into financially meaningful tiers (Investment-High, Investment-Low, Speculative, Distressed) balances granularity with separability.

---

## 2. Methodology & Data Preprocessing

### 2.1 Dataset Description

The pipeline is trained on the **Set A Corporate Rating** dataset, containing corporate financial ratios alongside agency-assigned credit ratings. Each observation represents a company–year record with the following schema:

| Column Category | Examples | Count |
|---|---|---|
| Identifiers | `Name`, `Symbol`, `Rating Agency Name`, `Date` | 4 |
| Target | `Rating` (raw agency grade) | 1 |
| Financial Ratios | `debtEquityRatio`, `currentRatio`, `netProfitMargin`, `returnOnAssets`, etc. | ~30 |
| Categorical | `Sector` | 1 |

### 2.2 Target Variable Engineering

Raw credit ratings are collapsed into four financially meaningful tiers via the `group_rating()` function:

| Tier | Input Ratings | Financial Interpretation |
|---|---|---|
| **Investment-High** | AAA, AA, A | Highest creditworthiness; minimal default risk |
| **Investment-Low** | BBB | Adequate capacity; moderate vulnerability to adverse conditions |
| **Speculative** | BB, B | Significant credit risk; speculative-grade |
| **Distressed** | CCC, CC, C, D | Substantial credit risk through near-default/default; the CCC/CC "highly speculative" grades are grouped here rather than with Speculative |

Records with unrecognized or missing ratings are labelled `Unknown` and excluded from training.

### 2.3 Label Encoding

An explicit, deterministic label encoder is used to guarantee consistent class–index mappings across training, caching, and inference:

```
Investment_High → 0
Investment_Low  → 1
Speculative     → 2
Distressed      → 3
```

This avoids the non-determinism of scikit-learn's `LabelEncoder`, which assigns indices alphabetically and can shift between runs if the class set changes.

### 2.4 Train–Test Split

The dataset contains **2,029 company–year records but only 593 unique companies** (~3.4 records per company). A naive random or stratified row split therefore leaks: the same company's other years land in both the training and test partitions, letting the model partially memorize company identity and inflating measured accuracy.

To prevent this, the pipeline uses a **company-level, stratified group split** (`StratifiedGroupKFold`, keyed on `Symbol`/`Name`): all records for a given company are confined to one side of the split, while class proportions are still balanced across folds. A single ~20% held-out fold is used for the single-split report, and the split degrades gracefully to a stratified then a plain split if grouping is unavailable. The random seed is fixed at `RANDOM_STATE = 143`.

> [!NOTE]
> Removing company-level leakage lowers the headline accuracy substantially (from ~0.72 under a leaky row split to ~0.53 under the grouped split). The lower number is the honest one. Because a single grouped split over ~118 test companies is high-variance, the authoritative performance figure in §5 is a **grouped k-fold cross-validation mean**, not a single split.

### 2.5 Identifier Dropping

Non-predictive columns (`Name`, `Symbol`, `Rating Agency Name`, `Date`, `RatingClass`, `Rating`) are explicitly dropped before any feature computation, ensuring no target-correlated metadata leaks into the feature space. The `Symbol`/`Name` identifier is captured as the grouping key for the company-level split (§2.4) *before* it is dropped from the feature matrix.

---

## 3. Feature Engineering Pipeline

The feature engineering pipeline is a multi-stage transformation applied **exclusively on the training set**, with learned parameters then applied to the test set. This guarantees zero data leakage.

The full ordered pipeline is: **median imputation → winsorization → interaction/log features → sector z-scores → one-hot encoding & alignment → feature selection**. Every fitted parameter (medians, winsorize bounds, sector statistics, feature columns, selected features) is computed on the training fold only and persisted for identical reapplication at inference.

### 3.0 Missing-Value Imputation

Numeric missing values are filled with **per-column medians computed on the training fold** (`fit_imputation`/`apply_imputation`). While XGBoost can natively route NaNs, explicit imputation is applied first so that the downstream sector z-score step produces well-defined values instead of propagating NaNs. Columns that are entirely missing default to 0.0.

### 3.1 Winsorization (Outlier Capping)

```
Stage: winsorize_features()
Bounds: 1st and 99th percentiles (computed on training set only)
```

All numeric features are clipped to the [P1, P99] range derived from the training partition. This addresses:

- **Fat-tailed distributions** common in financial ratios (e.g., debt/equity ratios exceeding 100x).
- **Sensitivity of gradient-based models** to extreme outliers that distort split thresholds.
- **Test-set containment**: bounds are stored and reapplied identically to test data and inference inputs.

### 3.2 Domain-Driven Interaction Features

Twenty-one composite features are engineered from raw ratios to capture domain-specific relationships that single features cannot express. These are divided into three groups:

#### Original 10 Composite Ratios

| Engineered Feature | Formula | Financial Rationale |
|---|---|---|
| `leverage_coverage` | debtEquityRatio / (\|operatingProfitMargin\| + ε) | How well profits offset leverage |
| `liquidity_score` | (currentRatio + quickRatio + cashRatio) / 3 | Aggregate short-term solvency |
| `cashflow_debt_coverage` | operatingCashFlowPerShare / (\|debtEquityRatio\| + ε) | Cash generation relative to debt load |
| `profitability_composite` | (netProfitMargin + operatingProfitMargin + grossProfitMargin) / 3 | Multi-layer profitability health |
| `debt_service_ratio` | operatingCashFlowSalesRatio / (debtRatio + ε) | Cash conversion vs. total debt |
| `efficiency_composite` | (assetTurnover + fixedAssetTurnover) / 2 | Asset utilization efficiency |
| `roa_leverage` | returnOnAssets / (debtRatio + ε) | Return quality relative to leverage |
| `margin_stability` | \|grossProfitMargin − netProfitMargin\| | Margin compression indicator |
| `cash_liquidity_ratio` | cashRatio / (currentRatio + ε) | Cash quality within liquidity |
| `equity_efficiency` | returnOnCapitalEmployed × assetTurnover | Capital deployment effectiveness |

#### 8 Credit-Risk Specific Features (New)

| Engineered Feature | Formula | Financial Rationale |
|---|---|---|
| `liquidity_leverage` | currentRatio / (debtEquityRatio + ε) | Short-term solvency relative to leverage |
| `roe_roa_spread` | returnOnEquity − returnOnAssets | Financial leverage amplification effect |
| `margin_compression` | operatingProfitMargin − netProfitMargin | Non-operating cost burden indicator |
| `fcf_cash_ratio` | freeCashFlowPerShare / (\|cashPerShare\| + ε) | Free cash flow quality relative to cash position |
| `roa_turnover` | returnOnAssets × assetTurnover | DuPont decomposition: asset productivity |
| `debt_tax_burden` | debtRatio × (1 − effectiveTaxRate) | After-tax cost of debt capacity |
| `cash_leverage` | cashRatio / (\|debtEquityRatio\| + ε) | Cash buffer relative to leverage exposure |
| `cash_quality` | operatingCashFlowSalesRatio / (\|netProfitMargin\| + ε) | Cash conversion quality vs. accrual profits |

#### 3 Polynomial Interaction Features

| Engineered Feature | Formula | Financial Rationale |
|---|---|---|
| `debtEquityRatio_sq` | debtEquityRatio² | Captures non-linear leverage effects (extreme leverage penalized quadratically) |
| `returnOnAssets_sq` | returnOnAssets² | Captures diminishing returns at high profitability |
| `currentRatio_sq` | currentRatio² | Captures non-linear liquidity effects |

All interaction features are clipped to [-1e6, 1e6] to prevent numeric overflow from division by near-zero denominators. The epsilon constant (1e-5) prevents division-by-zero errors while remaining negligibly small relative to financial ratio scales.

### 3.3 Log Transformations for Skew Reduction

Twelve financial ratios with known right-skewed or heavy-tailed distributions receive a sign-preserving log transform:

```python
X[f"{col}_log"] = sign(X[col]) × log1p(|X[col]|)
```

Target columns: `currentRatio`, `quickRatio`, `cashRatio`, `daysOfSalesOutstanding`, `debtEquityRatio`, `enterpriseValueMultiple`, `operatingCashFlowPerShare`, `freeCashFlowPerShare`, `cashPerShare`, `payablesTurnover`, `fixedAssetTurnover`, `companyEquityMultiplier`.

The sign-preserving formulation ensures that negative values (e.g., negative free cash flow) retain their directional meaning after transformation.

### 3.4 Sector-Relative Z-Score Normalization

```
Stage: add_zscore_features()
Statistics: sector-level means and standard deviations (computed on training set only)
```

For each numeric ratio `r` and each company with sector `s`:

```
z_score = (x_r − μ_r,s) / (σ_r,s + ε)
```

Where `μ_r,s` and `σ_r,s` are the sector-level mean and standard deviation of ratio `r`, computed exclusively on the training set. Companies in sectors unseen during training fall back to global statistics.

This produces a `_sec_z` variant of every numeric feature, enabling the model to learn **relative** financial health within an industry context. For example, a tech company with a debt ratio at the sector's 90th percentile will produce a high z-score, even if the absolute value is low compared to utilities.

### 3.5 One-Hot Encoding of Sector

The `Sector` categorical variable is one-hot encoded via `pd.get_dummies()`. Column alignment between train and test sets is enforced with `reindex(columns=feature_columns, fill_value=0)`, ensuring no feature-index mismatch during inference.

### 3.6 Total Feature Dimensionality

After the full pipeline, the feature space expands from approximately 30 raw features to **~130 engineered features**, including:

- ~30 original numeric ratios
- 18 domain-driven interaction features (10 original + 8 credit-risk specific)
- 3 polynomial interaction features
- 12 log-transformed features
- ~55 sector-relative z-score features (covering all numeric columns including new interactions)
- ~12 sector one-hot columns

### 3.7 Feature Selection

To curb the overfitting risk of a ~130-feature space on ~2,000 rows, a training-fold-only selection step (`fit_feature_selection`) prunes:

1. **Near-zero-variance columns** (variance ≤ 1e-8) — features with no discriminative signal.
2. **Redundant columns** — one of every pair with absolute Pearson correlation > 0.95.

The retained column list is persisted as `feature_columns` and reapplied at inference, so training and serving always see an identical, reduced feature set.

---

## 4. Training & Validation Architecture

### 4.1 Class Imbalance Handling

Sample weights are computed via scikit-learn's `compute_sample_weight(class_weight='balanced')`, which assigns inverse-frequency weights to each training sample. This ensures that the minority Distressed class contributes proportionally to the loss function during gradient boosting, despite having orders of magnitude fewer examples.

### 4.2 XGBoost Base Model Configuration

```python
XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    use_label_encoder=False,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,
    n_jobs=-1
)
```

Key design decisions:

| Parameter | Value | Rationale |
|---|---|---|
| `objective` | `multi:softprob` | Multiclass classification with probability outputs |
| `eval_metric` | `mlogloss` | Multinomial log-loss for proper scoring |
| `subsample` | 0.8 | Row subsampling for regularization against overfitting |
| `colsample_bytree` | 0.8 | Feature subsampling per tree for diversity |
| `colsample_bylevel` | 0.8 | Feature subsampling per tree level for additional regularization |
| `n_jobs` | -1 | Parallel tree construction on all available cores |

### 4.3 Hyperparameter Search

A `RandomizedSearchCV` with 3-fold cross-validation samples **15 configurations** from the following parameter distribution space:

| Hyperparameter | Search Values |
|---|---|
| `max_depth` | [5, 7, 9] |
| `n_estimators` | [150, 200] |
| `learning_rate` | [0.03, 0.05, 0.1] |
| `min_child_weight` | [1, 3, 5] |
| `gamma` | [0, 0.1, 0.2] |
| `reg_alpha` | [0, 0.1, 0.3] |
| `reg_lambda` | [1, 1.5] |

The full grid contains 3 × 2 × 3 × 3 × 3 × 3 × 2 = **972 candidate configurations**, of which 15 are randomly sampled. Each sampled configuration is evaluated across 3 folds for a total of **45 model fits**. The scoring metric is **Macro F1** (`f1_macro`), and the best configuration is selected by `RandomizedSearchCV`. Randomized search was chosen over exhaustive grid search to efficiently explore the larger hyperparameter space with aggressive regularization parameters (`gamma`, `reg_alpha`, `reg_lambda`).

### 4.4 Ensemble Architecture

The pipeline constructs a **3-model soft-voting ensemble** for robustness:

1. **Model 1**: Best estimator from `RandomizedSearchCV` (trained with `RANDOM_STATE = 143`).
2. **Model 2**: Retrained with the same best hyperparameters but `random_state = 144`.
3. **Model 3**: Retrained with the same best hyperparameters but `random_state = 145`.

The three models are combined via scikit-learn's `VotingClassifier` with `voting="soft"`, which averages predicted class probabilities across all three models before making a final decision. This reduces variance from individual tree randomness and produces more stable predictions than any single model.

All three models are fitted with balanced sample weights to ensure the ensemble inherits class-imbalance awareness.

### 4.5 Probability Calibration

The soft-voting ensemble is wrapped in a `CalibratedClassifierCV` with:

- **Method**: Isotonic regression — a non-parametric calibration method that fits a monotonically increasing step function on held-out predictions.
- **CV**: 3-fold (with automatic fallback to 2-fold if any fold lacks representation of a minority class).

Isotonic calibration was chosen over sigmoid (Platt) scaling because the ensemble's probability outputs are not necessarily sigmoidally miscalibrated; isotonic regression provides more flexible correction at the cost of requiring more calibration data, which the 3-fold CV provides.

### 4.6 Post-Hoc Threshold Optimization

After calibration, class-specific decision thresholds are optimized to maximize **Macro F1** on the training set:

```
adjusted_probabilities = calibrated_proba × thresholds
prediction = argmax(adjusted_probabilities)
```

The optimization uses Nelder-Mead simplex search with 10 random restarts to escape local optima. Thresholds are normalized to sum to `n_classes`, preserving the probability scale.

> [!NOTE]
> **Resolved leakage**: The **strategy selector** previously compared the Macro F1 of threshold-optimized vs. standard predictions **on the test set** to decide which strategy to persist — leaking test labels into that binary choice. It now makes the decision using **training-set Macro F1 only**, so neither the threshold values nor the decision to use them touches test data. Empirically this removed roughly 1–2 points of optimistic bias from the single-split accuracy.

### 4.7 Model Caching

All trained artifacts are persisted via `joblib` for instant reload:

| Artifact | File | Purpose |
|---|---|---|
| Calibrated ensemble | `calibrated_model.pkl` | Calibrated VotingClassifier for inference predictions |
| Base XGBoost model | `base_model.pkl` | Best single XGBoost for SHAP explanations (TreeExplainer requires uncalibrated model) |
| Imputer medians | `imputer_medians.pkl` | Per-column training medians for missing-value imputation |
| Winsorization bounds | `winsorize_bounds.pkl` | Consistent outlier capping |
| Sector statistics | `sector_stats.pkl` | Z-score normalization parameters |
| Feature columns | `feature_columns.pkl` | Selected feature list (post-selection) for column alignment during inference |
| Optimal thresholds | `optimal_thresholds.pkl` | Post-hoc decision boundaries |
| Prediction strategy | `prediction_strategy.pkl` | Whether thresholds improve performance |

Cache keys are derived from an MD5 hash of the uploaded file data, enabling per-dataset model isolation. File names are suffixed with the hash (e.g., `calibrated_model_a403fd46.pkl`) to support concurrent caching of multiple datasets.

---

## 5. Experimental Results

### 5.1 Test Set Performance

Authoritative figures are from **grouped, stratified 5-fold cross-validation** (company-level, leakage-safe; standard calibrated predictions via `evaluate_xgboost.py`):

| Metric | CV Mean ± Std |
|---|---|
| **Accuracy** | 0.4603 ± 0.0313 |
| **Macro F1** | 0.4181 ± 0.0308 |

Per-class F1 (CV mean): Investment-High 0.5546 · Investment-Low 0.3642 · Speculative 0.4997 · Distressed 0.2537.

For reference, the single representative held-out split (seed 143, leakage-free path) used for the confusion matrix (§7) and classification report (§5.2) scores **accuracy 0.5222 / macro F1 0.4350** — above the CV mean, which illustrates why a single grouped split over ~118 test companies is not a reliable estimate on its own.

### 5.2 Per-Class Classification Report

Representative held-out grouped split (seed 143), 406 test samples — consistent with the confusion matrix in §7:

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Investment-High | 0.5570 | 0.4490 | 0.4972 | 98 |
| Investment-Low | 0.4593 | 0.4627 | 0.4610 | 134 |
| Speculative | 0.6168 | 0.6478 | 0.6319 | 159 |
| Distressed | 0.1200 | 0.2000 | 0.1500 | 15 |
| **Macro Avg** | **0.4382** | **0.4399** | **0.4350** | **406** |
| **Weighted Avg** | **0.5320** | **0.5222** | **0.5252** | **406** |

### 5.3 Key Observations

1. **Speculative** achieves the highest F1 (0.6319), benefiting from the largest support and relatively distinct feature distributions.
2. **Investment-High** is moderate (F1 0.4972) with precision 0.5570 but recall only 0.4490 — many high-quality credits are misread as Investment-Low.
3. **Investment-Low** is a weak boundary tier (F1 0.4610), frequently misread as Speculative (42 cases) or Investment-High (29 cases).
4. **Distressed** is now genuinely measurable (F1 0.1500, recall 0.2000, 3/15 caught). Weak but real signal — no longer the unmeasurable n=1 artifact of the previous grouping.
5. These are single-split figures and vary meaningfully across folds; the cross-validation mean in §5.1 is the figure to cite.

---

## 6. Attribution of Improvement by Optimization

Each optimization stage contributed measurably to the final model performance. The following table summarizes the cumulative improvements:

| Optimization | Technique | Primary Impact |
|---|---|---|
| **1. Winsorization** | Percentile-based outlier capping (P1–P99) | Stabilized gradient estimates; reduced tree depth waste on outlier splits |
| **2. Rating Consolidation** | 10+ granular ratings → 4 financial tiers | Increased per-class sample size; improved class separability |
| **3. Interaction Features** | 18 domain-driven composite ratios (10 original + 8 credit-risk specific) | Captured cross-ratio financial signals invisible to single-feature splits |
| **4. Polynomial Features** | Squared terms for top 3 credit signals | Captured non-linear effects in leverage, profitability, and liquidity |
| **5. Sector Z-Scores** | Industry-relative normalization | Eliminated sector bias; enabled fair cross-industry comparison |
| **6. Log Transforms** | Sign-preserving log1p on 12 skewed ratios | Compressed heavy tails; improved tree split efficiency |
| **7. Ensemble Stacking** | 3-model soft-voting XGBoost ensemble with seed diversity | Reduced prediction variance; more stable probability estimates |
| **8. Calibration** | Isotonic calibration, 3-fold CV (fallback 2-fold) | Non-parametric probability alignment; more flexible than Platt scaling |
| **9. Threshold Optimization** | Nelder-Mead Macro F1 maximization, 10 restarts | Shifted decision boundaries to favor minority classes |

---

## 7. Confusion Matrix Analysis

The confusion matrix reveals the model's misclassification patterns across the four risk tiers:

```
                  Predicted
                  IH    IL    SP    DI
Actual IH     [  44   41    11     2 ]
       IL     [  29   62    42     1 ]
       SP     [   5   32   103    19 ]
       DI     [   1    0    11     3 ]
```

*(IH = Investment-High, IL = Investment-Low, SP = Speculative, DI = Distressed. Representative held-out grouped split, seed 143, leakage-free prediction path; 406 test samples; accuracy 212/406 = 0.5222. Exact single-split values vary ~±1 point between runs due to the multithreading non-determinism noted in §11.1; this snapshot matches the persisted `results/` artifacts. The authoritative figure is the 5-fold CV mean in §5.1.)*

### 7.1 Misclassification Patterns

- **Investment-High ↔ Investment-Low** boundary is a primary confusion zone (41 IH→IL, 29 IL→IH). These adjacent classes share overlapping financial profiles.
- **Investment-Low ↔ Speculative** boundary is a major source of error (42 IL→SP, 32 SP→IL), reflecting the inherent difficulty of the BBB/BB boundary — the "fallen angel" threshold in credit risk.
- **Distressed** is now a measurable class (15 test samples, 3 correctly identified — recall 0.20, F1 ≈ 0.15). Performance is weak but genuine, no longer a data-availability artifact: the CCC/CC/C/D grouping gives the tier enough support to learn from and evaluate.
- Most confusions are between **adjacent tiers**; extreme misclassifications are rare (IH→DI = 2, DI→IH = 1), indicating the learned feature space largely preserves ordinal credit-quality structure.

### 7.2 Cost-Weighted Analysis

In banking applications, not all misclassifications carry equal cost: **upgrades** (predicting higher credit quality than the truth) are riskier than **downgrades**, so they carry heavier weights. Applying illustrative asymmetric weights to the representative-split confusion matrix (§7):

| Misclassification Type | Cost Weight | Frequency | Weighted Cost |
|---|---|---|---|
| Upgrade by 1 tier | 2× | 72 | 144 |
| Upgrade by 2+ tiers | 5× | 6 | 30 |
| Downgrade by 1 tier | 1× | 102 | 102 |
| Downgrade by 2+ tiers | 2× | 14 | 28 |
| **Total** | | **194 errors** | **304** |

The error profile skews toward **downgrades (116) over upgrades (78)** — the safer direction in credit risk, since the model errs toward caution more often than toward over-optimism. (Weights are illustrative; a production deployment would calibrate them to the institution's actual loss matrix rather than report a single "cost-efficiency" percentage.)

---

## 8. Model Interpretability & Explainability (SHAP)

### 8.1 SHAP Framework

The pipeline integrates **SHAP (SHapley Additive exPlanations)** for model interpretability using `TreeExplainer`, which provides exact Shapley values for tree-based models in polynomial time.

SHAP values are computed on the **uncalibrated base XGBoost model** (not the CalibratedClassifierCV wrapper), as TreeExplainer requires direct access to the tree structure. The SHAP values for the predicted class are extracted and presented.

### 8.2 Top Feature Importances (Global)

The following table shows the top 15 features ranked by global XGBoost feature importance (from the best base model in the ensemble):

> [!WARNING]
> These importance values were produced by an **earlier training run** (pre-dating the company-level split, the CCC/CC→Distressed regrouping, and feature selection). They are retained here for illustration but should be **regenerated** from the current model before being cited. Instance-level SHAP served by the live dashboard already reflects the current model.

| Rank | Feature | Importance |
|---|---|---|
| 1 | Debt Ratio | 0.2545 |
| 2 | Cashflow Debt Coverage | 0.2422 |
| 3 | Effective Tax Rate | 0.1970 |
| 4 | Net Profit Margin | 0.1195 |
| 5 | Current Ratio | 0.1155 |
| 6 | Debt Service Ratio (Sector Z-Score) | 0.1095 |
| 7 | Debt Ratio (Sector Z-Score) | 0.1047 |
| 8 | Return On Capital Employed | 0.0836 |
| 9 | Operating Cash Flow Per Share | 0.0704 |
| 10 | Pretax Profit Margin | 0.0679 |
| 11 | Operating Cash Flow Sales Ratio (Sector Z-Score) | 0.0628 |
| 12 | Fixed Asset Turnover | 0.0555 |
| 13 | ROA Leverage | 0.0533 |
| 14 | Free Cash Flow Operating Cash Flow Ratio (Sector Z-Score) | 0.0530 |
| 15 | Enterprise Value Multiple (Log, Sector Z-Score) | 0.0506 |

### 8.3 Key Interpretability Findings

1. **Debt Ratio** is the single most important predictor (importance = 0.2545), closely followed by **Cashflow Debt Coverage** (0.2422), confirming that leverage and its relationship to cash generation are the strongest discriminators of creditworthiness.
2. **Engineered features are well-represented**: `Cashflow Debt Coverage`, `ROA Leverage`, and `Debt Service Ratio (Sector Z-Score)` all appear in the top 15 — features that did not exist in the raw dataset, validating the feature engineering pipeline.
3. **Sector z-scores complement raw features**: 3 of the top 15 features are sector-relative z-scores, demonstrating that industry-adjusted metrics carry meaningful predictive signal alongside absolute values.
4. **Sector one-hot features contribute minimally**: `Sector_Basic Industries` (rank 79, importance 0.0138) is the highest-ranked sector dummy, confirming that z-score normalization successfully captures sector effects through continuous features rather than categorical splits.
5. **New credit-risk features contribute**: The `ROA Leverage` interaction feature (rank 13) demonstrates that the expanded feature engineering captures additional predictive signal beyond the original 10 composite features.

### 8.4 Instance-Level SHAP Explanations

For each prediction served through the web interface, the pipeline computes instance-level SHAP values that explain **why a specific company received its predicted rating**. Features are classified as:

- **Pushes Toward** (positive SHAP): This feature increased the probability of the predicted class.
- **Pushes Away** (negative SHAP): This feature decreased the probability, and the prediction was made despite this opposing signal.

The top 8 features by absolute SHAP magnitude are displayed as interactive bar charts in the dashboard, with color coding to distinguish supportive and opposing features.

---

## 9. Visualizations & Diagnostic Plots

The pipeline generates the following diagnostic visualizations, stored in the `figures/` directory:

| Plot | File | Purpose |
|---|---|---|
| Class Distribution | `class_distribution.png` | Visualizes target class imbalance before training |
| Confusion Matrix | `confusion_matrix.png` | Heatmap of prediction errors across all class pairs |
| Feature Importance | `feature_importance.png` | Bar chart of top XGBoost feature importances |
| SHAP Bar Plot | `shap_bar.png` | Global mean SHAP values for top features |
| SHAP Beeswarm | `shap_beeswarm.png` | Distribution of SHAP values across all test samples |
| SHAP Waterfall | `shap_waterfall_company0.png` | Instance-level SHAP decomposition for the first test sample |

---

## 10. Inference API & Deployment Readiness

### 10.1 Architecture

The inference system is deployed as a **Node.js HTTP server** (`app.js`) that spawns Python child processes for prediction:

```
Browser → Node.js Server (port 3000) → Python subprocess → JSON response
```

### 10.2 API Endpoints

| Method | Endpoint | Model |
|---|---|---|
| POST | `/predict/decision-tree` | Decision Tree |
| POST | `/predict/random-forest` | Random Forest |
| POST | `/predict/xgboost` | XGBoost |

### 10.3 Request/Response Format

**Request** (POST body):
```json
{
  "fileName": "corporate_data.csv",
  "fileData": "<base64-encoded file content>",
  "fileEncoding": "base64"
}
```

**Response** (values from the representative held-out grouped split, seed 143; the confusion matrix and metrics match §5 and §7 exactly):
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
    "metrics": { "accuracy": "0.5222", "precision": "0.5320", "recall": "0.5222", "f1": "0.5252" },
    "matrix": [[44,41,11,2],[29,62,42,1],[5,32,103,19],[1,0,11,3]],
    "shap": [["Debt Service Ratio (Sector Z-Score)", 95.0, 1], ...],
    "shapStory": {
      "positive": ["Debt Service Ratio (Sector Z-Score)", "Cashflow Debt Coverage", "Gross Profit Margin (Sector Z-Score)"],
      "negative": ["Payables Turnover", "Free Cash Flow Operating Cash Flow Ratio (Sector Z-Score)"]
    }
  }
}
```

> [!NOTE]
> `precision`/`recall`/`f1` are **weighted averages** over the held-out test set (recall equals accuracy for weighted averaging). The `matrix` rows are actual-class counts summing to per-class support (98 / 134 / 159 / 15 = 406).

### 10.4 Model Caching for Production

The MD5-based caching system ensures that:

- **First request** for a new dataset: trains from scratch (~30–60 seconds).
- **Subsequent requests** with the same dataset: loads cached artifacts (~1–2 seconds).
- **Different datasets** maintain isolated model caches, preventing cross-contamination.

### 10.5 Web Dashboard

The frontend (`views/index.html` + `client.js`) provides a single-page dashboard with:

- **File upload** interface supporting CSV, XLSX, and XLS formats.
- **Model selector** dropdown for switching between Decision Tree, Random Forest, and XGBoost.
- **Confusion matrix** rendered as an interactive grid.
- **SHAP bar chart** showing the top 8 most impactful features per prediction, with directional color coding (push toward vs. push away).
- **Narrative explanation** summarizing the key drivers of each prediction in plain English.
- **Performance metrics** panel displaying accuracy, precision, recall, and F1 score.

---

## 11. Reproducibility & Technical Environment

### 11.1 Random Seed

All stochastic operations are seeded with `RANDOM_STATE = 143`:

- `StratifiedGroupKFold` shuffling for the company-level split (§2.4)
- All three `XGBClassifier` seeds (base model 143, ensemble members 144/145). *Note: the base model previously had no explicit seed — a reproducibility bug that has been fixed.*
- Nelder-Mead threshold optimization (10 restarts seeded via `np.random.default_rng(143)`)

> [!NOTE]
> **Residual non-determinism.** Even with all seeds fixed, XGBoost trained with `n_jobs=-1` exhibits minor run-to-run variance (observed ~±1 percentage point in single-split accuracy, e.g. 0.5148 vs 0.5222) due to non-deterministic floating-point reduction order across threads. This is why §5 reports a **cross-validation mean ± std** as the authoritative figure rather than any single run. Bit-exact reproducibility would require single-threaded training (`n_jobs=1`), traded off against speed.

### 11.2 Software Dependencies

| Package | Version | Role |
|---|---|---|
| Python | 3.x | Runtime |
| pandas | 2.3.1 | Data manipulation |
| numpy | 2.1.3 | Numerical operations |
| scikit-learn | 1.8.0 | ML pipeline, calibration, metrics |
| xgboost | latest | Gradient boosted classifier |
| shap | 0.52.0 | Model interpretability |
| openpyxl | 3.1.5 | Excel (.xlsx) file support |
| xlrd | 2.0.2 | Legacy Excel (.xls) file support |
| Node.js | LTS | Web server runtime |

### 11.3 Data Leakage Prevention Checklist

| Pipeline Stage | Leakage-Free? | Mechanism |
|---|---|---|
| Train/test split (company overlap) | ✅ | Company-level `StratifiedGroupKFold` on `Symbol`/`Name`; no company appears in both partitions (fixes prior row-level leakage across the 593 companies × 2,029 records) |
| Median imputation | ✅ | Medians computed on `X_train` only; stored and reapplied |
| Feature selection | ✅ | Variance/correlation pruning fit on `X_train` only; selected columns persisted |
| Winsorization bounds | ✅ | Computed on `X_train` only; stored and reapplied |
| Sector z-score statistics | ✅ | Sector means/stds from `X_train`; persisted in `sector_stats.pkl` |
| Interaction features | ✅ | Deterministic formula; no data-dependent parameters |
| Log transforms | ✅ | Deterministic formula; no fitted parameters |
| One-hot encoding | ✅ | Column alignment via `reindex` with `fill_value=0` |
| Threshold optimization (values) | ✅ | Nelder-Mead optimization uses `y_train` and `calibrated_model.predict_proba(X_train_enc)` only |
| Threshold strategy selector | ✅ | Fixed: the use-thresholds decision now compares Macro F1 on `y_train` only — see §4.6 |
| Sample weight computation | ✅ | Computed on `y_train` class frequencies only |

---

## 12. Experimental History: What Didn't Work

Throughout the development of this pipeline, several techniques were explored but ultimately abandoned or modified because they failed to improve performance, introduced instability, or were impractical for the dataset size. Documenting these negative results is critical for reproducibility and to prevent future rework.

### 12.1 SMOTE Oversampling (Abandoned)

**What was tried**: SMOTE (Synthetic Minority Over-sampling Technique) was implemented to synthetically augment the Distressed class, which has as few as 4–5 training samples. A `RandomOverSampler` pre-step boosted the Distressed count to a minimum of 6 (SMOTE's `k_neighbors` requirement), then SMOTE generated interpolated synthetic samples.

**Why it failed**:
- With only 4–5 real Distressed samples, SMOTE generated synthetic points in a near-degenerate feature subspace. The interpolated samples were effectively noisy copies rather than meaningful augmentations.
- The model overfitted to the synthetic Distressed cluster, learning an artificial decision boundary that did not generalize to the test set.
- **Balanced class weights** (`compute_sample_weight(class_weight='balanced')`) proved more effective — they up-weight the loss contribution of rare classes without fabricating new data points, avoiding the introduction of synthetic noise.

**Evidence**: The notebook includes `USE_SMOTE = False` as the final configuration, with the SMOTE code path preserved but disabled.

### 12.2 Optuna Bayesian Hyperparameter Optimization (Replaced)

**What was tried**: Optuna with 50 TPE (Tree-structured Parzen Estimator) trials was used for hyperparameter search in the experimental notebook, exploring a wide continuous parameter space including `n_estimators` (500–2000), `learning_rate` (0.005–0.3, log-scale), `max_delta_step` (0–3), and `reg_lambda` (1–50, log-scale). Each trial used 3-fold stratified CV with early stopping (50 rounds) and XGBoost pruning callbacks.

**Why it was replaced**:
- On a dataset of ~2,000 samples, the 50-trial search took approximately 8–9 minutes. While acceptable for notebook experimentation, this was too slow for the production inference pipeline where the model must retrain on a new dataset within 30–60 seconds.
- The TPE-optimized hyperparameters were often highly specific to the random seed and fold split, showing high variance across different data uploads. `RandomizedSearchCV` with 15 iterations from a curated discrete grid was more robust and completed in a fraction of the time.
- The Optuna solution also required additional dependencies (`optuna`, `optuna.integration`) that complicated the production deployment.

**Evidence**: The notebook uses `optuna.create_study()` with `XGBoostPruningCallback`. The production script uses `RandomizedSearchCV` with a fixed discrete grid.

### 12.3 Early Stopping in Production (Removed)

**What was tried**: The notebook uses `early_stopping_rounds=50` with a held-out validation set (`VAL_SIZE=0.15`) to prevent overfitting during XGBoost training.

**Why it was removed in production**:
- Early stopping requires a dedicated validation split, which further reduces the already small training set from ~1,600 to ~1,360 samples.
- The 3-model ensemble approach with `colsample_bylevel` and `gamma` regularization provides sufficient overfitting protection without sacrificing training data.
- In production, the full training set is used for both the `RandomizedSearchCV` cross-validation and the final ensemble training, maximizing data utilization.

### 12.4 GPU Acceleration (Abandoned)

**What was tried**: The notebook includes `DEVICE = "cpu"` with a comment "Forced to CPU — faster for small datasets", indicating GPU training was tested.

**Why it didn't help**: XGBoost's GPU acceleration (`device="cuda"`) provides speedups primarily on large datasets (>100K samples) where the GPU kernel launch overhead is amortized. On a ~2,000-sample dataset with ~130 features, CPU training is actually faster due to zero GPU overhead, no CUDA memory transfer latency, and efficient use of all CPU cores via `n_jobs=-1`.

### 12.5 StandardScaler Normalization (Not Used)

**What was tried**: The notebook imports `StandardScaler` from scikit-learn, suggesting feature normalization was considered.

**Why it wasn't used**: XGBoost is a tree-based ensemble that makes decisions based on feature value thresholds (split points), not distances or dot products. Standardizing features to zero mean and unit variance does not change the ranking of split candidates and therefore has no effect on tree-based model performance. The sector-relative z-score normalization serves a different purpose — it creates *new features* that encode relative standing within an industry, rather than normalizing existing features.

### 12.6 FrozenEstimator for Calibration (Compatibility Issues)

**What was tried**: The notebook includes a conditional import of `sklearn.frozen.FrozenEstimator`, which wraps a pre-fitted model to prevent `CalibratedClassifierCV` from refitting it during calibration.

**Why it wasn't used**: `FrozenEstimator` was introduced in scikit-learn 1.4+ and may not be available in all deployment environments. The production pipeline instead uses `CalibratedClassifierCV(estimator=ensemble, cv=3)` which trains new clones of the ensemble within each calibration fold — this is slightly more compute-intensive but universally compatible and produces better-calibrated probabilities since each calibration fold sees a freshly trained model.

### 12.7 Sigmoid (Platt) Calibration → Isotonic Calibration

**What was tried initially**: The notebook and early production versions used `method="sigmoid"` (Platt scaling) with `cv=2` for probability calibration.

**What changed**: The production pipeline switched to `method="isotonic"` with `cv=3`.

**Why the change**:
- Platt scaling assumes the miscalibration follows a sigmoid curve (two parameters: slope and intercept). This assumption holds well for SVMs but is overly restrictive for ensemble models whose probability outputs can have non-sigmoidal miscalibration patterns.
- The 3-model soft-voting ensemble produces smoother probability estimates than a single XGBoost, providing enough calibration data for isotonic regression to fit without overfitting.
- The fallback from 3-fold to 2-fold ensures robustness when minority classes have too few samples for stratified 3-fold splitting.

### 12.8 Single-Model vs. Ensemble (Ensemble Won)

**What was tried**: The notebook trains a single XGBoost model with Optuna-optimized hyperparameters.

**What changed**: The production pipeline trains 3 XGBoost models with identical hyperparameters but different random seeds (143, 144, 145) and combines them via `VotingClassifier(voting="soft")`.

**Why the change**:
- A single XGBoost model exhibits seed-dependent variance — the same hyperparameters can produce measurably different predictions depending on the random state.
- Soft-voting across 3 models averages out this variance, producing more stable probability estimates. This is particularly impactful for borderline cases near the Investment-Low / Speculative boundary (the "fallen angel" threshold).
- The ensemble reduces prediction variance relative to a single seed-dependent model, which is most valuable for borderline minority-class cases.

### 12.9 Exhaustive GridSearchCV → RandomizedSearchCV

**What was tried**: The initial production version used `GridSearchCV` with a small 8-configuration grid (`max_depth` [3,5], `learning_rate` [0.05, 0.1], `n_estimators` [150, 300]).

**What changed**: Replaced with `RandomizedSearchCV` sampling 15 configurations from a 972-configuration space that includes regularization parameters (`gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`).

**Why the change**:
- The original grid was too small and missed important regularization parameters that prevent overfitting on the ~130-feature space.
- Exhaustively searching the full 972-configuration grid would require ~2,900 model fits (972 × 3 folds), taking 30+ minutes — unacceptable for production.
- Random sampling with 15 iterations explores the space efficiently in ~45 model fits, and empirically finds near-optimal configurations because the performance landscape is relatively smooth for tree-based models.

---

## 13. Limitations & Future Work

### 13.1 Current Limitations

1. **Model performance is modest**: The honest cross-validated accuracy is ~46% (macro F1 ~0.42) on a hard, imbalanced 4-class problem with only 593 distinct companies. This is a realistic ceiling for this feature set and sample size, not a bug — but it is far from the inflated ~72% that a leaky split produced. Meaningful gains most likely require more data (especially low-grade issuers) and temporal features rather than further tuning. *(The strategy-selector test-set leakage noted in earlier revisions has been fixed — see §4.6.)*
2. **Weak Distressed-class performance**: Grouping CCC/CC/C/D together raises the Distressed tier to ~72 records (~15 per test fold), which makes it measurable — the model now actually predicts it (recall ≈ 0.20, F1 ≈ 0.15 on the representative split; CV-mean F1 ≈ 0.25). Performance is still weak and high-variance given the modest support, but this is now a genuine modeling/data-volume challenge rather than the earlier "unmeasurable at n=1" artifact. SMOTE was tested and abandoned (see §12.1); acquiring more genuine low-grade records remains the highest-leverage improvement.
3. **Static sector definitions**: The model does not account for sector reclassification or conglomerate companies spanning multiple industries.
4. **Single-period prediction**: The model treats each company–year record independently, without temporal modeling of credit trajectory.
5. **Calibration data requirements**: Isotonic calibration with 3-fold CV requires at least 3 samples per class per fold. The automatic fallback to 2-fold mitigates this but reduces calibration quality for rare classes.
6. **Threshold optimization coupling**: The strategy selector is determined once at training time and does not adapt to distribution shift at inference time.
7. **Feature space growth**: The pipeline generates ~130 features from ~30 raw inputs, increasing overfitting risk. No automated feature selection is currently applied.

### 13.2 Proposed Improvements

1. **Targeted data collection**: Acquiring more Distressed-class samples is the highest-leverage improvement. Even 10–20 additional C/D-rated observations would enable meaningful learning for this class.
2. **Temporal features**: Rolling 3-year and 5-year trends in key ratios (debt ratio trajectory, profit margin compression) as additional features.
3. **Ordinal regression head**: Replacing `multi:softprob` with an ordinal-aware loss function that respects the inherent ordering of credit tiers (IH > IL > SP > DI).
4. **Cross-model ensemble stacking**: Combining XGBoost predictions with Random Forest and Logistic Regression via a meta-learner for model diversity beyond seed variation.
5. **Feature selection**: Applying Boruta or recursive feature elimination to prune the ~130-feature space and reduce overfitting risk.
6. **K-fold cross-validation evaluation** *(implemented)*: `evaluate_xgboost.py` now runs grouped, stratified 5-fold cross-validation and reports mean ± std, replacing the fragile single-split estimate as the authoritative performance figure (§5.1).
7. **Dynamic threshold adaptation**: Implementing online threshold recalibration at inference time based on recent prediction distributions.

---

## 14. Conclusion

The optimized XGBoost credit rating prediction pipeline demonstrates that systematic, domain-informed engineering can meaningfully improve classification performance on imbalanced financial datasets. Key contributions include:

1. **A leakage-free, multi-stage feature engineering pipeline** that transforms ~30 raw financial ratios into ~130 engineered features through winsorization, 21 domain-driven interaction and polynomial features, log transforms, and sector-relative z-scores.
2. **A 3-model soft-voting ensemble** with isotonic probability calibration and post-hoc threshold optimization that adapts decision boundaries to class imbalance.
3. **Expanded hyperparameter search** via `RandomizedSearchCV` with aggressive regularization parameters (`gamma`, `reg_alpha`, `reg_lambda`) scored on Macro F1.
4. **Production-ready deployment** via a Node.js server with Python subprocess inference, MD5-based model caching, and an interactive web dashboard with SHAP-based explanations.
5. **Near-full reproducibility** through fixed random seeds, deterministic label encoding, and comprehensive artifact serialization (with minor residual variance from multithreaded XGBoost — see §11.1).

Under company-level, leakage-safe 5-fold cross-validation the model achieves a **Macro F1 of 0.42 ± 0.03** and **Accuracy of 0.46 ± 0.03**, strongest on the Speculative and Investment-High tiers. The Distressed tier (CCC/CC/C/D) is now measurable, with weak but genuine signal. This headline is deliberately conservative: it reflects performance with no company-level or threshold-selection leakage, unlike the ~72% figure reported before those issues were corrected.

---

## 15. References & Technical Appendix

### 15.1 References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
2. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
3. Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. *Advances in Large Margin Classifiers*, 61–74.
4. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632.
5. Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *The Journal of Finance*, 23(4), 589–609.
6. Basel Committee on Banking Supervision. (2006). International Convergence of Capital Measurement and Capital Standards: A Revised Framework. Bank for International Settlements.

### 15.2 File Structure

```
Credit-Risk-Analyzer/
├── app.js                          # Node.js HTTP server
├── client.js                       # Frontend JavaScript (dashboard logic)
├── package.json                    # Node.js dependencies
├── requirements.txt                # Python dependencies
├── set A corporate_rating.csv      # Default training dataset
├── views/
│   └── index.html                  # Single-page dashboard
├── models/
│   ├── decisionTree.js             # Decision Tree model config
│   ├── randomForest.js             # Random Forest model config
│   ├── logisticRegression.js       # Logistic Regression model config
│   └── xgboost.js                  # XGBoost model config
└── python_models/
    ├── decision_tree_stuff/
    │   └── predict_decision_tree.py
    ├── random_forest_stuff/
    │   └── predict_random_forest.py
    ├── logistic_regression_stuff/
    │   └── predict_logistic_regression.py
    └── xgboost_stuff/
        ├── preprocessing.py        # Shared, leakage-safe preprocessing (split, impute, features, selection)
        ├── predict_xgboost.py      # XGBoost ensemble training & single-split inference pipeline
        ├── evaluate_xgboost.py     # Grouped k-fold cross-validation (authoritative metrics)
        ├── results/                # Cached models + regenerated evaluation outputs
        │   ├── calibrated_model_<hash>.pkl
        │   ├── base_model_<hash>.pkl
        │   ├── imputer_medians_<hash>.pkl
        │   ├── winsorize_bounds_<hash>.pkl
        │   ├── sector_stats_<hash>.pkl
        │   ├── feature_columns_<hash>.pkl
        │   ├── optimal_thresholds_<hash>.pkl
        │   ├── prediction_strategy_<hash>.pkl
        │   ├── xgboost_metrics.txt              # single-split accuracy / macro-F1
        │   ├── xgboost_classification_report.csv
        │   ├── xgboost_test_predictions.csv
        │   └── xgboost_cv_metrics.txt           # authoritative CV mean ± std
        └── figures/                # Diagnostic visualizations (regenerate from the notebook after retraining)
            ├── class_distribution.png
            ├── confusion_matrix.png
            ├── feature_importance.png
            ├── shap_bar.png
            ├── shap_beeswarm.png
            └── shap_waterfall_company0.png
```

### 15.3 Glossary

| Term | Definition |
|---|---|
| **Macro F1** | Unweighted average of per-class F1 scores; treats all classes equally regardless of support |
| **Platt Scaling** | Sigmoid calibration method that fits a logistic regression on model outputs to produce calibrated probabilities |
| **SHAP Values** | Shapley values from cooperative game theory, adapted to explain individual model predictions by attributing the prediction to each feature's contribution |
| **Winsorization** | Statistical transformation that clips extreme values to specified percentiles to reduce outlier influence |
| **Z-Score** | Number of standard deviations a value lies from the group mean; used here for sector-relative normalization |
| **TreeExplainer** | SHAP's exact algorithm for computing Shapley values in tree ensembles in O(TLD²) time |
| **Nelder-Mead** | Derivative-free simplex optimization algorithm used for threshold tuning |

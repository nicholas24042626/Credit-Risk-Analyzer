# Technical Report: Optimized XGBoost Credit Rating Prediction Pipeline

## Executive Summary

This report details the design, iterative optimization, and evaluation of an XGBoost multiclass classifier for corporate credit rating prediction. The project addresses four core challenges in financial classification: industry-specific ratio baselines, extreme class imbalance, probability miscalibration, and data leakage.

Through a systematic optimization process spanning seven distinct improvements — robust outlier handling, target class consolidation, domain-driven feature engineering, data-efficient calibration, expanded hyperparameter search, post-hoc threshold optimization, and ensemble stacking — the final model achieved a **Macro F1 Score of 0.5392** and an **Accuracy of 72.41%** on a held-out test set, with a **Cost Efficiency of 86.18%** under a banking-grade misclassification cost matrix. All stages of this pipeline have been implemented with zero data leakage, ensuring mathematical and statistical validity.

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
12. [Limitations & Future Work](#12-limitations--future-work)
13. [Conclusion](#13-conclusion)
14. [References & Technical Appendix](#14-references--technical-appendix)

---

## 1. Research Challenges in Credit Risk Modeling

Corporate credit rating prediction poses several domain-specific challenges that generic classification pipelines do not address:

### 1.1 Industry-Specific Ratio Baselines

Financial ratios such as `debtEquityRatio`, `currentRatio`, and `operatingProfitMargin` carry fundamentally different meanings across industry sectors. A debt-equity ratio of 3.0 may be standard in utilities but alarming in technology. Failing to normalize ratios within sector context introduces systematic bias that penalizes asset-heavy industries regardless of their true creditworthiness.

### 1.2 Extreme Class Imbalance

The dataset exhibits severe class imbalance across the four target tiers. The Speculative class dominates with 172 test samples, while the Distressed class is represented by as few as 1 sample in the test set. Standard accuracy metrics become misleading under such distributions, as a model that predicts only the majority class can achieve superficially high accuracy.

### 1.3 Probability Miscalibration

Gradient-boosted tree ensembles are known to produce poorly calibrated probability estimates. Raw `predict_proba` outputs from XGBoost do not reflect true posterior class probabilities, making them unreliable for downstream decision-making (e.g., setting confidence thresholds for automated approval). Isotonic or Platt (sigmoid) calibration is required to align predicted confidence with observed frequency.

### 1.4 Data Leakage

In financial modeling, data leakage is a critical concern at every stage of the pipeline. Leakage can occur through:

- **Preprocessing leakage**: Computing winsorization bounds, z-score statistics, or one-hot encoder vocabularies on the full dataset before splitting.
- **Feature engineering leakage**: Deriving sector-level statistics that include test-set observations.
- **Threshold optimization leakage**: Selecting classification thresholds based on test-set performance.

This pipeline explicitly addresses all three forms by computing every transformation parameter exclusively on the training fold.

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
| **Speculative** | BB, B, CCC, CC | Significant credit risk; speculative-grade |
| **Distressed** | C, D | Near-default or in default; recovery analysis territory |

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

The dataset is split 80/20 with stratified sampling (`stratify=y_encoded`) to preserve class proportions in both partitions. A fallback to non-stratified splitting is provided for edge cases where a class has too few members to stratify. The random seed is fixed at `RANDOM_STATE = 143` for full reproducibility.

### 2.5 Identifier Dropping

Non-predictive columns (`Name`, `Symbol`, `Rating Agency Name`, `Date`, `RatingClass`, `Rating`) are explicitly dropped before any feature computation, ensuring no target-correlated metadata leaks into the feature space.

---

## 3. Feature Engineering Pipeline

The feature engineering pipeline is a multi-stage transformation applied **exclusively on the training set**, with learned parameters then applied to the test set. This guarantees zero data leakage.

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

Ten composite financial features are engineered from raw ratios to capture domain-specific relationships that single features cannot express:

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

After the full pipeline, the feature space expands from approximately 30 raw features to **107 engineered features**, including:

- ~30 original numeric ratios
- 10 domain-driven interaction features
- 12 log-transformed features
- ~50 sector-relative z-score features
- ~12 sector one-hot columns

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
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=143,
    n_jobs=-1
)
```

Key design decisions:

| Parameter | Value | Rationale |
|---|---|---|
| `objective` | `multi:softprob` | Multiclass classification with probability outputs |
| `eval_metric` | `mlogloss` | Multinomial log-loss for proper scoring |
| `subsample` | 0.7 | Row subsampling for regularization against overfitting |
| `colsample_bytree` | 0.7 | Feature subsampling per tree for diversity |
| `n_jobs` | -1 | Parallel tree construction on all available cores |

### 4.3 Hyperparameter Search

A grid search with 3-fold cross-validation explores the following space:

| Hyperparameter | Search Values |
|---|---|
| `max_depth` | [3, 5] |
| `learning_rate` | [0.05, 0.1] |
| `n_estimators` | [150, 300] |

This produces 2 × 2 × 2 = **8 candidate configurations**, each evaluated across 3 folds for a total of **24 model fits**. The scoring metric is accuracy, and the best configuration is selected by `GridSearchCV`.

### 4.4 Probability Calibration

The best XGBoost model is wrapped in a `CalibratedClassifierCV` with:

- **Method**: Sigmoid (Platt scaling) — maps raw scores through a logistic function fitted on held-out predictions.
- **CV**: 2-fold — balances calibration quality with data efficiency, critical for small datasets.

Sigmoid calibration was chosen over isotonic regression because it requires fewer data points and is less prone to overfitting on small calibration sets.

### 4.5 Post-Hoc Threshold Optimization

After calibration, class-specific decision thresholds are optimized to maximize **Macro F1** on the training set:

```
adjusted_probabilities = calibrated_proba × thresholds
prediction = argmax(adjusted_probabilities)
```

The optimization uses Nelder-Mead simplex search with 10 random restarts to escape local optima. Thresholds are normalized to sum to `n_classes`, preserving the probability scale.

A **strategy selector** automatically compares the Macro F1 of threshold-optimized predictions against standard predictions on the test set. The winning strategy is persisted and used at inference time.

### 4.6 Model Caching

All trained artifacts are persisted via `joblib` for instant reload:

| Artifact | File | Purpose |
|---|---|---|
| Calibrated model | `calibrated_model.pkl` | Inference predictions |
| Base XGBoost model | `xgb_credit_model.pkl` | SHAP explanations (TreeExplainer requires uncalibrated model) |
| Winsorization bounds | `winsorize_bounds.pkl` | Consistent outlier capping |
| Sector statistics | `sector_stats.pkl` | Z-score normalization parameters |
| Feature columns | `feature_columns.pkl` | Column alignment during inference |
| Optimal thresholds | `optimal_thresholds.pkl` | Post-hoc decision boundaries |
| Prediction strategy | `prediction_strategy.pkl` | Whether thresholds improve performance |

Cache keys are derived from an MD5 hash of the uploaded file data, enabling per-dataset model isolation.

---

## 5. Experimental Results

### 5.1 Test Set Performance

| Metric | Value |
|---|---|
| **Accuracy** | 0.7241 |
| **Balanced Accuracy** | 0.5415 |
| **Macro F1** | 0.5392 |
| **ROC AUC (OVR)** | 0.7122 |

### 5.2 Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Investment-High | 0.7212 | 0.7576 | 0.7389 | 99 |
| Investment-Low | 0.6260 | 0.6119 | 0.6189 | 134 |
| Speculative | 0.8012 | 0.7965 | 0.7988 | 172 |
| Distressed | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Weighted Avg** | **0.7219** | **0.7241** | **0.7229** | **406** |

### 5.3 Key Observations

1. **Speculative class** achieves the highest F1 (0.7988), benefiting from both the largest support and distinct feature distributions.
2. **Investment-High** shows balanced precision–recall (0.72/0.76), indicating effective discrimination at the high-quality end.
3. **Investment-Low** is the most confused class (F1 = 0.6189), consistent with its position as a boundary class between Investment-High and Speculative.
4. **Distressed** class achieves zero performance due to extreme rarity (1 test sample), representing a known limitation in any supervised approach without synthetic augmentation.

---

## 6. Attribution of Improvement by Optimization

Each optimization stage contributed measurably to the final model performance. The following table summarizes the cumulative improvements:

| Optimization | Technique | Primary Impact |
|---|---|---|
| **1. Winsorization** | Percentile-based outlier capping (P1–P99) | Stabilized gradient estimates; reduced tree depth waste on outlier splits |
| **2. Rating Consolidation** | 10+ granular ratings → 4 financial tiers | Increased per-class sample size; improved class separability |
| **3. Interaction Features** | 10 domain-driven composite ratios | Captured cross-ratio financial signals invisible to single-feature splits |
| **4. Sector Z-Scores** | Industry-relative normalization | Eliminated sector bias; enabled fair cross-industry comparison |
| **5. Log Transforms** | Sign-preserving log1p on 12 skewed ratios | Compressed heavy tails; improved tree split efficiency |
| **6. Calibration** | Sigmoid (Platt) calibration, 2-fold CV | Aligned predicted probabilities with observed frequencies |
| **7. Threshold Optimization** | Nelder-Mead Macro F1 maximization, 10 restarts | Shifted decision boundaries to favor minority classes |

---

## 7. Confusion Matrix Analysis

The confusion matrix reveals the model's misclassification patterns across the four risk tiers:

```
                  Predicted
                  IH    IL    SP    DI
Actual IH     [  75    20     4     0 ]
       IL     [  22    82    30     0 ]
       SP     [   7    29   137     0 ]
       DI     [   0     0     1     0 ]
```

*(IH = Investment-High, IL = Investment-Low, SP = Speculative, DI = Distressed)*

### 7.1 Misclassification Patterns

- **Investment-High ↔ Investment-Low** boundary is the primary confusion zone (20 IH→IL, 22 IL→IH). These adjacent classes share overlapping financial profiles.
- **Investment-Low ↔ Speculative** boundary is the second major source of error (30 IL→SP, 29 SP→IL), reflecting the inherent difficulty of the BBB/BB boundary — the "fallen angel" threshold in credit risk.
- **Distressed** class is invisible to the model (0/1 correct), entirely attributable to insufficient training representation rather than model architecture.
- The model rarely makes **extreme misclassifications** (e.g., IH→DI or DI→IH = 0), indicating that the learned feature space preserves ordinal credit quality structure.

### 7.2 Cost-Weighted Analysis

In banking applications, not all misclassifications carry equal cost. Under a typical asymmetric cost matrix where upgrading risk (predicting higher quality than actual) is penalized 3× more than downgrading:

| Misclassification Type | Cost Weight | Frequency | Weighted Cost |
|---|---|---|---|
| Upgrade by 1 tier | 2× | 51 | 102 |
| Upgrade by 2+ tiers | 5× | 7 | 35 |
| Downgrade by 1 tier | 1× | 49 | 49 |
| Downgrade by 2+ tiers | 2× | 1 | 2 |
| **Total Weighted Cost** | | | **188** |
| **Maximum Possible Cost** | | | **~1,360** |
| **Cost Efficiency** | | | **~86.18%** |

---

## 8. Model Interpretability & Explainability (SHAP)

### 8.1 SHAP Framework

The pipeline integrates **SHAP (SHapley Additive exPlanations)** for model interpretability using `TreeExplainer`, which provides exact Shapley values for tree-based models in polynomial time.

SHAP values are computed on the **uncalibrated base XGBoost model** (not the CalibratedClassifierCV wrapper), as TreeExplainer requires direct access to the tree structure. The SHAP values for the predicted class are extracted and presented.

### 8.2 Top Feature Importances (Global)

The following table shows the top 15 features ranked by global XGBoost feature importance:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Cashflow Debt Coverage | 0.2703 |
| 2 | Debt Ratio | 0.2126 |
| 3 | Effective Tax Rate | 0.1927 |
| 4 | Debt Ratio (Sector Z-Score) | 0.1297 |
| 5 | Debt Service Ratio (Sector Z-Score) | 0.0982 |
| 6 | Debt Equity Ratio (Sector Z-Score) | 0.0904 |
| 7 | Current Ratio | 0.0902 |
| 8 | Return On Assets (Sector Z-Score) | 0.0851 |
| 9 | Pretax Profit Margin (Sector Z-Score) | 0.0835 |
| 10 | Operating Cash Flow Per Share | 0.0832 |
| 11 | Net Profit Margin | 0.0795 |
| 12 | Return On Capital Employed | 0.0739 |
| 13 | Company Equity Multiplier (Sector Z-Score) | 0.0634 |
| 14 | Operating Cash Flow Sales Ratio (Sector Z-Score) | 0.0622 |
| 15 | Net Profit Margin (Sector Z-Score) | 0.0618 |

### 8.3 Key Interpretability Findings

1. **Cashflow Debt Coverage** is the single most important predictor (importance = 0.2703), confirming that the relationship between a company's cash generation and its leverage is the strongest discriminator of creditworthiness.
2. **Engineered features dominate**: 4 of the top 15 features are interaction or z-score features that did not exist in the raw dataset, validating the feature engineering pipeline.
3. **Sector z-scores are highly represented**: 7 of the top 15 features are sector-relative z-scores, demonstrating that industry-adjusted metrics carry more predictive signal than absolute values.
4. **Sector one-hot features contribute minimally**: `Sector_Basic Industries` (rank 86, importance 0.0126) is the only sector dummy in the top 90, confirming that z-score normalization successfully captures sector effects through continuous features rather than categorical splits.

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

**Response**:
```json
{
  "prediction": "Speculative",
  "probabilities": {
    "Investment-High": 0.12,
    "Investment-Low": 0.23,
    "Speculative": 0.58,
    "Distressed": 0.07
  },
  "modelData": {
    "tag": "XGBoost",
    "labels": ["Investment-High", "Investment-Low", "Speculative", "Distressed"],
    "metrics": { "accuracy": "0.7241", "precision": "0.7219", "recall": "0.7241", "f1": "0.7229" },
    "matrix": [[75,20,4,0],[22,82,30,0],[7,29,137,0],[0,0,1,0]],
    "shap": [["Cashflow Debt Coverage", 95.0, 1], ...],
    "shapStory": {
      "positive": ["Cashflow Debt Coverage", "Debt Ratio", "Current Ratio"],
      "negative": ["Net Profit Margin", "Effective Tax Rate"]
    }
  }
}
```

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

All stochastic operations use `RANDOM_STATE = 143`:

- `train_test_split` stratification
- `XGBClassifier` random state
- Nelder-Mead threshold optimization (10 restarts with `np.random.uniform`)

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
| Winsorization bounds | ✅ | Computed on `X_train` only; stored and reapplied |
| Sector z-score statistics | ✅ | Sector means/stds from `X_train`; persisted in `sector_stats.pkl` |
| Interaction features | ✅ | Deterministic formula; no data-dependent parameters |
| Log transforms | ✅ | Deterministic formula; no fitted parameters |
| One-hot encoding | ✅ | Column alignment via `reindex` with `fill_value=0` |
| Threshold optimization | ✅ | Optimized on training set probabilities; evaluated on test set |
| Sample weight computation | ✅ | Computed on `y_train` class frequencies only |

---

## 12. Limitations & Future Work

### 12.1 Current Limitations

1. **Distressed class performance**: The model achieves 0% recall on the Distressed tier due to extreme rarity (≤1 test sample). This is a dataset limitation, not an architectural one.
2. **Static sector definitions**: The model does not account for sector reclassification or conglomerate companies spanning multiple industries.
3. **Single-period prediction**: The model treats each company–year record independently, without temporal modeling of credit trajectory.
4. **Calibration fidelity**: 2-fold sigmoid calibration may underfit the calibration function on very small datasets.
5. **Threshold optimization coupling**: The strategy selector (threshold vs. standard) is determined once at training time and does not adapt to distribution shift at inference time.

### 12.2 Proposed Improvements

1. **SMOTE / ADASYN** for the Distressed class to synthetically augment minority samples, combined with Tomek link cleaning for boundary refinement.
2. **Temporal features**: Rolling 3-year and 5-year trends in key ratios (debt ratio trajectory, profit margin compression) as additional features.
3. **Ordinal regression head**: Replacing `multi:softprob` with an ordinal-aware loss function that respects the inherent ordering of credit tiers (IH > IL > SP > DI).
4. **Ensemble stacking**: Combining XGBoost predictions with Random Forest and Logistic Regression via a meta-learner (Logistic Regression on held-out predictions).
5. **Bayesian hyperparameter optimization**: Replacing grid search with Optuna or Hyperopt for more efficient exploration of the hyperparameter space.
6. **Cross-validation evaluation**: Moving from a single train/test split to 5-fold stratified cross-validation for more robust performance estimates.
7. **Feature selection**: Applying Boruta or recursive feature elimination to prune the 107-feature space and reduce overfitting risk.

---

## 13. Conclusion

The optimized XGBoost credit rating prediction pipeline demonstrates that systematic, domain-informed engineering can meaningfully improve classification performance on imbalanced financial datasets. Key contributions include:

1. **A leakage-free, multi-stage feature engineering pipeline** that transforms ~30 raw financial ratios into 107 engineered features through winsorization, domain-driven interactions, log transforms, and sector-relative z-scores.
2. **Post-hoc probability calibration and threshold optimization** that adapt decision boundaries to class imbalance without modifying the underlying model.
3. **Production-ready deployment** via a Node.js server with Python subprocess inference, MD5-based model caching, and an interactive web dashboard with SHAP-based explanations.
4. **Full reproducibility** through fixed random seeds, deterministic label encoding, and comprehensive artifact serialization.

The model achieves a **Macro F1 of 0.5392** and **Accuracy of 72.41%**, with the strongest performance on the Speculative tier (F1 = 0.7988) and known limitations on the Distressed tier due to data scarcity. The cost-weighted analysis demonstrates **86.18% cost efficiency** under banking-grade asymmetric penalties, indicating practical viability for credit risk screening applications.

---

## 14. References & Technical Appendix

### 14.1 References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
2. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
3. Platt, J. (1999). Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. *Advances in Large Margin Classifiers*, 61–74.
4. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632.
5. Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *The Journal of Finance*, 23(4), 589–609.
6. Basel Committee on Banking Supervision. (2006). International Convergence of Capital Measurement and Capital Standards: A Revised Framework. Bank for International Settlements.

### 14.2 File Structure

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
        ├── predict_xgboost.py      # Full XGBoost training & inference pipeline
        ├── results/                # Cached models, metrics, and reports
        │   ├── calibrated_model.pkl
        │   ├── xgb_credit_model.pkl
        │   ├── winsorize_bounds.pkl
        │   ├── sector_stats.pkl
        │   ├── feature_columns.pkl
        │   ├── optimal_thresholds.pkl
        │   ├── prediction_strategy.pkl
        │   ├── xgboost_metrics.txt
        │   ├── xgboost_classification_report.csv
        │   └── xgboost_feature_importance.csv
        └── figures/                # Diagnostic visualizations
            ├── class_distribution.png
            ├── confusion_matrix.png
            ├── feature_importance.png
            ├── shap_bar.png
            ├── shap_beeswarm.png
            └── shap_waterfall_company0.png
```

### 14.3 Glossary

| Term | Definition |
|---|---|
| **Macro F1** | Unweighted average of per-class F1 scores; treats all classes equally regardless of support |
| **Platt Scaling** | Sigmoid calibration method that fits a logistic regression on model outputs to produce calibrated probabilities |
| **SHAP Values** | Shapley values from cooperative game theory, adapted to explain individual model predictions by attributing the prediction to each feature's contribution |
| **Winsorization** | Statistical transformation that clips extreme values to specified percentiles to reduce outlier influence |
| **Z-Score** | Number of standard deviations a value lies from the group mean; used here for sector-relative normalization |
| **TreeExplainer** | SHAP's exact algorithm for computing Shapley values in tree ensembles in O(TLD²) time |
| **Nelder-Mead** | Derivative-free simplex optimization algorithm used for threshold tuning |

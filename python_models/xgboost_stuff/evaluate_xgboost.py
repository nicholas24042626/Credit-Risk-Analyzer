"""Authoritative offline evaluation for the XGBoost credit-risk pipeline.

This is the SINGLE SOURCE OF TRUTH for reported model performance (see
docs/XGBoost_Technical_Report.md and the dissertation write-up). A single
train/test split over ~593 companies is a high-variance estimate (the same
pipeline has been observed to swing from ~45% to ~70% single-split accuracy
purely based on which companies land in the held-out fold), so this script
reports **grouped, stratified k-fold cross-validation** — the honest, stable
performance number to cite.

Methodological alignment with the dashboard (predict_xgboost.py):
- Same model: train_model() (SMOTE oversampling + seeded XGBoost soft-voting
  ensemble blended with the ordinal cumulative-link view, §16.26) is imported
  from predict_xgboost.py and used unmodified here, so the CV number and the
  dashboard's single-split demo number are measuring the *same* model, not
  two different pipelines that happen to share a name.
- Same prediction rule: plain argmax over the blended probabilities. The
  nested threshold-multiplier protocol and the second-stage corrector were
  removed in §16.26 after both were measured to reduce the blended
  ensemble's accuracy and macro-F1.

Key properties:
- Company-level grouping (StratifiedGroupKFold on Symbol/Name): no company
  appears in both the training and the held-out fold, so there is no
  company-level leakage.
- Per-fold preprocessing: missingness-indicator columns, sector z-scores,
  categorical dtype casting, and feature selection are all fit on the
  training fold only. No imputation and no winsorisation — see
  preprocessing.py's module docstring for why those don't belong in a
  tree-based pipeline; NaN passes through for XGBoost's native missing-value
  split handling.
- Results (per-fold and mean/std) are persisted to xgboost_cv_metrics.json so
  the dashboard can display the CV mean ± std as the primary, authoritative
  accuracy figure without re-running this expensive CV on every request.

Run:  python evaluate_xgboost.py           (uses the default dataset, 3 folds ~= 70/30 split)
      python evaluate_xgboost.py 5          (override number of folds, e.g. 5 folds ~= 80/20 split)
"""

import json
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

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
    group_rating,
)
from predict_xgboost import (
    DEFAULT_DATASET_PATH,
    RESULTS_DIR,
    train_model,
)

warnings.filterwarnings("ignore")

CV_METRICS_JSON_PATH = RESULTS_DIR / "xgboost_cv_metrics.json"


def preprocess_fold(X_train, X_test):
    """Fit all transforms on the training fold, apply to both (leakage-free)."""
    indicator_cols = fit_missingness_indicators(X_train)
    X_train = apply_missingness_indicators(X_train, indicator_cols)
    X_test = apply_missingness_indicators(X_test, indicator_cols)

    X_train_i = add_interaction_features(X_train)
    X_test_i = add_interaction_features(X_test)

    if "Sector" in X_train_i.columns:
        stats = compute_sector_stats(X_train_i)
        apply_zscore_from_stats([X_train_i, X_test_i], stats)

    categories = fit_categorical_dtypes(X_train_i)
    X_train_cat = apply_categorical_dtypes(X_train_i, categories)
    X_test_cat = apply_categorical_dtypes(X_test_i, categories)

    X_train_enc, X_test_enc = align_features(X_train_cat, X_test_cat)
    selected = fit_feature_selection(X_train_enc)
    return X_train_enc[selected], X_test_enc.reindex(columns=selected)


def main():
    # Default of 3 folds gives each fold roughly a 67/33 train/test split
    # (1/3 ~= 0.33), matching the ~70/30 split now used by the dashboard's
    # single-split path (preprocessing.py's make_split default test_size).
    # Override via CLI arg, e.g. `python evaluate_xgboost.py 5` for 80/20.
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    df = pd.read_csv(DEFAULT_DATASET_PATH)
    df, _clean_report = clean_dataframe(df)
    # NOTE: add_temporal_features() was tried here and reverted -- see
    # docs/XGBoost_Technical_Report.md for the measured (negative) result.
    y_raw = df["Rating"].apply(group_rating)
    mask = y_raw != "Unknown"
    df = df[mask].copy()
    y = LABEL_ENCODER.transform(y_raw[mask])
    groups = extract_groups(df)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    n_classes = len(LABEL_ENCODER.classes_)

    accs, macro_f1s = [], []
    train_accs, train_macro_f1s = [], []
    per_class_f1 = {c: [] for c in LABEL_ENCODER.classes_}
    fold_records = []

    # Per-sector breakdown: accumulates (true, predicted) label pairs across
    # every fold's OUTER TEST predictions only, keyed by raw Sector value.
    # Motivation: real credit-rating agencies do not apply one global
    # threshold across sectors (leverage that's normal for a utility can be
    # alarming for a tech company), so a fair check of this pipeline is
    # whether accuracy is roughly comparable across sectors or concentrated
    # in a few. This is a MEASUREMENT only -- it does not change training,
    # splitting, or the reported headline CV accuracy above. See
    # docs/XGBoost_Technical_Report.md for the sector-fairness discussion
    # this was added to investigate.
    sector_true = {}
    sector_pred = {}

    for i, (tr, te) in enumerate(sgkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]
        groups_tr = groups[tr] if groups is not None else None

        # Capture each test row's raw Sector value BEFORE preprocessing
        # (preprocess_fold one-hot encodes Sector away). Purely for the
        # per-sector measurement below; never used as a training signal.
        sectors_te = X_te["Sector"].to_numpy() if "Sector" in X_te.columns else None

        X_tr_enc, X_te_enc = preprocess_fold(X_tr, X_te)

        # Same model as the dashboard: SMOTE + XGBoost soft-voting ensemble
        # blended with the ordinal cumulative-link view (train_model imported
        # directly from predict_xgboost.py -- one implementation, not two).
        # Plain argmax over the blended probabilities: the nested threshold
        # multipliers and second-stage corrector were removed in §16.26 (both
        # reduced the blended ensemble's accuracy and macro-F1).
        best_xgb, ensemble, best_params = train_model(X_tr_enc, y_tr, train_groups=groups_tr)

        preds = ensemble.predict_proba(X_te_enc).argmax(axis=1)
        train_preds = ensemble.predict_proba(X_tr_enc).argmax(axis=1)

        # Accumulate per-sector (true, predicted) pairs for this fold's test
        # rows into the running cross-fold collection.
        if sectors_te is not None:
            for sec, true_lbl, pred_lbl in zip(sectors_te, y_te, preds):
                sector_true.setdefault(sec, []).append(int(true_lbl))
                sector_pred.setdefault(sec, []).append(int(pred_lbl))

        acc = accuracy_score(y_te, preds)
        macro = f1_score(y_te, preds, average="macro")
        f1_by_class = f1_score(
            y_te, preds, average=None,
            labels=list(range(n_classes)), zero_division=0,
        )

        train_acc = accuracy_score(y_tr, train_preds)
        train_macro = f1_score(y_tr, train_preds, average="macro")

        accs.append(acc)
        macro_f1s.append(macro)
        train_accs.append(train_acc)
        train_macro_f1s.append(train_macro)
        for idx, cls in enumerate(LABEL_ENCODER.classes_):
            per_class_f1[cls].append(f1_by_class[idx])

        gap = train_acc - acc
        fold_records.append({
            "fold": i,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "train_accuracy": round(float(train_acc), 4),
            "test_accuracy": round(float(acc), 4),
            "train_test_gap": round(float(gap), 4),
            "train_macro_f1": round(float(train_macro), 4),
            "test_macro_f1": round(float(macro), 4),
        })
        print(
            f"fold {i}/{n_splits}: n_train={len(y_tr)} n_test={len(y_te)} "
            f"train_acc={train_acc:.4f} test_acc={acc:.4f} gap={gap:.4f} "
            f"train_macroF1={train_macro:.4f} test_macroF1={macro:.4f}",
            flush=True,
        )

    accs = np.array(accs)
    macro_f1s = np.array(macro_f1s)
    train_accs = np.array(train_accs)
    train_macro_f1s = np.array(train_macro_f1s)
    acc_gap = train_accs.mean() - accs.mean()
    macro_gap = train_macro_f1s.mean() - macro_f1s.mean()

    # Rough overfitting read: a large train/test gap points to overfitting
    # (too much model/feature capacity for the data); a small gap with a low
    # test score points to underfitting/data-starvation instead, where
    # trimming features is unlikely to help much.
    if acc_gap > 0.15:
        overfit_verdict = (
            f"LIKELY OVERFITTING: train accuracy exceeds test accuracy by "
            f"{acc_gap:.4f} (>{0.15:.2f} threshold). Feature-space reduction, "
            f"stronger regularization, or fewer boosting rounds are worth trying."
        )
    elif acc_gap > 0.08:
        overfit_verdict = (
            f"MILD OVERFITTING SIGNAL: train/test accuracy gap is {acc_gap:.4f}. "
            f"Some benefit possible from trimming features or adding regularization, "
            f"but this is not the dominant issue."
        )
    else:
        overfit_verdict = (
            f"NOT OVERFITTING: train/test accuracy gap is only {acc_gap:.4f}. "
            f"The model is closer to under-fit / data-starved than over-fit — "
            f"cutting features is unlikely to move the test score much. The "
            f"ceiling is more likely driven by sample size (593 companies) and "
            f"class imbalance (Distressed ~3.6%) than by feature-space size."
        )

    per_class_f1_means = {cls: float(np.array(vals).mean()) for cls, vals in per_class_f1.items()}

    # ── Per-sector accuracy/F1 breakdown (measurement only, see note above) ──
    # Small sectors produce noisy per-sector accuracy estimates the same way
    # small classes do (§16.13's MIN_SECTOR_GROUP_SIZE rationale applies
    # here too), so sectors are flagged as "low_support" below a threshold
    # rather than silently reported alongside well-supported sectors as if
    # equally reliable.
    MIN_SECTOR_SUPPORT_FOR_HEADLINE = 20
    sector_breakdown = []
    for sec in sorted(sector_true.keys()):
        y_true_sec = np.array(sector_true[sec])
        y_pred_sec = np.array(sector_pred[sec])
        n_sec = len(y_true_sec)
        sec_acc = float(accuracy_score(y_true_sec, y_pred_sec))
        sec_macro_f1 = float(f1_score(y_true_sec, y_pred_sec, average="macro", zero_division=0))
        sector_breakdown.append({
            "sector": str(sec),
            "n_test_rows": int(n_sec),
            "accuracy": round(sec_acc, 4),
            "macro_f1": round(sec_macro_f1, 4),
            "low_support": bool(n_sec < MIN_SECTOR_SUPPORT_FOR_HEADLINE),
        })

    reliable_sectors = [s for s in sector_breakdown if not s["low_support"]]
    if reliable_sectors:
        sector_accs_reliable = np.array([s["accuracy"] for s in reliable_sectors])
        sector_spread = float(sector_accs_reliable.max() - sector_accs_reliable.min())
        worst_sector = min(reliable_sectors, key=lambda s: s["accuracy"])
        best_sector = max(reliable_sectors, key=lambda s: s["accuracy"])
        if sector_spread > 0.30:
            sector_verdict = (
                f"LARGE SECTOR DISPARITY: accuracy spread across sectors with >= "
                f"{MIN_SECTOR_SUPPORT_FOR_HEADLINE} test rows is {sector_spread:.4f} "
                f"(worst: {worst_sector['sector']} at {worst_sector['accuracy']:.4f}, "
                f"best: {best_sector['sector']} at {best_sector['accuracy']:.4f}). "
                f"Sector-conditional thresholds or a sector-interaction feature may be "
                f"worth investigating for the weaker sector(s)."
            )
        elif sector_spread > 0.15:
            sector_verdict = (
                f"MODERATE SECTOR DISPARITY: accuracy spread across sectors with >= "
                f"{MIN_SECTOR_SUPPORT_FOR_HEADLINE} test rows is {sector_spread:.4f} "
                f"(worst: {worst_sector['sector']} at {worst_sector['accuracy']:.4f}, "
                f"best: {best_sector['sector']} at {best_sector['accuracy']:.4f}). "
                f"Present but not dominant; likely within normal sampling variation "
                f"given ~593 companies split across ~12 sectors."
            )
        else:
            sector_verdict = (
                f"NO MAJOR SECTOR DISPARITY: accuracy spread across sectors with >= "
                f"{MIN_SECTOR_SUPPORT_FOR_HEADLINE} test rows is only {sector_spread:.4f} "
                f"(worst: {worst_sector['sector']} at {worst_sector['accuracy']:.4f}, "
                f"best: {best_sector['sector']} at {best_sector['accuracy']:.4f}). "
                f"The existing sector-relative z-score features (compute_sector_stats) "
                f"appear to be normalising cross-sector differences adequately; "
                f"sector-conditional thresholds are not evidenced as necessary."
            )
    else:
        sector_spread = None
        sector_verdict = (
            f"INSUFFICIENT DATA: no sector had >= {MIN_SECTOR_SUPPORT_FOR_HEADLINE} "
            f"pooled test rows across all folds; per-sector accuracy cannot be reliably "
            f"compared."
        )

    lines = [
        f"cv_folds: {n_splits}",
        "split: grouped_stratified (company-level, leakage-safe)",
        "prediction_rule: plain argmax over blended softmax+ordinal probabilities "
        "(threshold multipliers and second-stage corrector removed, see report §16.26)",
        f"train_accuracy_mean: {train_accs.mean():.4f}",
        f"train_accuracy_std: {train_accs.std():.4f}",
        f"test_accuracy_mean: {accs.mean():.4f}",
        f"test_accuracy_std: {accs.std():.4f}",
        f"train_test_accuracy_gap: {acc_gap:.4f}",
        f"train_macro_f1_mean: {train_macro_f1s.mean():.4f}",
        f"test_macro_f1_mean: {macro_f1s.mean():.4f}",
        f"train_test_macro_f1_gap: {macro_gap:.4f}",
        # Kept for backward compatibility with anything parsing these keys.
        f"accuracy_mean: {accs.mean():.4f}",
        f"accuracy_std: {accs.std():.4f}",
        f"macro_f1_mean: {macro_f1s.mean():.4f}",
        f"macro_f1_std: {macro_f1s.std():.4f}",
        "",
        overfit_verdict,
    ]
    for cls in LABEL_ENCODER.classes_:
        lines.append(f"f1_{cls}_mean: {per_class_f1_means[cls]:.4f}")

    lines.append("")
    lines.append("=== Per-sector breakdown (pooled test predictions across all folds) ===")
    lines.append(sector_verdict)
    for s in sorted(sector_breakdown, key=lambda x: x["accuracy"]):
        flag = " [LOW SUPPORT]" if s["low_support"] else ""
        lines.append(
            f"  {s['sector']:<30} n={s['n_test_rows']:>4}  "
            f"accuracy={s['accuracy']:.4f}  macro_f1={s['macro_f1']:.4f}{flag}"
        )

    output = "\n".join(lines) + "\n"
    (RESULTS_DIR / "xgboost_cv_metrics.txt").write_text(output)
    print("\n" + output, flush=True)

    # ── JSON cache: consumed by predict_xgboost.py so the dashboard can show
    # the CV mean +/- std as the primary, authoritative accuracy figure
    # without re-running this (expensive) cross-validation on every request.
    cv_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cv_folds": n_splits,
        "split_strategy": "grouped_stratified",
        "prediction_rule": "argmax_blended_softmax_ordinal",
        "test_accuracy_mean": round(float(accs.mean()), 4),
        "test_accuracy_std": round(float(accs.std()), 4),
        "train_accuracy_mean": round(float(train_accs.mean()), 4),
        "train_accuracy_std": round(float(train_accs.std()), 4),
        "train_test_accuracy_gap": round(float(acc_gap), 4),
        "test_macro_f1_mean": round(float(macro_f1s.mean()), 4),
        "test_macro_f1_std": round(float(macro_f1s.std()), 4),
        "train_macro_f1_mean": round(float(train_macro_f1s.mean()), 4),
        "train_test_macro_f1_gap": round(float(macro_gap), 4),
        "per_class_f1_mean": {cls: round(v, 4) for cls, v in per_class_f1_means.items()},
        "overfit_verdict": overfit_verdict,
        "folds": fold_records,
        "sector_breakdown": sector_breakdown,
        "sector_accuracy_spread": round(sector_spread, 4) if sector_spread is not None else None,
        "sector_verdict": sector_verdict,
    }
    CV_METRICS_JSON_PATH.write_text(json.dumps(cv_summary, indent=2))
    print(f"CV metrics cached to {CV_METRICS_JSON_PATH}", flush=True)


if __name__ == "__main__":
    main()

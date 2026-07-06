"""Authoritative offline evaluation for the XGBoost credit-risk pipeline.

A single train/test split over ~593 companies is a high-variance estimate, so
this script reports **grouped, stratified k-fold cross-validation** — the
honest, stable performance number to cite in the technical report.

Key properties:
- Company-level grouping (StratifiedGroupKFold on Symbol/Name): no company
  appears in both the training and the held-out fold, so there is no
  company-level leakage.
- Per-fold preprocessing: imputation, winsorization, sector z-scores, and
  feature selection are all fit on the training fold only.
- Standard (calibrated) predictions are used — the threshold *strategy
  selector* is deliberately skipped here because it peeks at test labels
  (documented leakage), which would bias a CV estimate.

Run:  python evaluate_xgboost.py           (uses the default dataset)
      python evaluate_xgboost.py 5          (override number of folds)
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from preprocessing import (
    LABEL_ENCODER,
    RANDOM_STATE,
    DROP_COLS,
    add_interaction_features,
    apply_imputation,
    apply_zscore_from_stats,
    compute_sector_stats,
    encode_and_align,
    extract_groups,
    fit_feature_selection,
    fit_imputation,
    group_rating,
    winsorize_features,
)
from predict_xgboost import DEFAULT_DATASET_PATH, RESULTS_DIR, train_model

warnings.filterwarnings("ignore")


def preprocess_fold(X_train, X_test):
    """Fit all transforms on the training fold, apply to both (leakage-free)."""
    medians = fit_imputation(X_train)
    X_train = apply_imputation(X_train, medians)
    X_test = apply_imputation(X_test, medians)

    X_train_w, X_test_w, _ = winsorize_features(X_train, X_test)
    X_train_i = add_interaction_features(X_train_w)
    X_test_i = add_interaction_features(X_test_w)

    if "Sector" in X_train_i.columns:
        stats = compute_sector_stats(X_train_i)
        apply_zscore_from_stats([X_train_i, X_test_i], stats)

    X_train_enc, X_test_enc = encode_and_align(X_train_i, X_test_i)
    selected = fit_feature_selection(X_train_enc)
    return X_train_enc[selected], X_test_enc.reindex(columns=selected, fill_value=0)


def main():
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    df = pd.read_csv(DEFAULT_DATASET_PATH)
    y_raw = df["Rating"].apply(group_rating)
    mask = y_raw != "Unknown"
    df = df[mask].copy()
    y = LABEL_ENCODER.transform(y_raw[mask])
    groups = extract_groups(df)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    accs, macro_f1s = [], []
    per_class_f1 = {c: [] for c in LABEL_ENCODER.classes_}

    for i, (tr, te) in enumerate(sgkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        X_tr_enc, X_te_enc = preprocess_fold(X_tr, X_te)
        best_xgb, calibrated = train_model(X_tr_enc, y_tr)
        preds = calibrated.predict(X_te_enc)

        acc = accuracy_score(y_te, preds)
        macro = f1_score(y_te, preds, average="macro")
        f1_by_class = f1_score(y_te, preds, average=None, labels=[0, 1, 2, 3], zero_division=0)

        accs.append(acc)
        macro_f1s.append(macro)
        for idx, cls in enumerate(LABEL_ENCODER.classes_):
            per_class_f1[cls].append(f1_by_class[idx])

        print(f"fold {i}/{n_splits}: n_test={len(y_te)} acc={acc:.4f} macroF1={macro:.4f}", flush=True)

    accs = np.array(accs)
    macro_f1s = np.array(macro_f1s)

    lines = [
        f"cv_folds: {n_splits}",
        "split: grouped_stratified (company-level, leakage-safe)",
        "prediction: standard calibrated (no threshold strategy selector)",
        f"accuracy_mean: {accs.mean():.4f}",
        f"accuracy_std: {accs.std():.4f}",
        f"macro_f1_mean: {macro_f1s.mean():.4f}",
        f"macro_f1_std: {macro_f1s.std():.4f}",
    ]
    for cls in LABEL_ENCODER.classes_:
        vals = np.array(per_class_f1[cls])
        lines.append(f"f1_{cls}_mean: {vals.mean():.4f}")

    output = "\n".join(lines) + "\n"
    (RESULTS_DIR / "xgboost_cv_metrics.txt").write_text(output)
    print("\n" + output, flush=True)


if __name__ == "__main__":
    main()

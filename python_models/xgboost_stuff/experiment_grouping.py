"""Measurement-only experiment: how much does the target framing change accuracy?

Runs the SAME leakage-safe grouped 5-fold CV and the SAME training pipeline as
`evaluate_xgboost.py`, but under alternative label groupings:

  - binary  : Investment-Grade (AAA..BBB) vs High-Yield (BB..D)
  - three   : Investment-Grade / Speculative (BB,B) / Distressed (CCC,CC,C,D)

The current production framing is 4-class (baseline CV accuracy ~0.4603).
This script does NOT modify any product code — it only reports numbers so the
target framing can be chosen from data.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from preprocessing import RANDOM_STATE, DROP_COLS, extract_groups
from evaluate_xgboost import preprocess_fold
from predict_xgboost import DEFAULT_DATASET_PATH

warnings.filterwarnings("ignore")


def train_model_adaptive(X_train_enc, y_train):
    """Same ensemble as production, but the objective is inferred from the
    number of classes (so it works for binary as well as multiclass)."""
    n_classes = len(np.unique(y_train))
    obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
    metric = "logloss" if n_classes == 2 else "mlogloss"

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    base_model = XGBClassifier(
        objective=obj, eval_metric=metric, use_label_encoder=False,
        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    param_grid = {
        "max_depth": [5, 7, 9], "n_estimators": [150, 200],
        "learning_rate": [0.03, 0.05, 0.1], "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2], "reg_alpha": [0, 0.1, 0.3], "reg_lambda": [1, 1.5],
    }
    grid = RandomizedSearchCV(
        base_model, param_grid, n_iter=15, scoring="f1_macro", cv=3,
        n_jobs=-1, refit=True, random_state=RANDOM_STATE, verbose=0,
    )
    grid.fit(X_train_enc, y_train, sample_weight=sample_weights)
    best = grid.best_estimator_
    bp = grid.best_params_
    xgb2 = XGBClassifier(**{**bp, "objective": obj, "eval_metric": metric,
                            "random_state": RANDOM_STATE + 1, "n_jobs": -1})
    xgb3 = XGBClassifier(**{**bp, "objective": obj, "eval_metric": metric,
                            "random_state": RANDOM_STATE + 2, "n_jobs": -1})
    xgb2.fit(X_train_enc, y_train, sample_weight=sample_weights)
    xgb3.fit(X_train_enc, y_train, sample_weight=sample_weights)
    ensemble = VotingClassifier(
        estimators=[("xgb1", best), ("xgb2", xgb2), ("xgb3", xgb3)],
        voting="soft", n_jobs=-1,
    )
    ensemble.fit(X_train_enc, y_train, sample_weight=sample_weights)
    for cv_folds in (3, 2):
        try:
            calibrated = CalibratedClassifierCV(estimator=ensemble, method="isotonic", cv=cv_folds)
            calibrated.fit(X_train_enc, y_train, sample_weight=sample_weights)
            break
        except ValueError:
            if cv_folds == 2:
                raise
    return calibrated


def _norm(r):
    return str(r).strip().upper()


SCHEMES = {
    "binary (IG vs HY)": {
        "map": lambda r: 0 if _norm(r) in {"AAA", "AA", "A", "BBB"} else 1,
        "n": 2,
    },
    "three (IG / Spec / Distressed)": {
        "map": lambda r: (
            0 if _norm(r) in {"AAA", "AA", "A", "BBB"}
            else 1 if _norm(r) in {"BB", "B"}
            else 2
        ),
        "n": 3,
    },
}


def run_scheme(name, label_fn, df, groups):
    y = df["Rating"].apply(label_fn).to_numpy()
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    accs, macro_f1s = [], []
    for i, (tr, te) in enumerate(sgkf.split(X, y, groups), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]
        X_tr_enc, X_te_enc = preprocess_fold(X_tr, X_te)
        calibrated = train_model_adaptive(X_tr_enc, y_tr)
        preds = calibrated.predict(X_te_enc)
        acc = accuracy_score(y_te, preds)
        macro = f1_score(y_te, preds, average="macro")
        accs.append(acc)
        macro_f1s.append(macro)
        print(f"  [{name}] fold {i}/5: acc={acc:.4f} macroF1={macro:.4f}", flush=True)

    accs, macro_f1s = np.array(accs), np.array(macro_f1s)
    print(f"==> {name}: accuracy {accs.mean():.4f} ± {accs.std():.4f} | "
          f"macroF1 {macro_f1s.mean():.4f} ± {macro_f1s.std():.4f}\n", flush=True)
    return accs.mean(), accs.std(), macro_f1s.mean(), macro_f1s.std()


def main():
    df = pd.read_csv(DEFAULT_DATASET_PATH).copy()
    groups = extract_groups(df)

    results = {}
    for name, cfg in SCHEMES.items():
        results[name] = run_scheme(name, cfg["map"], df, groups)

    print("=" * 64)
    print(f"{'Scheme':<34}{'Accuracy':<18}{'Macro F1'}")
    print("-" * 64)
    print(f"{'4-class (current baseline)':<34}{'0.4603 ± 0.0313':<18}{'0.4181 ± 0.0308'}")
    for name, (am, as_, fm, fs) in results.items():
        print(f"{name:<34}{f'{am:.4f} ± {as_:.4f}':<18}{f'{fm:.4f} ± {fs:.4f}'}")
    print("=" * 64)


if __name__ == "__main__":
    main()

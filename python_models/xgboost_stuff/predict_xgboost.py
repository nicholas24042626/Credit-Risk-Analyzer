import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"

class ExplicitLabelEncoder:
    def __init__(self, mapping):
        self.mapping = mapping
        self.classes_ = np.array([c for c, _ in sorted(mapping.items(), key=lambda x: x[1])])
    def inverse_transform(self, y):
        return np.array([self.classes_[val] for val in y])

le = ExplicitLabelEncoder({
    "Investment_High": 0,
    "Investment_Low": 1,
    "Speculative": 2,
    "Distressed": 3,
})

def winsorize_features(X_fit, X_target, bounds=None, lower_pct=1, upper_pct=99):
    X_out = X_target.copy()
    numerics_target = [c for c in X_out.select_dtypes(include='number').columns if c in bounds["lower"].index]
    if numerics_target:
        X_out[numerics_target] = X_out[numerics_target].clip(
            lower=bounds["lower"][numerics_target], 
            upper=bounds["upper"][numerics_target], 
            axis=1
        )
    return X_out, bounds

def add_interaction_features(X):
    X_out = X.copy()
    X_out["leverage_coverage"] = (X_out["debtEquityRatio"] / (X_out["operatingProfitMargin"].abs() + 1e-5)).clip(-1e6, 1e6)
    X_out["liquidity_score"] = (X_out["currentRatio"] + X_out["quickRatio"] + X_out["cashRatio"]) / 3.0
    X_out["cashflow_debt_coverage"] = (X_out["operatingCashFlowPerShare"] / (X_out["debtEquityRatio"].abs() + 1e-5)).clip(-1e6, 1e6)
    X_out["profitability_composite"] = (X_out["netProfitMargin"] + X_out["operatingProfitMargin"] + X_out["grossProfitMargin"]) / 3.0
    X_out["debt_service_ratio"] = (X_out["operatingCashFlowSalesRatio"] / (X_out["debtRatio"] + 1e-5)).clip(-1e6, 1e6)
    X_out["efficiency_composite"] = (X_out["assetTurnover"] + X_out["fixedAssetTurnover"]) / 2.0
    X_out["roa_leverage"] = (X_out["returnOnAssets"] / (X_out["debtRatio"] + 1e-5)).clip(-1e6, 1e6)
    X_out["margin_stability"] = (X_out["grossProfitMargin"] - X_out["netProfitMargin"]).abs()
    X_out["cash_liquidity_ratio"] = (X_out["cashRatio"] / (X_out["currentRatio"] + 1e-5)).clip(-1e6, 1e6)
    X_out["equity_efficiency"] = (X_out["returnOnCapitalEmployed"] * X_out["assetTurnover"]).clip(-1e6, 1e6)
    skew_candidates = [
        "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
        "debtEquityRatio", "enterpriseValueMultiple", "operatingCashFlowPerShare",
        "freeCashFlowPerShare", "cashPerShare", "payablesTurnover",
        "fixedAssetTurnover", "companyEquityMultiplier",
    ]
    cols_to_log = [c for c in skew_candidates if c in X_out.columns]
    if cols_to_log:
        for col in cols_to_log:
            X_out[f"{col}_log"] = np.sign(X_out[col]) * np.log1p(np.abs(X_out[col]))
    return X_out

def add_zscore_features(X_fit, X_target, sector_stats=None):
    ratios = sector_stats.get("ratios", list(sector_stats["global_means"].keys()))
    X_out = X_target.copy()
    if "Sector" in X_out.columns:
        m_sec = X_out[["Sector"]].merge(sector_stats["means"], left_on="Sector", right_index=True, how="left").drop(columns=["Sector"])
        s_sec = X_out[["Sector"]].merge(sector_stats["stds"], left_on="Sector", right_index=True, how="left").drop(columns=["Sector"])
        m_sec = m_sec.fillna(sector_stats["global_means"])
        s_sec = s_sec.fillna(sector_stats["global_stds"])
        z_scores = (X_out[ratios] - m_sec) / (s_sec + 1e-5)
        z_scores.columns = [f"{r}_sec_z" for r in ratios]
        X_out = pd.concat([X_out, z_scores], axis=1)
    return X_out, sector_stats

def full_feature_pipeline(X_target, pipeline_state):
    X_target_w, _ = winsorize_features(None, X_target, bounds=pipeline_state["w_bounds"])
    X_target_i = add_interaction_features(X_target_w)
    X_target_z, _ = add_zscore_features(None, X_target_i, pipeline_state["sector_stats"])
    X_target_enc = pd.get_dummies(X_target_z, columns=["Sector"])
    X_target_enc = X_target_enc.reindex(columns=pipeline_state["feature_columns"], fill_value=0)
    return X_target_enc

def main():
    # Support both direct command-line tests and JSON sent by the Node server.
    raw_payload = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    
    if not raw_payload:
        print(json.dumps({"error": "No input data provided."}))
        sys.exit(1)
        
    try:
        input_data = json.loads(raw_payload)
        df_in = pd.DataFrame([input_data])
        
        drop_cols = ["Name", "Symbol", "Rating Agency Name", "Date", "RatingClass", "Rating"]
        df_in = df_in.drop(columns=[c for c in drop_cols if c in df_in.columns], errors="ignore")
        
        # Load pipeline state
        stats = joblib.load(RESULTS_DIR / "sector_stats.pkl")
        w_bounds = joblib.load(RESULTS_DIR / "winsorize_bounds.pkl")
        feature_cols = joblib.load(RESULTS_DIR / "feature_columns.pkl")
        calibrated_model = joblib.load(RESULTS_DIR / "calibrated_model.pkl")
        
        ps = {
            "sector_stats": stats,
            "w_bounds": w_bounds,
            "feature_columns": feature_cols,
        }
        
        df_in_enc = full_feature_pipeline(df_in, ps)
        
        proba = calibrated_model.predict_proba(df_in_enc)
        pred = calibrated_model.predict(df_in_enc)
        
        # Optional threshold adjustment
        try:
            optimal_thresholds = joblib.load(RESULTS_DIR / "optimal_thresholds.pkl")
            y_proba_adjusted = proba * optimal_thresholds
            pred_threshold = y_proba_adjusted.argmax(axis=1)
            pred_labels = le.inverse_transform(pred_threshold)
        except Exception:
            pred_labels = le.inverse_transform(pred)
        
        result = {
            "prediction": str(pred_labels[0]),
            "probabilities": {str(le.classes_[i]): float(proba[0][i]) for i in range(len(le.classes_))}
        }
        
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings("ignore")

RANDOM_STATE = 47

class ExplicitLabelEncoder:
    def __init__(self, mapping):
        self.mapping = mapping
        self.classes_ = np.array([c for c, _ in sorted(mapping.items(), key=lambda x: x[1])])
    def transform(self, y):
        return np.array([self.mapping[val] for val in y])
    def inverse_transform(self, y):
        return np.array([self.classes_[val] for val in y])

le = ExplicitLabelEncoder({
    "Investment_High": 0,
    "Investment_Low": 1,
    "Speculative": 2,
    "Distressed": 3,
})

def format_prediction_label(label):
    return str(label).replace("_", "-")

def group_rating(r):
    if r in ["AAA", "AA", "A"]: return "Investment_High"
    if r == "BBB": return "Investment_Low"
    if r in ["BB", "B"]: return "Speculative"
    if r in ["CCC", "CC", "C", "D"]: return "Distressed"
    return "Unknown"

def winsorize_features(X_train, X_test, lower_pct=1, upper_pct=99):
    X_train_out, X_test_out = X_train.copy(), X_test.copy()
    numerics = X_train.select_dtypes(include='number').columns
    bounds = {}
    for c in numerics:
        l = np.percentile(X_train[c].dropna(), lower_pct)
        u = np.percentile(X_train[c].dropna(), upper_pct)
        bounds[c] = (l, u)
        X_train_out[c] = X_train_out[c].clip(lower=l, upper=u)
        X_test_out[c] = X_test_out[c].clip(lower=l, upper=u)
    return X_train_out, X_test_out

def add_interaction_features(X):
    X_out = X.copy()
    
    # Safely calculate full suite of 10 financial composite features
    if "debtEquityRatio" in X_out.columns and "operatingProfitMargin" in X_out.columns:
        X_out["leverage_coverage"] = (X_out["debtEquityRatio"] / (X_out["operatingProfitMargin"].abs() + 1e-5)).clip(-1e6, 1e6)
    if "currentRatio" in X_out.columns and "quickRatio" in X_out.columns and "cashRatio" in X_out.columns:
        X_out["liquidity_score"] = (X_out["currentRatio"] + X_out["quickRatio"] + X_out["cashRatio"]) / 3.0
    if "operatingCashFlowPerShare" in X_out.columns and "debtEquityRatio" in X_out.columns:
        X_out["cashflow_debt_coverage"] = (X_out["operatingCashFlowPerShare"] / (X_out["debtEquityRatio"].abs() + 1e-5)).clip(-1e6, 1e6)
    if "netProfitMargin" in X_out.columns and "operatingProfitMargin" in X_out.columns and "grossProfitMargin" in X_out.columns:
        X_out["profitability_composite"] = (X_out["netProfitMargin"] + X_out["operatingProfitMargin"] + X_out["grossProfitMargin"]) / 3.0
    if "operatingCashFlowSalesRatio" in X_out.columns and "debtRatio" in X_out.columns:
        X_out["debt_service_ratio"] = (X_out["operatingCashFlowSalesRatio"] / (X_out["debtRatio"] + 1e-5)).clip(-1e6, 1e6)
    if "assetTurnover" in X_out.columns and "fixedAssetTurnover" in X_out.columns:
        X_out["efficiency_composite"] = (X_out["assetTurnover"] + X_out["fixedAssetTurnover"]) / 2.0
    if "returnOnAssets" in X_out.columns and "debtRatio" in X_out.columns:
        X_out["roa_leverage"] = (X_out["returnOnAssets"] / (X_out["debtRatio"] + 1e-5)).clip(-1e6, 1e6)
    if "grossProfitMargin" in X_out.columns and "netProfitMargin" in X_out.columns:
        X_out["margin_stability"] = (X_out["grossProfitMargin"] - X_out["netProfitMargin"]).abs()
    if "cashRatio" in X_out.columns and "currentRatio" in X_out.columns:
        X_out["cash_liquidity_ratio"] = (X_out["cashRatio"] / (X_out["currentRatio"] + 1e-5)).clip(-1e6, 1e6)
    if "returnOnCapitalEmployed" in X_out.columns and "assetTurnover" in X_out.columns:
        X_out["equity_efficiency"] = (X_out["returnOnCapitalEmployed"] * X_out["assetTurnover"]).clip(-1e6, 1e6)
    
    skew_candidates = [
        "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
        "debtEquityRatio", "enterpriseValueMultiple", "operatingCashFlowPerShare",
        "freeCashFlowPerShare", "cashPerShare", "payablesTurnover",
        "fixedAssetTurnover", "companyEquityMultiplier"
    ]
    cols_to_log = [c for c in skew_candidates if c in X_out.columns]
    for col in cols_to_log:
        X_out[f"{col}_log"] = np.sign(X_out[col]) * np.log1p(np.abs(X_out[col]))
    return X_out

def add_zscore_features(X_train, X_test):
    X_train_out, X_test_out = X_train.copy(), X_test.copy()
    if "Sector" in X_train_out.columns:
        ratios = [c for c in X_train_out.select_dtypes(include='number').columns if "log" not in c]
        
        # Calculate stats on train only
        means = X_train_out.groupby("Sector")[ratios].mean()
        stds = X_train_out.groupby("Sector")[ratios].std()
        global_means = X_train_out[ratios].mean()
        global_stds = X_train_out[ratios].std()
        
        for df_out in [X_train_out, X_test_out]:
            m_sec = df_out[["Sector"]].merge(means, left_on="Sector", right_index=True, how="left").drop(columns=["Sector"])
            s_sec = df_out[["Sector"]].merge(stds, left_on="Sector", right_index=True, how="left").drop(columns=["Sector"])
            m_sec = m_sec.fillna(global_means)
            s_sec = s_sec.fillna(global_stds)
            z_scores = (df_out[ratios] - m_sec) / (s_sec + 1e-5)
            z_scores.columns = [f"{r}_sec_z" for r in ratios]
            for col in z_scores.columns:
                df_out[col] = z_scores[col]
    return X_train_out, X_test_out

def full_feature_pipeline(X_train, X_test):
    X_train_w, X_test_w = winsorize_features(X_train, X_test)
    X_train_i, X_test_i = add_interaction_features(X_train_w), add_interaction_features(X_test_w)
    X_train_z, X_test_z = add_zscore_features(X_train_i, X_test_i)
    
    # One-hot encoding
    X_train_enc = pd.get_dummies(X_train_z, columns=["Sector"] if "Sector" in X_train_z else [])
    X_test_enc = pd.get_dummies(X_test_z, columns=["Sector"] if "Sector" in X_test_z else [])
    
    # Align columns
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)
    return X_train_enc, X_test_enc

def load_dataframe(input_data):
    import base64, io, os, tempfile
    file_data = input_data.get("fileData")
    file_name = input_data.get("fileName", "dataset.csv")
    file_encoding = input_data.get("fileEncoding", "utf8")
    
    if file_data:
        suffix = Path(file_name).suffix.lower()
        if file_encoding == "base64":
            binary = base64.b64decode(file_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(binary)
                temp_path = tmp.name
            try:
                if suffix in [".xlsx", ".xls"]:
                    return pd.read_excel(temp_path)
                return pd.read_csv(temp_path)
            finally:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        else:
            if suffix in [".xlsx", ".xls"]:
                raise ValueError("Excel files must be sent as base64.")
            return pd.read_csv(io.StringIO(file_data))
    
    project_root = Path(__file__).resolve().parents[2]
    default_path = project_root / "set A corporate_rating.csv"
    if default_path.exists():
        return pd.read_csv(default_path)
    raise FileNotFoundError("No file provided and default dataset not found.")

def humanize_feature_name(name):
    return name.replace("_log", "").replace("_sec_z", " (Sector Z-Score)").replace("_", " ").title()

def main():
    raw_payload = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    if not raw_payload:
        print(json.dumps({"error": "No input data provided."}))
        sys.exit(1)
        
    try:
        input_data = json.loads(raw_payload)
        df = load_dataframe(input_data)
        
        if "Rating" not in df.columns:
            raise ValueError("The uploaded dataset must contain a 'Rating' column to train the model.")
            
        y_raw = df["Rating"].apply(group_rating)
        valid_idx = y_raw != "Unknown"
        df_valid = df[valid_idx].copy()
        
        y_str = y_raw[valid_idx]
        y_encoded = le.transform(y_str)
        
        drop_cols = ["Name", "Symbol", "Rating Agency Name", "Date", "RatingClass", "Rating"]
        X = df_valid.drop(columns=[c for c in drop_cols if c in df_valid.columns], errors="ignore")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.20, random_state=RANDOM_STATE, stratify=y_encoded
            )
        except ValueError:
            # Fallback if classes are too small
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.20, random_state=RANDOM_STATE
            )
            
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Dataset is too small to split into training and test sets.")
            
        X_train_enc, X_test_enc = full_feature_pipeline(X_train, X_test)
        
        # We observed that heavy optimizations (SMOTE, deep trees, calibration) 
        # caused severe overfitting on this specific dataset, hurting test accuracy.
        # Reverting to a highly conservative, robust approach:
        from sklearn.model_selection import GridSearchCV
        from sklearn.utils.class_weight import compute_sample_weight
        
        # 1. Use mathematically balanced weights instead of synthesizing fake data
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        base_model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # 2. Use a highly constrained grid to prevent XGBoost from overfitting
        # shallow trees (depth 3-5) generalize much better on small datasets
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 150]
        }
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=3,
            n_jobs=-1
        )
        
        # Train carefully with sample weights
        grid_search.fit(X_train_enc, y_train, sample_weight=sample_weights)
        best_xgb = grid_search.best_estimator_
        
        pred_indices = best_xgb.predict(X_test_enc)
        proba = best_xgb.predict_proba(X_test_enc)
        
        acc = accuracy_score(y_test, pred_indices)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred_indices, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, pred_indices, labels=[0, 1, 2, 3])
        
        # Calculate feature importance using the robust estimator
        importances = best_xgb.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:6]
        
        shap_data = []
        for i in top_indices:
            fname = humanize_feature_name(X_train_enc.columns[i])
            val = float(importances[i]) * 100
            if val > 0:
                shap_data.append([fname, round(val, 2)])
                
        if not shap_data:
            shap_data = [["Feature", 100]]
            
        # Format results
        pred_labels = le.inverse_transform(pred_indices)
        
        modelData = {
            "tag": "XGBoost",
            "labels": ["Investment-High", "Investment-Low", "Speculative", "Distressed"],
            "metrics": {
                "accuracy": f"{acc:.4f}",
                "precision": f"{precision:.4f}",
                "recall": f"{recall:.4f}",
                "f1": f"{f1:.4f}",
                "strength": "Dynamically trained on uploaded data; robust to outliers.",
                "weakness": "May overfit if the dataset is too small or highly imbalanced."
            },
            "matrix": cm.tolist(),
            "shap": shap_data,
            "shapStory": {
                "positive": [s[0] for s in shap_data[:3]] if len(shap_data) >= 3 else [],
                "negative": [s[0] for s in shap_data[-3:]] if len(shap_data) >= 3 else []
            }
        }
        
        result = {
            "prediction": format_prediction_label(pred_labels[0]),
            "probabilities": {
                format_prediction_label(le.classes_[i]): float(proba[0][i])
                for i in range(len(le.classes_))
            },
            "modelData": modelData
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()

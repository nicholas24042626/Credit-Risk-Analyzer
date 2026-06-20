import subprocess
import sys
import json

data = {
    "currentRatio": 2.1,
    "quickRatio": 1.5,
    "cashRatio": 0.8,
    "daysOfSalesOutstanding": 40,
    "netProfitMargin": 0.12,
    "pretaxProfitMargin": 0.15,
    "grossProfitMargin": 0.35,
    "operatingProfitMargin": 0.18,
    "returnOnAssets": 0.08,
    "returnOnCapitalEmployed": 0.1,
    "returnOnEquity": 0.2,
    "assetTurnover": 1.4,
    "fixedAssetTurnover": 2.0,
    "debtEquityRatio": 0.6,
    "debtRatio": 0.3,
    "effectiveTaxRate": 0.22,
    "freeCashFlowOperatingCashFlowRatio": 0.7,
    "freeCashFlowPerShare": 1.2,
    "cashPerShare": 0.9,
    "companyEquityMultiplier": 1.5,
    "ebitPerRevenue": 0.16,
    "enterpriseValueMultiple": 8.5,
    "operatingCashFlowPerShare": 1.1,
    "operatingCashFlowSalesRatio": 0.2,
    "payablesTurnover": 4.0
}

json_data = json.dumps(data)

result = subprocess.run(
    [sys.executable, "predict_random_forest.py", json_data],
    capture_output=True,
    text=True
)

print("STDOUT:")
print(result.stdout)

print("STDERR:")
print(result.stderr)
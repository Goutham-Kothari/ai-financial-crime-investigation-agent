
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "paysim.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "paysim_random_forest.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "high_risk_transactions.csv"

print("Loading trained model...")
model = joblib.load(MODEL_PATH)

print("Loading sample transactions...")
df = pd.read_csv(DATA_PATH, nrows=10000)

features = df[["step", "type", "amount"]]
df["fraud_risk_score"] = model.predict_proba(features)[:, 1]

df["risk_level"] = pd.cut(
    df["fraud_risk_score"],
    bins=[-0.001, 0.30, 0.70, 1.001],
    labels=["Low", "Medium", "High"],
)

high_risk = df.sort_values(
    "fraud_risk_score", ascending=False
).head(100)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
high_risk.to_csv(REPORT_PATH, index=False)

print("\nTop 10 highest-risk transactions:")
print(
    high_risk[
        ["type", "amount", "fraud_risk_score", "risk_level", "isFraud"]
    ].head(10).to_string(index=False)
)

print(f"\nReport saved to: {REPORT_PATH}")


from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"

print("Loading PaySim data...")
df = pd.read_csv(DATA_PATH)

df["origin_balance_decrease"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)
df["origin_amount_difference"] = (
    df["origin_balance_decrease"] - df["amount"]
).abs()
df["origin_balance_matches_amount"] = (
    df["origin_amount_difference"] <= 0.01
)

# Compare fraud rates within each transaction type and balance pattern.
summary = (
    df.groupby(
        ["type", "origin_balance_matches_amount"],
        observed=True,
    )
    .agg(
        transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum"),
        fraud_rate=("isFraud", "mean"),
    )
    .reset_index()
)

summary["fraud_rate_percent"] = summary["fraud_rate"] * 100

print("\n===== FRAUD RATE BY TYPE AND BALANCE PATTERN =====")
print(
    summary[
        [
            "type",
            "origin_balance_matches_amount",
            "transactions",
            "fraud_cases",
            "fraud_rate_percent",
        ]
    ].round(4).to_string(index=False)
)

output_path = ROOT / "reports" / "balance_rule_analysis.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(output_path, index=False)

print(f"\nReport saved to: {output_path}")

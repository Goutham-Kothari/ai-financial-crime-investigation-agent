
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"
REPORT_PATH = ROOT / "reports" / "fraud_by_time.csv"

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

# Divide simulation steps into 10 chronological groups.
df["time_group"] = pd.qcut(
    df["step"].rank(method="first"),
    q=10,
    labels=False,
)

summary = df.groupby("time_group").agg(
    first_step=("step", "min"),
    last_step=("step", "max"),
    transactions=("isFraud", "size"),
    fraud_cases=("isFraud", "sum"),
    fraud_rate=("isFraud", "mean"),
    fraud_with_balance_match=(
        "origin_balance_matches_amount",
        lambda s: int((s & df.loc[s.index, "isFraud"].eq(1)).sum()),
    ),
).reset_index()

summary["fraud_rate_percent"] = summary["fraud_rate"] * 100

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(REPORT_PATH, index=False)

print("\n===== FRAUD PATTERNS OVER TIME =====")
print(
    summary[
        [
            "time_group",
            "first_step",
            "last_step",
            "transactions",
            "fraud_cases",
            "fraud_rate_percent",
            "fraud_with_balance_match",
        ]
    ].round(4).to_string(index=False)
)

print(f"\nReport saved to: {REPORT_PATH}")

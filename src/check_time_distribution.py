
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"

print("Loading PaySim data...")
df = pd.read_csv(DATA_PATH)

print("\n===== FRAUD RATE BY SIMULATION STEP =====")

summary = (
    df.groupby("step")
    .agg(
        transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum"),
    )
    .reset_index()
)

summary["fraud_rate_percent"] = (
    summary["fraud_cases"] / summary["transactions"] * 100
)

print("First 10 steps:")
print(summary.head(10).to_string(index=False))

print("\nLast 20 steps:")
print(summary.tail(20).to_string(index=False))

print("\n===== OVERALL CHECK =====")
print(f"Total transactions: {len(df):,}")
print(f"Total fraud cases: {int(df['isFraud'].sum()):,}")
print(f"First step: {df['step'].min()}")
print(f"Last step: {df['step'].max()}")

print("\nLargest fraud-rate steps:")
print(
    summary.sort_values("fraud_rate_percent", ascending=False)
    .head(10)
    .to_string(index=False)
)

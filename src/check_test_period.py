
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"

df = pd.read_csv(DATA_PATH)

# Use the same chronological boundary as the earlier time-split test.
cutoff_step = int(sorted(df["step"].unique())[int(len(df["step"].unique()) * 0.8)])

train = df[df["step"] < cutoff_step]
test = df[df["step"] >= cutoff_step]

for name, part in [("TRAINING PERIOD", train), ("TEST PERIOD", test)]:
    print(f"\n===== {name} =====")
    print(f"Step range: {part['step'].min()} to {part['step'].max()}")
    print(f"Transactions: {len(part):,}")
    print(f"Fraud cases: {int(part['isFraud'].sum()):,}")
    print(f"Fraud rate: {part['isFraud'].mean():.4%}")

    print("\nTransaction type counts by fraud label:")
    print(
        pd.crosstab(
            part["type"],
            part["isFraud"],
            margins=True,
        ).to_string()
    )

    print("\nFraud rate by transaction type:")
    print(
        part.groupby("type")["isFraud"]
        .agg(["count", "sum", "mean"])
        .to_string()
    )

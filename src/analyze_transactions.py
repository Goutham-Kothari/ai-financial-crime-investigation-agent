
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "paysim.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load a sample for initial analysis.
df = pd.read_csv(DATA_PATH, nrows=10000)

print("\n1. FRAUD VS NON-FRAUD")
print(df["isFraud"].value_counts())

print("\n2. FRAUD BY TRANSACTION TYPE")
fraud_by_type = (
    df.groupby("type")["isFraud"]
    .agg(["count", "sum", "mean"])
    .rename(columns={
        "count": "total_transactions",
        "sum": "fraud_transactions",
        "mean": "fraud_rate"
    })
)

fraud_by_type["fraud_rate"] *= 100
print(fraud_by_type.round(3))

print("\n3. AMOUNT COMPARISON")
print(
    df.groupby("isFraud")["amount"]
    .agg(["count", "mean", "median", "max"])
    .round(2)
)

print("\n4. EXAMPLE FRAUD TRANSACTIONS")
fraud_examples = df[df["isFraud"] == 1].head(10)

columns_to_show = [
    "step", "type", "amount",
    "nameOrig", "nameDest", "isFraud"
]
print(fraud_examples[columns_to_show].to_string(index=False))

# Save analysis results.
fraud_by_type.to_csv(REPORTS_DIR / "fraud_by_type.csv")
fraud_examples[columns_to_show].to_csv(
    REPORTS_DIR / "fraud_examples.csv",
    index=False
)

print("\nAnalysis complete. Results saved in reports/.")

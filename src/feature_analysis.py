
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"
REPORT_PATH = ROOT / "reports" / "feature_analysis.csv"

print("Loading PaySim data...")
df = pd.read_csv(DATA_PATH)

# Derive features from transaction details and balances.
df["origin_balance_decrease"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

df["destination_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

df["origin_amount_difference"] = (
    df["origin_balance_decrease"] - df["amount"]
).abs()

df["origin_balance_mismatch"] = (
    df["origin_amount_difference"] > 0.01
).astype(int)

# Use isFraud only to compare feature patterns for analysis.
summary = df.groupby("isFraud").agg(
    transaction_count=("amount", "size"),
    average_amount=("amount", "mean"),
    average_origin_balance_decrease=("origin_balance_decrease", "mean"),
    average_destination_balance_change=("destination_balance_change", "mean"),
    origin_balance_mismatch_rate=("origin_balance_mismatch", "mean"),
).reset_index()

summary["class_name"] = summary["isFraud"].map({
    0: "Not fraud",
    1: "Fraud",
})

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(REPORT_PATH, index=False)

print("\n===== FEATURE COMPARISON =====")
print(summary.round(4).to_string(index=False))
print(f"\nReport saved to: {REPORT_PATH}")
print(
    "\nNote: isFraud is used only to compare groups in this analysis, "
    "not as a model input."
)

print("\n===== FEATURE COMPARISON =====")

df["origin_balance_decrease"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

df["destination_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

df["origin_balance_mismatch"] = (
    (df["origin_balance_decrease"] - df["amount"]).abs() > 0.01
)

summary = df.groupby("isFraud").agg(
    transaction_count=("amount", "size"),
    average_amount=("amount", "mean"),
    average_origin_balance_decrease=(
        "origin_balance_decrease", "mean"
    ),
    average_destination_balance_change=(
        "destination_balance_change", "mean"
    ),
    origin_balance_mismatch_rate=(
        "origin_balance_mismatch", "mean"
    ),
)

summary["origin_balance_mismatch_rate"] *= 100
print(summary.round(3).to_string())

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

df["origin_balance_mismatch"] = (
    df["origin_amount_difference"] > 0.01
)

print("\n===== FRAUD RATE BY TRANSACTION TYPE =====")
print(
    df.groupby("type", observed=True)["isFraud"]
    .agg(transaction_count="size", fraud_count="sum", fraud_rate="mean")
    .sort_values("fraud_rate", ascending=False)
    .to_string()
)

print("\n===== FRAUD RATE BY BALANCE MISMATCH =====")
print(
    df.groupby("origin_balance_mismatch")["isFraud"]
    .agg(transaction_count="size", fraud_count="sum", fraud_rate="mean")
    .to_string()
)

print("\n===== FRAUD RATE BY AMOUNT MATCH =====")
df["origin_amount_matches"] = (
    df["origin_amount_difference"] <= 0.01
)

print(
    df.groupby("origin_amount_matches")["isFraud"]
    .agg(transaction_count="size", fraud_count="sum", fraud_rate="mean")
    .to_string()
)


from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"

df = pd.read_csv(DATA_PATH)

# Use the same chronological holdout boundary as before.
steps = sorted(df["step"].unique())
cutoff_step = steps[int(len(steps) * 0.8)]
test = df[df["step"] >= cutoff_step].copy()

test["origin_balance_decrease"] = (
    test["oldbalanceOrg"] - test["newbalanceOrig"]
)
test["origin_amount_difference"] = (
    test["origin_balance_decrease"] - test["amount"]
).abs()

# Simple, interpretable baseline:
# flag a transaction if it is TRANSFER or CASH_OUT
# and the origin balance decrease matches the amount.
test["rule_prediction"] = (
    test["type"].isin(["TRANSFER", "CASH_OUT"])
    & (test["origin_amount_difference"] <= 0.01)
).astype(int)

y_true = test["isFraud"]
y_pred = test["rule_prediction"]

print("===== RULE-BASED BASELINE =====")
print(f"Test transactions: {len(test):,}")
print(f"Fraud cases: {int(y_true.sum()):,}")
print(f"Alerts generated: {int(y_pred.sum()):,}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))

print(
    "\nNote: This is a simple baseline for comparison, "
    "not a validated production fraud rule."
)

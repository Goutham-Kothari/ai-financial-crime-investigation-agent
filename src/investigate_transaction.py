
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "paysim.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "paysim_random_forest.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "investigation_cases.csv"


def investigate(row, risk_score):
    reasons = []

    # Transaction type is contextual evidence, not proof of fraud.
    if row["type"] in ["TRANSFER", "CASH_OUT"]:
        reasons.append(
            f"Transaction type is {row['type']}; "
            "these types contained fraud cases in our dataset analysis."
        )

    # Check whether the sender's balance decreased by the transaction amount.
    sender_balance_change = (
        row["oldbalanceOrg"] - row["newbalanceOrig"]
    )
    balance_difference = abs(sender_balance_change - row["amount"])

    if balance_difference < 0.01:
        reasons.append(
            "Sender balance decreased by approximately the transaction amount."
        )
    else:
        reasons.append(
            "Sender balance change differs from the transaction amount; "
            "review the discrepancy."
        )

    # Check the destination account balance change.
    destination_balance_change = (
        row["newbalanceDest"] - row["oldbalanceDest"]
    )

    if row["type"] in ["TRANSFER", "CASH_OUT"]:
        if destination_balance_change == 0:
            reasons.append(
                "Destination balance did not increase in this record; "
                "verify the destination-side transaction details."
            )

    if risk_score >= 0.70:
        risk_level = "High"
        action = (
            "Prioritize analyst review. Verify both account balances, "
            "transaction history, and related transactions."
        )
    elif risk_score >= 0.30:
        risk_level = "Medium"
        action = "Review transaction context and related account activity."
    else:
        risk_level = "Low"
        action = "Retain for routine monitoring."

    return {
        "step": int(row["step"]),
        "transaction_type": row["type"],
        "amount": float(row["amount"]),
        "origin_account": row["nameOrig"],
        "destination_account": row["nameDest"],
        "origin_balance_before": float(row["oldbalanceOrg"]),
        "origin_balance_after": float(row["newbalanceOrig"]),
        "destination_balance_before": float(row["oldbalanceDest"]),
        "destination_balance_after": float(row["newbalanceDest"]),
        "risk_score": round(float(risk_score), 4),
        "risk_level": risk_level,
        "investigation_reasons": " ".join(reasons),
        "recommended_action": action,
    }


print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading transactions...")
df = pd.read_csv(DATA_PATH, nrows=10000)

features = df[["step", "type", "amount"]]
scores = model.predict_proba(features)[:, 1]

cases = []

for (_, row), score in zip(df.iterrows(), scores):
    if score >= 0.70:
        cases.append(investigate(row, score))

case_df = pd.DataFrame(cases)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
case_df.to_csv(REPORT_PATH, index=False)

print(f"\nHigh-risk cases generated: {len(case_df)}")
print(f"Saved report: {REPORT_PATH}")

if not case_df.empty:
    print("\nSample investigation case:")
    print(case_df.iloc[0].to_string())

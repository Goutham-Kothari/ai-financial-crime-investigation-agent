
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "paysim.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "paysim_random_forest.joblib"

@lru_cache(maxsize=1)
def load_resources():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Load the dataset and trained model
    data = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    return data, model


def build_evidence(row):
    amount = float(row["amount"])
    origin_before = float(row["oldbalanceOrg"])
    origin_after = float(row["newbalanceOrig"])
    destination_before = float(row["oldbalanceDest"])
    destination_after = float(row["newbalanceDest"])

    origin_decrease = origin_before - origin_after
    destination_change = destination_after - destination_before
    amount_difference = abs(origin_decrease - amount)

    findings = []

    if amount_difference < 0.01:
        findings.append({
            "code": "ORIGIN_BALANCE_MATCH",
            "finding": (
                "The origin balance decrease matches the transaction amount."
            ),
            "severity": "informational",
        })
    else:
        findings.append({
            "code": "ORIGIN_BALANCE_MISMATCH",
            "finding": (
                "The origin balance decrease differs from the transaction "
                "amount by {:.2f}.".format(amount_difference)
            ),
            "severity": "review",
        })

    if destination_change == 0:
        findings.append({
            "code": "NO_DESTINATION_BALANCE_CHANGE",
            "finding": (
                "No destination balance change is recorded for this row."
            ),
            "severity": "informational",
        })
    else:
        findings.append({
            "code": "DESTINATION_BALANCE_CHANGED",
            "finding": (
                "The destination balance changed by {:.2f}.".format(
                    destination_change
                )
            ),
            "severity": "informational",
        })

    return {
        "origin_balance_before": origin_before,
        "origin_balance_after": origin_after,
        "origin_balance_decrease": round(origin_decrease, 2),
        "destination_balance_before": destination_before,
        "destination_balance_after": destination_after,
        "destination_balance_change": round(destination_change, 2),
        "origin_amount_difference": round(amount_difference, 2),
        "amount_matches_origin_decrease": amount_difference < 0.01,
        "findings": findings,
        "interpretation_note": (
            "Balance mismatches are review signals, not proof of fraud. "
            "Interpret them in the context of transaction type and "
            "the dataset's recording rules."
        ),
    }

def investigate_dataset_transaction(transaction_id: int):
    data, model = load_resources()

    # Use the row number as a simple dataset transaction ID.
    if transaction_id < 1 or transaction_id > len(data):
        raise ValueError(
            f"Transaction ID must be between 1 and {len(data)}."
        )

    row = data.iloc[transaction_id - 1]

    amount = float(row["amount"])
    origin_decrease = float(
        row["oldbalanceOrg"] - row["newbalanceOrig"]
    )
    destination_change = float(
        row["newbalanceDest"] - row["oldbalanceDest"]
    )
    origin_amount_difference = abs(origin_decrease - amount)

    # These features must match Model V2's training features.
    features = pd.DataFrame([{
        "step": int(row["step"]),
        "type": str(row["type"]),
        "amount": amount,
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
        "origin_balance_decrease": origin_decrease,
        "destination_balance_change": destination_change,
        "origin_amount_difference": origin_amount_difference,
    }])

    risk_score = float(model.predict_proba(features)[0, 1])

    if risk_score >= 0.70:
        risk_level = "High"
        actions = [
            "Prioritize for human review.",
            "Check related account activity.",
            "Verify available supporting transaction records.",
        ]
    elif risk_score >= 0.30:
        risk_level = "Medium"
        actions = [
            "Review the transaction and related account activity.",
        ]
    else:
        risk_level = "Low"
        actions = [
            "Continue routine monitoring.",
            "Do not interpret a low score as proof of safety.",
        ]

    # Ground-truth label is for offline evaluation only.
    actual_label = int(row["isFraud"])

    
    # Build a concise, evidence-based investigation summary.
    finding_items = build_evidence(row)["findings"]
    review_findings = [
        item["finding"]
        for item in finding_items
        if item["severity"] == "review"
    ]

    if risk_level == "High":
        summary = (
            "The model assigned a high risk score. Prioritize this "
            "transaction for human review."
        )
    elif risk_level == "Medium":
        summary = (
            "The model assigned a medium risk score. Review the "
            "transaction alongside its supporting evidence."
        )
    else:
        summary = (
            "The model assigned a low risk score. Continue appropriate "
            "monitoring; a low score does not establish that the "
            "transaction is safe."
        )

    if review_findings:
        summary += " Evidence requiring review: " + " ".join(review_findings)


    return {
        "dataset_transaction_id": transaction_id,
        "transaction": {
            "step": int(row["step"]),
            "type": str(row["type"]),
            "amount": amount,
            "origin_account": str(row["nameOrig"]),
            "destination_account": str(row["nameDest"]),
        },
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "investigation_summary": summary,
        "evidence": build_evidence(row),
        "recommended_actions": actions,
        "offline_evaluation_only": {
            "isFraud": actual_label,
            "note": (
                "Dataset ground-truth label for offline testing only; "
                "not used as a model input or investigation evidence."
            ),
        },
        "warning": "A model score is an estimate, not proof of fraud.",
    }





if __name__ == "__main__":
    try:
        transaction_id = int(input("Enter dataset transaction ID (1-based): "))
        result = investigate_dataset_transaction(transaction_id)

        print("\n===== TRANSACTION INVESTIGATION =====")
        print(pd.Series(result).to_json(indent=2, force_ascii=False))

        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)

        output_path = reports_dir / "dataset_investigation.json"
        output_path.write_text(
            pd.Series(result).to_json(indent=2, force_ascii=False),
            encoding="utf-8",
        )

        print(f"\nReport saved to: {output_path}")

    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}")

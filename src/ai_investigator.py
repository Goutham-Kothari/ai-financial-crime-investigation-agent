
from pathlib import Path
import json

import joblib
import ollama
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "paysim.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "paysim_random_forest.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "ai_investigation_case.txt"

LLM_MODEL = "llama3.2:1b"
HIGH_RISK_THRESHOLD = 0.70


def build_evidence(row, risk_score):
    # Calculate facts in Python, not in the LLM.
    sender_change = float(row["oldbalanceOrg"] - row["newbalanceOrig"])
    destination_change = float(row["newbalanceDest"] - row["oldbalanceDest"])

    return {
        "transaction_step": int(row["step"]),
        "transaction_type": str(row["type"]),
        "amount": round(float(row["amount"]), 2),
        "origin_account": str(row["nameOrig"]),
        "destination_account": str(row["nameDest"]),
        "origin_balance_before": round(float(row["oldbalanceOrg"]), 2),
        "origin_balance_after": round(float(row["newbalanceOrig"]), 2),
        "destination_balance_before": round(float(row["oldbalanceDest"]), 2),
        "destination_balance_after": round(float(row["newbalanceDest"]), 2),
        "origin_balance_decrease": round(sender_change, 2),
        "destination_balance_change": round(destination_change, 2),
        "amount_vs_origin_decrease_difference": round(
            abs(sender_change - float(row["amount"])), 2
        ),
        "model_risk_score": round(float(risk_score), 4),
        "risk_level": (
            "High" if risk_score >= 0.70
            else "Medium" if risk_score >= 0.30
            else "Low"
        ),
        "balance_note": (
            "The destination balance did not change in this recorded row."
            if destination_change == 0
            else "The destination balance changed in this recorded row."
        ),
        "label_for_offline_evaluation_only": int(row["isFraud"]),
    }


def main():
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Loading transactions...")
    df = pd.read_csv(DATA_PATH, nrows=10000)

    features = df[["step", "type", "amount"]]
    scores = model.predict_proba(features)[:, 1]
    df["risk_score"] = scores

    candidates = (
        df[df["risk_score"] >= HIGH_RISK_THRESHOLD]
        .sort_values("risk_score", ascending=False)
        .head(20)
        .copy()
    )

    if candidates.empty:
        print("No transactions exceeded the high-risk threshold.")
        return

    candidates.insert(0, "case_number", range(1, len(candidates) + 1))

    print("\nAvailable high-risk transactions:")
    print(
        candidates[
            ["case_number", "type", "amount", "risk_score"]
        ].to_string(index=False)
    )

    try:
        choice = int(input("\nEnter a case number to investigate: "))
    except ValueError:
        print("Please enter a valid whole number.")
        return

    selected = candidates[candidates["case_number"] == choice]

    if selected.empty:
        print("Invalid case number. Run the script again and choose a listed number.")
        return

    row = selected.iloc[0]
    evidence = build_evidence(row, row["risk_score"])

    # Keep the dataset's fraud label out of the LLM prompt.
    evidence_for_llm = {
        key: value
        for key, value in evidence.items()
        if key != "label_for_offline_evaluation_only"
    }

    prompt = f"""
You are helping an analyst review a synthetic transaction dataset.

Write a concise report using ONLY the evidence provided below.

Rules:
- Do not invent dates, laws, thresholds, customer history, or facts.
- The transaction step is a dataset step, not a calendar date.
- The model score is an estimate, not proof of fraud.
- A zero balance or a particular transaction type alone does not prove
  money laundering or any other crime.
- Use the supplied balance calculations exactly as given.
- Clearly label possible explanations as hypotheses.
- If evidence is insufficient, say so.
- Recommend checks that an analyst can perform.

Evidence:
{json.dumps(evidence_for_llm, indent=2)}

Use these headings:
1. Case Summary
2. Verified Facts
3. Potential Concerns
4. Limitations and Unknowns
5. Recommended Investigation Steps
6. Preliminary Assessment
"""

    print("\nGenerating report with local LLM...")
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce cautious, factual transaction review "
                        "reports. Never turn a hypothesis into a fact."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:
        print("\nCould not connect to Ollama.")
        print("Check that Ollama is running and the model is installed.")
        print(f"Details: {exc}")
        return

    report = response["message"]["content"]

    validation_note = """
HUMAN REVIEW REQUIRED
This AI-generated report is decision support, not a fraud determination.
Check every statement against the source data before acting.
"""
    report = report + "\n\n" + validation_note

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

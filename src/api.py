

from src.dataset_investigator import investigate_dataset_transaction

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "paysim_random_forest_v2.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="AI Financial Crime Investigation Agent",
    description="Transaction risk scoring and evidence-based investigation API.",
    version="0.2.0",
)

ALLOWED_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}



from pydantic import BaseModel, Field, model_validator


class Transaction(BaseModel):
    step: int = Field(ge=1)
    type: str
    amount: float = Field(gt=0)

    origin_account: str | None = None
    destination_account: str | None = None

    oldbalanceOrg: float | None = Field(default=None, ge=0)
    newbalanceOrig: float | None = Field(default=None, ge=0)
    oldbalanceDest: float | None = Field(default=None, ge=0)
    newbalanceDest: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_balance_fields(self):
        balance_fields = [
            self.oldbalanceOrg,
            self.newbalanceOrig,
            self.oldbalanceDest,
            self.newbalanceDest,
        ]

        if any(value is not None for value in balance_fields):
            if any(value is None for value in balance_fields):
                raise ValueError(
                    "Provide all four balance fields together, or omit all four."
                )

        return self


class DatasetInvestigationRequest(BaseModel):
    transaction_id: int = Field(ge=1)


class BatchInvestigationRequest(BaseModel):
    transaction_ids: list[int] = Field(min_length=1, max_length=100)




def score_transaction(transaction: Transaction) -> float:
    balance_values = [
        transaction.oldbalanceOrg,
        transaction.newbalanceOrig,
        transaction.oldbalanceDest,
        transaction.newbalanceDest,
    ]

    if any(value is None for value in balance_values):
        raise HTTPException(
            status_code=422,
            detail=(
                "Model V2 requires all four balance fields: "
                "oldbalanceOrg, newbalanceOrig, oldbalanceDest, "
                "and newbalanceDest."
            ),
        )

    origin_balance_decrease = (
        transaction.oldbalanceOrg - transaction.newbalanceOrig
    )

    destination_balance_change = (
        transaction.newbalanceDest - transaction.oldbalanceDest
    )

    origin_amount_difference = abs(
        origin_balance_decrease - transaction.amount
    )

    features = pd.DataFrame([{
        "step": transaction.step,
        "type": transaction.type,
        "amount": transaction.amount,
        "oldbalanceOrg": transaction.oldbalanceOrg,
        "newbalanceOrig": transaction.newbalanceOrig,
        "oldbalanceDest": transaction.oldbalanceDest,
        "newbalanceDest": transaction.newbalanceDest,
        "origin_balance_decrease": origin_balance_decrease,
        "destination_balance_change": destination_balance_change,
        "origin_amount_difference": origin_amount_difference,
    }])

    return float(model.predict_proba(features)[0, 1])


def get_risk_level(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.30:
        return "Medium"
    return "Low"


@app.get("/")
def home():
    return {
        "message": "Financial Crime Investigation API is running",
        "docs": "/docs",
        "endpoints": ["/score", "/investigate"],
    }


@app.post("/score")
def score_endpoint(transaction: Transaction):
    if transaction.type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported transaction type. Use: {sorted(ALLOWED_TYPES)}",
        )

    risk_score = score_transaction(transaction)

    return {
        "transaction": transaction.model_dump(),
        "risk_score": round(risk_score, 4),
        "risk_level": get_risk_level(risk_score),
        "warning": "A model score is an estimate, not proof of fraud.",
    }


@app.post("/investigate")
def investigate_transaction(transaction: Transaction):
    if transaction.type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported transaction type. Use: {sorted(ALLOWED_TYPES)}",
        )

    risk_score = score_transaction(transaction)
    risk_level = get_risk_level(risk_score)

    evidence = [
        {
            "finding": "Model risk score",
            "details": (
                f"The model assigned a risk score of "
                f"{risk_score:.4f} ({risk_level} risk)."
            ),
        },
        {
            "finding": "Transaction details",
            "details": (
                f"Type: {transaction.type}; "
                f"amount: {transaction.amount:.2f}; "
                f"dataset step: {transaction.step}."
            ),
        },
    ]

    if transaction.origin_account:
        evidence.append({
            "finding": "Origin account",
            "details": f"Provided account identifier: {transaction.origin_account}",
        })

    if transaction.destination_account:
        evidence.append({
            "finding": "Destination account",
            "details": (
                f"Provided account identifier: {transaction.destination_account}"
            ),
        })

    balance_note = "Balance evidence was not provided."

    if (
        transaction.oldbalanceOrg is not None
        and transaction.newbalanceOrig is not None
    ):
        decrease = (
            transaction.oldbalanceOrg - transaction.newbalanceOrig
        )
        balance_note = (
            f"Origin balance decreased by {decrease:.2f}; "
            f"transaction amount is {transaction.amount:.2f}."
        )

        evidence.append({
            "finding": "Origin balance movement",
            "details": balance_note,
        })

        if abs(decrease - transaction.amount) > 0.01:
            evidence.append({
                "finding": "Balance discrepancy to review",
                "details": (
                    "The origin balance decrease does not exactly match "
                    "the transaction amount. Review the transaction context."
                ),
            })

    if (
        transaction.oldbalanceDest is not None
        and transaction.newbalanceDest is not None
    ):
        change = (
            transaction.newbalanceDest - transaction.oldbalanceDest
        )
        evidence.append({
            "finding": "Destination balance movement",
            "details": f"Destination balance changed by {change:.2f}.",
        })

    if risk_score >= 0.70:
        actions = [
            "Prioritize this transaction for human review.",
            "Verify transaction details and account activity.",
            "Check related transactions and available supporting records.",
        ]
    elif risk_score >= 0.30:
        actions = [
            "Review the transaction alongside account activity.",
            "Check for related unusual transactions.",
        ]
    else:
        actions = [
            "Continue routine monitoring.",
            "Do not treat a low score as proof that the transaction is safe.",
        ]

    return {
        "case_summary": {
            "transaction_type": transaction.type,
            "amount": transaction.amount,
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
        },
        "evidence": evidence,
        "balance_note": balance_note,
        "recommended_actions": actions,
        "limitations": [
            "The model uses only step, type, and amount for scoring.",
            "Balance findings use only the values supplied in this request.",
            "A risk score is not proof of fraud; human review is required.",
        ],
    }

@app.post("/investigate-dataset")
def investigate_dataset(request: DatasetInvestigationRequest):
    try:
        return investigate_dataset_transaction(request.transaction_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/investigate-batch")
def investigate_batch(request: BatchInvestigationRequest):
    # Reject invalid IDs before investigating anything.
    if any(transaction_id < 1 for transaction_id in request.transaction_ids):
        raise HTTPException(
            status_code=422,
            detail="Every transaction ID must be at least 1.",
        )

    if len(set(request.transaction_ids)) != len(request.transaction_ids):
        raise HTTPException(
            status_code=422,
            detail="Duplicate transaction IDs are not allowed.",
        )

    results = []

    try:
        for transaction_id in request.transaction_ids:
            result = investigate_dataset_transaction(transaction_id)

            # Keep the dataset's true label out of analyst-facing results.
            result.pop("offline_evaluation_only", None)

            results.append(result)

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    results.sort(
        key=lambda item: item["risk_score"],
        reverse=True,
    )

    return {
        "transactions_analyzed": len(results),
        "results_sorted_by_risk": results,
        "warning": (
            "Risk scores are estimates, not proof of fraud. "
            "Human review is required."
        ),
    }


@app.post("/risk-summary")
def risk_summary(request: BatchInvestigationRequest):
    if any(transaction_id < 1 for transaction_id in request.transaction_ids):
        raise HTTPException(
            status_code=422,
            detail="Every transaction ID must be at least 1.",
        )

    if len(set(request.transaction_ids)) != len(request.transaction_ids):
        raise HTTPException(
            status_code=422,
            detail="Duplicate transaction IDs are not allowed.",
        )

    results = []

    try:
        for transaction_id in request.transaction_ids:
            result = investigate_dataset_transaction(transaction_id)
            result.pop("offline_evaluation_only", None)
            results.append(result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))

    scores = [item["risk_score"] for item in results]
    counts = {
        "High": sum(item["risk_level"] == "High" for item in results),
        "Medium": sum(item["risk_level"] == "Medium" for item in results),
        "Low": sum(item["risk_level"] == "Low" for item in results),
    }

    return {
        "transactions_analyzed": len(results),
        "risk_level_counts": counts,
        "average_risk_score": round(sum(scores) / len(scores), 4),
        "highest_risk_transactions": sorted(
            results,
            key=lambda item: item["risk_score"],
            reverse=True,
        )[:5],
        "warning": "Model scores are estimates, not proof of fraud.",
    }

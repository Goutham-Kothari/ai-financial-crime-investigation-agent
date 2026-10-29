
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "paysim.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "paysim_random_forest.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_evaluation.json"

FEATURES = ["step", "type", "amount"]
TARGET = "isFraud"


def main():
    print("Loading PaySim dataset...")
    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    # Reproduce the original stratified 80/20 test split.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Evaluating model on held-out test data...")
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not fraud", "Fraud"],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    results = {
        "test_transactions": int(len(y_test)),
        "fraud_transactions": int(y_test.sum()),
        "fraud_rate": round(float(y_test.mean()), 6),
        "classification_report": report,
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "threshold": 0.5,
        "note": (
            "Evaluation uses isFraud as the ground-truth target only. "
            "Model inputs are step, type, and amount."
        ),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n===== MODEL EVALUATION =====")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not fraud", "Fraud"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(matrix)
    print(f"Average precision: {results['average_precision']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

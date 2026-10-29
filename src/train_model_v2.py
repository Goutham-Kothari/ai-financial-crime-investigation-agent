
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"
MODEL_PATH = ROOT / "models" / "paysim_random_forest_v2.joblib"
REPORT_PATH = ROOT / "reports" / "model_evaluation_v2.json"

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "origin_balance_decrease",
    "destination_balance_change",
    "origin_amount_difference",
]

CATEGORICAL_FEATURES = ["type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_features(df):
    result = df.copy()

    result["origin_balance_decrease"] = (
        result["oldbalanceOrg"] - result["newbalanceOrig"]
    )
    result["destination_balance_change"] = (
        result["newbalanceDest"] - result["oldbalanceDest"]
    )
    result["origin_amount_difference"] = (
        result["origin_balance_decrease"] - result["amount"]
    ).abs()

    return result


def main():
    print("Loading PaySim dataset...")
    data = pd.read_csv(DATA_PATH)

    print("Creating balance features...")
    data = add_features(data)

    X = data[FEATURES]
    y = data["isFraud"].astype(int)

    # Keep a reproducible stratified holdout for comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    print("Training Model V2. This may take some time...")
    pipeline.fit(X_train, y_train)

    print("Evaluating Model V2...")
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    report = {
        "test_transactions": int(len(y_test)),
        "fraud_transactions": int(y_test.sum()),
        "threshold": 0.5,
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not fraud", "Fraud"],
            output_dict=True,
            zero_division=0,
        ),
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
        "features": FEATURES,
        "note": (
            "isFraud is the target only, never an input feature. "
            "Threshold is 0.5. Compare these results with Model V1."
        ),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n===== MODEL V2 RESULTS =====")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not fraud", "Fraud"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(matrix)
    print(f"Average precision: {report['average_precision']:.4f}")
    print(f"ROC-AUC: {report['roc_auc']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Evaluation saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

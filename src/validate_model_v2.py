
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"

def evaluate_model(name, features, categorical, numeric, X_train, X_test, y_train, y_test):
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print(f"\nTraining {name}...")
    model.fit(X_train[features], y_train)

    probabilities = model.predict_proba(X_test[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n===== {name} =====")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Not fraud", "Fraud"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(f"Average precision: {average_precision_score(y_test, probabilities):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

def main():
    print("Loading PaySim dataset...")
    data = pd.read_csv(DATA_PATH)

    data["origin_balance_decrease"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["destination_balance_change"] = data["newbalanceDest"] - data["oldbalanceDest"]
    data["origin_amount_difference"] = (
        data["origin_balance_decrease"] - data["amount"]
    ).abs()

    y = data["isFraud"].astype(int)

    # Use the same split for both experiments.
    train_idx, test_idx = train_test_split(
        data.index,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # V1-style features.
    basic_features = ["step", "type", "amount"]
    basic_numeric = ["step", "amount"]

    # Balance features used by V2.
    v2_features = [
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "origin_balance_decrease",
        "destination_balance_change",
        "origin_amount_difference",
    ]
    v2_numeric = [f for f in v2_features if f != "type"]

    X_train = data.loc[train_idx]
    X_test = data.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    evaluate_model(
        "BASIC MODEL",
        basic_features,
        ["type"],
        basic_numeric,
        X_train, X_test, y_train, y_test,
    )

    evaluate_model(
        "BALANCE-FEATURE MODEL",
        v2_features,
        ["type"],
        v2_numeric,
        X_train, X_test, y_train, y_test,
    )

if __name__ == "__main__":
    main()

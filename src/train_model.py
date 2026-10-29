
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "paysim.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading PaySim dataset...")

df = pd.read_csv(DATA_PATH)

# Basic validation
required_columns = ["step", "type", "amount", "isFraud"]
missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required_columns)

# Exclude balance fields and isFlaggedFraud for this baseline.
X = df[["step", "type", "amount"]]
y = df["isFraud"].astype(int)

print("Total records:", len(df))
print("Fraud records:", int(y.sum()))
print("Fraud percentage:", round(y.mean() * 100, 4), "%")

# Stratification keeps the fraud proportion similar in each split.
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
            "transaction_type",
            OneHotEncoder(handle_unknown="ignore"),
            ["type"],
        ),
        ("numeric", "passthrough", ["step", "amount"]),
    ]
)

model = Pipeline(
    steps=[
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
    ]
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Predict probabilities so we can later tune the alert threshold.
probabilities = model.predict_proba(X_test)[:, 1]

# Initial threshold. This is a starting point, not a final decision rule.
threshold = 0.50
predictions = (probabilities >= threshold).astype(int)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not fraud", "Fraud"],
        zero_division=0,
    )
)

print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nFraud-class precision:",
      round(precision_score(y_test, predictions, zero_division=0), 4))
print("Fraud-class recall:",
      round(recall_score(y_test, predictions, zero_division=0), 4))
print("Fraud-class F1:",
      round(f1_score(y_test, predictions, zero_division=0), 4))
print("Average precision:",
      round(average_precision_score(y_test, probabilities), 4))

# Save the fitted pipeline for reuse.
model_path = MODELS_DIR / "paysim_random_forest.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")

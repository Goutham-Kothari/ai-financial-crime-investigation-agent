
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"

print("Loading PaySim data...")
df = pd.read_csv(DATA_PATH)

# Sort transactions by simulation step.
df = df.sort_values("step").reset_index(drop=True)

# Build the same balance features used by V2.
df["origin_balance_decrease"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)
df["destination_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)
df["origin_amount_difference"] = (
    df["origin_balance_decrease"] - df["amount"]
).abs()

numeric_features = [
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
categorical_features = ["type"]
features = numeric_features + categorical_features
target = "isFraud"

# Train on the earlier 80% of unique simulation steps;
# evaluate on the later 20%.
steps = sorted(df["step"].unique())
split_index = int(len(steps) * 0.8)
cutoff_step = steps[split_index]

train = df[df["step"] < cutoff_step]
test = df[df["step"] >= cutoff_step]

print(f"Training rows: {len(train):,}")
print(f"Testing rows:  {len(test):,}")
print(f"Training steps end before step {cutoff_step}")
print(f"Test fraud cases: {int(test[target].sum()):,}")

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

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

print("\nTraining model...")
model.fit(train[features], train[target])

print("Scoring later transactions...")
probabilities = model.predict_proba(test[features])[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print("\n===== TIME-BASED EVALUATION =====")
print(f"Average precision: {average_precision_score(test[target], probabilities):.4f}")
print(f"ROC-AUC: {roc_auc_score(test[target], probabilities):.4f}")
print(f"Fraud precision: {precision_score(test[target], predictions, zero_division=0):.4f}")
print(f"Fraud recall: {recall_score(test[target], predictions, zero_division=0):.4f}")
print("Confusion matrix:")
print(confusion_matrix(test[target], predictions))

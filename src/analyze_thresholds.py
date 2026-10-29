
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paysim.csv"
MODEL_PATH = ROOT / "models" / "paysim_random_forest.joblib"

print("Loading dataset...")
data = pd.read_csv(DATA_PATH)

features = ["step", "type", "amount"]
X = data[features]
y = data["isFraud"].astype(int)

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Calculating probabilities...")
probabilities = model.predict_proba(X_test)[:, 1]

thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

rows = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    rows.append({
        "threshold": threshold,
        "fraud_precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "fraud_recall": round(
            recall_score(y_test, predictions, zero_division=0), 4
        ),
        "fraud_f1": round(
            f1_score(y_test, predictions, zero_division=0), 4
        ),
        "alerts_generated": int(predictions.sum()),
        "fraud_cases_detected": int(
            ((predictions == 1) & (y_test.to_numpy() == 1)).sum()
        ),
        "false_alerts": int(
            ((predictions == 1) & (y_test.to_numpy() == 0)).sum()
        ),
    })

results = pd.DataFrame(rows)

print("\n===== THRESHOLD COMPARISON =====")
print(results.to_string(index=False))

output_path = ROOT / "reports" / "threshold_comparison.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")

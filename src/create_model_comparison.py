
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "model_comparison.csv"

results = pd.DataFrame([
    {
        "model": "Rule-based baseline",
        "precision": 0.2243,
        "recall": 0.9952,
        "false_positives": 5693,
        "false_negatives": 8,
        "alerts": 7339,
    },
    {
        "model": "V2 Random Forest",
        "precision": 1.0000,
        "recall": 1.0000,
        "false_positives": 0,
        "false_negatives": 0,
        "alerts": 1654,
    },
])

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(REPORT_PATH, index=False)

print("===== MODEL COMPARISON =====")
print(results.to_string(index=False))
print(f"\nReport saved to: {REPORT_PATH}")
print(
    "\nImportant: These results are from the PaySim simulation "
    "and should not be interpreted as real-world performance."
)


from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "paysim_random_forest_v2.joblib"

print("Loading V2 model...")
model = joblib.load(MODEL_PATH)

# Get the trained Random Forest and preprocessing pipeline.
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
importance = classifier.feature_importances_

results = pd.DataFrame({
    "feature": feature_names,
    "importance": importance,
})

results = results.sort_values("importance", ascending=False)

print("\n===== TOP 15 FEATURES =====")
print(results.head(15).to_string(index=False))

output_path = ROOT / "reports" / "feature_importance.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output_path, index=False)

print(f"\nReport saved to: {output_path}")

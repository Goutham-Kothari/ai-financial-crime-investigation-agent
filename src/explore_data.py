
from pathlib import Path
import pandas as pd

# Locate the project and dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "paysim.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Place paysim.csv inside the data folder."
    )

# Read a small sample first.
df = pd.read_csv(DATA_PATH, nrows=10000)

print("=" * 50)
print("PAYSIM DATASET EXPLORATION")
print("=" * 50)

print("\nSample shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five transactions:")
print(df.head())

print("\nTransaction types:")
print(df["type"].value_counts())

print("\nFraud label counts:")
print(df["isFraud"].value_counts())

print("\nFraud percentage in this sample:")
print(round(df["isFraud"].mean() * 100, 4), "%")

print("\nMissing values:")
print(df.isnull().sum())


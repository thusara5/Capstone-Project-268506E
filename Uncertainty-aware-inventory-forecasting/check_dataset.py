
from pathlib import Path
import pandas as pd

# Locate CSV files in the raw data directory
DATA_DIR = Path("data/raw")
csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV files found in data/raw/. "
        "Download and extract the Kaggle dataset first."
    )

for file_path in csv_files:
    print("\n" + "=" * 60)
    print(f"FILE: {file_path.name}")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"\nDataset shape: {df.shape}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nNumerical summary:")
    print(df.describe(include="number").to_string())

    print("\nDataset inspection completed.")

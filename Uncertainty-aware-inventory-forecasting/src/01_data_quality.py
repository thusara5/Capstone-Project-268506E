
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/sales_data.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Run this script from your project root."
    )

df = pd.read_csv(DATA_PATH)

# 1. Convert the date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

print("\n--- DATE RANGE ---")
print("Start date:", df["Date"].min())
print("End date:", df["Date"].max())
print("Unique dates:", df["Date"].nunique())
print("Invalid dates:", df["Date"].isna().sum())

# 2. Store and product coverage
print("\n--- STORE AND PRODUCT COVERAGE ---")
print("Unique stores:", df["Store ID"].nunique())
print("Unique products:", df["Product ID"].nunique())
print("Unique store-product pairs:",
      df[["Store ID", "Product ID"]].drop_duplicates().shape[0])

# 3. Check logical duplicates
keys = ["Date", "Store ID", "Product ID"]
duplicate_keys = df.duplicated(subset=keys, keep=False)

print("\n--- LOGICAL DUPLICATES ---")
print("Rows with repeated date-store-product keys:",
      duplicate_keys.sum())

if duplicate_keys.any():
    print(df.loc[duplicate_keys, keys].sort_values(keys).head(10))

# 4. Check observations per date
daily_counts = df.groupby("Date").size()

print("\n--- DAILY RECORD COUNTS ---")
print("Minimum records per date:", daily_counts.min())
print("Maximum records per date:", daily_counts.max())
print("Dates with fewer/more records than the median:",
      (daily_counts != daily_counts.median()).sum())

# 5. Check missing calendar dates
all_dates = pd.date_range(
    start=df["Date"].min(),
    end=df["Date"].max(),
    freq="D"
)
missing_dates = all_dates.difference(df["Date"].dropna().unique())

print("\n--- CALENDAR COVERAGE ---")
print("Missing calendar dates:", len(missing_dates))
if len(missing_dates):
    print("First missing dates:", list(missing_dates[:10]))

# 6. Demand statistics
print("\n--- DEMAND DISTRIBUTION ---")
print(df["Demand"].describe())
print("Zero-demand rows:", (df["Demand"] == 0).sum())
print("Negative-demand rows:", (df["Demand"] < 0).sum())

# 7. Numerical correlations for investigation
print("\n--- NUMERICAL CORRELATION WITH DEMAND ---")
numeric_cols = df.select_dtypes(include="number").columns
print(
    df[numeric_cols]
    .corr()["Demand"]
    .sort_values(ascending=False)
    .to_string()
)

# 8. Save a summary for future reference
Path("results").mkdir(parents=True, exist_ok=True)

summary = pd.DataFrame({
    "Check": [
        "Rows",
        "Columns",
        "Unique dates",
        "Unique stores",
        "Unique products",
        "Repeated date-store-product rows",
        "Missing calendar dates",
        "Missing values"
    ],
    "Result": [
        len(df),
        len(df.columns),
        df["Date"].nunique(),
        df["Store ID"].nunique(),
        df["Product ID"].nunique(),
        int(duplicate_keys.sum()),
        len(missing_dates),
        int(df.isna().sum().sum())
    ]
})

summary.to_csv("results/data_quality_summary.csv", index=False)
print("\nSaved results/data_quality_summary.csv")

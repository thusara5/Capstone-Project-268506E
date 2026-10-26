
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
DATA_PATH = Path("data/raw/sales_data.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], errors="raise")

required_columns = [
    "Date", "Store ID", "Product ID", "Category",
    "Demand", "Units Sold", "Price", "Discount",
    "Promotion", "Epidemic"
]

missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df = df.sort_values(["Date", "Store ID", "Product ID"])

# --------------------------------------------------
# 3. Descriptive statistics
# --------------------------------------------------
print("\n--- DEMAND SUMMARY ---")
print(df["Demand"].describe())

print("\n--- DEMAND BY CATEGORY ---")
print(
    df.groupby("Category")["Demand"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(2)
)

print("\n--- DEMAND BY STORE ---")
print(
    df.groupby("Store ID")["Demand"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

# Save descriptive summaries
df["Month"] = df["Date"].dt.to_period("M").astype(str)

df.groupby("Category")["Demand"].agg(
    ["count", "mean", "median", "std", "min", "max"]
).round(2).to_csv(RESULTS_DIR / "demand_by_category.csv")

df.groupby("Store ID")["Demand"].agg(
    ["count", "mean", "median", "std"]
).round(2).to_csv(RESULTS_DIR / "demand_by_store.csv")

# --------------------------------------------------
# 4. Chart 1: Overall demand distribution
# --------------------------------------------------
plt.figure(figsize=(9, 5))
sns.histplot(df["Demand"], bins=40, kde=True)
plt.title("Distribution of Retail Demand")
plt.xlabel("Demand")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_demand_distribution.png", dpi=150)
plt.close()

# --------------------------------------------------
# 5. Chart 2: Average demand over time
# Aggregate across stores and products per date
# --------------------------------------------------
daily = df.groupby("Date", as_index=False)["Demand"].mean()

plt.figure(figsize=(12, 5))
sns.lineplot(data=daily, x="Date", y="Demand")
plt.title("Average Daily Demand Across All Store-Product Pairs")
plt.xlabel("Date")
plt.ylabel("Mean demand per pair")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_daily_demand.png", dpi=150)
plt.close()

# --------------------------------------------------
# 6. Chart 3: Monthly average demand
# --------------------------------------------------
monthly = df.groupby("Month", as_index=False)["Demand"].mean()

plt.figure(figsize=(13, 5))
sns.lineplot(data=monthly, x="Month", y="Demand", marker="o")
plt.title("Monthly Average Demand")
plt.xlabel("Month")
plt.ylabel("Mean demand per observation")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_monthly_demand.png", dpi=150)
plt.close()

# --------------------------------------------------
# 7. Chart 4: Demand by category
# --------------------------------------------------
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Category", y="Demand")
plt.title("Demand Distribution by Product Category")
plt.xlabel("Category")
plt.ylabel("Demand")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_demand_by_category.png", dpi=150)
plt.close()

# --------------------------------------------------
# 8. Chart 5: Correlation between numeric variables
# --------------------------------------------------
numeric = df.select_dtypes(include="number")
correlations = numeric.corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlations,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_correlation_matrix.png", dpi=150)
plt.close()

# --------------------------------------------------
# 9. Chart 6: Units sold versus demand
# --------------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Units Sold",
    y="Demand",
    alpha=0.25,
    s=15
)
plt.title("Units Sold versus Demand")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "06_units_sold_vs_demand.png", dpi=150)
plt.close()

print("\nEDA completed.")
print(f"Charts saved to: {FIGURES_DIR}")
print(f"Summaries saved to: {RESULTS_DIR}")

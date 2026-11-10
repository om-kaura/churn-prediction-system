"""
Exploratory Data Analysis — IBM Telco Customer Churn Dataset

Cleans the raw data, prints headline churn statistics, and generates the
distribution / relationship plots used to motivate the modeling choices in
train_model.py. Run this before train_model.py — it doesn't depend on it,
but it's the natural first step in the pipeline.

Usage:
    python eda.py
"""
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_validation import validate_raw

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "telco_churn.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

sns.set_theme(style="whitegrid")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and apply the minimal cleaning every downstream step needs."""
    df = pd.read_csv(path)
    validate_raw(df)  # raises DataValidationError listing every problem found
    # TotalCharges is stored as a string and has 11 blank values (all tenure == 0,
    # i.e. brand-new customers who haven't been billed yet) — coerce to numeric and
    # impute with the median rather than dropping the rows.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def run_eda(df: pd.DataFrame, out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Overall churn rate
    churn_rate = df["Churn"].value_counts(normalize=True) * 100
    print("Overall churn rate:\n", churn_rate)

    plt.figure(figsize=(5, 5))
    df["Churn"].value_counts().plot.pie(autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"])
    plt.ylabel("")
    plt.title("Overall Churn Distribution")
    plt.savefig(os.path.join(out_dir, "01_churn_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Churn by contract type
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Contract", hue="Churn")
    plt.title("Churn by Contract Type")
    plt.savefig(os.path.join(out_dir, "02_churn_by_contract.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Tenure distribution by churn
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30)
    plt.title("Tenure Distribution by Churn Status")
    plt.savefig(os.path.join(out_dir, "03_tenure_by_churn.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Monthly charges by churn
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    plt.title("Monthly Charges by Churn Status")
    plt.savefig(os.path.join(out_dir, "04_monthlycharges_by_churn.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Churn by internet service type
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="InternetService", hue="Churn")
    plt.title("Churn by Internet Service Type")
    plt.savefig(os.path.join(out_dir, "05_churn_by_internet.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Payment method vs churn
    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, y="PaymentMethod", hue="Churn")
    plt.title("Churn by Payment Method")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06_churn_by_payment.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 7. Correlation heatmap (numeric + encoded churn)
    df_corr = df.copy()
    df_corr["Churn_num"] = df_corr["Churn"].map({"Yes": 1, "No": 0})
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_num"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_corr[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.savefig(os.path.join(out_dir, "07_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 8. Senior citizen churn rate
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="SeniorCitizen", hue="Churn")
    plt.title("Churn by Senior Citizen Status")
    plt.savefig(os.path.join(out_dir, "08_churn_by_senior.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Summary stats table
    summary = df.groupby("Churn").agg(
        avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_total_charges=("TotalCharges", "mean"),
        count=("Churn", "count"),
    ).round(2)
    summary.to_csv(os.path.join(out_dir, "churn_summary_stats.csv"))
    print("\nSummary stats by churn status:\n", summary)
    print("\nAll 8 EDA plots + summary stats saved to", out_dir)


if __name__ == "__main__":
    run_eda(load_data())

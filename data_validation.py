"""
Data validation — fail loudly on bad data *before* it reaches training.

Plain pandas on purpose (no pandera / Great Expectations): the dataset has one
table and ~20 columns, so a dependency would add more weight than it removes.
`validate_raw` is called by eda.load_data(), so every path that loads the CSV
(EDA, training, tests) gets checked. `build_dataset_summary` produces the
dataset facts the dashboard's About page shows, so you can show a teacher
exactly what data was used.
"""
import json
import os

import pandas as pd

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

_YES_NO = {"Yes", "No"}
_ADDON = {"Yes", "No", "No internet service"}
CATEGORICAL_VALUES = {
    "gender": {"Female", "Male"},
    "Partner": _YES_NO, "Dependents": _YES_NO, "PhoneService": _YES_NO,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": _ADDON, "OnlineBackup": _ADDON, "DeviceProtection": _ADDON,
    "TechSupport": _ADDON, "StreamingTV": _ADDON, "StreamingMovies": _ADDON,
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": {"Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"},
    "Churn": _YES_NO,
}


class DataValidationError(ValueError):
    """Raised with *every* problem found, not just the first."""


def validate_raw(df: pd.DataFrame) -> None:
    errors = []

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        errors.append(f"missing columns: {missing}")
    if extra:
        errors.append(f"unexpected columns: {extra}")
    if missing:  # nothing further can be checked reliably
        raise DataValidationError("; ".join(errors))

    if len(df) < 1000:
        errors.append(f"only {len(df)} rows — expected the full dataset (>= 1000)")
    if df["customerID"].duplicated().any():
        errors.append(f"{int(df['customerID'].duplicated().sum())} duplicate customerID values")

    for col, allowed in CATEGORICAL_VALUES.items():
        bad = set(df[col].dropna().unique()) - allowed
        if bad:
            errors.append(f"{col}: unexpected values {sorted(map(str, bad))}")
        if df[col].isna().any():
            errors.append(f"{col}: {int(df[col].isna().sum())} null values")

    if not set(df["SeniorCitizen"].unique()) <= {0, 1}:
        errors.append("SeniorCitizen must be 0/1")

    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    if tenure.isna().any() or (tenure < 0).any() or (tenure > 120).any():
        errors.append("tenure must be a number between 0 and 120")
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    if monthly.isna().any() or (monthly < 0).any() or (monthly > 1000).any():
        errors.append("MonthlyCharges must be a number between 0 and 1000")

    # Known quirk of this dataset: TotalCharges is blank for brand-new customers
    # (tenure == 0). Blanks anywhere else mean something is wrong.
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")
    bad_total = total.isna() & (tenure != 0)
    if bad_total.any():
        errors.append(f"TotalCharges is blank/non-numeric for {int(bad_total.sum())} customers with tenure > 0")
    if (total < 0).any():
        errors.append("TotalCharges has negative values")

    if "Churn" in df.columns and len(df):
        rate = (df["Churn"] == "Yes").mean()
        if not 0.05 <= rate <= 0.60:
            errors.append(f"churn rate {rate:.1%} is outside the plausible 5%-60% range")

    if errors:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(errors))


def build_dataset_summary(df: pd.DataFrame, file: str = "data/telco_churn.csv") -> dict:
    numeric = {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}
    columns = []
    for c in df.columns:
        s = df[c]
        blank = int((s.astype(str).str.strip() == "").sum())
        columns.append({
            "name": c,
            "kind": "id" if c == "customerID" else "number" if c in numeric else "category",
            "unique": int(s.nunique()),
            "missing": int(s.isna().sum()) + blank,
            "examples": [str(v) for v in s.dropna().unique()[:3]],
        })
    churned = int((df["Churn"] == "Yes").sum())
    return {
        "source": "IBM Telco Customer Churn (public sample dataset)",
        "file": file,
        "rows": int(len(df)),
        "column_count": int(df.shape[1]),
        "churned": churned,
        "retained": int(len(df)) - churned,
        "churn_rate": round(churned / len(df), 4),
        "columns": columns,
        "sample": json.loads(df.head(8).to_json(orient="records")),
    }


def write_dataset_summary(df: pd.DataFrame, out_dir: str) -> str:
    path = os.path.join(out_dir, "dataset_summary.json")
    with open(path, "w") as f:
        json.dump(build_dataset_summary(df), f)
    return path

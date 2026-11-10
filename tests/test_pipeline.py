"""
Pipeline tests — the checks that guard the *machine-learning* side, which API tests
can't see: data validation, artifact consistency, and two tripwires for mistakes
this project actually made or nearly made:

  * data leakage   -> an impossibly good AUC (SMOTE-before-split gave 0.93 here)
  * miscalibration -> probabilities inflated by class weighting (mean 0.40 vs 0.265)

They read the saved artifacts in outputs/ and rebuild the exact held-out split
(seed 42), so they run in seconds and never retrain.
"""
import importlib
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import data_validation as dv
from eda import DATA_PATH, load_data

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "outputs")
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]


def _json(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def holdout():
    """The exact 20% test split train_model.py used (same seed, stratification, encoding)."""
    df = load_data().drop(columns=["customerID"])
    y = (df.pop("Churn") == "Yes").astype(int)
    for col, le in joblib.load(os.path.join(OUT, "label_encoders.pkl")).items():
        df[col] = le.transform(df[col])
    X = df[joblib.load(os.path.join(OUT, "feature_columns.pkl"))]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def _raw_scores(name, X):
    if name == "XGBoost":
        m = xgb.XGBClassifier()
        m.load_model(os.path.join(OUT, "model_xgboost.json"))
    else:
        m = joblib.load(os.path.join(OUT, "model_random_forest.pkl" if name == "Random Forest" else "model_logistic_regression.pkl"))
    if name == "Logistic Regression":
        X = pd.DataFrame(joblib.load(os.path.join(OUT, "scaler.pkl")).transform(X), columns=X.columns)
    return m.predict_proba(X)[:, 1]


# ------------------------------ data validation ------------------------------
def test_shipped_dataset_passes_validation(raw):
    dv.validate_raw(raw)


def test_validation_reports_every_problem_not_just_the_first(raw):
    bad = raw.copy()
    bad.loc[0, "Contract"] = "Lifetime"
    bad.loc[1, "Churn"] = "Maybe"
    bad.loc[2, "tenure"] = -3
    with pytest.raises(dv.DataValidationError) as err:
        dv.validate_raw(bad)
    for needle in ("Contract", "Churn", "tenure"):
        assert needle in str(err.value)


def test_validation_rejects_a_missing_column(raw):
    with pytest.raises(dv.DataValidationError, match="missing columns"):
        dv.validate_raw(raw.drop(columns=["tenure"]))


def test_validation_rejects_duplicate_customer_ids(raw):
    with pytest.raises(dv.DataValidationError, match="duplicate"):
        dv.validate_raw(pd.concat([raw, raw.head(3)]))


def test_blank_total_charges_is_only_tolerated_for_brand_new_customers(raw):
    bad = raw.copy()
    bad.loc[bad.index[bad["tenure"] > 0][0], "TotalCharges"] = " "
    with pytest.raises(dv.DataValidationError, match="TotalCharges"):
        dv.validate_raw(bad)


# --------------------------- artifact consistency ---------------------------
def test_label_encoders_match_the_validation_vocabulary():
    enc = joblib.load(os.path.join(OUT, "label_encoders.pkl"))
    assert set(enc) == set(dv.CATEGORICAL_VALUES) - {"Churn"}
    for col, le in enc.items():
        assert set(le.classes_) == dv.CATEGORICAL_VALUES[col], col


def test_dataset_summary_matches_the_csv(raw):
    s = _json("dataset_summary.json")
    assert s["rows"] == len(raw)
    assert abs(s["churn_rate"] - (raw["Churn"] == "Yes").mean()) < 1e-3
    assert {c["name"] for c in s["columns"]} == set(raw.columns)


def test_winner_was_chosen_by_cross_validation_not_test_score():
    comp = pd.read_csv(os.path.join(OUT, "model_comparison.csv"))
    info = _json("best_model_info.json")
    assert info["best_model"] == comp.loc[comp["CV_ROC_AUC"].idxmax(), "Model"]
    assert "CV" in info["selection_criterion"]


def test_threshold_table_agrees_with_the_reported_metrics():
    td = _json("threshold_data.json")
    comp = pd.read_csv(os.path.join(OUT, "model_comparison.csv")).set_index("Model")
    for name in MODEL_NAMES:
        for _, tp, fp, tn, fn in td["models"][name]:
            assert tp + fp + tn + fn == td["n_test"] and tp + fn == td["n_positive"]
        _, tp, fp, tn, fn = next(r for r in td["models"][name] if abs(r[0] - 0.265) < 1e-9)
        assert abs(tp / (tp + fn) - comp.loc[name, "Recall"]) < 0.01
        assert abs(tp / (tp + fp) - comp.loc[name, "Precision"]) < 0.01
        assert abs((tp + tn) / td["n_test"] - comp.loc[name, "Accuracy"]) < 0.01


def test_balanced_threshold_is_derived_from_the_class_weight_ratio():
    cal = _json("calibration.json")
    assert abs(cal["balanced_threshold"] - 1 / (1 + cal["class_weight_ratio"])) < 1e-3


# ----------------------------------- tripwires -----------------------------------
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_leakage_tripwire_held_out_auc_is_plausible(name, holdout):
    """On this dataset an honest pipeline lands near 0.84-0.85. SMOTE-before-split
    scored 0.93 in testing. If a change makes this exceed 0.90, suspect leakage
    before celebrating; if it drops below 0.80, something broke."""
    _, Xte, _, yte = holdout
    auc = roc_auc_score(yte, _raw_scores(name, Xte))
    assert 0.80 <= auc <= 0.90, f"{name} held-out AUC {auc:.3f} is outside the plausible range"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_probabilities_are_calibrated(name, holdout):
    """Class weighting inflates raw scores (mean ~0.40 vs a true churn rate of 0.265).
    After correction the mean must match reality and beat the base-rate Brier (~0.195)."""
    _, Xte, _, yte = holdout
    w = _json("calibration.json")["class_weight_ratio"]
    raw_p = _raw_scores(name, Xte)
    p = raw_p / (raw_p + w * (1 - raw_p))
    assert abs(p.mean() - yte.mean()) < 0.03, f"mean probability {p.mean():.3f} vs actual {yte.mean():.3f}"
    assert brier_score_loss(yte, p) < 0.15


def test_importing_train_model_has_no_side_effects():
    """Importing the module (as tests do) must not touch the MLflow database."""
    db = os.path.join(ROOT, "mlruns.db")
    before = os.path.getmtime(db) if os.path.exists(db) else None
    importlib.import_module("train_model")
    after = os.path.getmtime(db) if os.path.exists(db) else None
    assert before == after

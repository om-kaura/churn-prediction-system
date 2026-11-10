"""
Basic tests for the Churn Prediction API.
Run with: pytest api/test_main.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "models_loaded" in response.json()
    assert len(response.json()["models_loaded"]) == 3


def test_predict_high_risk_customer():
    """Month-to-month, low tenure, no security add-ons -> should predict high churn risk."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert 0 <= data["churn_probability"] <= 1
    assert len(data["top_risk_factors"]) == 6


def test_predict_low_risk_customer():
    """Two-year contract, long tenure -> should predict low churn risk."""
    low_risk = SAMPLE_CUSTOMER.copy()
    low_risk.update({
        "tenure": 60,
        "Contract": "Two year",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0
    })
    response = client.post("/predict", json=low_risk)
    assert response.status_code == 200
    data = response.json()
    # Low-risk profile should have lower probability than high-risk one
    high_risk_resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert data["churn_probability"] < high_risk_resp.json()["churn_probability"]


def test_predict_invalid_category():
    bad_customer = SAMPLE_CUSTOMER.copy()
    bad_customer["Contract"] = "Invalid Contract Type"
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 400


def test_predict_missing_field():
    incomplete = {"gender": "Female"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422  # FastAPI validation error


def test_list_models():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "best_model" in data
    assert len(data["models"]) == 3
    names = [m["name"] for m in data["models"]]
    assert "Logistic Regression" in names
    assert "Random Forest" in names
    assert "XGBoost" in names


def test_predict_with_specific_model():
    customer = SAMPLE_CUSTOMER.copy()
    customer["model"] = "Random Forest"
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    assert response.json()["model_used"] == "Random Forest"


def test_predict_all_models():
    response = client.post("/predict_all", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    model_names = [r["model"] for r in results]
    assert "Logistic Regression" in model_names
    assert "Random Forest" in model_names
    assert "XGBoost" in model_names
    best_flags = [r["is_best"] for r in results]
    assert sum(best_flags) == 1  # exactly one model flagged as best


def test_predict_compare_matches_predict_all():
    """New documented route (system_architecture.md: /predict/compare) should
    behave identically to the legacy /predict_all alias."""
    response = client.post("/predict/compare", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    assert sum(r["is_best"] for r in results) == 1


def test_predict_path_based_model_selection():
    """New documented route: POST /predict/{model} instead of a body field."""
    response = client.post("/predict/XGBoost", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    assert response.json()["model_used"] == "XGBoost"


def test_predict_path_based_unknown_model():
    response = client.post("/predict/NotAModel", json=SAMPLE_CUSTOMER)
    assert response.status_code == 400


def test_predict_compare_not_shadowed_by_wildcard_route():
    """Regression guard: /predict/compare must resolve to the compare handler,
    not be swallowed by /predict/{model_name} with model_name='compare'."""
    response = client.post("/predict/compare", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    assert "results" in response.json()  # compare's shape, not a single PredictionResponse


def test_roc_curve_endpoint():
    response = client.get("/models/roc-curve")
    assert response.status_code == 200
    data = response.json()
    for model in ["Logistic Regression", "Random Forest", "XGBoost"]:
        assert model in data
        assert "fpr" in data[model] and "tpr" in data[model] and "auc" in data[model]


def test_prediction_gets_logged_to_database():
    """Every /predict call should land in the SQLite history log."""
    before = client.get("/predictions/history").json()["count"]
    client.post("/predict", json=SAMPLE_CUSTOMER)
    after = client.get("/predictions/history").json()["count"]
    assert after == before + 1
    latest = client.get("/predictions/history").json()["predictions"][0]
    assert latest["model_used"] in ("Logistic Regression", "Random Forest", "XGBoost")
    assert latest["churn_prediction"] in ("Yes", "No")
    assert "tenure" in latest["input_features"]


# ---------------------------------------------------------------------------
# Calibration, threshold, config, and access-control tests
# ---------------------------------------------------------------------------
import pytest
from api.config import settings


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """Each test gets a clean rate-limit budget, so tests can't starve each other."""
    app.state.limiter.reset()
    yield


def _features(**overrides):
    d = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "model"}
    d.update(overrides)
    return d


def test_probability_is_calibrated_not_raw():
    """Class weighting inflates raw scores; the API must report the corrected value."""
    from api import main as api_main
    raw = float(api_main.MODELS["XGBoost"].predict_proba(api_main._encode_row(_features()))[0][1])
    expected = raw / (raw + api_main.CLASS_WEIGHT_RATIO * (1 - raw))
    body = client.post("/predict/XGBoost", json=_features()).json()
    assert abs(body["churn_probability"] - expected) < 1e-3
    assert body["churn_probability"] < raw


def test_threshold_controls_the_decision_and_is_echoed():
    p = client.post("/predict", json=_features()).json()["churn_probability"]
    checked = 0
    if p + 0.05 <= 0.99:
        r = client.post(f"/predict?threshold={round(p + 0.05, 3)}", json=_features()).json()
        assert r["churn_prediction"] == "No" and r["threshold_used"] == round(p + 0.05, 3)
        checked += 1
    if p - 0.05 >= 0.01:
        r = client.post(f"/predict?threshold={round(p - 0.05, 3)}", json=_features()).json()
        assert r["churn_prediction"] == "Yes"
        checked += 1
    assert checked >= 1


def test_default_threshold_is_the_balanced_operating_point():
    body = client.post("/predict", json=_features()).json()
    assert body["threshold_used"] == client.get("/health").json()["default_threshold"]
    assert 0.2 < body["threshold_used"] < 0.35   # ~ the 26.5% base rate, not 0.5


@pytest.mark.parametrize("bad", ["0", "1.5", "-0.2", "abc"])
def test_invalid_threshold_is_rejected(bad):
    assert client.post(f"/predict?threshold={bad}", json=_features()).status_code == 422


def test_compare_reports_threshold_used():
    body = client.post("/predict/compare?threshold=0.4", json=_features()).json()
    assert body["threshold_used"] == 0.4 and len(body["results"]) == 3


@pytest.mark.parametrize("field,value", [("tenure", -5), ("tenure", 500), ("MonthlyCharges", -1), ("SeniorCitizen", 2)])
def test_numeric_ranges_are_validated(field, value):
    assert client.post("/predict", json=_features(**{field: value})).status_code == 422


def test_thresholds_endpoint_shape():
    data = client.get("/models/thresholds").json()
    assert data["columns"] == ["threshold", "tp", "fp", "tn", "fn"]
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        for _, tp, fp, tn, fn in data["models"][name]:
            assert tp + fp + tn + fn == data["n_test"]


def test_dataset_summary_endpoint():
    data = client.get("/dataset/summary").json()
    assert data["rows"] == 7043 and data["column_count"] == 21
    assert abs(data["churn_rate"] - 0.2654) < 0.001
    assert len(data["sample"]) == 8


def test_models_endpoint_explains_selection_and_calibration():
    body = client.get("/models").json()
    assert "CV" in body["selection_criterion"]
    assert body["calibration"]["class_weight_ratio"] > 1


def test_history_requires_key_only_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "")
    assert client.get("/predictions/history").status_code == 200          # unset = open (dev default)
    monkeypatch.setattr(settings, "admin_api_key", "s3cret")
    assert client.get("/predictions/history").status_code == 401
    assert client.get("/predictions/history", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/predictions/history", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/predictions/export.csv").status_code == 401


def test_export_csv_has_inputs_and_outputs():
    client.post("/predict", json=_features())
    r = client.get("/predictions/export.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    header = r.text.splitlines()[0].split(",")
    assert {"created_at", "model_used", "churn_probability", "tenure", "Contract"} <= set(header)
    assert len(r.text.splitlines()) >= 2


def test_rate_limit_returns_429_after_the_configured_budget():
    budget = int(settings.rate_limit_predict.split("/")[0])
    codes = [client.post("/predict", json=_features()).status_code for _ in range(budget + 5)]
    assert codes.count(200) == budget and codes.count(429) == 5

"""
Churn Prediction API
Serves predictions from a selectable trained model (Logistic Regression,
Random Forest, or XGBoost) with SHAP explanations, plus /predict/compare to
score a customer with all three at once. Every prediction is logged to SQLite
(database.py), predict routes are rate-limited, and config comes from .env.

Probabilities: the models are trained with class weighting, which inflates raw
scores, so every reported probability is re-calibrated (see _calibrate) — a
returned 0.30 really does mean "about 30% of customers like this churn".
"""
import csv
import io
import json
import os
import secrets
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api import database
from api.config import settings

app = FastAPI(
    title=settings.app_name,
    description="Predicts customer churn probability with SHAP-based explanations, with selectable model backend",
    version=settings.api_version,
)

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_default])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", settings.model_dir)
DB_PATH = os.path.join(os.path.dirname(__file__), "..", settings.database_path)
database.init_db(DB_PATH)


def _load_json(name, default=None):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

with open(os.path.join(MODEL_DIR, "model_comparison.csv")) as f:
    _comparison_lines = f.read().strip().split("\n")
_header = _comparison_lines[0].split(",")
MODEL_METRICS = {}
for line in _comparison_lines[1:]:
    parts = line.split(",")
    MODEL_METRICS[parts[0]] = dict(zip(_header[1:], [float(p) for p in parts[1:]]))

MODEL_FILES = {
    "Logistic Regression": "model_logistic_regression.pkl",
    "Random Forest": "model_random_forest.pkl",
}
MODELS = {name: joblib.load(os.path.join(MODEL_DIR, fname)) for name, fname in MODEL_FILES.items()}

# XGBoost is loaded from its own native format, not pickle — pickle/joblib of an
# XGBoost booster is not reliably portable across xgboost versions or platforms.
_xgb_model = xgb.XGBClassifier()
_xgb_model.load_model(os.path.join(MODEL_DIR, "model_xgboost.json"))
MODELS["XGBoost"] = _xgb_model

background_sample = joblib.load(os.path.join(MODEL_DIR, "background_sample.pkl"))

# Logistic Regression is trained on standardized features (see train_model.py);
# Random Forest and XGBoost on the raw label-encoded ones. Anything scored by
# Logistic Regression must go through the same scaler first.
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
scaled_background_sample = pd.DataFrame(scaler.transform(background_sample), columns=background_sample.columns)

TREE_MODELS = {"Random Forest", "XGBoost"}
EXPLAINERS = {}
for name, m in MODELS.items():
    EXPLAINERS[name] = shap.TreeExplainer(m) if name in TREE_MODELS else shap.LinearExplainer(m, scaled_background_sample)

BEST_INFO = _load_json("best_model_info.json", {})
BEST_MODEL_NAME = BEST_INFO["best_model"]
ROC_CURVE_DATA = _load_json("roc_curve_data.json", {})
THRESHOLD_DATA = _load_json("threshold_data.json")
DATASET_SUMMARY = _load_json("dataset_summary.json")

# Class weighting inflates raw scores; this ratio (saved by train_model.py) undoes it.
CALIBRATION = _load_json("calibration.json", {"class_weight_ratio": 1.0, "balanced_threshold": 0.5})
CLASS_WEIGHT_RATIO = float(CALIBRATION["class_weight_ratio"])
DEFAULT_THRESHOLD = (
    settings.default_threshold if settings.default_threshold is not None
    else round(float(CALIBRATION["balanced_threshold"]), 3)
)


def _calibrate(raw: float) -> float:
    """Raw (class-weight-inflated) score -> real-world churn probability."""
    return raw / (raw + CLASS_WEIGHT_RATIO * (1 - raw))


def require_admin(x_api_key: Optional[str] = Header(default=None)):
    """No-op while ADMIN_API_KEY is unset (local dev); otherwise a valid X-API-Key
    header is required. Guards the routes that expose logged customer inputs."""
    expected = settings.admin_api_key
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


THRESHOLD_QUERY = Query(
    default=None, ge=0.01, le=0.99,
    description="Flag as churn when estimated probability >= this. Defaults to the balanced operating point.",
)


class CustomerFeatures(BaseModel):
    gender: str = Field(..., json_schema_extra={"example": "Female"})
    SeniorCitizen: int = Field(..., ge=0, le=1, json_schema_extra={"example": 0})
    Partner: str = Field(..., json_schema_extra={"example": "Yes"})
    Dependents: str = Field(..., json_schema_extra={"example": "No"})
    tenure: int = Field(..., ge=0, le=120, json_schema_extra={"example": 12})
    PhoneService: str = Field(..., json_schema_extra={"example": "Yes"})
    MultipleLines: str = Field(..., json_schema_extra={"example": "No"})
    InternetService: str = Field(..., json_schema_extra={"example": "DSL"})
    OnlineSecurity: str = Field(..., json_schema_extra={"example": "No"})
    OnlineBackup: str = Field(..., json_schema_extra={"example": "Yes"})
    DeviceProtection: str = Field(..., json_schema_extra={"example": "No"})
    TechSupport: str = Field(..., json_schema_extra={"example": "No"})
    StreamingTV: str = Field(..., json_schema_extra={"example": "No"})
    StreamingMovies: str = Field(..., json_schema_extra={"example": "No"})
    Contract: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    PaperlessBilling: str = Field(..., json_schema_extra={"example": "Yes"})
    PaymentMethod: str = Field(..., json_schema_extra={"example": "Electronic check"})
    MonthlyCharges: float = Field(..., ge=0, le=1000, json_schema_extra={"example": 70.35})
    TotalCharges: float = Field(..., ge=0, le=100000, json_schema_extra={"example": 845.5})
    model: Optional[str] = Field(default=None, json_schema_extra={"example": "XGBoost"})


class PredictionResponse(BaseModel):
    model_used: str
    churn_prediction: str
    churn_probability: float
    threshold_used: float
    top_risk_factors: list


def _encode_row(data: dict) -> pd.DataFrame:
    row = {}
    for col in feature_columns:
        val = data[col]
        if col in encoders:
            le = encoders[col]
            if val not in le.classes_:
                raise HTTPException(status_code=400, detail=f"Invalid value '{val}' for {col}")
            val = le.transform([val])[0]
        row[col] = val
    return pd.DataFrame([row])[feature_columns]


def _predict_with(model_name: str, X: pd.DataFrame, threshold: float):
    model = MODELS[model_name]
    explainer = EXPLAINERS[model_name]

    # Logistic Regression expects standardized features; the tree models expect
    # the raw label-encoded ones they were trained on.
    if model_name not in TREE_MODELS:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    prob = _calibrate(float(model.predict_proba(X)[0][1]))
    pred = "Yes" if prob >= threshold else "No"

    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim == 3:
        # shape (n_samples, n_features, n_classes) -> take positive class
        shap_vals = shap_vals[:, :, 1]
    shap_row = shap_vals[0]

    top_idx = np.argsort(np.abs(shap_row))[::-1][:6]
    top_factors = [{"feature": feature_columns[i], "impact": round(float(shap_row[i]), 4)} for i in top_idx]
    return pred, round(prob, 4), top_factors


def _predict_and_log(model_name: str, data: dict, threshold: float):
    """Shared by every predict route: encode, score, explain, and persist to SQLite."""
    X = _encode_row(data)
    pred, prob, top_factors = _predict_with(model_name, X, threshold)
    database.log_prediction(model_name, pred, prob, data, db_path=DB_PATH)
    return pred, prob, top_factors


def _resolve_threshold(threshold: Optional[float]) -> float:
    return DEFAULT_THRESHOLD if threshold is None else float(threshold)


def _predict_all_impl(customer: CustomerFeatures, threshold: Optional[float] = None):
    t = _resolve_threshold(threshold)
    data = customer.model_dump()
    data.pop("model", None)
    results = []
    for model_name in MODELS.keys():
        pred, prob, top_factors = _predict_and_log(model_name, data, t)
        results.append({
            "model": model_name,
            "churn_prediction": pred,
            "churn_probability": prob,
            "top_risk_factors": top_factors,
            "is_best": model_name == BEST_MODEL_NAME,
        })
    return {"threshold_used": t, "results": results}


@app.get("/")
def root():
    return {"status": "ok", "message": f"{settings.app_name} is running", "version": settings.api_version}


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": list(MODELS.keys()), "default_threshold": DEFAULT_THRESHOLD}


@app.get("/models")
def list_models():
    """Models with their evaluation metrics, which one is best and why, and how
    probabilities are calibrated."""
    return {
        "best_model": BEST_MODEL_NAME,
        "selection_criterion": BEST_INFO.get("selection_criterion"),
        "default_threshold": DEFAULT_THRESHOLD,
        "calibration": CALIBRATION,
        "models": [{"name": name, "metrics": MODEL_METRICS.get(name, {})} for name in MODELS.keys()],
    }


@app.get("/models/roc-curve")
def roc_curve_data():
    """fpr/tpr/auc per model, for rendering an ROC chart client-side instead of a static PNG."""
    return ROC_CURVE_DATA


@app.get("/models/thresholds")
def model_thresholds():
    """Confusion-matrix counts [threshold, tp, fp, tn, fn] at every threshold, per model,
    on the held-out test set — powers the dashboard's threshold explorer."""
    if THRESHOLD_DATA is None:
        raise HTTPException(status_code=404, detail="threshold_data.json not found — run train_model.py")
    return THRESHOLD_DATA


@app.get("/dataset/summary")
def dataset_summary():
    """Facts about the training data (size, churn rate, columns, sample rows)."""
    if DATASET_SUMMARY is None:
        raise HTTPException(status_code=404, detail="dataset_summary.json not found — run train_model.py")
    return DATASET_SUMMARY


@app.get("/predictions/history", dependencies=[Depends(require_admin)])
def prediction_history(limit: int = Query(50, ge=1, le=500)):
    """Recent predictions from the SQLite log — a real, inspectable database of
    what this API has actually been asked, separate from the training CSV."""
    return {"count": database.count_predictions(DB_PATH), "predictions": database.get_history(limit, DB_PATH)}


@app.get("/predictions/export.csv", dependencies=[Depends(require_admin)])
def export_predictions(limit: int = Query(5000, ge=1, le=50000)):
    """The prediction log as a CSV download (one row per prediction, one column per input)."""
    fields = ["id", "created_at", "model_used", "churn_prediction", "churn_probability"] + list(feature_columns)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in database.get_history(limit, DB_PATH):
        writer.writerow({**record, **record["input_features"]})
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=predictions.csv"})


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit(settings.rate_limit_predict)
def predict(request: Request, customer: CustomerFeatures, threshold: Optional[float] = THRESHOLD_QUERY):
    try:
        data = customer.model_dump()
        model_name = data.pop("model", None) or BEST_MODEL_NAME
        if model_name not in MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'. Choose from {list(MODELS.keys())}")
        t = _resolve_threshold(threshold)
        pred, prob, top_factors = _predict_and_log(model_name, data, t)
        return PredictionResponse(model_used=model_name, churn_prediction=pred, churn_probability=prob,
                                  threshold_used=t, top_risk_factors=top_factors)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: /predict/compare must be declared BEFORE /predict/{model_name}, or the
# wildcard route swallows "compare" as a model name.
@app.post("/predict/compare")
@limiter.limit(settings.rate_limit_predict)
def predict_compare(request: Request, customer: CustomerFeatures, threshold: Optional[float] = THRESHOLD_QUERY):
    """Runs the same customer through all three models and returns a comparison."""
    try:
        return _predict_all_impl(customer, threshold)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_all")
def predict_all(customer: CustomerFeatures):
    """Deprecated alias for /predict/compare — kept so nothing depending on this path breaks."""
    return _predict_all_impl(customer)


@app.post("/predict/{model_name}", response_model=PredictionResponse)
@limiter.limit(settings.rate_limit_predict)
def predict_specific_model(request: Request, model_name: str, customer: CustomerFeatures,
                           threshold: Optional[float] = THRESHOLD_QUERY):
    """Path-based model selection, e.g. POST /predict/XGBoost. The path segment wins
    over any `model` field in the body."""
    try:
        if model_name not in MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'. Choose from {list(MODELS.keys())}")
        data = customer.model_dump()
        data.pop("model", None)
        t = _resolve_threshold(threshold)
        pred, prob, top_factors = _predict_and_log(model_name, data, t)
        return PredictionResponse(model_used=model_name, churn_prediction=pred, churn_probability=prob,
                                  threshold_used=t, top_risk_factors=top_factors)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)

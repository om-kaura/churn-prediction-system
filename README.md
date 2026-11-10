# Customer Churn Multi-Model Predictor

Predicts customer churn risk and shows *why*: three models (Logistic Regression, Random
Forest, XGBoost) score the same customer side by side, and SHAP explains each prediction.
Trained on the IBM Telco Customer Churn dataset (7,043 customers, `data/telco_churn.csv`).

See **[MODEL_CARD.md](MODEL_CARD.md)** for performance, calibration, subgroup behaviour and
limitations.

## Quick start

```bash
pip install -r requirements.txt
python run.py            # API on :8000 + dashboard on :8080, opens your browser
```

or with Docker: `docker compose up --build` (older installs: `docker-compose`) — API at
`localhost:8000/docs`, dashboard at `localhost:8080`.

The dashboard is real ES6 modules, which browsers won't load from `file://`, so it must be
served over HTTP (`run.py` and Docker both do this).

## The dashboard

1. **Predict** — a step-by-step questionnaire (with Reset), the result, a SHAP chart of what drove
   it, and all three models side by side.
2. **Model Performance** — metrics, ROC curves, and a **threshold explorer**: slide the alert
   threshold and watch the confusion matrix and precision/recall change. The same threshold
   drives the Predict page.
3. **About** — what it is, plus the *actual dataset*: size, churn rate, every column, sample rows,
   and a CSV download of the prediction log.

## Project structure

```
├── data/telco_churn.csv        # the dataset
├── data_validation.py          # schema/range/category checks, run on every load
├── eda.py                      # validated loading + 8 exploratory plots
├── train_model.py              # tune, cross-validate, calibrate, explain, track in MLflow
├── run.py                      # one-command runner
├── api/                        # FastAPI service (main, config, database, tests)
├── web_ui/                     # dashboard (index.html, css/, js/ ES6 modules)
├── tests/test_pipeline.py      # ML-side tests incl. leakage + calibration tripwires
├── outputs/                    # trained models, metrics, predictions.db (ships populated)
├── mlruns.db, mlruns/          # MLflow tracking (3 runs)
├── MODEL_CARD.md   .env.example   pytest.ini   .github/workflows/ci.yml
├── Dockerfile  docker-compose.yml  requirements.txt  requirements-api.txt
```

## Results

| Model | Accuracy | Recall | F1 | Test AUC | CV AUC (±std) | Brier |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.749 | 0.794 | 0.627 | 0.848 | 0.849 ± 0.012 | 0.134 |
| Random Forest | 0.740 | 0.802 | 0.621 | 0.842 | 0.845 ± 0.010 | 0.139 |
| Logistic Regression | 0.739 | 0.797 | 0.618 | 0.840 | 0.844 ± 0.012 | 0.139 |

How to read it:
- **The three models are tied.** XGBoost's cross-validated lead is smaller than the fold noise.
  The winner is picked by 5-fold CV on the training data, *not* by test score, so the held-out
  numbers stay an honest report.
- **Accuracy is low on purpose.** Predicting "nobody churns" already scores 73.5%. Models are
  class-weighted to catch ~79% of churners (vs ~51%) at the cost of precision; the threshold
  explorer lets you move along that trade-off.
- **Probabilities are calibrated.** Class weighting inflates raw scores (mean 0.40 vs a true
  rate of 0.265), so they are corrected before display. A shown "30%" means roughly 30 in 100
  similar customers churn (held-out check in the model card).

### Why not 85–92% accuracy?
Many tutorials report that. Reproduced on this data:

| Pipeline | Accuracy | ROC-AUC |
|---|---|---|
| Predict "no churn" for everyone | 73.5% | — |
| XGBoost, honest, default settings | 78.1% | 0.816 |
| SMOTE on all data, **then** split (leaky) | 84.4% | **0.928** |
| Random Forest, same leak | 85.6% | **0.931** |

Oversampling before splitting puts blends of training customers into the test set; the model has
effectively seen the answers. An AUC jump from 0.82 to 0.93 is the fingerprint, and
`tests/test_pipeline.py` fails if any model's held-out AUC exceeds 0.90.
A popular public Kaggle notebook on this dataset (clean pipeline, no leakage) scores 81.6%
accuracy — but with recall of 57%, versus 79% here. Re-running its code on this data, its best
model reaches AUC 0.864 vs 0.860 for this project's XGBoost on the same split, and moving only
its decision threshold to match this project's recall gives the same accuracy and F1. The
accuracy gap is where the threshold sits, not model quality.

### Tested and rejected
Feature engineering (−0.0001 CV AUC), early stopping, stacking all three models (AUC 0.839) and
SMOTE all failed to beat the tuned single model. Details in the model card.

## API

| Route | Notes |
|---|---|
| `POST /predict` | Best model by default; `?threshold=` optional |
| `POST /predict/{model}` | e.g. `/predict/XGBoost` |
| `POST /predict/compare` | All three models at once (the dashboard uses this) |
| `GET /models`, `/models/roc-curve`, `/models/thresholds` | metrics, ROC points, confusion counts per threshold |
| `GET /dataset/summary` | rows, churn rate, columns, sample |
| `GET /predictions/history`, `/predictions/export.csv` | logged predictions (optionally key-protected) |
| `GET /health` | liveness, loaded models, default threshold |

Inputs are range-checked (`tenure` 0–120, `MonthlyCharges` ≥ 0, …) and categories validated.
`/predict*` is rate-limited (default 30/min/IP → HTTP 429). Interactive docs at `/docs`.

## Configuration (`.env`, see `.env.example`)

| Variable | Default | Purpose |
|---|---|---|
| `CORS_ORIGINS` | `*` | allowed browser origins — **tighten before deploying** |
| `RATE_LIMIT_PREDICT` / `RATE_LIMIT_DEFAULT` | `30/minute` / `120/minute` | per-IP limits |
| `DEFAULT_THRESHOLD` | unset → ≈0.265 | alert cut-off (the class base rate = the balanced point) |
| `ADMIN_API_KEY` | empty (open) | if set, history/CSV routes need an `X-API-Key` header — **set before going public** |
| `MODEL_DIR`, `DATABASE_PATH`, `PORT` | `outputs`, `outputs/predictions.db`, `8000` | paths/port |

## Testing

```bash
pytest -q        # 50 tests, ~20 s: 33 API + 17 pipeline
```
API tests cover routing, validation, calibration, thresholds, rate limiting and access control.
Pipeline tests cover data validation, artifact consistency, and the leakage/calibration tripwires.
CI (`.github/workflows/ci.yml`) runs them and builds the Docker image on every push.

Retrain: `python eda.py && python train_model.py` (≈1 min). MLflow:
`mlflow ui --backend-store-uri sqlite:///mlruns.db`.

## Known limitations / next steps
- **No drift monitoring** — retrain on fresh data before trusting it later.
- **Auth is minimal**: an optional key on two routes plus rate limiting. A browser app can't hold a
  secret, so public prediction endpoints stay open by design; add real auth if that matters.
- **SQLite** logging is fine for a demo, not for heavy concurrent writes.
- **DVC + S3** for versioning data/artifacts is not set up (needs cloud credentials); MLflow covers
  experiment tracking. Hyperparameters are tuned on the same rows used for CV, so CV scores are
  slightly optimistic (equally for every model); nested CV would remove that.
- The `gender` and `SeniorCitizen` inputs produce different flag rates across groups — see the
  model card before any sensitive use.

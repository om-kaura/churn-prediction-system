# Model Card — Customer Churn Multi-Model Predictor

## What it is
Three binary classifiers (Logistic Regression, Random Forest, XGBoost) that estimate the
probability a telecom customer will churn, with per-prediction SHAP explanations.
XGBoost is served by default. Everything here was measured by `train_model.py` and is
re-checked by `tests/test_pipeline.py`.

## Intended use
- **Yes:** ranking customers for *retention outreach* (a discount call, a check-in), where a
  human decides what to do and a wrong flag costs one unnecessary offer.
- **No:** automated decisions that disadvantage a customer (price rises, service
  restrictions, credit), anything about individuals' creditworthiness or eligibility, or any
  other industry/country without retraining and re-validating on that data.

## Data
IBM Telco Customer Churn sample dataset — 7,043 customers, 19 features (demographics,
subscribed services, contract and billing), 26.5% churned. Validated by
`data_validation.py` on every load. Split 80/20, stratified, seed 42: 5,634 train / 1,409
held-out test. Model selection uses 5-fold cross-validation on the training rows only.

## Performance (held-out test set, 1,409 customers)

| Model | Accuracy | Precision | Recall | F1 | Test ROC-AUC | CV ROC-AUC (mean ± std) | Brier |
|---|---|---|---|---|---|---|---|
| **XGBoost** | 0.749 | 0.517 | 0.794 | 0.627 | 0.848 | 0.849 ± 0.012 | 0.134 |
| Random Forest | 0.740 | 0.507 | 0.802 | 0.621 | 0.842 | 0.845 ± 0.010 | 0.139 |
| Logistic Regression | 0.739 | 0.505 | 0.797 | 0.618 | 0.840 | 0.844 ± 0.012 | 0.139 |

Accuracy/precision/recall/F1 are at the balanced operating point (alert threshold ≈ 26.5%
estimated churn probability). Predicting "nobody churns" scores 73.5% accuracy, and
predicting the base rate for everyone scores a Brier of ~0.195.

**How to read this honestly:** XGBoost's cross-validated lead (0.004) is *smaller than the
fold-to-fold noise* (±0.012). The three models are statistically tied; plain Logistic
Regression is within 0.005 AUC of the winner. XGBoost is the default for consistency with the
SHAP explanations, not because it is meaningfully more accurate.

## Operating point
Accuracy is deliberately not maximised. Models are trained with class weighting so they catch
~79% of churners (vs ~51% unweighted) at the cost of precision (~52%: about half of flagged
customers would have stayed). That suits retention outreach; if a flag were expensive you would
raise the threshold — the dashboard's threshold explorer shows the trade-off live.

## Calibration
Class weighting inflates raw scores (mean raw score 0.40 vs a true churn rate of 0.265), so
probabilities are corrected by dividing the odds by the class-weight ratio. Held-out check,
XGBoost, by risk quintile (predicted → actual churn): 1.3%→1.1%, 5.8%→6.4%, 17.9%→21.4%,
39.2%→35.5%, 66.5%→68.4%. Exact for XGBoost and Logistic Regression, approximate for Random
Forest. Guarded by a test.

## Subgroup behaviour (XGBoost, threshold 0.265, held-out test)

| Group | n | Actual churn | Flagged | Recall | Precision |
|---|---|---|---|---|---|
| Female | 687 | 28.1% | 40.0% | 78.8% | 55.3% |
| Male | 722 | 25.1% | 41.4% | 80.1% | 48.5% |
| Not senior | 1,187 | 23.3% | 36.0% | 75.4% | 48.7% |
| Senior citizen | 222 | 44.1% | 66.2% | 90.8% | 60.5% |

Gender gaps are small. Senior citizens are flagged ~1.8× as often, tracking their ~1.9× higher
actual churn rate, but this is a real disparity in who gets flagged. For retention *offers*
that may be acceptable; for anything adverse it would not be. `gender` and `SeniorCitizen`
are model inputs — consider removing them and re-measuring before any sensitive use. The
senior group is small (n=222), so these figures are noisy.

## Limitations
- One dataset, one telecom, a snapshot in time; 7k rows, so differences under ~0.01 AUC are noise.
- No usage, billing-dispute, support-ticket or price-change data — the 19 features cap
  accuracy (everything tried plateaus near 0.85 AUC).
- Feature importance explains *the model*, not causes: Contract type driving predictions does
  not mean moving customers to long contracts would reduce churn.
- No drift monitoring is implemented; retrain on fresh data before relying on it later.

## Tested and rejected
- **Engineered features** (service count, average monthly spend, spend gap, fiber-without-support,
  autopay): −0.0001 CV AUC, better in 7 of 15 folds → no effect.
- **Early stopping**: found 138 trees, test AUC 0.845 vs 0.848 → not adopted.
- **Soft-voting all three models**: test AUC 0.839, no better than the single tuned model.
- **SMOTE**: changes recall, not ranking (AUC 0.817 vs 0.816); weighting is simpler.

## Reproduce
`python eda.py && python train_model.py && pytest` (training ≈ 1 minute on one CPU core).

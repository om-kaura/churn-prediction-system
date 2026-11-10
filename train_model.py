"""
Model Training & Explainability — Customer Churn Prediction

Trains Logistic Regression, Random Forest, and XGBoost (each hyperparameter-tuned
via GridSearchCV on ROC-AUC — XGBoost in two stages: structure, then regularization),
compares their performance, picks the best model, and runs SHAP explainability on
it. This is the pipeline the FastAPI service in api/main.py actually loads at
startup — its artifact names are part of the API's contract, so don't rename them
without updating api/main.py to match.

Class imbalance: churn is ~26.5% of the dataset. All three models are trained with
balanced class weighting (class_weight="balanced" for LR/RF, scale_pos_weight for
XGBoost) rather than left at the unweighted default. This was a deliberate,
verified choice, not the library default:
  - ROC-AUC (used for model selection above) barely moves either way (+/-0.0002),
    since it's a ranking metric and largely insensitive to class weighting.
  - Recall on the churn class goes from ~0.51 to ~0.79-0.80 across all three models,
    at the cost of precision (more false-positive "at risk" flags). For a retention
    tool, missing an actual churner is normally costlier than one unnecessary
    outreach, so this trade favors recall.
  - Reported accuracy drops as a direct, expected side effect (it drops for all
    three models, not just the one this pipeline picks) — accuracy is a poor
    headline metric under class imbalance in the first place; F1/Recall/ROC-AUC
    tell the real story here. Don't read the lower accuracy number as a bug.

Model selection: the winner is chosen by mean 5-fold cross-validated ROC-AUC on
the TRAINING data (identical folds for every model), not by test-set score — so
the held-out test metrics stay an unbiased report instead of doubling as the
selection criterion. Raw data is validated (data_validation.py) before use.

Probabilities: class weighting inflates raw scores, so they are re-calibrated
(calibrate_probs) before being shown or thresholded — see that function.

Usage:
    python eda.py            # optional, but produces the plots referenced in README
    python train_model.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_validation import write_dataset_summary
from eda import DATA_PATH, load_data

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
sns.set_theme(style="whitegrid")


def log_to_mlflow(results: list, trained_models: dict, out_dir: str) -> None:
    """Logs each model's params, metrics, and the fitted model artifact as its own
    MLflow run, tracked in a local SQLite database (mlruns.db — no server/S3
    needed; MLflow's newer versions require a DB backend rather than the older
    plain-folder file store). Inspect with:
        mlflow ui --backend-store-uri sqlite:///mlruns.db
    This answers the "version/track your artifacts externally" critique in the
    spirit it's usually meant — real experiment tracking — without requiring cloud
    infra (DVC+S3) this project doesn't have set up yet; see README for that path.
    """
    # Set up here, not at import time: importing this module (e.g. from a test)
    # must not touch the tracking database.
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(os.path.dirname(__file__), "mlruns.db"))
    mlflow.set_experiment("customer-churn-prediction")
    metrics_by_name = {r["Model"]: r for r in results}

    for name, model in trained_models.items():
        with mlflow.start_run(run_name=name):
            params = {k: v for k, v in model.get_params().items() if v is not None}
            mlflow.log_params(params)

            row = metrics_by_name[name]
            mlflow.log_metrics({
                "accuracy": row["Accuracy"], "precision": row["Precision"],
                "recall": row["Recall"], "f1": row["F1"], "roc_auc": row["ROC_AUC"],
                "cv_roc_auc": row["CV_ROC_AUC"], "cv_roc_auc_std": row["CV_ROC_AUC_STD"], "brier": row["Brier"],
            })
            mlflow.set_tag("dataset", "telco_churn.csv (IBM Telco Customer Churn, 7043 rows)")

            if name == "XGBoost":
                mlflow.xgboost.log_model(model, name="model")
            else:
                # RandomForestClassifier's internal sklearn.tree._tree.Tree is not in
                # mlflow's default trusted-type allowlist for its skops-based
                # serializer; this model was trained in-process by this same script,
                # so it's safe to trust here.
                mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])
    print(f"\nMLflow runs logged under {mlflow.get_tracking_uri()} "
          f"— view with: mlflow ui --backend-store-uri sqlite:///mlruns.db")


def prepare_features(df: pd.DataFrame):
    """Drop the ID column, encode Churn and every categorical feature.

    Returns (X, y, feature_names, encoders). `encoders` and the feature order are
    saved to disk so api/main.py can encode a live request identically to how
    training data was encoded — this is the coupling that makes /predict correct.
    """
    df = df.drop("customerID", axis=1).copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    encoders = {}
    for col in df.select_dtypes(include=["object", "str"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y, list(X.columns), encoders


def evaluate(name, model, X_test, y_test, results, roc_data, out_dir):
    """Score a fitted model, record its metrics, and save its confusion matrix."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_data[name] = (fpr, tpr, roc_auc_score(y_test, probs))
    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds), 4),
        "ROC_AUC": round(roc_auc_score(y_test, probs), 4),
    })
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(os.path.join(out_dir, f"cm_{name.replace(' ', '_')}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def train_models(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, out_dir):
    results, trained_models, roc_data = [], {}, {}

    # Logistic Regression is the only model here that's sensitive to feature scale —
    # tree ensembles split on raw thresholds and are unaffected by it, so scaling is
    # applied only for this model (this is also what fixes the lbfgs convergence
    # warning you get if you fit LR directly on the label-encoded features).
    print("Tuning Logistic Regression...")
    # C tops out at 10 here — verified separately that widening this to 1000 doesn't
    # move cv ROC-AUC past the 5th decimal (0.84463 at C=10 vs 0.84464 at C=1000), so
    # the curve has already flattened well before this edge. Not worth the extra fits.
    # class_weight="balanced" reweights the loss by inverse class frequency (~73.5/26.5
    # split here) — see the module-level note below on why this is applied to all three
    # models rather than left at sklearn's unweighted default.
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        param_grid={"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]},
        cv=5, scoring="roc_auc", n_jobs=-1,
    )
    lr_grid.fit(X_train_scaled, y_train)
    print("Best LR params:", lr_grid.best_params_)
    evaluate("Logistic Regression", lr_grid.best_estimator_, X_test_scaled, y_test, results, roc_data, out_dir)
    trained_models["Logistic Regression"] = lr_grid.best_estimator_

    print("Tuning Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid={"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]},
        cv=3, scoring="roc_auc", n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    print("Best RF params:", rf_grid.best_params_)
    evaluate("Random Forest", rf_grid.best_estimator_, X_test, y_test, results, roc_data, out_dir)
    trained_models["Random Forest"] = rf_grid.best_estimator_

    # XGBoost is tuned in two stages rather than one joint grid: structure first,
    # then regularization conditional on the winning structure. A full 7-way joint
    # grid (structure x reg_alpha x reg_lambda x subsample x colsample_bytree) is
    # ~650 combinations — combinatorially wasteful (and prone to overfitting the CV
    # folds) for a 5-model comparison on a single core. Staging it is the standard,
    # cheaper alternative and is what was actually run to validate this approach.
    #
    # scale_pos_weight is XGBoost's native equivalent of class_weight="balanced" —
    # it upweights the minority (churn) class in the loss by the class ratio below.
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {pos_weight:.3f} (class imbalance ratio)")

    print("Tuning XGBoost (stage 1: model structure)...")
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(eval_metric="logloss", random_state=42, scale_pos_weight=pos_weight),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
        },
        cv=3, scoring="roc_auc", n_jobs=-1,
    )
    xgb_grid.fit(X_train, y_train)
    xgb_structural_params = xgb_grid.best_params_
    print("Best XGB structural params:", xgb_structural_params)

    # Stage 2: with structure fixed, search the knobs XGBoost otherwise leaves at
    # library defaults (reg_lambda=1, reg_alpha=0, subsample=1, colsample_bytree=1).
    print("Tuning XGBoost (stage 2: regularization)...")
    xgb_reg_grid = GridSearchCV(
        xgb.XGBClassifier(eval_metric="logloss", random_state=42, scale_pos_weight=pos_weight, **xgb_structural_params),
        param_grid={
            "reg_lambda": [0.5, 1, 2, 5],
            "reg_alpha": [0, 0.5, 1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        cv=3, scoring="roc_auc", n_jobs=-1,
    )
    xgb_reg_grid.fit(X_train, y_train)
    print("Best XGB regularization params:", xgb_reg_grid.best_params_)
    evaluate("XGBoost", xgb_reg_grid.best_estimator_, X_test, y_test, results, roc_data, out_dir)
    trained_models["XGBoost"] = xgb_reg_grid.best_estimator_

    return results, trained_models, roc_data


def run_shap(best_name, best_model, feature_names, X_train, X_test, out_dir):
    print("\nRunning SHAP explainability...")
    if best_name in ("XGBoost", "Random Forest"):
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.LinearExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_bar_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked_idx = np.argsort(mean_abs_shap)[::-1]
    top_feature, second_feature = feature_names[ranked_idx[0]], feature_names[ranked_idx[1]]

    plt.figure()
    shap.dependence_plot(top_feature, shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_dependence_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.dependence_plot(second_feature, shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_dependence_plot_2.png"), dpi=150, bbox_inches="tight")
    plt.close()

    importance_df = pd.DataFrame({
        "Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values("Mean_Abs_SHAP", ascending=False)
    importance_df.to_csv(os.path.join(out_dir, "shap_feature_importance.csv"), index=False)

    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = base_val[0] if len(base_val) > 1 else base_val[0]

    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[0], base_values=base_val,
                          data=X_test.iloc[0], feature_names=feature_names),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_individual_explanation.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Top feature driving churn: {top_feature}")
    print(f"Second most important: {second_feature}")
    print("\nTop 10 features by SHAP importance:")
    print(importance_df.head(10).to_string(index=False))
    return importance_df, explainer


THRESHOLDS = np.round(np.arange(1, 200) * 0.005, 3)   # 0.005 .. 0.995


def calibrate_probs(p, class_weight_ratio):
    """Undo the odds inflation caused by class weighting.

    Training with class_weight="balanced" / scale_pos_weight=w multiplies the
    predicted odds of churn by w, so a raw 0.84 does NOT mean "84% chance" — the
    average raw score was 0.40 when the true churn rate is 0.265. Dividing the
    odds by w puts the number back on the real-world scale. Exact for the
    logistic-loss models (XGBoost, Logistic Regression), approximate for Random
    Forest. Monotonic, so ROC-AUC and the model ranking are unchanged.
    """
    p = np.asarray(p, dtype=float)
    return p / (p + class_weight_ratio * (1 - p))


def cross_validate_models(trained_models, X_train, X_train_scaled, y_train, n_splits=5, seed=42):
    """Mean +/- std 5-fold CV ROC-AUC on the TRAINING data, with identical folds
    for every model. Used to pick the winner: choosing it by test-set score (as
    this pipeline used to) means the "unbiased" test number was partly used up
    on selection. The test set is now only ever used to report.
    (Hyperparameters were tuned on these same training rows, so the CV numbers
    are slightly optimistic — equally so for every model, which is what matters
    for ranking them.)"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {}
    for name, model in trained_models.items():
        X_for_model = X_train_scaled if name == "Logistic Regression" else X_train
        scores = cross_val_score(model, X_for_model, y_train, cv=cv, scoring="roc_auc")
        out[name] = (float(scores.mean()), float(scores.std()))
        print(f"  CV ROC-AUC  {name:20s} {scores.mean():.4f} +/- {scores.std():.4f}")
    return out


def threshold_table(y_true, prob):
    """[threshold, tp, fp, tn, fn] for every threshold on the calibrated-probability
    scale — everything the dashboard needs to draw a confusion matrix and
    precision/recall at any cut-off."""
    y = np.asarray(y_true).astype(int)
    rows = []
    for t in THRESHOLDS:
        pred = prob >= t
        rows.append([float(t), int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()),
                     int((~pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())])
    return rows


def main(out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    df = load_data()

    X, y, feature_names, encoders = prepare_features(df)
    joblib.dump(encoders, os.path.join(out_dir, "label_encoders.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    results, trained_models, roc_data = train_models(
        X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, out_dir
    )

    print("\nCross-validating (same 5 folds for every model)...")
    cv_scores = cross_validate_models(trained_models, X_train, X_train_scaled, y_train)
    for r in results:
        r["CV_ROC_AUC"], r["CV_ROC_AUC_STD"] = (round(v, 4) for v in cv_scores[r["Model"]])

    # Class weighting inflates predicted probabilities — undo it so "probability"
    # means probability (see calibrate_probs). Brier score = mean squared error of
    # the probabilities; lower is better, and predicting the base rate for everyone
    # scores ~0.195 on this data.
    class_weight_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    cal_probs = {}
    for name, model in trained_models.items():
        X_for_model = X_test_scaled if name == "Logistic Regression" else X_test
        cal_probs[name] = calibrate_probs(model.predict_proba(X_for_model)[:, 1], class_weight_ratio)
    for r in results:
        r["Brier"] = round(float(brier_score_loss(y_test, cal_probs[r["Model"]])), 4)

    log_to_mlflow(results, trained_models, out_dir)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False)
    print("\n" + "=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    # Combined ROC curve
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison — All Models")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "roc_curve_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Best model selection
    best_row = results_df.loc[results_df["CV_ROC_AUC"].idxmax()]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]
    print(f"\nBest model selected by CV: {best_name} (CV ROC-AUC={best_row['CV_ROC_AUC']}, "
          f"held-out test ROC-AUC={best_row['ROC_AUC']})")

    joblib.dump(best_model, os.path.join(out_dir, "best_model.pkl"))
    joblib.dump(feature_names, os.path.join(out_dir, "feature_columns.pkl"))
    with open(os.path.join(out_dir, "best_model_info.json"), "w") as f:
        json.dump({"best_model": best_name, "selection_criterion": "mean 5-fold CV ROC-AUC on training data",
                   "metrics": best_row.to_dict()}, f, indent=2, default=str)

    # Save each model individually so the API can offer a model selector. XGBoost is
    # the one exception: it's saved via save_model() below in its own portable JSON
    # format instead of pickle, so a duplicate .pkl of it would just be dead weight.
    model_filename_map = {
        "Logistic Regression": "model_logistic_regression.pkl",
        "Random Forest": "model_random_forest.pkl",
    }
    for name, filename in model_filename_map.items():
        joblib.dump(trained_models[name], os.path.join(out_dir, filename))
    print("\nSaved individual models:", list(model_filename_map.values()))

    # XGBoost's own JSON format is more portable across versions/platforms than pickle
    trained_models["XGBoost"].save_model(os.path.join(out_dir, "model_xgboost.json"))
    print("Also saved model_xgboost.json (native format, portable across xgboost versions)")

    # Small real background sample, used by SHAP's LinearExplainer if LR is ever the best model
    background_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
    # pandas 3 gives column labels an Arrow-backed string dtype when pyarrow happens
    # to be installed, and pickling that makes this artifact un-loadable anywhere
    # pyarrow ISN'T installed — including the API's Docker image, which deliberately
    # doesn't ship it (see requirements-api.txt). Plain object labels pickle portably.
    background_sample.columns = pd.Index(list(background_sample.columns), dtype=object)
    joblib.dump(background_sample, os.path.join(out_dir, "background_sample.pkl"))
    print("Saved background_sample.pkl for LinearExplainer")

    # Downsampled ROC points for the frontend's interactive chart
    roc_export = {}
    for name, (fpr, tpr, auc) in roc_data.items():
        n = len(fpr)
        step = max(1, n // 60)
        roc_export[name] = {
            "fpr": fpr[::step].tolist(),
            "tpr": tpr[::step].tolist(),
            "auc": round(float(auc), 4),
        }
    with open(os.path.join(out_dir, "roc_curve_data.json"), "w") as f:
        json.dump(roc_export, f)
    print("Saved roc_curve_data.json for interactive charting")

    # Confusion-matrix counts at every threshold, per model, for the dashboard's
    # threshold explorer (each model scored in the feature space it was trained on).
    threshold_export = {"n_test": int(len(y_test)), "n_positive": int(np.asarray(y_test).sum()),
                        "columns": ["threshold", "tp", "fp", "tn", "fn"], "models": {}}
    for name in trained_models:
        threshold_export["models"][name] = threshold_table(y_test, cal_probs[name])
    with open(os.path.join(out_dir, "threshold_data.json"), "w") as f:
        json.dump(threshold_export, f)
    print("Saved threshold_data.json for the threshold explorer")

    # The API needs the same ratio to put its probabilities on the real-world scale.
    # balanced_threshold is the calibrated cut-off equivalent to a raw 0.5, i.e. the
    # operating point the reported Accuracy/Precision/Recall/F1 were measured at.
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump({"method": "prior shift: odds divided by the class-weight ratio",
                   "class_weight_ratio": round(class_weight_ratio, 4),
                   "balanced_threshold": round(1 / (1 + class_weight_ratio), 4)}, f, indent=2)
    print("Saved calibration.json")

    write_dataset_summary(pd.read_csv(DATA_PATH), out_dir)
    print("Saved dataset_summary.json for the About page")

    # SHAP must see the model in the same feature space it was trained on.
    shap_train = X_train_scaled if best_name == "Logistic Regression" else X_train
    shap_test = X_test_scaled if best_name == "Logistic Regression" else X_test
    run_shap(best_name, best_model, feature_names, shap_train, shap_test, out_dir)

    print("\n" + "=" * 60)
    print("ALL OUTPUTS SAVED TO", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

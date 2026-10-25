from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.data_pipeline import dataset_audit, load_consumption, load_households
from src.feature_engineering import CATEGORICAL, MODEL_FEATURES, NUMERICAL_FEATURES, TARGET_KWH, make_model_table
from src.model_bundle import EnergyRiskModelBundle


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "models"
SEED = 42


def score(y_true, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, predicted),
        "Balanced Accuracy": balanced_accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probability),
        "PR-AUC": average_precision_score(y_true, probability),
    }


def prepare_frames(table):
    train = table[(table["month"] >= "2023-12-31") & (table["month"] <= "2024-06-30")].copy()
    validation = table[(table["month"] >= "2024-07-31") & (table["month"] <= "2024-08-31")].copy()
    test = table[table["month"] == "2024-09-30"].copy()
    numeric_fill = train[NUMERICAL_FEATURES].median(numeric_only=True).to_dict()
    categorical_fill = {}
    for column in CATEGORICAL:
        mode = train[column].astype("string").dropna().mode()
        categorical_fill[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"
    for frame in (train, validation, test):
        for column in NUMERICAL_FEATURES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(numeric_fill[column])
        for column in CATEGORICAL:
            frame[column] = frame[column].astype("string").fillna(categorical_fill[column]).replace("<NA>", categorical_fill[column]).astype(str)
    return train, validation, test, numeric_fill, categorical_fill


def select_threshold(y_true, probability):
    rows = []
    for threshold in np.arange(0.30, 0.701, 0.005):
        result = score(y_true, probability, float(threshold))
        rows.append({"Threshold": float(threshold), **result})
    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(["Accuracy", "Balanced Accuracy"], ascending=False)
    return float(comparison.iloc[0]["Threshold"]), comparison


def evaluate_additional_models(train, validation, test):
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]), NUMERICAL_FEATURES),
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ], sparse_threshold=0)

    x_train = preprocessor.fit_transform(train[MODEL_FEATURES])
    x_validation = preprocessor.transform(validation[MODEL_FEATURES])
    x_test = preprocessor.transform(test[MODEL_FEATURES])
    y_train = train["high_risk"].to_numpy()
    y_validation = validation["high_risk"].to_numpy()
    y_test = test["high_risk"].to_numpy()

    candidates = [
        ("Gaussian Naive Bayes", GaussianNB(var_smoothing=1e-7)),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis(reg_param=0.50)),
    ]
    rows = []
    for name, candidate in candidates:
        start = time.perf_counter()
        candidate.fit(x_train, y_train)
        training_time = time.perf_counter() - start
        validation_probability = candidate.predict_proba(x_validation)[:, 1]
        threshold, _ = select_threshold(y_validation, validation_probability)
        start = time.perf_counter()
        test_probability = candidate.predict_proba(x_test)[:, 1]
        prediction_time = time.perf_counter() - start
        metrics = score(y_test, test_probability, threshold)
        metrics["Training Time"] = training_time
        metrics["Prediction Time"] = prediction_time
        rows.append({"Model": name, **metrics})
    return rows


def save_plots(y_test, probability, predicted, model, feature_names):
    sns.set_theme(style="whitegrid")
    matrix = confusion_matrix(y_test, predicted)
    plt.figure(figsize=(6.2, 5.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Very high"], yticklabels=["Normal", "Very high"])
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("CatBoost confusion matrix: September 2024")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probability)
    plt.figure(figsize=(6.5, 5.2))
    plt.plot(fpr, tpr, color="#146b63", linewidth=2.2, label=f"CatBoost AUC = {roc_auc_score(y_test, probability):.4f}")
    plt.plot([0, 1], [0, 1], "--", color="#888888")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES / "roc_curve.png", dpi=180)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, probability)
    plt.figure(figsize=(6.5, 5.2))
    plt.plot(recall, precision, color="#8d153a", linewidth=2.2,
             label=f"CatBoost PR-AUC = {average_precision_score(y_test, probability):.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES / "precision_recall_curve.png", dpi=180)
    plt.close()

    importance = pd.DataFrame({"Feature": feature_names, "Importance": model.get_feature_importance()})
    importance = importance.sort_values("Importance", ascending=False)
    importance.to_csv(TABLES / "feature_importance.csv", index=False)
    top = importance.head(15).sort_values("Importance")
    plt.figure(figsize=(8.2, 6.2))
    plt.barh(top["Feature"].str.replace("_", " "), top["Importance"], color="#146b63")
    plt.xlabel("CatBoost feature importance")
    plt.title("Most influential model features")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=180)
    plt.close()


def main():
    for directory in (PROCESSED, TABLES, FIGURES, MODELS):
        directory.mkdir(parents=True, exist_ok=True)

    consumption = load_consumption(RAW)
    households = load_households(RAW)
    table, build_report = make_model_table(consumption, households)
    table.to_csv(PROCESSED / "model_table.csv", index=False)
    dataset_audit(RAW).to_csv(TABLES / "dataset_audit.csv", index=False)

    train, validation, test, numeric_fill, categorical_fill = prepare_frames(table)
    split_rows = []
    for name, frame in [("Training", train), ("Validation", validation), ("Testing", test)]:
        split_rows.append({
            "Split": name,
            "Start month": frame["month"].min().date().isoformat(),
            "End month": frame["month"].max().date().isoformat(),
            "Records": len(frame),
            "Households": frame["household_ID"].nunique(),
            "Normal": int((frame["high_risk"] == 0).sum()),
            "Very high": int((frame["high_risk"] == 1).sum()),
            "Very-high percentage": frame["high_risk"].mean() * 100,
        })
    pd.DataFrame(split_rows).to_csv(TABLES / "split_summary.csv", index=False)

    model = CatBoostClassifier(
        iterations=1000,
        depth=5,
        learning_rate=0.025,
        loss_function="Logloss",
        eval_metric="Logloss",
        l2_leaf_reg=8,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    start = time.perf_counter()
    model.fit(
        train[MODEL_FEATURES],
        train["high_risk"],
        cat_features=CATEGORICAL,
        eval_set=(validation[MODEL_FEATURES], validation["high_risk"]),
        early_stopping_rounds=100,
        verbose=False,
    )
    training_time = time.perf_counter() - start
    validation_probability = model.predict_proba(validation[MODEL_FEATURES])[:, 1]
    decision_threshold, threshold_table = select_threshold(validation["high_risk"], validation_probability)
    threshold_table.to_csv(TABLES / "validation_thresholds.csv", index=False)

    start = time.perf_counter()
    test_probability = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    prediction_time = time.perf_counter() - start
    test_metrics = score(test["high_risk"], test_probability, decision_threshold)
    test_metrics["Training Time"] = training_time
    test_metrics["Prediction Time"] = prediction_time
    predicted = (test_probability >= decision_threshold).astype(int)

    previous_rule_probability = (test["previous_month_consumption"] >= TARGET_KWH).astype(float).to_numpy()
    previous_rule = score(test["high_risk"], previous_rule_probability, 0.5)
    dummy_probability = np.full(len(test), train["high_risk"].mean())
    dummy = score(test["high_risk"], dummy_probability, 0.5)
    additional_rows = evaluate_additional_models(train, validation, test)
    comparison = pd.DataFrame([
        {"Model": "CatBoost", **test_metrics},
        *additional_rows,
        {"Model": "Previous-month rule", **previous_rule},
        {"Model": "Dummy majority model", **dummy},
    ])
    comparison.to_csv(TABLES / "model_comparison.csv", index=False)
    save_plots(test["high_risk"], test_probability, predicted, model, MODEL_FEATURES)

    bundle = EnergyRiskModelBundle(
        model=model,
        model_name="CatBoost",
        feature_names=MODEL_FEATURES,
        categorical_features=CATEGORICAL,
        numeric_fill_values=numeric_fill,
        categorical_fill_values=categorical_fill,
        threshold=decision_threshold,
        native_catboost=True,
    )
    joblib.dump(bundle, MODELS / "energy_risk_model.pkl")

    categories = {
        column: sorted(table[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL
    }
    schema = {
        "model_features": MODEL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL,
        "categories": categories,
        "target_threshold_kwh": TARGET_KWH,
        "decision_threshold": decision_threshold,
    }
    (MODELS / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    metadata = {
        "model_name": "CatBoost",
        "model_version": "2.0",
        "dataset_source": "RECON-SL, Kaggle version 2 / Zenodo record 15023048",
        "selected_target_threshold_kwh": TARGET_KWH,
        "threshold_reason": "200 kWh or more is a clear very-high-use boundary close to the training 85th percentile (202 kWh). The positive class remains above 10% in every split.",
        "decision_threshold": decision_threshold,
        "best_iteration": int(model.get_best_iteration()),
        "input_features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL,
        "numerical_features": NUMERICAL_FEATURES,
        "class_names": {"0": "Normal electricity consumption risk", "1": "Very high electricity consumption risk"},
        "test_metrics": test_metrics,
        "records_used": int(len(table)),
        "households_used": int(table["household_ID"].nunique()),
        "training_date_range": [train["month"].min().date().isoformat(), train["month"].max().date().isoformat()],
        "validation_date_range": [validation["month"].min().date().isoformat(), validation["month"].max().date().isoformat()],
        "test_date_range": [test["month"].min().date().isoformat(), test["month"].max().date().isoformat()],
        "selection_method": "CatBoost configuration and probability threshold selected using the chronological validation period. The final test month was not used for tuning.",
        "comparison_models": [
            "CatBoost",
            "Gaussian Naive Bayes",
            "Linear Discriminant Analysis",
            "Quadratic Discriminant Analysis",
            "Previous-month rule",
            "Dummy majority model",
        ],
        "build_report": build_report,
        "library_versions": {"python": platform.python_version(), "pandas": pd.__version__},
        "known_limitations": [
            "The data covers selected Sri Lankan service areas.",
            "The final test period contains one calendar month.",
            "The model predicts a risk category rather than an exact bill.",
            "The positive class is smaller, so balanced accuracy and recall must be reported with accuracy.",
        ],
    }
    (MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(json.dumps({
        "model": "CatBoost",
        "target_kwh": TARGET_KWH,
        "decision_threshold": decision_threshold,
        "records": len(table),
        "households": table["household_ID"].nunique(),
        "test_metrics": test_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()

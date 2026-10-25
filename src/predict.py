from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from .feature_engineering import MODEL_FEATURES, derived_from_inputs


NON_NEGATIVE = {
    "previous_month_consumption", "consumption_two_months_ago", "consumption_three_months_ago",
    "household_size", "floor_area", "monthly_expenditure", "total_appliances",
    "total_appliance_hours_week", "high_energy_appliance_count", "ac_count", "ac_hours_week",
}


def load_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file was not found: {path}")
    return joblib.load(path)


def predict_household(record: dict, model=None, model_path: str | Path | None = None) -> dict:
    try:
        completed = derived_from_inputs(record)
        missing = sorted(set(MODEL_FEATURES) - set(completed))
        if missing:
            raise ValueError("Missing required features: " + ", ".join(missing))
        for feature in NON_NEGATIVE:
            if completed.get(feature) is not None and float(completed[feature]) < 0:
                raise ValueError(f"{feature} cannot be negative")
        if int(completed["month_number"]) not in range(1, 13):
            raise ValueError("month_number must be between 1 and 12")
        bundle = model if model is not None else load_model(model_path)
        frame = pd.DataFrame([{key: completed[key] for key in MODEL_FEATURES}])
        probability = float(bundle.predict_proba(frame)[0, 1])
        predicted = int(probability >= 0.5)
        return {
            "predicted_class": predicted,
            "probability": probability,
            "risk_label": "High risk" if predicted else "Normal risk",
            "error": None,
        }
    except (ValueError, TypeError, KeyError) as exc:
        return {"predicted_class": None, "probability": None, "risk_label": None, "error": str(exc)}


def explain_prediction(record: dict, model) -> list[dict]:
    """Return the strongest local SHAP contributions for supported tree pipelines."""
    import numpy as np
    import shap

    completed = derived_from_inputs(record)
    frame = pd.DataFrame([{key: completed[key] for key in MODEL_FEATURES}])
    prepared = model.prepare(frame)
    pipeline = model.model
    if not hasattr(pipeline, "named_steps"):
        return []
    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocess.transform(prepared)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = [name.split("__", 1)[-1] for name in preprocess.get_feature_names_out()]
    values = shap.TreeExplainer(estimator).shap_values(transformed)
    if isinstance(values, list):
        values = values[1]
    elif np.asarray(values).ndim == 3:
        values = np.asarray(values)[:, :, 1]
    ranked = np.argsort(np.abs(values[0]))[::-1]
    readable = {
        "previous_month_consumption": "Previous-month consumption",
        "three_month_average": "Three-month average",
        "three_month_max": "Three-month maximum",
        "consumption_two_months_ago": "Consumption two months ago",
        "consumption_three_months_ago": "Consumption three months ago",
        "consumption_change": "Recent consumption change",
        "rolling_standard_deviation": "Variation across three months",
    }
    factors = []
    for index in ranked:
        raw_name = names[index]
        label = readable.get(raw_name, raw_name.replace("_", " ").title())
        contribution = float(values[0, index])
        factors.append({"feature": label, "direction": "higher" if contribution > 0 else "lower", "shap_value": contribution})
        if len(factors) == 3:
            break
    return factors

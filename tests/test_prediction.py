from pathlib import Path

from src.feature_engineering import CONSUMPTION_FEATURES
from src.predict import load_model, predict_household


ROOT = Path(__file__).resolve().parents[1]


def sample_record(level=260.0):
    record = {name: level - index * 2 for index, name in enumerate(CONSUMPTION_FEATURES)}
    record.update({
        "prediction_month": 10,
        "household_size": 4,
        "floor_area": 1200.0,
        "monthly_expenditure": 65000.0,
        "total_appliances": 14,
        "total_appliance_hours_week": 170.0,
        "high_energy_appliance_count": 3,
        "ac_count": 1,
        "ac_hours_week": 20.0,
        "has_refrigerator": 1,
        "has_washing_machine": 1,
        "has_rice_cooker": 1,
        "has_water_heater": 0,
        "has_computer": 1,
        "has_television": 1,
        "has_air_conditioner": 1,
        "home_business": 0,
        "solar_generation": 0,
        "grid_cooking": 0,
        "csc_area": "GALLE",
        "housing_type": "Single House - Single Floor",
        "socio_economic_class": "SEC C",
        "meter_type": "Non smart meter",
        "water_heating_method": "Unknown",
    })
    return record


def test_saved_model_loads_and_predicts_probability():
    model = load_model(ROOT / "models" / "energy_risk_model.pkl")
    result = predict_household(sample_record(), model=model)
    assert result["error"] is None
    assert result["predicted_class"] in {0, 1}
    assert 0 <= result["probability"] <= 1


def test_negative_consumption_is_rejected():
    model = load_model(ROOT / "models" / "energy_risk_model.pkl")
    record = sample_record()
    record["previous_month_consumption"] = -1
    result = predict_household(record, model=model)
    assert result["predicted_class"] is None
    assert "cannot be negative" in result["error"]


def test_missing_history_is_reported():
    model = load_model(ROOT / "models" / "energy_risk_model.pkl")
    record = sample_record()
    record.pop("consumption_thirteen_months_ago")
    result = predict_household(record, model=model)
    assert result["predicted_class"] is None
    assert "consumption_thirteen_months_ago" in result["error"]

import numpy as np
import pandas as pd

from src.feature_engineering import CONSUMPTION_FEATURES, TARGET_KWH, add_time_features, derived_from_inputs


def test_latest_reading_is_used_for_next_month_prediction():
    frame = pd.DataFrame({
        "household_ID": ["H1"] * 4,
        "month": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"]),
        "consumption": [100.0, 120.0, 140.0, 160.0],
    })
    result = add_time_features(frame)
    april = result.iloc[-1]
    assert april["previous_month_consumption"] == 160.0
    assert april["consumption_two_months_ago"] == 140.0
    assert april["consumption_three_months_ago"] == 120.0


def test_target_boundary_is_two_hundred_kwh_or_more():
    assert TARGET_KWH == 200.0
    assert int(200.0 >= TARGET_KWH) == 1
    assert int(199.9 >= TARGET_KWH) == 0


def test_input_derivations_use_only_supplied_history():
    record = {name: float(260 - index * 5) for index, name in enumerate(CONSUMPTION_FEATURES)}
    record["prediction_month"] = 10
    result = derived_from_inputs(record)
    assert result["four_month_average"] == np.mean([260, 255, 250, 245])
    assert result["same_month_last_year"] == record[CONSUMPTION_FEATURES[-1]]
    assert result["prediction_month"] == 10

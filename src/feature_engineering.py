from __future__ import annotations

import numpy as np
import pandas as pd


BASE_NUMERIC = [
    "previous_month_consumption", "consumption_two_months_ago", "consumption_three_months_ago",
    "three_month_average", "three_month_max", "three_month_min", "consumption_change",
    "consumption_percentage_change", "rolling_standard_deviation", "month_number", "quarter",
    "household_size", "floor_area", "monthly_expenditure", "total_appliances",
    "total_appliance_hours_week", "high_energy_appliance_count", "ac_count", "ac_hours_week",
]
BINARY = [
    "has_refrigerator", "has_washing_machine", "has_rice_cooker", "has_water_heater",
    "has_computer", "has_television", "has_air_conditioner", "home_business",
    "solar_generation", "grid_cooking",
]
CATEGORICAL = ["csc_area", "housing_type", "socio_economic_class", "meter_type", "water_heating_method"]
MODEL_FEATURES = BASE_NUMERIC + BINARY + CATEGORICAL


def add_time_features(consumption: pd.DataFrame) -> pd.DataFrame:
    df = consumption.sort_values(["household_ID", "month"]).copy()
    group = df.groupby("household_ID")["consumption"]
    df["previous_month_consumption"] = group.shift(1)
    df["consumption_two_months_ago"] = group.shift(2)
    df["consumption_three_months_ago"] = group.shift(3)
    past = df[["previous_month_consumption", "consumption_two_months_ago", "consumption_three_months_ago"]]
    df["three_month_average"] = past.mean(axis=1, skipna=False)
    df["three_month_max"] = past.max(axis=1, skipna=False)
    df["three_month_min"] = past.min(axis=1, skipna=False)
    df["rolling_standard_deviation"] = past.std(axis=1, ddof=0, skipna=False)
    df["consumption_change"] = df["previous_month_consumption"] - df["consumption_two_months_ago"]
    denominator = df["consumption_two_months_ago"].replace(0, np.nan)
    df["consumption_percentage_change"] = (df["consumption_change"] / denominator * 100).clip(-500, 500)
    df["month_number"] = df["month"].dt.month
    df["quarter"] = df["month"].dt.quarter
    df["next_month_consumption"] = group.shift(-1)
    return df


def make_model_table(consumption: pd.DataFrame, household: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = add_time_features(consumption)
    before = len(df)
    df = df.merge(household, on="household_ID", how="left", validate="many_to_one", indicator=True)
    matched = int((df["_merge"] == "both").sum())
    unmatched = int((df["_merge"] != "both").sum())
    df = df.drop(columns="_merge")
    # Survey answers are used only from the month-end after they were collected.
    available = df["wave1_date"].notna() & (df["wave1_date"] <= df["month"])
    history = df[["previous_month_consumption", "consumption_two_months_ago", "consumption_three_months_ago"]].notna().all(axis=1)
    target_available = df["next_month_consumption"].notna()
    df = df.loc[available & history & target_available].copy()
    report = {
        "consumption_rows_before_merge": before,
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "rows_after_availability_history_and_target_filters": int(len(df)),
        "duplicate_household_months_after_merge": int(df.duplicated(["household_ID", "month"]).sum()),
        "households_used": int(df["household_ID"].nunique()),
    }
    return df.sort_values(["month", "household_ID"]).reset_index(drop=True), report


def derived_from_inputs(record: dict) -> dict:
    result = dict(record)
    values = np.array([
        float(result["previous_month_consumption"]),
        float(result["consumption_two_months_ago"]),
        float(result["consumption_three_months_ago"]),
    ])
    result["three_month_average"] = float(values.mean())
    result["three_month_max"] = float(values.max())
    result["three_month_min"] = float(values.min())
    result["rolling_standard_deviation"] = float(values.std(ddof=0))
    result["consumption_change"] = float(values[0] - values[1])
    result["consumption_percentage_change"] = float(
        np.clip((values[0] - values[1]) / values[1] * 100, -500, 500) if values[1] else 0
    )
    result["quarter"] = int((int(result["month_number"]) - 1) // 3 + 1)
    return result


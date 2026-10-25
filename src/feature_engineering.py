from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_KWH = 200.0

CONSUMPTION_FEATURES = ["previous_month_consumption"] + [
    f"consumption_{name}_months_ago"
    for name in ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen")
]

DERIVED_NUMERIC = [
    "two_month_average", "four_month_average", "four_month_median", "four_month_min",
    "four_month_max", "four_month_standard_deviation", "six_month_average",
    "six_month_standard_deviation", "consumption_change", "two_month_consumption_change",
    "consumption_ratio", "months_above_90_last_four", "months_above_90_last_six",
    "all_last_four_above_90", "all_last_four_below_90", "distance_from_90",
    "same_month_last_year", "prediction_month", "prediction_month_sine",
    "prediction_month_cosine", "historical_average", "historical_standard_deviation",
    "historical_above_90_rate", "latest_difference_from_history", "four_month_trend",
]

HOUSEHOLD_NUMERIC = [
    "household_size", "floor_area", "monthly_expenditure", "total_appliances",
    "total_appliance_hours_week", "high_energy_appliance_count", "ac_count", "ac_hours_week",
]

BINARY = [
    "has_refrigerator", "has_washing_machine", "has_rice_cooker", "has_water_heater",
    "has_computer", "has_television", "has_air_conditioner", "home_business",
    "solar_generation", "grid_cooking",
]

CATEGORICAL = ["csc_area", "housing_type", "socio_economic_class", "meter_type", "water_heating_method"]
NUMERICAL_FEATURES = CONSUMPTION_FEATURES + DERIVED_NUMERIC + HOUSEHOLD_NUMERIC + BINARY
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL


def _trend(values: pd.DataFrame) -> pd.Series:
    array = values.to_numpy(dtype=float)
    x = np.array([-1.5, -0.5, 0.5, 1.5])
    counts = np.sum(~np.isnan(array), axis=1, keepdims=True)
    means = np.divide(np.nansum(array, axis=1, keepdims=True), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
    centered = array - means
    return pd.Series(np.nansum(centered * x, axis=1) / np.sum(x * x), index=values.index)


def add_time_features(consumption: pd.DataFrame) -> pd.DataFrame:
    df = consumption.sort_values(["household_ID", "month"]).copy()
    group = df.groupby("household_ID", sort=False)["consumption"]
    df["previous_month_consumption"] = df["consumption"]
    for lag, name in enumerate(CONSUMPTION_FEATURES[1:], start=1):
        df[name] = group.shift(lag)

    recent2 = df[CONSUMPTION_FEATURES[:2]]
    recent4 = df[CONSUMPTION_FEATURES[:4]]
    recent6 = df[CONSUMPTION_FEATURES[:6]]
    df["two_month_average"] = recent2.mean(axis=1)
    df["four_month_average"] = recent4.mean(axis=1)
    df["four_month_median"] = recent4.median(axis=1)
    df["four_month_min"] = recent4.min(axis=1)
    df["four_month_max"] = recent4.max(axis=1)
    df["four_month_standard_deviation"] = recent4.std(axis=1, ddof=0)
    df["six_month_average"] = recent6.mean(axis=1)
    df["six_month_standard_deviation"] = recent6.std(axis=1, ddof=0)
    df["consumption_change"] = df[CONSUMPTION_FEATURES[0]] - df[CONSUMPTION_FEATURES[1]]
    df["two_month_consumption_change"] = df[CONSUMPTION_FEATURES[0]] - df[CONSUMPTION_FEATURES[2]]
    denominator = df[CONSUMPTION_FEATURES[1]].replace(0, np.nan)
    df["consumption_ratio"] = (df[CONSUMPTION_FEATURES[0]] / denominator).clip(0, 10)
    df["months_above_90_last_four"] = (recent4 > 90).sum(axis=1)
    df["months_above_90_last_six"] = (recent6 > 90).sum(axis=1)
    df["all_last_four_above_90"] = (recent4 > 90).all(axis=1).astype(int)
    df["all_last_four_below_90"] = (recent4 <= 90).all(axis=1).astype(int)
    df["distance_from_90"] = df[CONSUMPTION_FEATURES[0]] - 90
    df["same_month_last_year"] = df[CONSUMPTION_FEATURES[-1]]
    df["prediction_month"] = (df["month"].dt.month % 12) + 1
    df["prediction_month_sine"] = np.sin(2 * np.pi * df["prediction_month"] / 12)
    df["prediction_month_cosine"] = np.cos(2 * np.pi * df["prediction_month"] / 12)
    df["historical_average"] = group.expanding().mean().reset_index(level=0, drop=True)
    df["historical_standard_deviation"] = group.expanding().std(ddof=0).reset_index(level=0, drop=True)
    above = df.assign(above=(df["consumption"] > 90).astype(float))
    df["historical_above_90_rate"] = above.groupby("household_ID")["above"].expanding().mean().reset_index(level=0, drop=True)
    df["latest_difference_from_history"] = df[CONSUMPTION_FEATURES[0]] - df["historical_average"]
    df["four_month_trend"] = _trend(recent4.iloc[:, ::-1])
    df["next_month_consumption"] = group.shift(-1)
    return df


def make_model_table(consumption: pd.DataFrame, households: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = add_time_features(consumption)
    before = len(df)
    df = df.merge(households, on="household_ID", how="left", validate="many_to_one", indicator=True)
    matched = int((df["_merge"] == "both").sum())
    df = df.drop(columns="_merge")
    available = df["wave1_date"].notna() & (df["wave1_date"] <= df["month"])
    history = df[CONSUMPTION_FEATURES[:3]].notna().all(axis=1)
    target_available = df["next_month_consumption"].notna()
    df = df.loc[available & history & target_available].copy()
    df["high_risk"] = (df["next_month_consumption"] >= TARGET_KWH).astype(int)
    report = {
        "consumption_rows_before_merge": before,
        "matched_rows": matched,
        "rows_after_filters": int(len(df)),
        "households_used": int(df["household_ID"].nunique()),
        "duplicate_household_months": int(df.duplicated(["household_ID", "month"]).sum()),
    }
    return df.sort_values(["month", "household_ID"]).reset_index(drop=True), report


def derived_from_inputs(record: dict) -> dict:
    result = dict(record)
    values = np.array([float(result[name]) for name in CONSUMPTION_FEATURES], dtype=float)
    recent2, recent4, recent6 = values[:2], values[:4], values[:6]
    result["two_month_average"] = float(recent2.mean())
    result["four_month_average"] = float(recent4.mean())
    result["four_month_median"] = float(np.median(recent4))
    result["four_month_min"] = float(recent4.min())
    result["four_month_max"] = float(recent4.max())
    result["four_month_standard_deviation"] = float(recent4.std(ddof=0))
    result["six_month_average"] = float(recent6.mean())
    result["six_month_standard_deviation"] = float(recent6.std(ddof=0))
    result["consumption_change"] = float(values[0] - values[1])
    result["two_month_consumption_change"] = float(values[0] - values[2])
    result["consumption_ratio"] = float(np.clip(values[0] / values[1], 0, 10) if values[1] else 0)
    result["months_above_90_last_four"] = int((recent4 > 90).sum())
    result["months_above_90_last_six"] = int((recent6 > 90).sum())
    result["all_last_four_above_90"] = int((recent4 > 90).all())
    result["all_last_four_below_90"] = int((recent4 <= 90).all())
    result["distance_from_90"] = float(values[0] - 90)
    result["same_month_last_year"] = float(values[-1])
    prediction_month = int(result.get("prediction_month", result.get("month_number", 1)))
    result["prediction_month"] = prediction_month
    result["prediction_month_sine"] = float(np.sin(2 * np.pi * prediction_month / 12))
    result["prediction_month_cosine"] = float(np.cos(2 * np.pi * prediction_month / 12))
    result.setdefault("historical_average", float(values.mean()))
    result.setdefault("historical_standard_deviation", float(values.std(ddof=0)))
    result.setdefault("historical_above_90_rate", float((values > 90).mean()))
    result["latest_difference_from_history"] = float(values[0] - result["historical_average"])
    x = np.array([-1.5, -0.5, 0.5, 1.5])
    ordered = recent4[::-1]
    result["four_month_trend"] = float(np.sum((ordered - ordered.mean()) * x) / np.sum(x * x))
    return result

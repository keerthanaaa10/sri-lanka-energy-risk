from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


YES_VALUES = {"yes", "y", "true", "1"}
HIGH_ENERGY_APPLIANCES = {
    "Refrigerator",
    "Washing Machine",
    "Electric Oven",
    "Microwave",
    "Geyser / Hot water systems for bathrooms which operate from electricity",
    "Electric water heater to heat water for drinking purposes",
    "Electric cook tops (induction cookers, Infra-red cookers, hot plates)",
    "Clothes dryer",
    "Dish washer",
    "Separate Freezer",
    "Electric heater (to control room temperature)",
}


def yes_no(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin(YES_VALUES).astype(int)


def load_consumption(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "consumption" / "monthly_consumption.csv"
    data = pd.read_csv(path, low_memory=False)
    data["month"] = pd.to_datetime(data["month"], errors="coerce")
    cleaned = data["consumption"].astype("string").str.replace(",", "", regex=False)
    data["consumption"] = pd.to_numeric(cleaned, errors="coerce")
    data = data.dropna(subset=["household_ID", "month"])
    data = data.drop_duplicates(["household_ID", "month"], keep="last")
    return data.sort_values(["household_ID", "month"]).reset_index(drop=True)


def load_households(raw_dir: Path) -> pd.DataFrame:
    survey = raw_dir / "survey"
    wave = survey / "wave_1"

    dates = pd.read_csv(survey / "survey_dates.csv", low_memory=False)
    # The source uses month/day/year text (for example 01/08/2024 is January 8).
    dates["wave1_date"] = pd.to_datetime(dates["wave1"], dayfirst=False, errors="coerce")
    dates = dates[["household_ID", "wave1_date"]]

    info = pd.read_csv(wave / "w1_household_information_and_history.csv", low_memory=False)
    info = info.rename(columns={
        "electricity_provider_csc_area": "csc_area",
        "type_of_house": "housing_type",
        "no_of_household_members": "household_size",
        "total_monthly_expenditure_of_last_month": "monthly_expenditure",
        "type_of_electricity_meter": "meter_type",
    })
    info["home_business"] = yes_no(info["is_there_business_carried_out_in_the_household"])
    info = info[[
        "household_ID", "csc_area", "housing_type", "floor_area", "household_size",
        "home_business", "socio_economic_class", "monthly_expenditure", "meter_type",
    ]]

    appliances = pd.read_csv(wave / "w1_appliances.csv", low_memory=False)
    appliances["hours"] = pd.to_numeric(appliances["no_of_hours_used_during_last_week"], errors="coerce")
    appliances["is_high_energy"] = appliances["appliance_type"].isin(HIGH_ENERGY_APPLIANCES).astype(int)
    appliance_summary = appliances.groupby("household_ID").agg(
        total_appliances=("appliance_ID", "count"),
        total_appliance_hours_week=("hours", "sum"),
        high_energy_appliance_count=("is_high_energy", "sum"),
    ).reset_index()
    appliance_names = {
        "has_refrigerator": {"Refrigerator"},
        "has_washing_machine": {"Washing Machine"},
        "has_rice_cooker": {"Rice cooker"},
        "has_water_heater": {
            "Geyser / Hot water systems for bathrooms which operate from electricity",
            "Electric water heater to heat water for drinking purposes",
        },
        "has_computer": {"Computers", "Laptops"},
        "has_television": {"TV"},
    }
    for output, names in appliance_names.items():
        flag = appliances.assign(flag=appliances["appliance_type"].isin(names).astype(int))
        values = flag.groupby("household_ID")["flag"].max().rename(output).reset_index()
        appliance_summary = appliance_summary.merge(values, on="household_ID", how="left")

    ac = pd.read_csv(wave / "w1_ac_roster.csv", low_memory=False)
    ac["day_hours"] = pd.to_numeric(ac["no_of_hours_ac_was_on_during_daytime_last_week"], errors="coerce").fillna(0)
    ac["night_hours"] = pd.to_numeric(ac["no_of_hours_ac_was_on_during_night_last_week"], errors="coerce").fillna(0)
    ac["hours"] = ac["day_hours"] + ac["night_hours"]
    ac_summary = ac.groupby("household_ID").agg(
        ac_count=("ac_ID", "count"),
        ac_hours_week=("hours", "sum"),
    ).reset_index()
    ac_summary["has_air_conditioner"] = (ac_summary["ac_count"] > 0).astype(int)

    energy = pd.read_csv(wave / "w1_electricity_generation_water_heating_cooking.csv", low_memory=False)
    energy["solar_generation"] = yes_no(energy["generate_electicity_using_solar_energy"])
    energy["grid_cooking"] = yes_no(energy["electricity_from_national_grid_used_for_cooking"])
    energy = energy.rename(columns={"water_heating_method_for_bathing": "water_heating_method"})
    energy = energy[["household_ID", "solar_generation", "grid_cooking", "water_heating_method"]]

    households = dates.merge(info, on="household_ID", how="left", validate="one_to_one")
    households = households.merge(appliance_summary, on="household_ID", how="left", validate="one_to_one")
    households = households.merge(ac_summary, on="household_ID", how="left", validate="one_to_one")
    households = households.merge(energy, on="household_ID", how="left", validate="one_to_one")

    numeric_zero = [
        "total_appliances", "total_appliance_hours_week", "high_energy_appliance_count",
        "has_refrigerator", "has_washing_machine", "has_rice_cooker", "has_water_heater",
        "has_computer", "has_television", "ac_count", "ac_hours_week",
        "has_air_conditioner", "solar_generation", "grid_cooking",
    ]
    households[numeric_zero] = households[numeric_zero].fillna(0)
    households["water_heating_method"] = households["water_heating_method"].fillna("Unknown")
    return households


def dataset_audit(raw_dir: Path) -> pd.DataFrame:
    consumption = load_consumption(raw_dir)
    return pd.DataFrame([
        {"Check": "Monthly records", "Value": len(consumption)},
        {"Check": "Households", "Value": consumption["household_ID"].nunique()},
        {"Check": "Missing consumption", "Value": int(consumption["consumption"].isna().sum())},
        {"Check": "Duplicate household-month rows", "Value": int(consumption.duplicated(["household_ID", "month"]).sum())},
        {"Check": "Negative readings", "Value": int((consumption["consumption"] < 0).sum())},
    ])

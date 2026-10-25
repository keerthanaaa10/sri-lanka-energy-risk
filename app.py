from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.feature_engineering import CONSUMPTION_FEATURES
from src.predict import explain_prediction, load_model, predict_household


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "energy_risk_model.pkl"
SCHEMA_PATH = ROOT / "models" / "feature_schema.json"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(
    page_title="Electricity Risk Checker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
    --navy: #173b61;
    --blue: #245d8f;
    --green: #237a57;
    --red: #b33b32;
    --yellow: #e8b632;
    --page: #f4f6f8;
    --line: #d7dde3;
    --text: #20262d;
    --muted: #63707d;
}

.stApp { background: var(--page); color: var(--text); }
.block-container { max-width: 1120px; padding-top: 1.2rem; padding-bottom: 3rem; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
html, body, [class*="css"] { font-family: Arial, Helvetica, sans-serif; }

.colour-line {
    height: 5px;
    background: linear-gradient(90deg, #8d153a 0 38%, #ffbe29 38% 44%, #00534e 44% 100%);
}
.site-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 20px;
    background: var(--navy);
    color: white;
}
.site-header .mark {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    background: white;
    color: var(--navy);
    border-radius: 4px;
    font-size: 1.35rem;
}
.site-header h1 { margin: 0; font-size: 1.45rem; font-weight: 650; letter-spacing: 0; }
.site-header p { margin: 4px 0 0; font-size: .83rem; opacity: .82; }

.intro {
    background: white;
    border: 1px solid var(--line);
    border-top: 0;
    padding: 18px 20px;
    margin-bottom: 18px;
}
.intro p { margin: 0 0 12px; color: #3f4a54; line-height: 1.55; max-width: 820px; }
.model-info { display: flex; gap: 24px; flex-wrap: wrap; color: var(--muted); font-size: .82rem; }
.model-info b { color: var(--text); }

.form-heading {
    margin: 1.7rem 0 .85rem;
    padding-bottom: .45rem;
    border-bottom: 2px solid var(--blue);
    color: var(--navy);
    font-size: 1.05rem;
    font-weight: 650;
}
.form-heading small { float: right; color: var(--muted); font-size: .75rem; font-weight: 400; }
div[data-testid="stForm"] { background: white; border: 1px solid var(--line); border-radius: 3px; padding: 0 1.4rem 1.4rem; }
label[data-testid="stWidgetLabel"] p { font-size: .82rem; }
div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 3px !important; }
div[data-testid="stFormSubmitButton"] button {
    min-height: 2.8rem;
    background: var(--blue);
    color: white;
    border: 0;
    border-radius: 3px;
    font-weight: 600;
}
div[data-testid="stFormSubmitButton"] button:hover { background: var(--navy); color: white; border: 0; }

.demo-line {
    background: #fff8df;
    border: 1px solid #ead58b;
    padding: 12px 14px;
    margin: 1rem 0;
    font-size: .9rem;
}
.result-box {
    margin: 1.5rem 0;
    display: grid;
    grid-template-columns: 165px 1fr;
    background: white;
    border: 1px solid var(--line);
    border-left: 7px solid var(--green);
}
.result-box.high { border-left-color: var(--red); }
.score { padding: 18px; background: #eef6f1; border-right: 1px solid var(--line); }
.high .score { background: #faeeee; }
.score strong { display: block; font-size: 2.7rem; line-height: 1; color: var(--green); }
.high .score strong { color: var(--red); }
.score span { display: block; margin-top: 7px; color: var(--muted); font-size: .72rem; line-height: 1.35; }
.result-text { padding: 18px 20px; }
.result-text h2 { margin: 0 0 .45rem !important; color: var(--navy); font-size: 1.45rem !important; }
.result-text p { margin: 0 0 13px; color: #48535d; font-size: .88rem; }
.risk-track { height: 8px; background: #e2e7eb; border-radius: 5px; overflow: hidden; }
.risk-fill { height: 100%; background: var(--green); }
.high .risk-fill { background: var(--red); }

.subheading { margin: 1.4rem 0 .4rem; color: var(--navy); font-size: 1.08rem; font-weight: 650; }
.factor { display: flex; justify-content: space-between; gap: 1rem; padding: .65rem 0; border-bottom: 1px solid var(--line); font-size: .85rem; }
.factor b { color: var(--blue); font-size: .78rem; }
.notice { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .78rem; line-height: 1.5; }

@media (max-width: 700px) {
    .block-container { padding: .7rem .8rem 2rem; }
    .site-header { padding: 14px; }
    .site-header h1 { font-size: 1.15rem; }
    .model-info { gap: 8px 18px; }
    .form-heading small { display: none; }
    .result-box { grid-template-columns: 1fr; }
    .score { border-right: 0; border-bottom: 1px solid var(--line); }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def resources():
    model = load_model(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    metadata = json.loads(METADATA_PATH.read_text())
    return model, schema, metadata


def defaults(kind: str) -> dict:
    common = {
        "prediction_month": 10,
        "csc_area": "GALLE",
        "housing_type": "Single House - Single Floor",
        "socio_economic_class": "SEC C",
        "meter_type": "Non smart meter",
        "water_heating_method": "None, we do not use hot water for bathing or body wash purposes.",
        "home_business": 0,
        "solar_generation": 0,
        "grid_cooking": 0,
    }
    low = {
        **common,
        "household_size": 3,
        "floor_area": 700.0,
        "monthly_expenditure": 45000.0,
        "total_appliances": 8,
        "total_appliance_hours_week": 120.0,
        "high_energy_appliance_count": 1,
        "ac_count": 0,
        "ac_hours_week": 0.0,
        "has_refrigerator": 1,
        "has_washing_machine": 0,
        "has_rice_cooker": 1,
        "has_water_heater": 0,
        "has_computer": 0,
        "has_television": 1,
        "has_air_conditioner": 0,
    }
    for index, feature in enumerate(CONSUMPTION_FEATURES):
        low[feature] = float(54 - index)
    if kind == "high":
        high = {
            **common,
            "household_size": 6,
            "floor_area": 2100.0,
            "monthly_expenditure": 145000.0,
            "total_appliances": 25,
            "total_appliance_hours_week": 390.0,
            "high_energy_appliance_count": 7,
            "ac_count": 2,
            "ac_hours_week": 72.0,
            "has_refrigerator": 1,
            "has_washing_machine": 1,
            "has_rice_cooker": 1,
            "has_water_heater": 1,
            "has_computer": 1,
            "has_television": 1,
            "has_air_conditioner": 1,
        }
        for index, feature in enumerate(CONSUMPTION_FEATURES):
            high[feature] = float(260 - index * 4)
        return high
    if kind == "invalid":
        return {**low, "previous_month_consumption": -20.0}
    return low


def heading(title: str, note: str = "") -> None:
    note_html = f"<small>{note}</small>" if note else ""
    st.markdown(f'<div class="form-heading">{title}{note_html}</div>', unsafe_allow_html=True)


model, schema, metadata = resources()
demo = st.query_params.get("demo", "")
record = defaults(demo if demo in {"low", "high", "invalid"} else "low")
accuracy = metadata["test_metrics"]["Accuracy"] * 100
balanced_accuracy = metadata["test_metrics"]["Balanced Accuracy"] * 100

st.markdown(
    f"""
<div class="colour-line"></div>
<div class="site-header">
  <div class="mark">⚡</div>
  <div><h1 class="brand">Household Electricity Risk Checker</h1><p>Sri Lanka residential electricity project</p></div>
</div>
<div class="intro">
  <p>This tool estimates whether a household's electricity use could reach <b>200 kWh or more next month</b>. Enter the billing history and household details below.</p>
  <div class="model-info">
    <span><b>Model:</b> {metadata['model_name']}</span>
    <span><b>Test accuracy:</b> {accuracy:.2f}%</span>
    <span><b>Balanced accuracy:</b> {balanced_accuracy:.2f}%</span>
    <span><b>Data:</b> {metadata['households_used']:,} households</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if demo not in {"low", "high", "invalid"}:
    with st.form("risk_form"):
        heading("1. Electricity-use history", "Use the values shown on the last 13 bills")
        a, b, c, d = st.columns([1, 1, 1, .75])
        record["consumption_three_months_ago"] = a.number_input("Three months ago (kWh)", value=record["consumption_three_months_ago"], step=1.0)
        record["consumption_two_months_ago"] = b.number_input("Two months ago (kWh)", value=record["consumption_two_months_ago"], step=1.0)
        record["previous_month_consumption"] = c.number_input("Previous month (kWh)", value=record["previous_month_consumption"], step=1.0)
        record["prediction_month"] = d.number_input("Prediction month", min_value=1, max_value=12, value=record["prediction_month"])
        with st.expander("Earlier billing history (required for the trained model)"):
            earlier = CONSUMPTION_FEATURES[3:]
            labels = ["Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen"]
            for start in range(0, len(earlier), 2):
                cols = st.columns(2)
                for offset, feature in enumerate(earlier[start:start + 2]):
                    label = labels[start + offset] + " months ago (kWh)"
                    record[feature] = cols[offset].number_input(label, value=record[feature], step=1.0)

        heading("2. Household information")
        a, b, c = st.columns(3)
        record["household_size"] = a.number_input("Number of household members", min_value=1, value=record["household_size"])
        record["floor_area"] = b.number_input("Floor area (sq ft)", min_value=0.0, value=record["floor_area"], step=50.0)
        record["monthly_expenditure"] = c.number_input("Monthly expenditure (LKR)", min_value=0.0, value=record["monthly_expenditure"], step=1000.0)
        a, b, c = st.columns(3)
        record["csc_area"] = a.selectbox("Electricity service area", schema["categories"]["csc_area"], index=schema["categories"]["csc_area"].index(record["csc_area"]))
        record["housing_type"] = b.selectbox("Housing type", schema["categories"]["housing_type"], index=schema["categories"]["housing_type"].index(record["housing_type"]))
        record["socio_economic_class"] = c.selectbox("Survey economic group", schema["categories"]["socio_economic_class"], index=schema["categories"]["socio_economic_class"].index(record["socio_economic_class"]))
        a, b = st.columns(2)
        record["meter_type"] = a.selectbox("Meter type", schema["categories"]["meter_type"], index=schema["categories"]["meter_type"].index(record["meter_type"]))
        record["water_heating_method"] = b.selectbox("Water heating method", schema["categories"]["water_heating_method"], index=0)

        heading("3. Appliances and weekly use")
        a, b, c, d, e = st.columns(5)
        record["total_appliances"] = a.number_input("Total appliances", min_value=0, value=record["total_appliances"])
        record["high_energy_appliance_count"] = b.number_input("High-energy items", min_value=0, value=record["high_energy_appliance_count"])
        record["total_appliance_hours_week"] = c.number_input("Use hours/week", min_value=0.0, value=record["total_appliance_hours_week"])
        record["ac_count"] = d.number_input("AC units", min_value=0, value=record["ac_count"])
        record["ac_hours_week"] = e.number_input("AC hours/week", min_value=0.0, value=record["ac_hours_week"])

        appliance_labels = {
            "has_refrigerator": "Refrigerator",
            "has_washing_machine": "Washing machine",
            "has_rice_cooker": "Rice cooker",
            "has_water_heater": "Electric water heater",
            "has_computer": "Computer or laptop",
            "has_television": "Television",
            "has_air_conditioner": "Air conditioner",
        }
        cols = st.columns(4)
        for i, (key, label) in enumerate(appliance_labels.items()):
            record[key] = int(cols[i % 4].checkbox(label, value=bool(record[key])))

        heading("4. Other electricity use")
        a, b, c = st.columns(3)
        record["home_business"] = int(a.checkbox("Business operated at home"))
        record["solar_generation"] = int(b.checkbox("Solar electricity generation"))
        record["grid_cooking"] = int(c.checkbox("Electricity used for cooking"))
        st.caption("Personal details such as name, address and account number are not required.")
        submitted = st.form_submit_button("Check next-month risk", use_container_width=True)
else:
    submitted = True
    st.markdown(
        "<div class='demo-line'><b>Demonstration input:</b> "
        + "{:.0f} → {:.0f} → {:.0f} kWh, {} people, {} appliances and {} AC unit(s).</div>".format(
            record["consumption_three_months_ago"],
            record["consumption_two_months_ago"],
            record["previous_month_consumption"],
            record["household_size"],
            record["total_appliances"],
            record["ac_count"],
        ),
        unsafe_allow_html=True,
    )

if submitted:
    outcome = predict_household(record, model=model)
    if outcome["error"]:
        st.error("Please correct the input: " + outcome["error"], icon="⚠️")
    else:
        high = outcome["predicted_class"] == 1
        result_class = "high" if high else ""
        probability = outcome["probability"] * 100
        message = (
            "The entered information suggests that usage may reach 200 kWh or more next month."
            if high
            else "The entered information suggests that usage is less likely to reach 200 kWh next month."
        )
        st.markdown(
            f"""
<div class="result-box {result_class}">
  <div class="score"><strong>{probability:.1f}%</strong><span>probability of high electricity-use risk</span></div>
  <div class="result-text"><h2>{outcome['risk_label']}</h2><p>{message}</p><div class="risk-track"><div class="risk-fill" style="width:{probability:.1f}%"></div></div></div>
</div>
""",
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="subheading">Main factors in this result</div>', unsafe_allow_html=True)
            with st.spinner("Checking the strongest model factors..."):
                factors = explain_prediction(record, model)
            for factor in factors:
                arrow = "↑" if factor["direction"] == "higher" else "↓"
                st.markdown(
                    f'<div class="factor"><span>{factor["feature"]}</span><b>{arrow} {factor["direction"]} risk</b></div>',
                    unsafe_allow_html=True,
                )
            st.caption("These are model associations and do not prove cause and effect.")
        with right:
            st.markdown('<div class="subheading">Suggested checks</div>', unsafe_allow_html=True)
            suggestions = []
            if record["previous_month_consumption"] >= 200:
                suggestions.append("Check which appliances were used more than usual last month.")
            if record["has_air_conditioner"]:
                suggestions.append("Use an AC timer and close doors and windows while it is running.")
            if record["has_water_heater"]:
                suggestions.append("Limit electric water-heating time and switch it off after use.")
            suggestions.extend(
                [
                    "Compare the next bill with the recent three-month average.",
                    "Switch off unused devices instead of leaving them on standby.",
                ]
            )
            for item in suggestions[:4]:
                st.markdown(f"- {item}")

st.markdown(
    """<div class="notice"><b>Important:</b> This is a machine-learning estimate, not an exact bill forecast. The application does not save the values entered in the form.</div>""",
    unsafe_allow_html=True,
)

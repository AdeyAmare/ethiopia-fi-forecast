# app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -------------------------------
# Load Data
# -------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models" / "forecasts"

# Forecast files
forecast_files = [
    MODELS_DIR / "ACC_OWNERSHIP_forecast.csv",
    MODELS_DIR / "USG_TELEBIRR_USERS_forecast.csv"
]

dfs = []
for f in forecast_files:
    if f.exists():
        df = pd.read_csv(f, parse_dates=["forecast_date"])
        # infer indicator_code from filename if missing
        if "indicator_code" not in df.columns:
            df["indicator_code"] = f.stem.replace("_forecast", "")
        # create forecast column if missing
        if "forecast" not in df.columns and "event_augmented" in df.columns:
            df["forecast"] = df["event_augmented"]
        dfs.append(df)
df_forecasts = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# P2P / ATM ratio files
ratio_files = {
    "base": MODELS_DIR / "P2P_ATM_Ratio_base.csv",
    "optimistic": MODELS_DIR / "P2P_ATM_Ratio_optimistic.csv",
    "pessimistic": MODELS_DIR / "P2P_ATM_Ratio_pessimistic.csv"
}

ratios = {}
for scenario, path in ratio_files.items():
    if path.exists():
        df = pd.read_csv(path, parse_dates=["forecast_date"])
        ratios[scenario] = df

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Ethiopia Financial Inclusion Dashboard")
page = st.sidebar.radio("Select Page:", ["Overview", "Trends", "Forecasts", "Projections"])

# -------------------------------
# Overview Page
# -------------------------------
if page == "Overview":
    st.title("Overview of Financial Inclusion Indicators")
    
    if not df_forecasts.empty:
        latest = df_forecasts.groupby("indicator_code").last().reset_index()
        st.subheader("Key Metrics")
        cols = st.columns(len(latest))
        for i, indicator in enumerate(latest["indicator_code"]):
            value = latest.loc[latest["indicator_code"] == indicator, "trend_forecast"].values[0]
            cols[i].metric(label=indicator.replace("_", " "), value=f"{value:,.0f}")
    else:
        st.info("Forecast data not available.")

    st.subheader("P2P / ATM Crossover Ratio")
    selected_scenario = st.selectbox("Select Scenario", options=list(ratios.keys()))
    if selected_scenario in ratios:
        df_ratio = ratios[selected_scenario]
        fig = px.line(df_ratio, x="forecast_date", y="P2P_ATM_Ratio",
                      title=f"P2P / ATM Crossover Ratio ({selected_scenario.capitalize()})")
        st.plotly_chart(fig)
    else:
        st.info("P2P / ATM ratio data not available for this scenario.")

# -------------------------------
# Trends Page
# -------------------------------
elif page == "Trends":
    st.title("Trends in Financial Inclusion Indicators")
    if not df_forecasts.empty:
        indicators = df_forecasts["indicator_code"].unique().tolist()
        selected = st.multiselect("Select Indicators", options=indicators, default=indicators[:2])
        start_date = st.date_input("Start Date", df_forecasts["forecast_date"].min())
        end_date = st.date_input("End Date", df_forecasts["forecast_date"].max())
        
        df_filtered = df_forecasts[df_forecasts["indicator_code"].isin(selected)]
        df_filtered = df_filtered[(df_filtered["forecast_date"] >= pd.Timestamp(start_date)) &
                                  (df_filtered["forecast_date"] <= pd.Timestamp(end_date))]
        fig = px.line(df_filtered, x="forecast_date", y="trend_forecast", color="indicator_code",
                      title="Indicator Trends")
        st.plotly_chart(fig)
    else:
        st.info("Forecast data not available.")

# -------------------------------
# Forecasts Page
# -------------------------------
elif page == "Forecasts":
    st.title("Forecasts and Scenarios")
    scenario_options = df_forecasts["scenario"].unique() if "scenario" in df_forecasts.columns else list(ratios.keys())
    selected_scenario = st.selectbox("Select Scenario", scenario_options)
    
    if "scenario" in df_forecasts.columns:
        df_scenario = df_forecasts[df_forecasts["scenario"] == selected_scenario]
    else:
        df_scenario = df_forecasts.copy()

    if not df_scenario.empty:
        fig = px.line(df_scenario, x="forecast_date", y="forecast", color="indicator_code",
                      title=f"Forecasts ({selected_scenario.capitalize()} Scenario)")
        st.plotly_chart(fig)
        st.download_button(
            "Download Forecast CSV",
            df_scenario.to_csv(index=False).encode('utf-8'),
            file_name=f"forecast_{selected_scenario}.csv",
            mime="text/csv"
        )
    else:
        st.info("No forecast data available for this scenario.")

# -------------------------------
# Inclusion Projections Page
# -------------------------------
elif page == "Projections":
    st.title("Financial Inclusion Projections")
    target = 60  # percent
    if "ACC_OWNERSHIP" in df_forecasts["indicator_code"].unique():
        df_target = df_forecasts[df_forecasts["indicator_code"] == "ACC_OWNERSHIP"]
        fig = px.line(df_target, x="forecast_date", y="forecast", title="Account Ownership Projection")
        fig.add_hline(y=target, line_dash="dash", line_color="red",
                      annotation_text="60% Target", annotation_position="top right")
        st.plotly_chart(fig)
        st.markdown("This projection shows progress toward the 60% financial inclusion goal under selected scenarios.")
    else:
        st.info("ACC_OWNERSHIP forecast not available.")

# app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import List, Dict

# -------------------------------
# Configuration & Constants
# -------------------------------
DATA_DIR: Path = Path(__file__).parent.parent / "data"
MODELS_DIR: Path = Path(__file__).parent.parent / "models" / "forecasts"

TARGET_ACCOUNT_OWNERSHIP: float = 60.0  # percent

INDICATOR_LABELS: Dict[str, str] = {
    "ACC_OWNERSHIP": "Account Ownership (%)",
    "USG_TELEBIRR_USERS": "Telebirr Usage (Users)",
}

# -------------------------------
# Load Forecast Data
# -------------------------------
forecast_files: List[Path] = [
    MODELS_DIR / "ACC_OWNERSHIP_forecast.csv",
    MODELS_DIR / "USG_TELEBIRR_USERS_forecast.csv",
]

dfs: List[pd.DataFrame] = []

for f in forecast_files:
    try:
        if f.exists():
            df = pd.read_csv(f, parse_dates=["forecast_date"])

            if "indicator_code" not in df.columns:
                df["indicator_code"] = f.stem.replace("_forecast", "")

            if "forecast" not in df.columns and "event_augmented" in df.columns:
                df["forecast"] = df["event_augmented"]

            dfs.append(df)
    except Exception as e:
        st.error(f"Error loading forecast file {f.name}: {e}")

df_forecasts: pd.DataFrame = (
    pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
)

if not df_forecasts.empty:
    df_forecasts["indicator_label"] = (
        df_forecasts["indicator_code"]
        .map(INDICATOR_LABELS)
        .fillna(df_forecasts["indicator_code"])
    )

# -------------------------------
# Load P2P / ATM Ratio Data
# -------------------------------
ratio_files: Dict[str, Path] = {
    "base": MODELS_DIR / "P2P_ATM_Ratio_base.csv",
    "optimistic": MODELS_DIR / "P2P_ATM_Ratio_optimistic.csv",
    "pessimistic": MODELS_DIR / "P2P_ATM_Ratio_pessimistic.csv",
}

ratios: Dict[str, pd.DataFrame] = {}

for scenario, path in ratio_files.items():
    try:
        if path.exists():
            ratios[scenario] = pd.read_csv(
                path, parse_dates=["forecast_date"]
            )
    except Exception as e:
        st.error(f"Error loading ratio file {path.name}: {e}")

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Ethiopia Financial Inclusion Dashboard")
page: str = st.sidebar.radio(
    "Select Page:",
    ["Overview", "Trends", "Forecasts", "Projections"],
)

# -------------------------------
# Overview Page
# -------------------------------
if page == "Overview":
    st.title("Overview of Financial Inclusion Indicators")

    st.markdown(
        """
        This dashboard tracks Ethiopia’s financial inclusion trajectory using
        forecasted indicators on **account access**, **digital usage**, and
        **payment channel substitution**.
        """
    )

    if not df_forecasts.empty:
        try:
            latest = (
                df_forecasts
                .sort_values("forecast_date")
                .groupby("indicator_code")
                .tail(1)
            )

            st.subheader("Key Metrics")
            cols = st.columns(len(latest))

            for i, row in latest.iterrows():
                unit = "%" if row["indicator_code"] == "ACC_OWNERSHIP" else ""
                cols[list(latest.index).index(i)].metric(
                    label=row["indicator_label"],
                    value=f"{row['forecast']:,.1f}{unit}",
                )

        except Exception as e:
            st.error(f"Error displaying key metrics: {e}")
    else:
        st.info("Forecast data not available.")

    st.subheader("P2P / ATM Crossover Ratio")

    st.markdown(
        "This chart shows how digital peer-to-peer transactions are projected "
        "to substitute traditional ATM usage over time."
    )

    selected_scenario: str = st.selectbox(
        "Select Scenario", options=list(ratios.keys())
    )

    if selected_scenario in ratios:
        try:
            fig = px.line(
                ratios[selected_scenario],
                x="forecast_date",
                y="P2P_ATM_Ratio",
                title=f"P2P / ATM Ratio ({selected_scenario.capitalize()} Scenario)",
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error displaying P2P/ATM ratio chart: {e}")
    else:
        st.info("Ratio data not available for this scenario.")

# -------------------------------
# Trends Page
# -------------------------------
elif page == "Trends":
    st.title("Trends in Financial Inclusion Indicators")

    st.markdown(
        """
        Explore how access and usage indicators evolve over time.
        Divergence between trends may indicate **usage growth without new inclusion**.
        """
    )

    if not df_forecasts.empty:
        try:
            indicators = (
                df_forecasts[["indicator_code", "indicator_label"]]
                .drop_duplicates()
            )

            selected = st.multiselect(
                "Select Indicators",
                options=indicators["indicator_code"],
                default=indicators["indicator_code"].tolist(),
                format_func=lambda x: INDICATOR_LABELS.get(x, x),
            )

            start_date = st.date_input(
                "Start Date", df_forecasts["forecast_date"].min()
            )
            end_date = st.date_input(
                "End Date", df_forecasts["forecast_date"].max()
            )

            df_filtered = df_forecasts[
                (df_forecasts["indicator_code"].isin(selected))
                & (df_forecasts["forecast_date"] >= pd.Timestamp(start_date))
                & (df_forecasts["forecast_date"] <= pd.Timestamp(end_date))
            ]

            fig = px.line(
                df_filtered,
                x="forecast_date",
                y="forecast",
                color="indicator_label",
                title="Indicator Trends Over Time",
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error displaying trends: {e}")
    else:
        st.info("Forecast data not available.")

# -------------------------------
# Forecasts Page
# -------------------------------
elif page == "Forecasts":
    st.title("Forecasts and Scenarios")

    st.markdown(
        "Compare forecast trajectories under different macroeconomic "
        "and policy scenarios."
    )

    try:
        scenario_options = (
            df_forecasts["scenario"].unique().tolist()
            if "scenario" in df_forecasts.columns
            else list(ratios.keys())
        )

        selected_scenario: str = st.selectbox(
            "Select Scenario", scenario_options
        )

        if "scenario" in df_forecasts.columns:
            df_scenario = df_forecasts[
                df_forecasts["scenario"] == selected_scenario
            ]
        else:
            df_scenario = df_forecasts.copy()

        if not df_scenario.empty:
            fig = px.line(
                df_scenario,
                x="forecast_date",
                y="forecast",
                color="indicator_label",
                title=f"Forecasts ({selected_scenario.capitalize()} Scenario)",
            )

            st.plotly_chart(fig)

            st.download_button(
                "Download Forecast CSV",
                df_scenario.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_{selected_scenario}.csv",
                mime="text/csv",
            )
        else:
            st.info("No forecast data available for this scenario.")

    except Exception as e:
        st.error(f"Error displaying forecasts: {e}")

# -------------------------------
# Projections Page
# -------------------------------
elif page == "Projections":
    st.title("Financial Inclusion Projections")

    st.markdown(
        "This projection evaluates progress toward Ethiopia’s "
        "**60% account ownership target**."
    )

    try:
        if "ACC_OWNERSHIP" in df_forecasts["indicator_code"].unique():
            df_target = df_forecasts[
                df_forecasts["indicator_code"] == "ACC_OWNERSHIP"
            ]

            fig = px.line(
                df_target,
                x="forecast_date",
                y="forecast",
                title="Account Ownership Projection (%)",
            )

            fig.add_hline(
                y=TARGET_ACCOUNT_OWNERSHIP,
                line_dash="dash",
                line_color="red",
                annotation_text="60% Target",
                annotation_position="top right",
            )

            st.plotly_chart(fig)

            st.markdown(
                "Closing the gap to the target depends on sustained growth "
                "in both **access expansion** and **digital usage adoption**."
            )
        else:
            st.info("Account ownership forecast not available.")

    except Exception as e:
        st.error(f"Error displaying projections: {e}")

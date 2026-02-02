import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.linear_model import LinearRegression
import logging
import matplotlib.pyplot as plt
import scipy.stats as stats

from src.event_impact_modeling import EventImpactModeler

class FinancialInclusionForecaster:
    """
    Task 4: Forecasting Access and Usage Indicators (2025-2027)

    Features:
    - Trend-based regression (with confidence intervals)
    - Event-augmented forecast
    - Scenario analysis (optimistic, base, pessimistic)
    - Forecast visualization
    - Event impact summary
    """

    def __init__(self, model: EventImpactModeler, forecast_horizon: int = 36):
        self.model = model
        self.forecast_horizon = forecast_horizon  # months
        self.logger = logging.getLogger("FI_Forecaster")
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(handler)

        self.forecasts = {}
        self.models = {}

        # Clean indicator codes for consistency
        self.model.obs["indicator_code"] = (
            self.model.obs["indicator_code"].astype(str)
            .str.strip()
            .str.upper()
        )
        self.model.links["related_indicator"] = (
            self.model.links["related_indicator"].astype(str)
            .str.strip()
            .str.upper()
        )

    # ------------------------------------------------------------
    # 1. Fit baseline trend with linear regression
    # ------------------------------------------------------------
    def fit_trend(self, indicator_code: str):
        df = self.model.obs[self.model.obs["indicator_code"] == indicator_code].copy()
        df = df.sort_values("observation_date")
        if df.empty:
            self.logger.warning(
                f"No historical observations found for {indicator_code}. "
                f"Trend-based forecast will be skipped; using event effects only."
            )
            return None

        df["months_since_start"] = ((df["observation_date"] - df["observation_date"].min())
                                    .dt.days // 30)
        X = df[["months_since_start"]].values
        y = df["value_numeric"].values

        reg = LinearRegression()
        reg.fit(X, y)
        self.models[indicator_code] = reg

        # Save residuals for confidence intervals
        df["residuals"] = y - reg.predict(X)
        self.models[f"{indicator_code}_residuals"] = df["residuals"]

        self.logger.info(f"Fitted trend for {indicator_code} using {len(df)} observations.")
        return reg

    # ------------------------------------------------------------
    # 2. Generate trend forecast with confidence intervals
    # ------------------------------------------------------------
    def trend_forecast(self, indicator_code: str):
        reg = self.models.get(indicator_code) or self.fit_trend(indicator_code)

        if reg is None:
            last_date = self.model.events["event_date"].max()
            if pd.isna(last_date):
                last_date = pd.Timestamp.today()
            forecast_index = [last_date + pd.DateOffset(months=m) for m in range(1, self.forecast_horizon+1)]
            return pd.DataFrame(index=forecast_index, columns=["trend_forecast", "lower_ci", "upper_ci"]).fillna(0)

        last_date = self.model.obs["observation_date"].max()
        start_date = self.model.obs["observation_date"].min()
        months = np.arange(1, self.forecast_horizon + 1)
        forecast_index = [last_date + pd.DateOffset(months=int(m)) for m in months]
        months_since_start = np.array([
            (d - start_date).days // 30 for d in forecast_index
        ]).reshape(-1, 1)
        forecast_values = reg.predict(months_since_start)

        # Confidence intervals
        residuals = self.models.get(f"{indicator_code}_residuals")
        if residuals is not None and len(residuals) > 1:
            std_err = residuals.std()
            ci = 1.96 * std_err  # 95% CI
        else:
            ci = 0

        return pd.DataFrame({
            "forecast_date": forecast_index,
            "trend_forecast": forecast_values,
            "lower_ci": forecast_values - ci,
            "upper_ci": forecast_values + ci
        }).set_index("forecast_date")

    # ------------------------------------------------------------
    # 3. Event-augmented forecast
    # ------------------------------------------------------------
    def event_augmented_forecast(self, indicator_code: str):
        trend_df = self.trend_forecast(indicator_code)
        if trend_df.empty:
            return trend_df

        trend_df["event_augmented"] = trend_df["trend_forecast"].copy()
        links = self.model.links[self.model.links["related_indicator"] == indicator_code]

        for _, link in links.iterrows():
            event = self.model.events[self.model.events["record_id"] == link["parent_id"]]
            if event.empty:
                continue
            event_date = event["event_date"].iloc[0]
            lag = pd.DateOffset(months=int(link["lag_months"]))
            start = event_date + lag
            max_effect = self.model.MAGNITUDE_MAP.get(link["impact_magnitude"], 0)
            if link["impact_direction"] == "decrease":
                max_effect = -max_effect

            for date in trend_df.index:
                if date >= start:
                    months_since = (date.year - start.year) * 12 + (date.month - start.month)
                    ramp = min(months_since / self.model.ramp_months, 1)
                    trend_df.loc[date, "event_augmented"] += max_effect * ramp

        return trend_df

    # ------------------------------------------------------------
    # 4. Scenario analysis
    # ------------------------------------------------------------
    def scenario_forecast(self, indicator_code: str):
        base_df = self.event_augmented_forecast(indicator_code)
        if base_df.empty:
            self.logger.warning(f"No forecast generated for {indicator_code}.")
            return {}

        scenarios = {"optimistic": 1.2, "base": 1.0, "pessimistic": 0.8}
        scenario_dfs = {}
        for name, factor in scenarios.items():
            scenario_dfs[name] = base_df.copy()
            scenario_dfs[name]["forecast"] = base_df["event_augmented"] * factor

        self.forecasts[indicator_code] = base_df
        return scenario_dfs

    # ------------------------------------------------------------
    # 5. Save forecast artifacts
    # ------------------------------------------------------------
    def save_forecast_artifacts(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        for indicator, df in self.forecasts.items():
            df.to_csv(output_path / f"{indicator}_forecast.csv", index=True)
        with open(output_path / "fitted_models.pkl", "wb") as f:
            pickle.dump(self.models, f)
        self.logger.info(f"Saved forecast artifacts to {output_path}")

    # ------------------------------------------------------------
    # 6. Visualization of forecasts and scenarios
    # ------------------------------------------------------------
    def plot_forecast(self, indicator_code: str):
        if indicator_code not in self.forecasts:
            self.logger.warning(f"No forecast data for {indicator_code}")
            return

        df = self.forecasts[indicator_code]
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["trend_forecast"], label="Trend")
        plt.plot(df.index, df["event_augmented"], label="Event-Augmented")
        plt.fill_between(df.index, df["lower_ci"], df["upper_ci"], color='gray', alpha=0.2, label="95% CI")
        plt.title(f"{indicator_code} Forecast")
        plt.xlabel("Date")
        plt.ylabel("Indicator Value")
        plt.legend()
        plt.show()

    # ------------------------------------------------------------
    # 7. Event impact summary
    # ------------------------------------------------------------
    def summarize_event_impacts(self, indicator_code: str):
        links = self.model.links[self.model.links["related_indicator"] == indicator_code]
        impacts = []
        for _, link in links.iterrows():
            event = self.model.events[self.model.events["record_id"] == link["parent_id"]]
            if event.empty:
                continue
            event_date = event["event_date"].iloc[0]
            magnitude = self.model.MAGNITUDE_MAP.get(link["impact_magnitude"], 0)
            direction = link["impact_direction"]
            impacts.append({
                "event_name": event["event_name"].iloc[0],
                "event_date": event_date,
                "magnitude": magnitude,
                "direction": direction,
                "lag_months": link["lag_months"]
            })
        return pd.DataFrame(impacts)

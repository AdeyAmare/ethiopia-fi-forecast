import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.linear_model import LinearRegression
import logging
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict

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
        - Supports all indicators in the dataset

    Attributes:
        model (EventImpactModeler): Event impact modeler instance
        forecast_horizon (int): Number of months to forecast
        logger (Logger): Logger for information and warnings
        forecasts (dict): Stores event-augmented forecasts for each indicator
        models (dict): Stores fitted regression models and residuals
    """

    def __init__(self, model: EventImpactModeler, forecast_horizon: int = 36):
        self.model = model
        self.forecast_horizon: int = forecast_horizon  # months
        self.logger = logging.getLogger("FI_Forecaster")
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(handler)

        self.forecasts: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, Union[LinearRegression, pd.Series]] = {}

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
    def fit_trend(self, indicator_code: str) -> Optional[LinearRegression]:
        """
        Fit a linear regression trend model for the given indicator.

        Args:
            indicator_code (str): Indicator code to fit

        Returns:
            LinearRegression | None: Fitted model or None if no data
        """
        try:
            df = self.model.obs[self.model.obs["indicator_code"] == indicator_code].copy()
            df = df.sort_values("observation_date")
            if df.empty:
                self.logger.warning(
                    f"No historical observations found for {indicator_code}. "
                    "Trend-based forecast will be skipped; using event effects only."
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
        except Exception as e:
            self.logger.error(f"Error fitting trend for {indicator_code}: {e}")
            return None

    # ------------------------------------------------------------
    # 2. Generate trend forecast with confidence intervals
    # ------------------------------------------------------------
    def trend_forecast(self, indicator_code: str) -> pd.DataFrame:
        """
        Generate a trend-based forecast for the indicator with 95% confidence intervals.

        Args:
            indicator_code (str): Indicator code

        Returns:
            DataFrame: Indexed by forecast_date, includes trend_forecast, lower_ci, upper_ci
        """
        try:
            reg = self.models.get(indicator_code) or self.fit_trend(indicator_code)

            if reg is None:
                last_date = self.model.events["event_date"].max()
                if pd.isna(last_date):
                    last_date = pd.Timestamp.today()
                forecast_index = [last_date + pd.DateOffset(months=m) for m in range(1, self.forecast_horizon + 1)]
                return pd.DataFrame(index=forecast_index, columns=["trend_forecast", "lower_ci", "upper_ci"]).fillna(0)

            last_date = self.model.obs["observation_date"].max()
            start_date = self.model.obs["observation_date"].min()
            months = np.arange(1, self.forecast_horizon + 1)
            forecast_index = [last_date + pd.DateOffset(months=int(m)) for m in months]
            months_since_start = np.array([
                (d - start_date).days // 30 for d in forecast_index
            ]).reshape(-1, 1)
            forecast_values = reg.predict(months_since_start)

            residuals = self.models.get(f"{indicator_code}_residuals")
            ci = 1.96 * residuals.std() if residuals is not None and len(residuals) > 1 else 0

            return pd.DataFrame({
                "forecast_date": forecast_index,
                "trend_forecast": forecast_values,
                "lower_ci": forecast_values - ci,
                "upper_ci": forecast_values + ci
            }).set_index("forecast_date")
        except Exception as e:
            self.logger.error(f"Error generating trend forecast for {indicator_code}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------
    # 3. Event-augmented forecast
    # ------------------------------------------------------------
    def event_augmented_forecast(self, indicator_code: str) -> pd.DataFrame:
        """
        Generate an event-augmented forecast based on trend + event effects.

        Args:
            indicator_code (str): Indicator code

        Returns:
            DataFrame: Indexed by forecast_date, includes trend_forecast, event_augmented
        """
        try:
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
        except Exception as e:
            self.logger.error(f"Error generating event-augmented forecast for {indicator_code}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------
    # 4. Scenario analysis
    # ------------------------------------------------------------
    def scenario_forecast(self, indicator_code: str) -> Dict[str, pd.DataFrame]:
        """
        Generate scenario forecasts (optimistic, base, pessimistic) for an indicator.

        Args:
            indicator_code (str): Indicator code

        Returns:
            dict: Scenario name -> DataFrame
        """
        try:
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
        except Exception as e:
            self.logger.error(f"Error generating scenario forecasts for {indicator_code}: {e}")
            return {}

    # ------------------------------------------------------------
    # 5. Forecast all indicators in the dataset
    # ------------------------------------------------------------
    def forecast_all_indicators(self):
        """
        Generate forecasts for all indicators in the dataset.
        """
        try:
            all_indicators = self.model.obs["indicator_code"].unique()
            self.logger.info(f"Generating forecasts for {len(all_indicators)} indicators...")
            for indicator in all_indicators:
                self.scenario_forecast(indicator)
        except Exception as e:
            self.logger.error(f"Error forecasting all indicators: {e}")

    # ------------------------------------------------------------
    # 6. Save forecast artifacts
    # ------------------------------------------------------------
    def save_forecast_artifacts(self, output_path: Union[str, Path]):
        """
        Save forecast CSVs and fitted models to disk.

        Args:
            output_path (str | Path): Directory to save artifacts
        """
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            for indicator, df in self.forecasts.items():
                df.to_csv(output_path / f"{indicator}_forecast.csv", index=True)
            with open(output_path / "fitted_models.pkl", "wb") as f:
                pickle.dump(self.models, f)
            self.logger.info(f"Saved forecast artifacts to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving forecast artifacts: {e}")

    # ------------------------------------------------------------
    # 7. Visualization of forecasts and scenarios
    # ------------------------------------------------------------
    def plot_forecast(self, indicator_code: str):
        """
        Plot trend and event-augmented forecasts with confidence intervals.

        Args:
            indicator_code (str): Indicator code
        """
        try:
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
        except Exception as e:
            self.logger.error(f"Error plotting forecast for {indicator_code}: {e}")

    # ------------------------------------------------------------
    # 8. Event impact summary
    # ------------------------------------------------------------
    def summarize_event_impacts(self, indicator_code: str) -> pd.DataFrame:
        """
        Return a summary of events affecting an indicator, including magnitude and lag.

        Args:
            indicator_code (str): Indicator code

        Returns:
            DataFrame: Columns include event_name, event_date, magnitude, direction, lag_months
        """
        try:
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
        except Exception as e:
            self.logger.error(f"Error summarizing event impacts for {indicator_code}: {e}")
            return pd.DataFrame()

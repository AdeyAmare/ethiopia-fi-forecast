import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Union, Optional


class EventImpactModeler:
    """
    Task 3: Event Impact Modeling

    Models how events (policies, product launches, infrastructure investments)
    affect financial inclusion indicators over time using impact_link metadata.

    Attributes:
        MAGNITUDE_MAP (dict): Maps impact magnitude strings to numeric point changes.
        data_path (Path): Path to the main dataset CSV.
        impact_links_path (Path): Path to the impact links CSV.
        ramp_months (int): Number of months over which event effects ramp up.
        df (DataFrame): Full dataset loaded from data_path.
        links (DataFrame): Impact link metadata loaded from impact_links_path.
        events (DataFrame): Subset of df corresponding to events.
        obs (DataFrame): Subset of df corresponding to observations.
        logger (Logger): Logger for info and warning messages.
    """

    MAGNITUDE_MAP = {
        "low": 0.5,
        "medium": 1.5,
        "high": 3.0,
    }

    def __init__(
        self,
        data_path: Union[str, Path],
        impact_links_path: Union[str, Path],
        ramp_months: int = 24,
    ):
        self.data_path = Path(data_path)
        self.impact_links_path = Path(impact_links_path)
        self.ramp_months = ramp_months

        # -------------------------------
        # Setup logger
        # -------------------------------
        self.logger = logging.getLogger("EventImpactModel")
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(handler)

        # -------------------------------
        # Load data robustly
        # -------------------------------
        try:
            self.df = pd.read_csv(self.data_path, dtype=str)
        except Exception as e:
            self.logger.error(f"Error loading data file {self.data_path}: {e}")
            self.df = pd.DataFrame()

        try:
            self.links = pd.read_csv(self.impact_links_path, dtype=str)
        except Exception as e:
            self.logger.error(f"Error loading impact links file {self.impact_links_path}: {e}")
            self.links = pd.DataFrame()

        # -------------------------------
        # Separate events and observations
        # -------------------------------
        self.events = self.df[self.df.get("record_type") == "event"].copy()
        self.obs = self.df[self.df.get("record_type") == "observation"].copy()

        # -------------------------------
        # Parse dates safely
        # -------------------------------
        self.events["event_date"] = pd.to_datetime(
            self.events.get("observation_date", None),
            errors="coerce",
        )
        self.obs["observation_date"] = pd.to_datetime(
            self.obs.get("observation_date", None),
            errors="coerce",
        )

        # Convert numeric columns safely
        self.obs["value_numeric"] = pd.to_numeric(
            self.obs.get("value_numeric", np.nan), errors="coerce"
        )
        self.links["lag_months"] = pd.to_numeric(
            self.links.get("lag_months", 0), errors="coerce"
        )

        self.logger.info(
            f"Loaded {len(self.events)} events, {len(self.obs)} observations, {len(self.links)} impact links"
        )

    # ------------------------------------------------------------------
    # 1. EVENT–INDICATOR SUMMARY
    # ------------------------------------------------------------------
    def event_indicator_summary(self) -> pd.DataFrame:
        """
        Join impact_links with events to show which events affect which indicators.

        Returns:
            DataFrame: Columns include parent_id, indicator, category, event_date,
                       pillar, related_indicator, impact_direction, impact_magnitude,
                       lag_months, confidence.
        """
        try:
            summary = self.links.merge(
                self.events[["record_id", "indicator", "event_date", "category"]],
                left_on="parent_id",
                right_on="record_id",
                how="left",
                suffixes=("", "_event"),
            )
            cols = [
                "parent_id",
                "indicator",
                "category",
                "event_date",
                "pillar",
                "related_indicator",
                "impact_direction",
                "impact_magnitude",
                "lag_months",
                "confidence",
            ]
            return summary[cols]
        except Exception as e:
            self.logger.error(f"Error building event-indicator summary: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 2. ASSOCIATION MATRIX
    # ------------------------------------------------------------------
    def build_association_matrix(self) -> pd.DataFrame:
        """
        Create an Event × Indicator matrix with estimated effect sizes (pp).

        Returns:
            DataFrame: Rows are events, columns are indicators, values are effect sizes.
        """
        try:
            rows = []

            for _, link in self.links.iterrows():
                magnitude = self.MAGNITUDE_MAP.get(link.get("impact_magnitude", "").lower(), np.nan)

                if link.get("impact_direction", "").lower() == "decrease":
                    magnitude = -magnitude

                rows.append(
                    {
                        "event_id": link.get("parent_id"),
                        "indicator": link.get("related_indicator"),
                        "effect_pp": magnitude,
                        "lag_months": float(link.get("lag_months", 0)),
                    }
                )

            df = pd.DataFrame(rows)

            matrix = df.pivot_table(
                index="event_id",
                columns="indicator",
                values="effect_pp",
                aggfunc="sum",
                fill_value=0,
            )

            return matrix
        except Exception as e:
            self.logger.error(f"Error building association matrix: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. SIMULATE INDICATOR PATH
    # ------------------------------------------------------------------
    def simulate_indicator(self, indicator_code: str) -> Optional[pd.DataFrame]:
        """
        Simulate an indicator path based on historical baseline + event effects.

        Args:
            indicator_code (str): The code of the indicator to simulate.

        Returns:
            DataFrame: Observed + simulated values over time.
        """
        try:
            base = (
                self.obs[self.obs["indicator_code"] == indicator_code]
                .sort_values("observation_date")
                .copy()
            )

            if base.empty:
                self.logger.warning(f"No observations found for {indicator_code}")
                return None

            base = base.set_index("observation_date")[["value_numeric"]].copy()
            base["simulated"] = base["value_numeric"].iloc[0]

            # Loop through links affecting this indicator
            for _, link in self.links[self.links["related_indicator"] == indicator_code].iterrows():
                event = self.events[self.events["record_id"] == link["parent_id"]]
                if event.empty:
                    continue

                event_date = event["event_date"].iloc[0]
                if pd.isna(event_date):
                    self.logger.warning(f"Event {link['parent_id']} has invalid date; skipping")
                    continue

                lag = pd.DateOffset(months=int(link.get("lag_months", 0)))
                start = event_date + lag

                max_effect = self.MAGNITUDE_MAP.get(link.get("impact_magnitude", "").lower(), 0)
                if str(link.get("impact_direction", "")).lower() == "decrease":
                    max_effect = -max_effect

                for date in base.index:
                    if date < start:
                        continue
                    months_since = (date.year - start.year) * 12 + (date.month - start.month)
                    ramp = min(months_since / self.ramp_months, 1)
                    base.loc[date, "simulated"] += max_effect * ramp

            return base
        except Exception as e:
            self.logger.error(f"Error simulating indicator {indicator_code}: {e}")
            return None

    # ------------------------------------------------------------------
    # 4. VALIDATION
    # ------------------------------------------------------------------
    def validate_against_actual(self, indicator_code: str) -> Optional[pd.DataFrame]:
        """
        Compare simulated vs observed values.

        Args:
            indicator_code (str): The code of the indicator to validate.

        Returns:
            DataFrame: Contains observed, simulated, and error_pp columns.
        """
        try:
            sim = self.simulate_indicator(indicator_code)
            if sim is None:
                return None
            sim["error_pp"] = sim["simulated"] - sim["value_numeric"]
            return sim
        except Exception as e:
            self.logger.error(f"Error validating indicator {indicator_code}: {e}")
            return None

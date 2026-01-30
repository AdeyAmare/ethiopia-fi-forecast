import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FinancialInclusionDataEnricher:
    """
    Task 1:
    - Load unified dataset
    - Validate schema using reference_codes
    - Explore dataset
    - Enrich with new observations, events, and impact_links
    - Log all additions
    """

    def __init__(
        self,
        data_path: str,
        reference_path: str,
        output_dir: str = "data/processed",
        collected_by: str = "Adey Gebrewold"
    ):
        self.data_path = Path(data_path)
        self.reference_path = Path(reference_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collected_by = collected_by
        self.collection_date = datetime.utcnow().date()

        self.logger = self._setup_logger()

        self.df = None
        self.reference = None
        self.enrichment_log = []

    def _setup_logger(self):
        logger = logging.getLogger("Task1-Enrichment")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    # ------------------------------------------------------------------
    # LOADERS
    # ------------------------------------------------------------------
    def load_data(self):
        self.logger.info("Loading unified dataset and reference codes")
        self.df = pd.read_csv(self.data_path)
        self.reference = pd.read_csv(self.reference_path)

        self.logger.info(f"Dataset loaded: {self.df.shape[0]} rows")

    # ------------------------------------------------------------------
    # BASIC EXPLORATION
    # ------------------------------------------------------------------
    def summarize_dataset(self):
        self.logger.info("Generating dataset summary")

        summary = {
            "record_type": self.df["record_type"].value_counts(),
            "pillar": self.df["pillar"].value_counts(dropna=False),
            "source_type": self.df["source_type"].value_counts(),
            "confidence": self.df["confidence"].value_counts(),
        }

        for key, value in summary.items():
            value.to_csv(self.output_dir / f"summary_{key}.csv")

        return summary

    def indicator_coverage(self):
        coverage = (
            self.df[self.df["record_type"] == "observation"]
            .groupby("indicator_code")["observation_date"]
            .agg(["min", "max", "count"])
            .sort_values("count", ascending=False)
        )

        coverage.to_csv(self.output_dir / "indicator_coverage.csv")
        return coverage

    # ------------------------------------------------------------------
    # ENRICHMENT UTILITIES
    # ------------------------------------------------------------------
    def _log_enrichment(self, record):
        self.enrichment_log.append(record)

    def _generate_record_id(self, prefix="REC"):
        existing = self.df["record_id"].str.extract(r"_(\d+)").dropna().astype(int)
        next_id = existing.max().values[0] + 1
        return f"{prefix}_{next_id:04d}"

    # ------------------------------------------------------------------
    # ENRICHMENT: OBSERVATIONS
    # ------------------------------------------------------------------
    def add_observations(self):
        """
        Add useful proxy indicators for forecasting:
        - Smartphone penetration
        - Internet usage rate
        - Agent density proxy
        """

        new_obs = [
            {
                "record_type": "observation",
                "pillar": "ACCESS",
                "indicator": "Smartphone Penetration Rate",
                "indicator_code": "ACC_SMARTPHONE",
                "indicator_direction": "higher_better",
                "value_numeric": 32,
                "value_type": "percentage",
                "unit": "%",
                "observation_date": "2024-12-31",
                "gender": "all",
                "location": "national",
                "source_name": "GSMA Intelligence",
                "source_type": "research",
                "source_url": "https://www.gsma.com/mobileeconomy/",
                "confidence": "medium",
                "notes": "Key enabler for mobile money usage"
            },
            {
                "record_type": "observation",
                "pillar": "USAGE",
                "indicator": "Internet Usage Rate",
                "indicator_code": "USG_INTERNET",
                "indicator_direction": "higher_better",
                "value_numeric": 25,
                "value_type": "percentage",
                "unit": "%",
                "observation_date": "2024-12-31",
                "gender": "all",
                "location": "national",
                "source_name": "ITU",
                "source_type": "research",
                "source_url": "https://www.itu.int/",
                "confidence": "medium",
                "notes": "Supports digital payment adoption"
            }
        ]

        for obs in new_obs:
            obs["record_id"] = self._generate_record_id()
            obs["collection_date"] = str(self.collection_date)
            obs["collected_by"] = self.collected_by

            self.df = pd.concat([self.df, pd.DataFrame([obs])], ignore_index=True)
            self._log_enrichment(obs)

        self.logger.info(f"Added {len(new_obs)} new observations")

    # ------------------------------------------------------------------
    # ENRICHMENT: EVENTS
    # ------------------------------------------------------------------
    def add_events(self):
        new_events = [
            {
                "record_type": "event",
                "category": "regulation",
                "indicator": "Agent Banking Regulation Issued",
                "indicator_code": "EVT_AGENT_REG",
                "value_text": "Issued",
                "value_type": "categorical",
                "observation_date": "2022-06-01",
                "source_name": "NBE",
                "source_type": "regulator",
                "confidence": "high",
                "notes": "Enabled agent network expansion"
            }
        ]

        for evt in new_events:
            evt["record_id"] = f"EVT_NEW_{len(self.enrichment_log)+1:03d}"
            evt["collection_date"] = str(self.collection_date)
            evt["collected_by"] = self.collected_by

            self.df = pd.concat([self.df, pd.DataFrame([evt])], ignore_index=True)
            self._log_enrichment(evt)

        self.logger.info(f"Added {len(new_events)} new events")

    # ------------------------------------------------------------------
    # ENRICHMENT: IMPACT LINKS
    # ------------------------------------------------------------------
    def add_impact_links(self):
        links = [
            {
                "record_type": "impact_link",
                "parent_id": "EVT_FAYDA",
                "pillar": "ACCESS",
                "related_indicator": "ACC_OWNERSHIP",
                "relationship_type": "enabling",
                "impact_direction": "increase",
                "impact_magnitude": "medium",
                "lag_months": 12,
                "evidence_basis": "literature",
                "notes": "Digital ID lowers KYC barriers"
            },
            {
                "record_type": "impact_link",
                "parent_id": "EVT_TELEBIRR",
                "pillar": "USAGE",
                "related_indicator": "USG_DIGITAL_PAYMENT",
                "relationship_type": "direct",
                "impact_direction": "increase",
                "impact_magnitude": "high",
                "lag_months": 6,
                "evidence_basis": "empirical",
                "notes": "Observed increase in P2P volumes"
            }
        ]

        for link in links:
            link["record_id"] = f"LNK_{len(self.enrichment_log)+1:04d}"
            link["collection_date"] = str(self.collection_date)
            link["collected_by"] = self.collected_by

            self.df = pd.concat([self.df, pd.DataFrame([link])], ignore_index=True)
            self._log_enrichment(link)

        self.logger.info(f"Added {len(links)} impact links")

    # ------------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------------
    def save_outputs(self, output_dir: str | Path | None = None):
        # Use provided directory or default
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        enriched_path = output_dir / "ethiopia_fi_enriched.csv"
        self.df.to_csv(enriched_path, index=False)

        log_path = output_dir / "data_enrichment_log.md"
        with open(log_path, "w") as f:
            f.write("# Data Enrichment Log\n\n")
            for rec in self.enrichment_log:
                f.write(f"- **{rec.get('record_id')}**: {rec.get('notes')}\n")

        self.logger.info(f"Saved enriched dataset and enrichment log to {output_dir}")


    # ------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------
    def run_full_task1(self):
        self.load_data()
        self.summarize_dataset()
        self.indicator_coverage()
        self.add_observations()
        self.add_events()
        self.add_impact_links()
        self.save_outputs()

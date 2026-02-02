import pandas as pd
import logging
from pathlib import Path
from datetime import datetime


class FinancialInclusionDataEnricher:
    """
    Task 1: Data Exploration and Enrichment (Fully Schema-Compliant)

    - Explores unified financial inclusion dataset
    - Enriches with new observations and events (Sheet 1)
    - Enriches impact_links separately (Sheet 2)
    - Documents all additions with full provenance
    """

    def __init__(
        self,
        data_path: str,
        reference_path: str,
        output_dir: str = "data/processed",
        collected_by: str = "Adey Gebrewold",
    ):
        self.data_path = Path(data_path)
        self.reference_path = Path(reference_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collected_by = collected_by
        self.collection_date = datetime.utcnow().date().isoformat()

        self.df = None
        self.reference = None
        self.impact_links_df = None

        self.enrichment_log = []
        self.logger = self._setup_logger()

    # ------------------------------------------------------------------
    # LOGGER
    # ------------------------------------------------------------------
    def _setup_logger(self):
        logger = logging.getLogger("Task1-Enrichment")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        return logger

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------
    def load_data(self):
        self.logger.info("Loading unified dataset and reference codes")
        self.df = pd.read_csv(self.data_path)
        self.reference = pd.read_csv(self.reference_path)

        self.impact_links_df = self.df[self.df["record_type"] == "impact_link"].copy()
        self.df = self.df[self.df["record_type"] != "impact_link"].copy()

        self.logger.info(f"Main dataset loaded: {self.df.shape[0]} records")
        self.logger.info(f"Impact links loaded: {self.impact_links_df.shape[0]} records")

    # ------------------------------------------------------------------
    # EXPLORATION
    # ------------------------------------------------------------------
    def summarize_dataset(self):
        summary_cols = ["record_type", "pillar", "source_type", "confidence"]
        for col in summary_cols:
            self.df[col].value_counts(dropna=False).to_csv(
                self.output_dir / f"summary_{col}.csv"
            )

    def indicator_coverage(self):
        coverage = (
            self.df[self.df["record_type"] == "observation"]
            .groupby("indicator_code")["observation_date"]
            .agg(["min", "max", "count"])
            .sort_values("count", ascending=False)
        )
        coverage.to_csv(self.output_dir / "indicator_coverage.csv")

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------
    def _generate_record_id(self, prefix: str, existing_series: pd.Series):
        nums = (
            existing_series.dropna()
            .str.extract(r"(\d+)$")[0]
            .dropna()
            .astype(int)
        )
        next_num = nums.max() + 1 if not nums.empty else 1
        return f"{prefix}_{next_num:04d}"

    def _log_enrichment(self, record: dict):
        self.enrichment_log.append(record)

    # ------------------------------------------------------------------
    # ADD OBSERVATIONS
    # ------------------------------------------------------------------
    def add_observations(self):
        observations = [
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
                "source_url": "https://www.gsma.com/mobileeconomy/",
                "original_text": "Smartphone adoption in Ethiopia reached approximately 32% in 2024.",
                "source_type": "research",
                "confidence": "medium",
                "notes": "Smartphone access is a prerequisite for mobile money and digital financial services.",
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
                "source_url": "https://www.itu.int/",
                "original_text": "About 25% of Ethiopia’s population used the internet in 2024.",
                "source_type": "research",
                "confidence": "medium",
                "notes": "Internet usage supports digital payment adoption and platform-based finance.",
            },
        ]

        for obs in observations:
            obs["record_id"] = self._generate_record_id("OBS", self.df["record_id"])
            obs["collected_by"] = self.collected_by
            obs["collection_date"] = self.collection_date

            self.df = pd.concat([self.df, pd.DataFrame([obs])], ignore_index=True)
            self._log_enrichment(obs)

        self.logger.info(f"Added {len(observations)} observations")

    # ------------------------------------------------------------------
    # ADD EVENTS
    # ------------------------------------------------------------------
    def add_events(self):
        events = [
            # EXISTING
            {
                "record_type": "event",
                "category": "policy",
                "indicator": "Agent Banking Regulation Issued",
                "indicator_code": "EVT_AGENT_REG",
                "value_text": "Issued",
                "value_type": "categorical",
                "observation_date": "2022-06-01",
                "source_name": "National Bank of Ethiopia",
                "source_url": "https://www.nbe.gov.et/",
                "original_text": "NBE issued a directive enabling agent banking operations in Ethiopia.",
                "source_type": "regulator",
                "confidence": "high",
                "notes": "Enables agent-based account access.",
            },
            # NEW
            {
                "record_type": "event",
                "category": "product_launch",
                "indicator": "Telebirr Mobile Money Launched",
                "indicator_code": "EVT_TELEBIRR",
                "value_text": "Launched",
                "value_type": "categorical",
                "observation_date": "2021-05-11",
                "source_name": "Ethio Telecom",
                "source_url": "https://www.ethiotelecom.et/",
                "original_text": "Ethio Telecom launched Telebirr mobile money platform.",
                "source_type": "operator",
                "confidence": "high",
                "notes": "Major driver of mobile money adoption.",
            },
            {
                "record_type": "event",
                "category": "infrastructure",
                "indicator": "4G Network Expansion",
                "indicator_code": "EVT_4G_EXPANSION",
                "value_text": "Expanded",
                "value_type": "categorical",
                "observation_date": "2019-01-01",
                "source_name": "Ethio Telecom",
                "source_url": "https://www.ethiotelecom.et/",
                "original_text": "Expansion of 4G LTE infrastructure in major cities.",
                "source_type": "operator",
                "confidence": "medium",
                "notes": "Improves digital access and service reliability.",
            },
            {
                "record_type": "event",
                "category": "policy",
                "indicator": "Digital ID (Fayda) Rollout",
                "indicator_code": "EVT_FAYDA",
                "value_text": "Rolled out",
                "value_type": "categorical",
                "observation_date": "2023-01-01",
                "source_name": "NID Program",
                "source_url": "https://id.gov.et/",
                "original_text": "Launch of Fayda digital ID system.",
                "source_type": "government",
                "confidence": "medium",
                "notes": "Reduces KYC barriers to account ownership.",
            },
        ]

        for evt in events:
            evt["record_id"] = self._generate_record_id("EVT", self.df["record_id"])
            evt["collected_by"] = self.collected_by
            evt["collection_date"] = self.collection_date

            self.df = pd.concat([self.df, pd.DataFrame([evt])], ignore_index=True)
            self._log_enrichment(evt)

        self.logger.info(f"Added {len(events)} events")
    # ------------------------------------------------------------------
    # ADD IMPACT LINKS (SHEET 2)
    # ------------------------------------------------------------------
    def add_impact_links(self):
        links = [
            # Agent banking
            {
                "record_type": "impact_link",
                "parent_id": "EVT_AGENT_REG",
                "pillar": "ACCESS",
                "related_indicator": "ACC_ACCOUNT_OWNERSHIP",
                "relationship_type": "enabling",
                "impact_direction": "increase",
                "impact_magnitude": "medium",
                "lag_months": 12,
                "evidence_basis": "literature",
                "original_text": "Agent banking reduces physical access barriers.",
                "confidence": "medium",
                "notes": "Common effect in SSA countries.",
            },
            # Telebirr
            {
                "record_type": "impact_link",
                "parent_id": "EVT_TELEBIRR",
                "pillar": "USAGE",
                "related_indicator": "ACC_MM_ACCOUNT",
                "relationship_type": "direct",
                "impact_direction": "increase",
                "impact_magnitude": "high",
                "lag_months": 6,
                "evidence_basis": "observed",
                "original_text": "Mobile money platforms rapidly increase account ownership.",
                "confidence": "high",
                "notes": "Aligned with Kenya, Ghana evidence.",
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
                "evidence_basis": "observed",
                "original_text": "Telebirr increased P2P and merchant payments.",
                "confidence": "high",
                "notes": "Explains post-2021 jump.",
            },
            # Infrastructure
            {
                "record_type": "impact_link",
                "parent_id": "EVT_4G_EXPANSION",
                "pillar": "ACCESS",
                "related_indicator": "ACC_MOBILE_PEN",
                "relationship_type": "enabling",
                "impact_direction": "increase",
                "impact_magnitude": "medium",
                "lag_months": 18,
                "evidence_basis": "literature",
                "original_text": "Mobile broadband supports mobile service uptake.",
                "confidence": "medium",
                "notes": "Gradual diffusion effect.",
            },
            # Digital ID
            {
                "record_type": "impact_link",
                "parent_id": "EVT_FAYDA",
                "pillar": "ACCESS",
                "related_indicator": "ACC_ACCOUNT_OWNERSHIP",
                "relationship_type": "enabling",
                "impact_direction": "increase",
                "impact_magnitude": "medium",
                "lag_months": 12,
                "evidence_basis": "cross_country",
                "original_text": "Digital ID simplifies KYC requirements.",
                "confidence": "medium",
                "notes": "Observed in India, Nigeria.",
            },
        ]

        for link in links:
            link["record_id"] = self._generate_record_id(
                "LNK", self.impact_links_df["record_id"]
            )
            link["collected_by"] = self.collected_by
            link["collection_date"] = self.collection_date

            self.impact_links_df = pd.concat(
                [self.impact_links_df, pd.DataFrame([link])],
                ignore_index=True,
            )
            self._log_enrichment(link)

        self.logger.info(f"Added {len(links)} impact links")

    # ------------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------------
    def save_outputs(self):
        self.df.to_csv(
            self.output_dir / "ethiopia_fi_enriched_data.csv", index=False
        )
        self.impact_links_df.to_csv(
            self.output_dir / "ethiopia_fi_impact_links.csv", index=False
        )

        log_path = self.output_dir / "data_enrichment_log.md"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Data Enrichment Log\n\n")
            for rec in self.enrichment_log:
                f.write(
                    f"- **{rec['record_id']}** ({rec['record_type']}): "
                    f"{rec.get('notes', '')}\n"
                )

        self.logger.info("All outputs saved successfully")

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

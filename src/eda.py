import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging


class EthiopiaFinancialInclusionEDA:
    """
    Exploratory Data Analysis (EDA) class for Ethiopia Financial Inclusion data.

    This class provides methods to explore, visualize, and generate insights 
    from Ethiopia's financial inclusion dataset, including:
    - Dataset overview and quality checks
    - Temporal coverage analysis
    - Pillar, access, usage, and infrastructure trends
    - Event timelines and overlays
    - Correlation analysis
    - Key insights and data quality reporting
    - Analysis of new indicators (DIR_ / IND_)
    
    Attributes
    ----------
    df : pd.DataFrame
        The raw dataset loaded from CSV.
    obs : pd.DataFrame
        Subset of the data containing observations.
    events : pd.DataFrame
        Subset of the data containing events.
    links : pd.DataFrame
        Subset of the data containing impact links.
    output_dir : Path
        Directory path to save figures.
    logger : logging.Logger
        Logger for informational messages.
    """

    def __init__(self, data_path: str | Path, output_dir: str | Path = "reports/figures"):
        """
        Initialize the EDA class by loading data and setting up subsets.

        Parameters
        ----------
        data_path : str | Path
            Path to the CSV file containing the dataset.
        output_dir : str | Path, optional
            Directory to save generated figures (default is "reports/figures").
        """
        self.df = pd.read_csv(data_path)

        # Core subsets
        self.obs = self.df[self.df["record_type"] == "observation"].copy()
        self.events = self.df[self.df["record_type"] == "event"].copy()
        self.links = self.df[self.df["record_type"] == "impact_link"].copy()

        # Dates
        self.obs["observation_date"] = pd.to_datetime(
            self.obs["observation_date"], errors="coerce", format="mixed"
        )
        self.obs = self.obs.dropna(subset=["observation_date"])
        self.obs["year"] = self.obs["observation_date"].dt.year

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Unique indicator codes in the dataset:")
        print(self.obs["indicator_code"].unique())

        # Logger
        self.logger = logging.getLogger("Task2EDA")
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(handler)

        sns.set_style("whitegrid")

    # ------------------------------------------------------------------
    # Dataset Overview & Quality
    # ------------------------------------------------------------------
    def summarize_dataset(self):
        """
        Summarize the dataset by record type, pillar, source type, and confidence.

        Returns
        -------
        dict
            Dictionary containing counts for record_type, pillar, source_type, and confidence.
        """
        summary = {
            "record_type": self.df["record_type"].value_counts(),
            "pillar": self.df["pillar"].value_counts(dropna=False),
            "source_type": self.df["source_type"].value_counts(dropna=False),
            "confidence": self.df["confidence"].value_counts(dropna=False),
        }

        self.logger.info("=== DATASET OVERVIEW ===")
        for k, v in summary.items():
            self.logger.info(f"\n{k.upper()}:\n{v}")

        return summary

    def load_impact_links(self, links_path: str | Path):
        """
        Load impact links from a separate CSV file after enrichment.
        """
        self.links = pd.read_csv(links_path)
        self.logger.info(f"Impact links loaded: {len(self.links)} records")


    def dataset_summary_table(self):
        """
        Generate a summary table of the dataset including counts, date range, and indicators.

        Returns
        -------
        dict
            Dictionary containing dataset statistics and temporal coverage.
        """
        summary = {
            "total_records": len(self.df),
            "observations": len(self.obs),
            "events": len(self.events),
            "impact_links": len(self.links),
            "indicators": self.obs["indicator_code"].nunique(),
            "date_min": self.obs["observation_date"].min(),
            "date_max": self.obs["observation_date"].max(),
            "years_covered": sorted(self.obs["year"].unique().tolist()),
        }

        self.logger.info("=== DATASET SUMMARY TABLE ===")
        for k, v in summary.items():
            self.logger.info(f"{k}: {v}")

        return summary

    def plot_confidence_distribution(self, save_fig=True):
        """
        Plot a bar chart showing the distribution of confidence levels in the dataset.

        Parameters
        ----------
        save_fig : bool, optional
            Whether to save the figure to the output directory (default is True).

        Returns
        -------
        None
        """
        plt.figure(figsize=(6, 4))
        self.df["confidence"].value_counts().plot(kind="bar")
        plt.title("Confidence Level Distribution")
        plt.ylabel("Record Count")

        if save_fig:
            path = self.output_dir / "confidence_distribution.png"
            plt.savefig(path, dpi=300)

        plt.show()

    # ------------------------------------------------------------------
    # Temporal Coverage
    # ------------------------------------------------------------------
    def temporal_coverage_heatmap(self, save_fig=True):
        """
        Generate a heatmap showing temporal coverage of indicators by year.

        Parameters
        ----------
        save_fig : bool, optional
            Whether to save the figure (default is True).

        Returns
        -------
        pd.DataFrame
            Pivot table of counts of observations per indicator per year.
        """
        coverage = self.obs.pivot_table(
            index="indicator_code",
            columns="year",
            values="value_numeric",
            aggfunc="count",
            fill_value=0,
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(coverage, annot=True, fmt="d", cmap="YlOrRd")
        plt.title("Temporal Coverage: Indicators by Year")

        if save_fig:
            path = self.output_dir / "temporal_coverage_heatmap.png"
            plt.savefig(path, dpi=300)

        plt.show()
        return coverage

    def identify_sparse_indicators(self, min_points=2):
        """
        Identify indicators with fewer than a specified minimum number of observations.

        Parameters
        ----------
        min_points : int, optional
            Minimum number of observations required to not be considered sparse (default is 2).

        Returns
        -------
        pd.Series
            Series of sparse indicators with their observation counts.
        """
        counts = self.obs.groupby("indicator_code").size()
        sparse = counts[counts < min_points]

        self.logger.info("=== SPARSE INDICATORS ===")
        self.logger.info(sparse)

        return sparse

    # ------------------------------------------------------------------
    # Pillar & Access Analysis
    # ------------------------------------------------------------------
    def pillar_distribution(self, save_fig=True):
        """
        Plot a bar chart showing the distribution of observations by pillar.

        Parameters
        ----------
        save_fig : bool, optional
            Whether to save the figure (default is True).

        Returns
        -------
        pd.Series
            Counts of observations per pillar.
        """
        plt.figure(figsize=(6, 4))
        self.obs["pillar"].value_counts().plot(kind="bar")
        plt.title("Observation Distribution by Pillar")

        if save_fig:
            path = self.output_dir / "pillar_distribution.png"
            plt.savefig(path, dpi=300)

        plt.show()
        return self.obs["pillar"].value_counts()

    def plot_account_ownership(self, save_fig=True):
        """
        Plot the trend of account ownership over time.

        Parameters
        ----------
        save_fig : bool, optional
            Whether to save the figure (default is True).

        Returns
        -------
        pd.DataFrame
            Filtered account ownership observations sorted by date.
        """
        acc = self.obs[self.obs["indicator_code"] == "ACC_OWNERSHIP"].sort_values(
            "observation_date"
        )

        plt.figure(figsize=(7, 4))
        plt.plot(acc["observation_date"], acc["value_numeric"], marker="o", linewidth=2)
        plt.ylabel("Percentage (%)")
        plt.title("Account Ownership Trend")
        plt.grid(alpha=0.3)

        if save_fig:
            path = self.output_dir / "account_ownership_trend.png"
            plt.savefig(path, dpi=300)

        plt.show()
        return acc

    def calculate_growth_rates(self, indicator_code):
        """
        Calculate period-to-period growth in percentage points for a given indicator.

        Parameters
        ----------
        indicator_code : str
            The indicator code to calculate growth rates for.

        Returns
        -------
        pd.DataFrame
            DataFrame including a new 'growth_pp' column representing period-to-period growth.
        """
        df = (
            self.obs[self.obs["indicator_code"] == indicator_code]
            .sort_values("observation_date")
            .copy()
        )
        df["growth_pp"] = df["value_numeric"].diff()
        return df

    def calculate_cagr(self, indicator_code):
        """
        Calculate the compound annual growth rate (CAGR) for a given indicator.

        Parameters
        ----------
        indicator_code : str
            The indicator code to calculate CAGR for.

        Returns
        -------
        float | np.nan
            CAGR as a decimal fraction or np.nan if insufficient data.
        """
        df = (
            self.obs[self.obs["indicator_code"] == indicator_code]
            .sort_values("observation_date")
            .copy()
        )

        if len(df) < 2:
            return np.nan

        start, end = df["value_numeric"].iloc[[0, -1]]
        years = (df["observation_date"].iloc[-1] -
                 df["observation_date"].iloc[0]).days / 365.25

        if start <= 0 or years <= 0:
            return np.nan

        return (end / start) ** (1 / years) - 1

    # ------------------------------------------------------------------
    # Usage Analysis
    # ------------------------------------------------------------------
    def plot_usage_trends(self, indicators, save_fig=True):
        """
        Plot usage trends over time for a list of indicators.

        Parameters
        ----------
        indicators : list
            List of indicator codes to include in the plot.
        save_fig : bool, optional
            Whether to save the figure (default is True).

        Returns
        -------
        pd.DataFrame
            Filtered data used for plotting.
        """
        usage = self.obs[self.obs["indicator_code"].isin(indicators)].copy()

        plt.figure(figsize=(7, 4))
        sns.lineplot(
            data=usage,
            x="observation_date",
            y="value_numeric",
            hue="indicator_code",
            marker="o",
        )
        plt.title("Usage Trends")
        plt.grid(alpha=0.3)

        if save_fig:
            path = self.output_dir / "usage_trends.png"
            plt.savefig(path, dpi=300)

        plt.show()
        return usage

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------
    def plot_infrastructure_indicators(self, infra_codes, save_fig=True):
        """
        Plot one or two infrastructure indicators on a single figure, potentially with dual y-axis.

        Parameters
        ----------
        infra_codes : list
            List of 1 or 2 indicator codes to plot.
        save_fig : bool, optional
            Whether to save the figure (default is True).

        Returns
        -------
        pd.DataFrame
            Filtered infrastructure data used for plotting.
        """
        infra = self.obs[self.obs["indicator_code"].isin(infra_codes)].sort_values(
            "observation_date"
        )
        
        print("Filtered data:")
        print(infra)

        if infra.empty:
            print("No data to plot!")
            return infra

        fig, ax1 = plt.subplots(figsize=(7, 4))
        first = infra_codes[0]
        d1 = infra[infra["indicator_code"] == first]

        print(self.obs["indicator_code"].unique())

        
        print(f"Plotting {first} with {len(d1)} points")
        ax1.plot(d1["observation_date"], d1["value_numeric"], "o-", label=first)
        ax1.set_ylabel(first)

        if len(infra_codes) > 1:
            second = infra_codes[1]
            d2 = infra[infra["indicator_code"] == second]
            print(f"Plotting {second} with {len(d2)} points")
            ax2 = ax1.twinx()
            ax2.plot(d2["observation_date"], d2["value_numeric"], "o-", label=second)

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2)
        else:
            ax1.legend()

        plt.title("Infrastructure Indicators")
        plt.grid(alpha=0.3)

        if save_fig:
            path = self.output_dir / "infrastructure_trends.png"
            plt.savefig(path, dpi=300)

        plt.show()
        return infra

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def plot_event_timeline(self, save_fig=True):
        """
        Plot a scatter timeline of all events.

        Parameters
        ----------
        save_fig : bool, optional
            Whether to save the figure (default is True).

        Returns
        -------
        pd.DataFrame
            Event data used for plotting.
        """
        events = self.events.copy()
        events["event_date"] = pd.to_datetime(
            events["observation_date"], errors="coerce"
        )
        events = events.dropna(subset=["event_date"])

        plt.figure(figsize=(10, 3))
        plt.scatter(events["event_date"], range(len(events)))
        for i, label in enumerate(events["indicator"]):
            plt.text(events["event_date"].iloc[i], i, label, fontsize=8)

        plt.yticks([])
        plt.title("Event Timeline")

        if save_fig:
            path = self.output_dir / "event_timeline.png"
            plt.savefig(path, dpi=300)

        plt.show()
        return events

    def plot_indicator_with_events(self, indicator_code, save_fig=True):
        """
        Plot an indicator series over time with event overlays.

        Parameters
        ----------
        indicator_code : str
            The indicator code to plot.
        save_fig : bool, optional
            Whether to save the figure (default is True).

        Returns
        -------
        None
        """
        series = self.obs[self.obs["indicator_code"] == indicator_code]
        events = self.events.copy()
        events["event_date"] = pd.to_datetime(
            events["observation_date"], errors="coerce"
        )

        plt.figure(figsize=(8, 4))
        plt.plot(series["observation_date"], series["value_numeric"], marker="o")

        for _, row in events.iterrows():
            if pd.notna(row["event_date"]):
                plt.axvline(row["event_date"], color="red", alpha=0.2)

        plt.title(f"{indicator_code} with Events")

        if save_fig:
            path = self.output_dir / f"{indicator_code}_event_overlay.png"
            plt.savefig(path, dpi=300)

        plt.show()

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------
    def correlation_matrix(self, indicators):
        """
        Compute correlation matrix for a list of indicators.

        Parameters
        ----------
        indicators : list
            List of indicator codes.

        Returns
        -------
        pd.DataFrame
            Correlation matrix of indicators.
        """
        df = self.obs[self.obs["indicator_code"].isin(indicators)].copy()
        df["value_numeric"] = pd.to_numeric(df["value_numeric"], errors="coerce")

        pivot = df.pivot_table(
            index="observation_date",
            columns="indicator_code",
            values="value_numeric",
            aggfunc="mean",
        )

        return pivot.corr()

    def plot_correlation(self, indicators, save_fig=True):
        """
        Plot a heatmap of the correlation matrix for selected indicators.

        Parameters
        ----------
        indicators : list
            List of indicator codes.
        save_fig : bool, optional
            Whether to save the figure (default is True).

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        corr = self.correlation_matrix(indicators)

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            corr, annot=True, cmap="RdBu_r", center=0, square=True
        )
        plt.title("Correlation Matrix")

        if save_fig:
            path = self.output_dir / "correlation_heatmap.png"
            plt.savefig(path, dpi=300)

        plt.show()
        return corr

    def strongest_correlations(self, indicators, top_n=5):
        """
        Identify the strongest correlations between indicators.

        Parameters
        ----------
        indicators : list
            List of indicator codes.
        top_n : int, optional
            Number of top correlations to return (default is 5).

        Returns
        -------
        pd.DataFrame
            Top correlated indicator pairs.
        """
        corr = self.correlation_matrix(indicators)

        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                pairs.append(
                    {
                        "indicator_1": corr.columns[i],
                        "indicator_2": corr.columns[j],
                        "correlation": corr.iloc[i, j],
                    }
                )

        return (
            pd.DataFrame(pairs)
            .dropna()
            .sort_values("correlation", key=abs, ascending=False)
            .head(top_n)
        )

    # ------------------------------------------------------------------
    # Insights & Data Quality
    # ------------------------------------------------------------------
    def generate_key_insights(self):
        """
        Generate key insights based on access, usage, and infrastructure indicators.

        Returns
        -------
        dict
            Dictionary of calculated insights such as growth rates, gaps, and coverage.
        """
        insights = {}

        acc = self.obs[self.obs["indicator_code"] == "ACC_OWNERSHIP"].sort_values(
            "observation_date"
        )
        if len(acc) >= 3:
            recent = acc["value_numeric"].iloc[-1] - acc["value_numeric"].iloc[-2]
            earlier = acc["value_numeric"].iloc[-2] - acc["value_numeric"].iloc[-3]
            insights["access_growth_deceleration_pp"] = earlier - recent

        mm = self.obs[self.obs["indicator_code"] == "ACC_MM_ACCOUNT"]
        if not mm.empty and not acc.empty:
            insights["mobile_money_gap_pp"] = (
                acc["value_numeric"].iloc[-1] - mm["value_numeric"].iloc[-1]
            )

        p2p = self.obs[self.obs["indicator_code"] == "USG_P2P_COUNT"]
        if len(p2p) >= 2:
            insights["p2p_absolute_growth"] = (
                p2p["value_numeric"].iloc[-1] - p2p["value_numeric"].iloc[0]
            )

        infra = self.obs[self.obs["indicator_code"] == "INF_4G_COVERAGE"]
        if not infra.empty:
            insights["latest_4g_coverage"] = infra["value_numeric"].iloc[-1]

        return insights

    def data_quality_report(self):
        """
        Generate a data quality report including counts and shares of records.

        Returns
        -------
        dict
            Dictionary of data quality metrics.
        """
        return {
            "findex_points": self.obs[
                self.obs["indicator_code"] == "ACC_OWNERSHIP"
            ].shape[0],
            "high_confidence_share": (self.df["confidence"] == "high").mean(),
            "single_point_indicators": (
                self.obs.groupby("indicator_code").size() == 1
            ).sum(),
            "impact_links": len(self.links),
        }

    # ------------------------------------------------------------------
    # New Indicators (DIR_ / IND_)
    # ------------------------------------------------------------------
    def analyze_new_indicators(self):
        """
        Analyze newly added direct (DIR_) and indirect (IND_) indicators.

        Returns
        -------
        dict
            Dictionary containing counts of new indicators and observations.
        """
        direct = self.obs[self.obs["indicator_code"].str.startswith("DIR_", na=False)]
        indirect = self.obs[self.obs["indicator_code"].str.startswith("IND_", na=False)]

        return {
            "direct_indicators": direct["indicator_code"].nunique(),
            "indirect_indicators": indirect["indicator_code"].nunique(),
            "direct_observations": len(direct),
            "indirect_observations": len(indirect),
        }

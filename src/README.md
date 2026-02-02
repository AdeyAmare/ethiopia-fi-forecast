# `src` Directory

This folder contains the core source code for **Ethiopia Financial Inclusion Forecasting & Analysis**, including data enrichment, exploratory data analysis (EDA), event impact modeling, and forecasting. The code is designed to support financial inclusion research, indicator tracking, and insights generation for policy and market stakeholders.

## 📁 Directory Structure

```
src/
│
├── data_enrichment.py         # Task 1: Load, validate, and enrich financial inclusion datasets
├── eda.py                     # Task 2: Exploratory Data Analysis and visualization
├── event_impact_modeling.py   # Task 3: Model how events affect financial inclusion indicators
├── forecasting.py             # Task 4: Forecast access and usage indicators (2025-2027)
└── __init__.py                # Makes this folder a Python package
```

---

## 💻 Modules Overview

### 1. `data_enrichment.py` – Task 1

**Purpose:**

* Load unified financial inclusion datasets
* Validate schema against reference codes
* Add new observations, events, and impact links
* Track and log all enrichment activities

**Key Class:** `FinancialInclusionDataEnricher`

**Usage Example:**

```python
from src.data_enrichment import FinancialInclusionDataEnricher

enricher = FinancialInclusionDataEnricher(
    data_path="data/raw/ethiopia_fi.csv",
    reference_path="data/raw/reference_codes.csv"
)

enricher.run_full_task1()
```

**Outputs:**

* `ethiopia_fi_enriched.csv` – Enriched dataset
* `data_enrichment_log.md` – Markdown log of added observations, events, and links
* Summaries of dataset, indicator coverage, and quality

---

### 2. `eda.py` – Task 2

**Purpose:**

* Perform Exploratory Data Analysis (EDA) on the enriched dataset
* Generate visualizations and insights for access, usage, infrastructure, and events
* Calculate correlations, growth rates, and key indicators

**Key Class:** `EthiopiaFinancialInclusionEDA`

**Usage Example:**

```python
from src.eda import EthiopiaFinancialInclusionEDA

eda = EthiopiaFinancialInclusionEDA(data_path="data/processed/ethiopia_fi_enriched.csv")

eda.summarize_dataset()
eda.temporal_coverage_heatmap()
eda.plot_account_ownership()
eda.plot_usage_trends(indicators=["USG_P2P_COUNT", "USG_MOBILE_PAYMENT"])
eda.plot_correlation(indicators=["ACC_OWNERSHIP", "USG_P2P_COUNT", "INF_4G_COVERAGE"])
insights = eda.generate_key_insights()
dq_report = eda.data_quality_report()
```

**Outputs:**

* Figures saved in `reports/figures` (configurable)
* Insight dictionaries and data quality reports for further analysis

---

### 3. `event_impact_modeling.py` – Task 3

**Purpose:**

* Model how events (policies, product launches, infrastructure investments) affect financial inclusion indicators over time
* Use `impact_links` metadata to quantify direction, magnitude, and lag of event effects
* Simulate indicator paths and validate against observed values

**Key Class:** `EventImpactModeler`

**Usage Example:**

```python
from src.event_impact_modeling import EventImpactModeler

modeler = EventImpactModeler(
    data_path="data/processed/ethiopia_fi_enriched.csv",
    impact_links_path="data/processed/impact_links.csv",
    ramp_months=24
)

summary = modeler.event_indicator_summary()
matrix = modeler.build_association_matrix()
simulated = modeler.simulate_indicator("ACC_OWNERSHIP")
validation = modeler.validate_against_actual("ACC_OWNERSHIP")
```

**Outputs:**

* Event–indicator summary tables
* Event × Indicator association matrices
* Simulated indicator paths with ramped event effects
* Validation tables comparing simulated vs observed values

---

### 4. `forecasting.py` – Task 4

**Purpose:**

* Forecast access and usage indicators from 2025–2027
* Combine trend-based regression with event-augmented forecasts
* Generate scenario analyses: optimistic, base, pessimistic
* Visualize forecasts and summarize event impacts

**Key Class:** `FinancialInclusionForecaster`

**Usage Example:**

```python
from src.event_impact_modeling import EventImpactModeler
from src.forecasting import FinancialInclusionForecaster

modeler = EventImpactModeler(
    data_path="data/processed/ethiopia_fi_enriched.csv",
    impact_links_path="data/processed/impact_links.csv"
)

forecaster = FinancialInclusionForecaster(model=modeler, forecast_horizon=36)

forecaster.forecast_all_indicators()
forecaster.save_forecast_artifacts("data/forecasts")
forecaster.plot_forecast("ACC_OWNERSHIP")
scenario_dfs = forecaster.scenario_forecast("ACC_OWNERSHIP")
event_impacts = forecaster.summarize_event_impacts("ACC_OWNERSHIP")
```

**Outputs:**

* Forecast CSVs for all indicators
* Scenario forecast DataFrames (optimistic, base, pessimistic)
* Event-augmented trend plots with confidence intervals
* Pickled regression models for reuse

---

## 🛠️ Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## ⚡ Workflow Summary

1. **Data Enrichment (Task 1)**

   * Load raw datasets
   * Validate and enrich with new observations, events, and impact links
   * Save enriched dataset and logs

2. **Exploratory Data Analysis (Task 2)**

   * Load enriched dataset
   * Generate summaries, trends, coverage, and correlations
   * Produce visualizations and key insights

3. **Event Impact Modeling (Task 3)**

   * Load enriched dataset + impact links
   * Summarize events affecting indicators
   * Build association matrices
   * Simulate indicator paths and validate

4. **Forecasting (Task 4)**

   * Fit trend-based models for all indicators
   * Generate event-augmented forecasts
   * Conduct scenario analyses
   * Save forecast artifacts and visualize results



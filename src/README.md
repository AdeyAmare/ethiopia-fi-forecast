# `src` Directory

This folder contains the core source code for **Ethiopia Financial Inclusion Forecasting & Analysis**, including data enrichment and exploratory data analysis (EDA). The code is designed to support financial inclusion research, indicator tracking, and insights generation for policy and market stakeholders.

## 📁 Directory Structure

```
src/
│
├── data_enrichment.py   # Task 1: Load, validate, and enrich financial inclusion datasets
├── eda.py               # Task 2: Exploratory Data Analysis and visualization
└── __init__.py          # Makes this folder a Python package
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

# Initialize the enrichment pipeline
enricher = FinancialInclusionDataEnricher(
    data_path="data/raw/ethiopia_fi.csv",
    reference_path="data/raw/reference_codes.csv"
)

# Run full enrichment pipeline
enricher.run_full_task1()

# The enriched dataset and log are saved in 'data/processed'
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

# Initialize EDA pipeline
eda = EthiopiaFinancialInclusionEDA(data_path="data/processed/ethiopia_fi_enriched.csv")

# Summary and dataset overview
eda.summarize_dataset()
eda.dataset_summary_table()

# Plot trends and coverage
eda.temporal_coverage_heatmap()
eda.plot_account_ownership()
eda.plot_usage_trends(indicators=["USG_P2P_COUNT", "USG_MOBILE_PAYMENT"])

# Correlation analysis
eda.plot_correlation(indicators=["ACC_OWNERSHIP", "USG_P2P_COUNT", "INF_4G_COVERAGE"])

# Generate insights and data quality report
insights = eda.generate_key_insights()
dq_report = eda.data_quality_report()
```

**Outputs:**

* Figures saved in `reports/figures` (configurable)
* Insight dictionaries and data quality reports for further analysis

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
   * Save enriched dataset and enrichment logs

2. **Exploratory Data Analysis (Task 2)**

   * Load enriched dataset
   * Generate summaries, trends, coverage, and correlations
   * Produce visualizations and key insights

---
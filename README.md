# Ethiopia Financial Inclusion Analysis

This project analyzes Ethiopia’s digital financial landscape, providing a **data enrichment pipeline, exploratory analysis, event impact modeling, forecasting, and interactive visualizations** to support decision-making and policy design for financial inclusion.

---

## 🏗 Project Overview

Ethiopia is undergoing rapid digital financial transformation, with mobile money platforms, internet penetration, and agent networks driving financial inclusion. This project tracks these changes by:

1. **Enriching datasets** – Integrating new observations, regulatory events, and impact links.
2. **Exploratory Data Analysis (EDA)** – Summarizing trends, gaps, and correlations in financial access and usage.
3. **Event Impact Modeling** – Quantifying how events (policies, product launches, infrastructure investments) affect indicators over time.
4. **Forecasting** – Projecting access and usage indicators (2025–2027) with trend + event-augmented models and scenario analysis.
5. **Visualization & Interactive Dashboard** – Generating figures and an interactive Streamlit dashboard for stakeholders.

The workflow combines **data engineering, statistical modeling, and visualization** to create actionable intelligence.

---

## 📁 Project Structure

```
project_root/
│
├── data/
│   ├── raw/             # Raw unified datasets & reference codes
│   └── processed/       # Enriched datasets, impact links, and forecasts
│
├── src/                 # Core modules
│   ├── data_enrichment.py       # Task 1: Enrichment pipeline
│   ├── eda.py                   # Task 2: Exploratory Data Analysis
│   ├── event_impact_modeling.py # Task 3: Event impact modeling
│   └── forecasting.py           # Task 4: Forecasting indicators
│
├── notebooks/           # Jupyter notebooks
│   ├── data_enrichment_exploration.ipynb  # Enrichment exploration & visualization
│   ├── eda.ipynb                           # Comprehensive EDA & reporting
│   ├── event_impact_modeling.ipynb        # Event-impact modeling and simulation
│   └── forecaster.ipynb                    # Forecasting and scenario analysis
│
├── reports/
│   └── figures/         # Figures and visualizations from notebooks and forecasts
│
├── app/                 # Streamlit interactive dashboard
│   └── dashboard.py
│
├── tests/               # Unit tests for src modules
│   ├── test_data_enrichment.py
│   ├── test_eda.py
│   └── test_event_impact_modeling.py
│
└── README.md            # Project-level README
```

---

## ⚡ Requirements

* Python ≥ 3.10
* Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `pytest`, `streamlit`
* Jupyter Notebook for interactive exploration

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠 Setup & Quick Start

1. **Clone the repository**

```bash
git clone <repo_url>
cd <project_root>
```

2. **Prepare raw data**

Place the following in `data/raw/`:

* `ethiopia_fi_unified_data.csv`
* `reference_codes.csv`

3. **Run the Data Enrichment Pipeline (Task 1)**

```python
from src.data_enrichment import FinancialInclusionDataEnricher

enricher = FinancialInclusionDataEnricher(
    data_path="data/raw/ethiopia_fi_unified_data.csv",
    reference_path="data/raw/reference_codes.csv"
)
enricher.run_full_task1()
```

4. **Perform Exploratory Data Analysis (Task 2)**

```python
from src.eda import EthiopiaFinancialInclusionEDA

eda = EthiopiaFinancialInclusionEDA(
    data_path="data/processed/ethiopia_fi_enriched.csv",
    output_dir="reports/figures/eda"
)
eda.summarize_dataset()
eda.plot_account_ownership()
eda.plot_usage_trends(["ACC_OWNERSHIP","USG_P2P_COUNT"])
eda.plot_correlation(["ACC_OWNERSHIP","ACC_MM_ACCOUNT","USG_P2P_COUNT"])
```

5. **Event Impact Modeling (Task 3)**

```python
from src.event_impact_modeling import EventImpactModeler

modeler = EventImpactModeler(
    data_path="data/processed/ethiopia_fi_enriched.csv",
    impact_links_path="data/processed/impact_links.csv"
)

summary = modeler.event_indicator_summary()
matrix = modeler.build_association_matrix()
simulated = modeler.simulate_indicator("ACC_OWNERSHIP")
validation = modeler.validate_against_actual("ACC_OWNERSHIP")
```

6. **Forecasting Indicators (Task 4)**

```python
from src.forecasting import FinancialInclusionForecaster

forecaster = FinancialInclusionForecaster(model=modeler, forecast_horizon=36)
forecaster.forecast_all_indicators()
forecaster.save_forecast_artifacts("data/forecasts")
forecaster.plot_forecast("ACC_OWNERSHIP")
scenario_dfs = forecaster.scenario_forecast("ACC_OWNERSHIP")
```

7. **Launch Interactive Dashboard**

```bash
streamlit run app/dashboard.py
```

---

## 📊 Workflow Overview

### 1. Data Enrichment (Task 1)

* Load raw dataset & reference codes
* Validate schema and indicator coverage
* Add **observations** (smartphone, internet, mobile money adoption)
* Add **events** (policies, regulations)
* Create **impact links** between events and indicators
* Save enriched dataset and log

### 2. Exploratory Data Analysis (Task 2)

* Dataset overview and quality checks
* Temporal coverage, gaps, and trends
* Access, usage, and infrastructure analyses
* Correlation analysis and key insights

### 3. Event Impact Modeling (Task 3)

* Build Event × Indicator association matrices
* Simulate indicator paths using ramped event effects
* Validate simulated values against observations
* Generate visualizations of event impacts

### 4. Forecasting (Task 4)

* Fit trend-based regression models for all indicators
* Generate event-augmented forecasts
* Perform scenario analysis (optimistic, base, pessimistic)
* Export forecast CSVs and visualize scenarios

### 5. Interactive Dashboard

* Overview of key metrics and latest forecasts
* Trends visualization with date range filtering
* Forecasts and scenario comparison
* Account ownership projections vs strategic targets
* Downloadable CSVs for stakeholders

---

## 📝 Outputs

* **Enriched Dataset:** `data/processed/ethiopia_fi_enriched.csv`
* **Impact Links:** `data/processed/impact_links.csv`
* **Forecasts:** `data/forecasts/`
* **Data Enrichment Log:** `data/processed/data_enrichment_log.md`
* **Visualizations:** `reports/figures/`

---

## ✅ Testing

Run all unit tests:

```bash
pytest tests/
```

---

## 🔹 Key Insights

* **Account Ownership** is increasing but growth is slowing.
* **Mobile Money Gap** persists, showing room for digital expansion.
* **Digital Transactions (P2P)** are growing rapidly.
* **Infrastructure** like 4G coverage is limited and requires investment.
* **Event Impacts** are quantifiable and inform scenario-based forecasts.
* **Forecasting** provides projected trends under multiple scenarios to guide strategy.

---

## ⚙️ Notes

* Analysis is fully reproducible using notebooks and scripts.
* Future work: add new indicators, improve temporal coverage, expand event-impact modeling, and update dashboard interactivity.


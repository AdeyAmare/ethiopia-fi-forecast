# `notebooks`

This folder contains Jupyter notebooks for **exploring, visualizing, and analyzing Ethiopia’s financial inclusion data**. The notebooks demonstrate how enriched datasets are transformed into insights, figures, and reports.

## 📁 Notebook Overview

1. **`data_enrichment_exploration.ipynb`**

   * Explores the dataset after enrichment (Task 1)
   * Visualizes key enablers such as smartphone penetration and internet usage
   * Tracks event timelines and highlights newly added events
   * Displays impact links showing relationships between events and indicators

2. **`eda.ipynb`**

   * Performs comprehensive Exploratory Data Analysis (Task 2)
   * Summarizes dataset quality and temporal coverage
   * Visualizes trends for account ownership, mobile money adoption, and infrastructure
   * Computes correlations between indicators and highlights key insights
   * Generates figures and tables for reporting

---

## 🔄 Workflow

1. **Load Enriched Dataset** – Reads the output of the `src/data_enrichment.py` pipeline.
2. **Visualize Key Indicators** – Create bar charts, line plots, and heatmaps to explore trends and gaps.
3. **Analyze Events** – Timeline plots show financial inclusion events and regulatory changes.
4. **Correlations & Insights** – Identify strong relationships between indicators, calculate growth rates, and summarize key findings.
5. **Export Figures** – All visualizations are saved under `reports/figures` for reporting or presentation purposes.

---

## 📊 Outputs

* **Figures & Plots** – Stored in `reports/figures`

  * Enabler vs outcome comparisons
  * Indicator coverage before and after enrichment
  * Event timelines with enriched events highlighted
  * Infrastructure and access trends
  * Correlation heatmaps and strongest correlations
* **Insights & Data Quality Summaries** – Key trends, gaps, and limitations identified for decision-making

---

## ⚡ Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```


---
Here’s an updated version of your `notebooks` README that includes **Task 3 (Event Impact Modeling)** and **Task 4 (Forecasting)** notebooks:

---

# `notebooks`

This folder contains Jupyter notebooks for **exploring, visualizing, modeling, and forecasting Ethiopia’s financial inclusion data**. The notebooks demonstrate how enriched datasets are transformed into insights, figures, and reports.

## 📁 Notebook Overview

1. **`data_enrichment_exploration.ipynb`**

   * Explores the dataset after enrichment (Task 1)
   * Visualizes key enablers such as smartphone penetration and internet usage
   * Tracks event timelines and highlights newly added events
   * Displays impact links showing relationships between events and indicators

2. **`eda.ipynb`**

   * Performs comprehensive Exploratory Data Analysis (Task 2)
   * Summarizes dataset quality and temporal coverage
   * Visualizes trends for account ownership, mobile money adoption, and infrastructure
   * Computes correlations between indicators and highlights key insights
   * Generates figures and tables for reporting

3. **`event_impact_modeling.ipynb`**

   * Implements Task 3: Event Impact Modeling
   * Demonstrates how events (policies, product launches, infrastructure investments) affect indicators over time
   * Builds Event × Indicator association matrices
   * Simulates indicator paths based on historical baseline + event effects
   * Validates simulated values against observed data
   * Visualizes event impacts and ramps

4. **`forecaster.ipynb`**

   * Implements Task 4: Forecasting Access and Usage Indicators (2025–2027)
   * Fits trend-based regression models for all indicators
   * Generates event-augmented forecasts and scenario analyses (optimistic, base, pessimistic)
   * Visualizes forecasts with confidence intervals
   * Summarizes event impacts on forecasted indicators
   * Exports forecast data and figures for reporting

---

## 🔄 Workflow

1. **Load Enriched Dataset** – Reads the output of the `src/data_enrichment.py` pipeline.
2. **Visualize Key Indicators** – Create bar charts, line plots, and heatmaps to explore trends and gaps.
3. **Analyze Events** – Timeline plots show financial inclusion events and regulatory changes.
4. **Model Event Impacts** – Build association matrices, simulate indicator paths, and validate against observed data.
5. **Forecast Indicators** – Generate trend and event-augmented forecasts, conduct scenario analyses, and visualize results.
6. **Correlations & Insights** – Identify strong relationships between indicators, calculate growth rates, and summarize key findings.
7. **Export Figures & Data** – All visualizations are saved under `reports/figures`, and forecast data is saved for reporting or presentation purposes.

---

## 📊 Outputs

* **Figures & Plots** – Stored in `reports/figures`

  * Enabler vs outcome comparisons
  * Indicator coverage before and after enrichment
  * Event timelines with enriched events highlighted
  * Infrastructure and access trends
  * Event impact ramp plots and association matrices
  * Forecasts and scenario visualizations
  * Correlation heatmaps and strongest correlations

* **Insights & Data Quality Summaries** – Key trends, gaps, and limitations identified for decision-making

* **Forecast Data** – Event-augmented and scenario forecasts for all indicators

---

## ⚡ Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

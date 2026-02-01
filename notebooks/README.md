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

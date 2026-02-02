# `app` Directory

This folder contains the **Streamlit dashboard application** for visualizing, exploring, and projecting Ethiopia’s financial inclusion indicators. The dashboard integrates forecast data, scenario analyses, and key metrics for interactive reporting.

## 📁 File Overview

```
app/
└── dashboard.py      # Streamlit app for interactive visualization of forecasts, trends, and projections
```

---

## 💻 `dashboard.py` – Financial Inclusion Dashboard

**Purpose:**

* Display key financial inclusion metrics in Ethiopia
* Visualize indicator trends over time
* Show forecasted values with scenario analysis (optimistic, base, pessimistic)
* Project progress toward inclusion targets (e.g., 60% account ownership)
* Compare P2P / ATM crossover ratios under different scenarios

**Features / Pages:**

1. **Overview**

   * Key metrics for selected indicators
   * Latest forecast values
   * P2P / ATM crossover ratio charts per scenario

2. **Trends**

   * Interactive multi-indicator line plots
   * Date range selection for visualizing trends
   * Compare historical and forecasted values

3. **Forecasts**

   * Event-augmented and trend-based forecasts
   * Scenario selection (optimistic, base, pessimistic)
   * Download forecast CSVs

4. **Projections**

   * Visualize account ownership progress toward strategic targets
   * Horizontal target lines and annotations for easy interpretation

---

## 🔄 Workflow

1. **Load Forecast Data** – Reads forecast CSVs from the `models/forecasts` directory
2. **Load P2P / ATM Ratio Data** – Reads scenario-based ratio files from the same folder
3. **Interactive Navigation** – Users can select pages via sidebar radio buttons
4. **Dynamic Visualizations** – Trend, forecast, and projection charts rendered with Plotly
5. **Downloadable Data** – Users can download scenario-specific forecast CSVs directly from the dashboard

---

## ⚡ Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run app/dashboard.py
```

---

## 📊 Inputs & Outputs

**Inputs:**

* Forecast CSVs (e.g., `ACC_OWNERSHIP_forecast.csv`, `USG_TELEBIRR_USERS_forecast.csv`)
* Scenario CSVs for P2P / ATM ratios (`P2P_ATM_Ratio_base.csv`, etc.)

**Outputs:**

* Interactive dashboards for key metrics, trends, forecasts, and projections
* Downloadable scenario-specific forecast CSVs
* Plots rendered within the dashboard (Plotly)

---

## 🛠️ Notes

* Handles missing or corrupt files gracefully with error messages in the Streamlit app


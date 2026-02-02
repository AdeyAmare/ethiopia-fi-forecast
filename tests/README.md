# `tests` – Unit Tests for Financial Inclusion Modules

This folder contains **unit tests** for the core source code in `src/`. The tests validate functionality, edge cases, and data handling for both **data enrichment**, **exploratory data analysis**, and **event impact modeling** pipelines.

## 📁 Structure

* `test_data_enrichment.py` – Tests the `FinancialInclusionDataEnricher` class:

  * Loading and validating datasets
  * Adding observations, events, and impact links
  * Running the full enrichment pipeline and saving outputs

* `test_eda.py` – Tests the `EthiopiaFinancialInclusionEDA` class:

  * Handling missing or invalid data
  * Growth rate and CAGR calculations
  * Correlation analysis
  * New indicators analysis
  * Ensuring plotting functions run without errors

* `test_event_impact_modeler.py` – Tests the `EventImpactModeler` class:

  * Loading events, observations, and impact links, including missing or invalid data
  * Generating event-indicator summaries
  * Building association matrices with unknown magnitudes or missing events
  * Simulating indicator paths with ramping effects and empty observation sets
  * Validating simulated vs actual values and calculating errors
  * Edge case handling: missing dates, invalid magnitudes, nonexistent indicators, and NaNs

## ⚡ Running Tests

Use **pytest** to run all tests:

```bash
cd <project_root>
pytest tests/
````

* Temporary CSV files are used for testing; no permanent data is modified.
* Tests cover normal scenarios, edge cases, and plotting function execution.

## 📝 Notes

* Ensure that `src/` is in your Python path before running tests.
* Tests are designed to catch regressions and ensure the reliability of enrichment, EDA, and event impact modeling pipelines.


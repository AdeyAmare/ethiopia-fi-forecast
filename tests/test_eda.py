import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from src.eda import EthiopiaFinancialInclusionEDA  # replace with your module name

# -----------------------------
# Sample data for testing edge cases
# -----------------------------
EDGE_CASE_DATA = pd.DataFrame([
    # Normal observation
    {
        "record_id": "REC_0001",
        "record_type": "observation",
        "pillar": "ACCESS",
        "indicator": "Financial Account Ownership",
        "indicator_code": "ACC_OWNERSHIP",
        "value_numeric": 30,
        "observation_date": "2024-01-01",
        "confidence": "high",
        "source_type": "survey",
    },
    # Observation with missing date
    {
        "record_id": "REC_0002",
        "record_type": "observation",
        "pillar": "USAGE",
        "indicator": "Mobile Money Accounts",
        "indicator_code": "ACC_MM_ACCOUNT",
        "value_numeric": 20,
        "observation_date": "",
        "confidence": "medium",
        "source_type": "survey",
    },
    # Observation with non-numeric value
    {
        "record_id": "REC_0003",
        "record_type": "observation",
        "pillar": "USAGE",
        "indicator": "P2P Transactions",
        "indicator_code": "USG_P2P_COUNT",
        "value_numeric": "NaN",
        "observation_date": "2024-01-02",
        "confidence": "low",
        "source_type": "survey",
    },
    # Event
    {
        "record_id": "EVT_0001",
        "record_type": "event",
        "indicator": "Agent Banking Regulation Issued",
        "indicator_code": "EVT_AGENT_REG",
        "observation_date": "2023-06-01",
        "confidence": "high",
        "pillar": "ACCESS",
        "source_type": "regulator",
    },
    # Impact link
    {
        "record_id": "LNK_0001",
        "record_type": "impact_link",
        "indicator": "",
        "indicator_code": "",
        "observation_date": "2023-06-01",
        "confidence": "high",
        "pillar": "ACCESS",
        "source_type": "literature",
    },
])

# -----------------------------
# Pytest fixture
# -----------------------------
@pytest.fixture
def temp_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "edge_case_data.csv"
        EDGE_CASE_DATA.to_csv(data_path, index=False)
        yield data_path, Path(tmpdir)

# -----------------------------
# Tests
# -----------------------------
def test_empty_and_missing_values(temp_csv):
    data_path, tmpdir = temp_csv
    eda = EthiopiaFinancialInclusionEDA(data_path, output_dir=tmpdir)

    # Dataset subsets
    assert len(eda.obs) == 2  # Only 2 valid observation dates
    assert len(eda.events) == 1
    assert len(eda.links) == 1

    # Sparse indicators (single data point)
    sparse = eda.identify_sparse_indicators(min_points=2)
    assert "ACC_OWNERSHIP" in sparse.index
    assert sparse["ACC_OWNERSHIP"] == 1

    # Check data quality report
    dq = eda.data_quality_report()
    assert dq["findex_points"] == 1
    assert dq["impact_links"] == 1

def test_growth_and_cagr_edge_cases(temp_csv):
    data_path, tmpdir = temp_csv
    eda = EthiopiaFinancialInclusionEDA(data_path, output_dir=tmpdir)

    # Growth per period
    growth = eda.calculate_growth_rates("ACC_OWNERSHIP")
    assert "growth_pp" in growth.columns
    assert pd.isna(growth["growth_pp"].iloc[0])  # first difference should be NaN

    # CAGR with single point
    cagr = eda.calculate_cagr("ACC_OWNERSHIP")
    assert np.isnan(cagr)

    # CAGR with invalid start or zero duration
    eda.obs.loc[eda.obs["indicator_code"] == "ACC_OWNERSHIP", "observation_date"] = "2024-01-01"
    cagr2 = eda.calculate_cagr("ACC_OWNERSHIP")
    assert np.isnan(cagr2)

def test_correlation_edge_cases(temp_csv):
    data_path, tmpdir = temp_csv
    eda = EthiopiaFinancialInclusionEDA(data_path, output_dir=tmpdir)

    # Correlation with only one valid numeric column
    corr = eda.correlation_matrix(["ACC_OWNERSHIP", "USG_P2P_COUNT"])
    # USG_P2P_COUNT is NaN, correlation should be NaN
    assert np.isnan(corr.loc["ACC_OWNERSHIP", "USG_P2P_COUNT"])

def test_new_indicators_analysis(temp_csv):
    data_path, tmpdir = temp_csv
    eda = EthiopiaFinancialInclusionEDA(data_path, output_dir=tmpdir)

    result = eda.analyze_new_indicators()
    assert result["direct_indicators"] == 0
    assert result["indirect_indicators"] == 0
    assert result["direct_observations"] == 0
    assert result["indirect_observations"] == 0

def test_plot_functions_do_not_fail(temp_csv):
    data_path, tmpdir = temp_csv
    eda = EthiopiaFinancialInclusionEDA(data_path, output_dir=tmpdir)

    # Just test that plotting functions run without errors
    eda.plot_confidence_distribution(save_fig=False)
    eda.temporal_coverage_heatmap(save_fig=False)
    eda.pillar_distribution(save_fig=False)
    eda.plot_account_ownership(save_fig=False)
    eda.plot_usage_trends(["ACC_OWNERSHIP"], save_fig=False)
    eda.plot_infrastructure_indicators(["ACC_OWNERSHIP"], save_fig=False)
    eda.plot_event_timeline(save_fig=False)
    eda.plot_indicator_with_events("ACC_OWNERSHIP", save_fig=False)
    eda.plot_correlation(["ACC_OWNERSHIP"], save_fig=False)

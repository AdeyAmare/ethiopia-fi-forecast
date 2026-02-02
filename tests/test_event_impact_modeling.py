import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from src.event_impact_modeling import EventImpactModeler

# -----------------------------
# EDGE CASE DATA
# -----------------------------
EDGE_CASE_DATA = pd.DataFrame([
    # Observation with valid value
    {"record_id": "OBS_001", "record_type": "observation", "indicator_code": "IND_001",
     "observation_date": "2026-01-01", "value_numeric": "10"},
    
    # Observation with missing value
    {"record_id": "OBS_002", "record_type": "observation", "indicator_code": "IND_002",
     "observation_date": "2026-01-01", "value_numeric": ""},
    
    # Event with valid date
    {"record_id": "EVT_001", "record_type": "event", "indicator": "IND_001",
     "observation_date": "2026-02-01", "category": "Policy"},
    
    # Event with missing date
    {"record_id": "EVT_002", "record_type": "event", "indicator": "IND_002",
     "observation_date": ""},
])

EDGE_CASE_LINKS = pd.DataFrame([
    # Valid link
    {"parent_id": "EVT_001", "related_indicator": "IND_001",
     "impact_magnitude": "high", "impact_direction": "increase", "lag_months": "1"},
    
    # Invalid magnitude
    {"parent_id": "EVT_002", "related_indicator": "IND_002",
     "impact_magnitude": "unknown", "impact_direction": "decrease", "lag_months": "abc"},
    
    # Nonexistent event
    {"parent_id": "EVT_999", "related_indicator": "IND_001",
     "impact_magnitude": "medium", "impact_direction": "increase", "lag_months": "0"},
])

# -----------------------------
# FIXTURE TO CREATE TEMP FILES
# -----------------------------
@pytest.fixture
def temp_csv_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        links_path = Path(tmpdir) / "links.csv"
        EDGE_CASE_DATA.to_csv(data_path, index=False)
        EDGE_CASE_LINKS.to_csv(links_path, index=False)
        yield data_path, links_path

# -----------------------------
# TESTS
# -----------------------------
def test_loading_edge_cases(temp_csv_files):
    data_path, links_path = temp_csv_files
    modeler = EventImpactModeler(data_path, links_path)
    
    # Ensure events and observations are loaded correctly
    assert len(modeler.events) == 2
    assert len(modeler.obs) == 2
    
    # Ensure invalid numeric values are converted to NaN
    assert np.isnan(modeler.obs.loc[modeler.obs["indicator_code"] == "IND_002", "value_numeric"]).all()
    
    # Ensure invalid lag_months are converted to NaN
    assert np.isnan(modeler.links.loc[modeler.links["parent_id"] == "EVT_002", "lag_months"]).all()

def test_event_indicator_summary(temp_csv_files):
    data_path, links_path = temp_csv_files
    modeler = EventImpactModeler(data_path, links_path)
    summary = modeler.event_indicator_summary()
    
    # Check that summary contains expected columns
    expected_cols = [
        "parent_id", "indicator", "category", "event_date",
        "pillar", "related_indicator", "impact_direction",
        "impact_magnitude", "lag_months", "confidence"
    ]
    for col in expected_cols:
        assert col in summary.columns

def test_association_matrix(temp_csv_files):
    data_path, links_path = temp_csv_files
    modeler = EventImpactModeler(data_path, links_path)
    matrix = modeler.build_association_matrix()
    
    # Matrix should have correct row/column names
    assert "EVT_001" in matrix.index
    assert "IND_001" in matrix.columns
    
    # EVT_002 has unknown magnitude, should result in 0
    assert matrix.get("IND_002", pd.Series([0])).sum() == 0

def test_simulate_indicator_edge_cases(temp_csv_files):
    data_path, links_path = temp_csv_files
    modeler = EventImpactModeler(data_path, links_path)
    
    # Existing indicator with event
    sim = modeler.simulate_indicator("IND_001")
    assert sim is not None
    assert "simulated" in sim.columns
    assert sim["simulated"].iloc[0] == 10  # starts from first observed value
    
    # Indicator with no observations
    sim_none = modeler.simulate_indicator("IND_999")
    assert sim_none is None

def test_validate_against_actual(temp_csv_files):
    data_path, links_path = temp_csv_files
    modeler = EventImpactModeler(data_path, links_path)
    
    validation = modeler.validate_against_actual("IND_001")
    assert validation is not None
    assert "error_pp" in validation.columns
    
    # Nonexistent indicator
    validation_none = modeler.validate_against_actual("IND_999")
    assert validation_none is None

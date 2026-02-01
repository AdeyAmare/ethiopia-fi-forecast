import pytest
import pandas as pd
from pathlib import Path
from src.data_enrichment import FinancialInclusionDataEnricher  # replace with your actual module name
import tempfile

# -----------------------------
# Sample data for testing
# -----------------------------
SAMPLE_DATA = pd.DataFrame([
    {
        "record_id": "REC_0001",
        "record_type": "observation",
        "pillar": "ACCESS",
        "indicator": "Financial Account Ownership",
        "indicator_code": "ACC_OWNERSHIP",
        "indicator_direction": "higher_better",
        "value_numeric": 30,
        "value_type": "percentage",
        "unit": "%",
        "observation_date": "2024-01-01",
        "gender": "all",
        "location": "national",
        "source_name": "Survey",
        "source_type": "survey",
        "confidence": "high",
        "notes": ""
    }
])

REFERENCE_DATA = pd.DataFrame([
    {"indicator_code": "ACC_OWNERSHIP", "description": "Account ownership"}
])

# -----------------------------
# Pytest fixtures
# -----------------------------
@pytest.fixture
def temp_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "data.csv"
        ref_path = Path(tmpdir) / "reference.csv"

        SAMPLE_DATA.to_csv(data_path, index=False)
        REFERENCE_DATA.to_csv(ref_path, index=False)

        yield data_path, ref_path, Path(tmpdir)

# -----------------------------
# Tests
# -----------------------------
def test_load_data(temp_files):
    data_path, ref_path, _ = temp_files
    enricher = FinancialInclusionDataEnricher(data_path, ref_path)
    enricher.load_data()
    
    assert enricher.df is not None
    assert enricher.reference is not None
    assert enricher.df.shape[0] == SAMPLE_DATA.shape[0]

def test_add_observations(temp_files):
    data_path, ref_path, _ = temp_files
    enricher = FinancialInclusionDataEnricher(data_path, ref_path)
    enricher.load_data()
    enricher.add_observations()
    
    # Check new observations added
    assert enricher.df.shape[0] == SAMPLE_DATA.shape[0] + 2
    assert any(enricher.df["indicator_code"] == "ACC_SMARTPHONE")
    assert len(enricher.enrichment_log) == 2

def test_add_events(temp_files):
    data_path, ref_path, _ = temp_files
    enricher = FinancialInclusionDataEnricher(data_path, ref_path)
    enricher.load_data()
    enricher.add_events()
    
    assert any(enricher.df["record_type"] == "event")
    assert len(enricher.enrichment_log) == 1

def test_add_impact_links(temp_files):
    data_path, ref_path, _ = temp_files
    enricher = FinancialInclusionDataEnricher(data_path, ref_path)
    enricher.load_data()
    enricher.add_impact_links()
    
    assert any(enricher.df["record_type"] == "impact_link")
    assert len(enricher.enrichment_log) == 2  # 2 links added

def test_run_full_task1(temp_files):
    data_path, ref_path, tmpdir = temp_files
    enricher = FinancialInclusionDataEnricher(data_path, ref_path, output_dir=tmpdir)
    enricher.run_full_task1()
    
    # Check enriched dataset saved
    enriched_file = Path(tmpdir) / "ethiopia_fi_enriched.csv"
    log_file = Path(tmpdir) / "data_enrichment_log.md"
    
    assert enriched_file.exists()
    assert log_file.exists()
    
    # Check all enrichments present
    df = pd.read_csv(enriched_file)
    assert any(df["indicator_code"] == "ACC_SMARTPHONE")
    assert any(df["record_type"] == "event")
    assert any(df["record_type"] == "impact_link")

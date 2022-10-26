import geopandas as gpd
import pandas as pd
import pytest

from dp_mobility_report import DpMobilityReport
from dp_mobility_report.benchmark.similarity_measures import compute_similarity_measures
from dp_mobility_report import constants as const



@pytest.fixture
def proposal_dpmreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return DpMobilityReport(test_data, test_tessellation, privacy_budget=1000).report

@pytest.fixture
def benchmark_dpmreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return DpMobilityReport(test_data, test_tessellation, privacy_budget=None).report

@pytest.fixture
def test_tessellation():
    """Load a test tessellation."""
    return gpd.read_file("tests/test_files/test_tessellation.geojson")



#def test_histogram_bin_sizes():
#    pass


def test_similarity_measures(proposal_dpmreport, benchmark_dpmreport, test_tessellation):

    test_tessellation.loc[:, const.TILE_ID] = test_tessellation.tile_id.astype(str)

    sim_dict = compute_similarity_measures(proposal_dpmreport, benchmark_dpmreport, test_tessellation, cost_matrix=None)

    pass
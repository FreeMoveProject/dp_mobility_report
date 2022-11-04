import geopandas as gpd
import pandas as pd
import pytest
import numpy as np

from dp_mobility_report import DpMobilityReport
from dp_mobility_report.benchmark.similarity_measures import (
    compute_similarity_measures,
    earth_movers_distance1D
)
from dp_mobility_report.benchmark import benchmarkreport
from dp_mobility_report.benchmark.preprocessing import combine_analysis_exclusion
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

@pytest.fixture
def benchmark_report():
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_data_proposal = pd.read_csv("tests/test_files/test_data.csv", nrows=50)
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return benchmarkreport.BenchmarkReport(test_data, test_data_proposal, test_tessellation)


def test_histogram_bin_sizes(benchmark_report):

    assert list(benchmark_report.report_proposal.report[const.TRAVEL_TIME].data[1]) == list(benchmark_report.report_benchmark.report[const.TRAVEL_TIME].data[1])
    assert list(benchmark_report.report_proposal.report[const.JUMP_LENGTH].data[1]) == list(benchmark_report.report_benchmark.report[const.JUMP_LENGTH].data[1])
    assert list(benchmark_report.report_proposal.report[const.TRIPS_PER_USER].data[1]) == list(benchmark_report.report_benchmark.report[const.TRIPS_PER_USER].data[1])
    assert list(benchmark_report.report_proposal.report[const.RADIUS_OF_GYRATION].data[1]) == list(benchmark_report.report_benchmark.report[const.RADIUS_OF_GYRATION].data[1])
    assert list(benchmark_report.report_proposal.report[const.MOBILITY_ENTROPY].data[1]) == list(benchmark_report.report_benchmark.report[const.MOBILITY_ENTROPY].data[1])


def test_earth_movers_distance1D():
    assert earth_movers_distance1D((np.array([4,2,5,3]), np.array([0,2,5,7,9])), (np.array([4,2,5,3]), np.array([0,2,5,7,9]))) == 0
    assert earth_movers_distance1D((np.array([4,2,5,3]), np.array([0,2,5,7,9])), (np.array([8,2,3,5]), np.array([0,2,5,7,9]))) == 0.8412698412698414
    assert earth_movers_distance1D((np.array([4,2,5,3,7]), np.array([0,1,2,3,4])), (np.array([4,2,5,3,7]), np.array([0,1,2,3,4]))) == 0
    assert earth_movers_distance1D((np.array([4,2,5,3,7]), np.array([0,1,2,3,4])), (np.array([12,4,8,2,7]), np.array([0,1,2,3,4]))) == 0.696969696969697


def test_similarity_measures(
    proposal_dpmreport, benchmark_dpmreport, test_tessellation
):

    test_tessellation.loc[:, const.TILE_ID] = test_tessellation.tile_id.astype(str)

    analysis_exclusion = [const.MOBILITY_ENTROPY]
    #TODO analysis_exclusion = [const.VISITS_PER_TILE]
    (
        relative_error_dict,
        kld_dict,
        jsd_dict,
        emd_dict,
        smape_dict,
    ) = compute_similarity_measures(
        analysis_exclusion,
        proposal_dpmreport,
        benchmark_dpmreport,
        test_tessellation,
        cost_matrix=None,
    )

    assert isinstance(relative_error_dict, dict)
    assert isinstance(kld_dict, dict)
    assert isinstance(jsd_dict, dict)
    assert isinstance(emd_dict, dict)
    assert isinstance(smape_dict, dict)


def test_combine_analysis_exclusion():
    excluded_analyses_proposal = [const.VISITS_PER_TILE, const.RADIUS_OF_GYRATION]
    excluded_analyses_benchmark = [const.RADIUS_OF_GYRATION]

    combined_exclusion = combine_analysis_exclusion(
        excluded_analyses_proposal, excluded_analyses_benchmark
    )
    assert combined_exclusion == [const.VISITS_PER_TILE, const.RADIUS_OF_GYRATION]


def test_unify_histogram_bins():
    # TODO
    pass
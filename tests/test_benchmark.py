import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from dp_mobility_report import DpMobilityReport
from dp_mobility_report import constants as const
from dp_mobility_report.benchmark import b_utils, benchmarkreport
from dp_mobility_report.benchmark.preprocessing import combine_analysis_exclusion, validate_measure_selection
from dp_mobility_report.benchmark.similarity_measures import (
    compute_similarity_measures,
    earth_movers_distance1D,
    get_selected_measures,
)


@pytest.fixture
def alternative_dpmreport():
    """Create an alternative test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return DpMobilityReport(test_data, test_tessellation, privacy_budget=1000).report


@pytest.fixture
def base_dpmreport():
    """Create a base test report."""
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
    test_data_alternative = pd.read_csv("tests/test_files/test_data.csv", nrows=50)
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return benchmarkreport.BenchmarkReport(
        df_base=test_data,
        tessellation=test_tessellation,
        df_alternative=test_data_alternative,
    )


def test_histogram_bin_sizes(benchmark_report):

    assert list(
        benchmark_report.report_alternative.report[const.TRAVEL_TIME].data[1]
    ) == list(benchmark_report.report_base.report[const.TRAVEL_TIME].data[1])
    assert list(
        benchmark_report.report_alternative.report[const.JUMP_LENGTH].data[1]
    ) == list(benchmark_report.report_base.report[const.JUMP_LENGTH].data[1])
    # assert list(
    #     benchmark_report.report_alternative.report[const.TRIPS_PER_USER].data[1]
    # ) == list(benchmark_report.report_base.report[const.TRIPS_PER_USER].data[1])
    assert list(
        benchmark_report.report_alternative.report[const.RADIUS_OF_GYRATION].data[1]
    ) == list(benchmark_report.report_base.report[const.RADIUS_OF_GYRATION].data[1])
    # assert list(
    #     benchmark_report.report_alternative.report[const.MOBILITY_ENTROPY].data[1]
    # ) == list(benchmark_report.report_base.report[const.MOBILITY_ENTROPY].data[1])


def test_earth_movers_distance1D():
    assert (
        earth_movers_distance1D(
            (np.array([4, 2, 5, 3]), np.array([0, 2, 5, 7, 9])),
            (np.array([4, 2, 5, 3]), np.array([0, 2, 5, 7, 9])),
        )
        == 0
    )
    assert (
        earth_movers_distance1D(
            (np.array([4, 2, 5, 3]), np.array([0, 2, 5, 7, 9])),
            (np.array([8, 2, 3, 5]), np.array([0, 2, 5, 7, 9])),
        )
        == 0.8412698412698414
    )
    assert (
        earth_movers_distance1D(
            (np.array([4, 2, 5, 3, 7]), np.array([0, 1, 2, 3, 4])),
            (np.array([4, 2, 5, 3, 7]), np.array([0, 1, 2, 3, 4])),
        )
        == 0
    )
    assert (
        earth_movers_distance1D(
            (np.array([4, 2, 5, 3, 7]), np.array([0, 1, 2, 3, 4])),
            (np.array([12, 4, 8, 2, 7]), np.array([0, 1, 2, 3, 4])),
        )
        == 0.696969696969697
    )


def test_similarity_measures(alternative_dpmreport, base_dpmreport, test_tessellation):

    test_tessellation.loc[:, const.TILE_ID] = test_tessellation.tile_id.astype(str)

    analysis_exclusion = [const.MOBILITY_ENTROPY]
    # TODO analysis_exclusion = [const.VISITS_PER_TILE]
    (
        relative_error_dict,
        kld_dict,
        jsd_dict,
        emd_dict,
        smape_dict,
    ) = compute_similarity_measures(
        analysis_exclusion, alternative_dpmreport, base_dpmreport, test_tessellation
    )

    assert isinstance(relative_error_dict, dict)
    assert isinstance(kld_dict, dict)
    assert isinstance(jsd_dict, dict)
    assert isinstance(emd_dict, dict)
    assert isinstance(smape_dict, dict)


def test_combine_analysis_exclusion():
    excluded_analyses_alternative = [const.VISITS_PER_TILE, const.RADIUS_OF_GYRATION]
    excluded_analyses_base = [const.RADIUS_OF_GYRATION]

    combined_exclusion = combine_analysis_exclusion(
        excluded_analyses_alternative, excluded_analyses_base
    )
    assert combined_exclusion == [const.VISITS_PER_TILE, const.RADIUS_OF_GYRATION]


def test_unify_histogram_bins():
    # TODO
    pass


def test_get_selected_measures(benchmark_report):

    similarity_measures = get_selected_measures(benchmark_report)
    assert isinstance(similarity_measures, dict)
    assert not None in similarity_measures.values()

    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_data_alternative = pd.read_csv("tests/test_files/test_data.csv", nrows=50)
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    benchmark_report = benchmarkreport.BenchmarkReport(
        test_data, test_tessellation, test_data_alternative, measure_selection={const.TRAVEL_TIME_QUARTILES: const.JSD}
    )

    assert const.JSD == benchmark_report.measure_selection[const.TRAVEL_TIME_QUARTILES]
    with pytest.warns(Warning):
        similarity_measures = get_selected_measures(benchmark_report)
    assert similarity_measures[const.TRAVEL_TIME_QUARTILES] == None


def test_benchmark_report(benchmark_report):

    assert isinstance(benchmark_report.emd, dict)
    assert isinstance(benchmark_report.jsd, dict)
    assert isinstance(benchmark_report.kld, dict)
    assert isinstance(benchmark_report.re, dict)
    assert isinstance(benchmark_report.smape, dict)
    assert isinstance(benchmark_report.measure_selection, dict)
    assert isinstance(benchmark_report.similarity_measures, dict)


def test_measure_selection():

    with pytest.warns(Warning):
        validate_measure_selection(measure_selection={const.JUMP_LENGTH: 'klld'}, analysis_exclusion=[const.VISITS_PER_TILE])
    with pytest.warns(Warning):
        validate_measure_selection(measure_selection={'jump_lengthh': const.KLD}, analysis_exclusion=[const.VISITS_PER_TILE])

    assert validate_measure_selection(measure_selection={const.OD_FLOWS: const.SMAPE}, analysis_exclusion=[const.VISITS_PER_TILE, const.DS_STATISTICS, const.MISSING_VALUES, const.TRIPS_OVER_TIME, const.TRIPS_PER_WEEKDAY, const.TRIPS_PER_HOUR, const.TRAVEL_TIME, const.JUMP_LENGTH, const.TRIPS_PER_USER, const.USER_TIME_DELTA, const.RADIUS_OF_GYRATION, const.USER_TILE_COUNT, const.MOBILITY_ENTROPY, const.VISITS_PER_TILE_TIMEWINDOW]) == {const.OD_FLOWS: const.SMAPE} 

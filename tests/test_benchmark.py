import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from dp_mobility_report import BenchmarkReport, DpMobilityReport
from dp_mobility_report import constants as const
from dp_mobility_report.benchmark.preprocessing import (
    combine_analysis_exclusion,
    validate_measure_selection,
    validate_top_n_ranking
)
from dp_mobility_report.benchmark.similarity_measures import (
    compute_similarity_measures,
    earth_movers_distance1D,
    get_selected_measures,
)


@pytest.fixture
def alternative_dpmreport():
    """Create an alternative test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv", nrows=50)
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return DpMobilityReport(test_data, test_tessellation, privacy_budget=None).report


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
def test_data():
    """Load a test dataset."""
    return pd.read_csv("tests/test_files/test_data.csv")

@pytest.fixture
def test_data_alternative():
    """Load an alternative test dataset."""
    return pd.read_csv("tests/test_files/test_data_new_dates.csv", nrows=50)

@pytest.fixture
def test_data_sequence():
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_data["datetime"] = (
        test_data.groupby("tid").rank(method="first").uid.astype(int)
    )
    return test_data

@pytest.fixture
def test_data_sequence_alternative():
    test_data = pd.read_csv("tests/test_files/test_data_new_dates.csv", nrows=50)
    test_data["datetime"] = (
        test_data.groupby("tid").rank(method="first").uid.astype(int)
    )
    return test_data

@pytest.fixture
def benchmark_report():
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_data_alternative = pd.read_csv(
        "tests/test_files/test_data_new_dates.csv", nrows=50
    )
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return BenchmarkReport(
        df_base=test_data,
        tessellation=test_tessellation,
        df_alternative=test_data_alternative,
        privacy_budget_alternative=10,
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
    assert list(
        benchmark_report.report_alternative.report[const.USER_TILE_COUNT].data[1]
    ) == list(benchmark_report.report_base.report[const.USER_TILE_COUNT].data[1])
    assert list(
        benchmark_report.report_alternative.report[const.MOBILITY_ENTROPY].data[1]
    ) == list(benchmark_report.report_base.report[const.MOBILITY_ENTROPY].data[1])


def test_earth_movers_distance1D():
    assert (
        earth_movers_distance1D(
            (np.array([4, 2, 5, 3]), np.array([0, 2, 5, 7, 9])),
            (np.array([4, 2, 5, 3]), np.array([0, 2, 5, 7, 9])),
            9,
            9
        )
        == 0
    )
    assert (
        earth_movers_distance1D(
            (np.array([4, 2, 5, 3]), np.array([0, 2, 5, 7, 9])),
            (np.array([8, 2, 3, 5]), np.array([0, 2, 5, 7, 9])),
            9,
            9
        )
        == 0.8412698412698414
    )
    assert (
        earth_movers_distance1D(
            (np.array([4, 2, 5, 3, 7]), np.array([0, 1, 2, 3, 4])),
            (np.array([4, 2, 5, 3, 7]), np.array([0, 1, 2, 3, 4])),
            4,
            4
        )
        == 0
    )
    assert (
        earth_movers_distance1D(
            (np.array([4, 2, 5, 3, 7]), np.array([0, 1, 2, 3, 4])),
            (np.array([12, 4, 8, 2, 7]), np.array([0, 1, 2, 3, 4])),
            4,
            4
        )
        == 0.696969696969697
    )


def test_similarity_measures(alternative_dpmreport, base_dpmreport, test_tessellation):

    test_tessellation.loc[:, const.TILE_ID] = test_tessellation.tile_id.astype(str)

    analysis_exclusion = [const.MOBILITY_ENTROPY]
    (
        relative_error_dict,
        kld_dict,
        jsd_dict,
        emd_dict,
        smape_dict,
        kendall_dict,
        top_n_cov_dict,
    ) = compute_similarity_measures(
        analysis_exclusion, alternative_dpmreport, base_dpmreport, test_tessellation, top_n_ranking=[10, 100], disable_progress_bar=True
    )

    assert isinstance(relative_error_dict, dict)
    assert isinstance(kld_dict, dict)
    assert isinstance(jsd_dict, dict)
    assert isinstance(emd_dict, dict)
    assert isinstance(smape_dict, dict)
    assert isinstance(kendall_dict, dict)
    assert isinstance(top_n_cov_dict, dict)
    assert round(jsd_dict[const.VISITS_PER_TILE], 3) == 0.041
    assert round(emd_dict[const.VISITS_PER_TILE], 2) == 315.75
    assert round(jsd_dict[const.VISITS_PER_TIME_TILE], 3) == 0.268
    assert np.isnan(emd_dict[const.VISITS_PER_TIME_TILE])

    analysis_exclusion = [const.VISITS_PER_TILE]
    (
        relative_error_dict,
        kld_dict,
        jsd_dict,
        emd_dict,
        smape_dict,
        kendall_dict,
        top_n_cov_dict,
    ) = compute_similarity_measures(
        analysis_exclusion, alternative_dpmreport, base_dpmreport, test_tessellation, top_n_ranking=[10, 100], disable_progress_bar=True
    )

    assert isinstance(relative_error_dict, dict)
    assert isinstance(kld_dict, dict)
    assert isinstance(jsd_dict, dict)
    assert isinstance(emd_dict, dict)
    assert isinstance(smape_dict, dict)
    assert isinstance(kendall_dict, dict)
    assert isinstance(top_n_cov_dict, dict)

    # TODO: test if emd for time_tile is nan if both (base and alternative) have no values in the same time_window (should then be excluded for computation)


def test_combine_analysis_exclusion():
    excluded_analyses_alternative = [const.VISITS_PER_TILE, const.RADIUS_OF_GYRATION]
    excluded_analyses_base = [const.RADIUS_OF_GYRATION]

    combined_exclusion = combine_analysis_exclusion(
        excluded_analyses_alternative, excluded_analyses_base
    )
    assert combined_exclusion == [const.VISITS_PER_TILE, const.RADIUS_OF_GYRATION]


def test_get_selected_measures(benchmark_report):

    similarity_measures = get_selected_measures(benchmark_report)
    assert isinstance(similarity_measures, dict)
    assert None not in similarity_measures.values()

    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_data_alternative = pd.read_csv("tests/test_files/test_data.csv", nrows=50)
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    benchmark_report = BenchmarkReport(
        test_data,
        test_tessellation,
        test_data_alternative,
        measure_selection={const.TRAVEL_TIME_QUARTILES: const.JSD},
    )

    assert const.JSD == benchmark_report.measure_selection[const.TRAVEL_TIME_QUARTILES]
    with pytest.warns(Warning):
        similarity_measures = get_selected_measures(benchmark_report)
    assert similarity_measures[const.TRAVEL_TIME_QUARTILES] is None


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
        validate_measure_selection(
            measure_selection={const.JUMP_LENGTH: "klld"},
            analysis_exclusion=[const.VISITS_PER_TILE],
        )
    with pytest.warns(Warning):
        validate_measure_selection(
            measure_selection={"jump_lengthh": const.KLD},
            analysis_exclusion=[const.VISITS_PER_TILE],
        )

    assert validate_measure_selection(
        measure_selection={const.OD_FLOWS: const.SMAPE},
        analysis_exclusion=[
            const.VISITS_PER_TILE,
            const.DS_STATISTICS,
            const.MISSING_VALUES,
            const.TRIPS_OVER_TIME,
            const.TRIPS_PER_WEEKDAY,
            const.TRIPS_PER_HOUR,
            const.TRAVEL_TIME,
            const.JUMP_LENGTH,
            const.TRIPS_PER_USER,
            const.USER_TIME_DELTA,
            const.RADIUS_OF_GYRATION,
            const.USER_TILE_COUNT,
            const.MOBILITY_ENTROPY,
            const.VISITS_PER_TIME_TILE,
            const.VISITS_PER_TILE_RANKING,
            const.OD_FLOWS_RANKING,
            const.OD_FLOWS_QUARTILES
        ],
    ) == {const.OD_FLOWS: const.SMAPE}


def test_top_n_ranking_input():
    assert validate_top_n_ranking([1, 10]) == [1, 10]
    
    assert validate_top_n_ranking([10]) == [10]

    with pytest.raises(Exception):
        validate_top_n_ranking(["not", "ints"])

    with pytest.raises(Exception):
        validate_top_n_ranking([0.4, 10])

    with pytest.raises(Exception):
        validate_top_n_ranking([0, 10])

    with pytest.raises(Exception):
        validate_top_n_ranking([-10, 10])



def test_benchmark_to_file(benchmark_report):

    benchmark_report.to_file("test_benchmark.html")
    # test linechart

def test_to_html_file(test_data, test_data_alternative, test_data_sequence, test_data_sequence_alternative, test_tessellation, tmp_path):

    file_name = tmp_path / "html/test_output1.html"
    file_name.parent.mkdir()
    BenchmarkReport(df_base=test_data, tessellation=test_tessellation, df_alternative=test_data_alternative, privacy_budget_base=None, privacy_budget_alternative=15).to_file(
        file_name
    )
    assert file_name.is_file()

    file_name = tmp_path / "html/test_output2.html"
    BenchmarkReport(df_base=test_data, tessellation=test_tessellation, df_alternative=test_data_alternative, privacy_budget_base=15, privacy_budget_alternative=None).to_file(
        file_name
    )
    assert file_name.is_file()

    file_name = tmp_path / "html/test_output3.html"
    BenchmarkReport(
        df_base=test_data, 
        tessellation=test_tessellation, 
        df_alternative=test_data_alternative, 
        privacy_budget_base=100, 
        privacy_budget_alternative=None,
        analysis_exclusion=[
            const.RADIUS_OF_GYRATION,
            const.MOBILITY_ENTROPY,
            const.TRIPS_PER_USER,
        ],
    ).to_file(file_name)
    assert file_name.is_file()

    # without tessellation
    file_name = tmp_path / "html/test_output4.html"
    BenchmarkReport(
        df_base=test_data, 
        df_alternative=test_data_alternative, 
        ).to_file(file_name)
    assert file_name.is_file()

    # without timestamps
    file_name = tmp_path / "html/test_output5.html"
    BenchmarkReport(
        df_base=test_data_sequence,
        df_alternative=test_data_sequence_alternative,
        ).to_file(file_name)
    assert file_name.is_file()
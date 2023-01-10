import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame
from pandas import DataFrame

from dp_mobility_report import DpMobilityReport
from dp_mobility_report import constants as const


@pytest.fixture
def test_data():
    """Load a test dataset."""
    return pd.read_csv("tests/test_files/test_data.csv")


@pytest.fixture
def test_data_sequence():
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_data["datetime"] = (
        test_data.groupby("tid").rank(method="first").uid.astype(int)
    )
    return test_data


@pytest.fixture
def test_tessellation():
    """Load a test tessellation."""
    return gpd.read_file("tests/test_files/test_tessellation.geojson")


def test_DpMobilityReport(test_data, test_data_sequence, test_tessellation):
    """Test instance of DpMobilityReport is created properly with valid input and default values."""
    mob_report = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        analysis_exclusion=[const.JUMP_LENGTH],
    )
    assert isinstance(mob_report, DpMobilityReport)
    assert isinstance(mob_report.max_trips_per_user, (int, np.integer))
    assert isinstance(mob_report._budget_split, dict)
    assert isinstance(mob_report.analysis_exclusion, list)
    assert mob_report.analysis_exclusion == [const.JUMP_LENGTH]
    assert mob_report.privacy_budget is None
    assert isinstance(mob_report.tessellation, GeoDataFrame)
    assert isinstance(mob_report.df, DataFrame)

    mob_report = DpMobilityReport(test_data, test_tessellation, privacy_budget=0.1)
    assert isinstance(mob_report, DpMobilityReport)
    mob_report = DpMobilityReport(
        test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user=None
    )
    assert isinstance(mob_report, DpMobilityReport)

    # only one trip
    mob_report = DpMobilityReport(
        test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user=1
    )
    assert isinstance(mob_report, DpMobilityReport)

    # reasonable number of trips
    mob_report = DpMobilityReport(
        test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user=3
    )
    assert isinstance(mob_report, DpMobilityReport)

    # more trips than present
    mob_report = DpMobilityReport(
        test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user=1000
    )
    assert isinstance(mob_report, DpMobilityReport)

    # TODO: test variations of analysis_inclusion
    mob_report = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        analysis_selection=[const.OVERVIEW],
    )
    assert isinstance(mob_report, DpMobilityReport)

    # TODO: test variations of exclude_analysis
    # variations of exclude_analysis
    mob_report = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        analysis_exclusion=[const.MISSING_VALUES],
    )
    assert isinstance(mob_report, DpMobilityReport)

    # test without datetime
    mob_report = DpMobilityReport(
        test_data_sequence,
        test_tessellation,
        privacy_budget=None,
    )
    assert isinstance(mob_report, DpMobilityReport)

    # test sampling
    mob_report1 = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        max_trips_per_user=1,
        seed_sampling=100,
    )

    mob_report2 = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        max_trips_per_user=1,
        seed_sampling=100,
    )
    assert mob_report1.df.equals(mob_report2.df)


def test_wrong_input_params_DpMobilityReport(
    test_data, test_data_sequence, test_tessellation
):
    """Test if wrong input parameters are caught correctly."""
    # wrong input for dataframe
    with pytest.raises(TypeError):
        DpMobilityReport("not a DataFrame", test_tessellation, privacy_budget=None)

    with pytest.warns(Warning):
        DpMobilityReport(
            test_data_sequence,
            test_tessellation,
            privacy_budget=None,
        )

    # wrong input for tessellation
    with pytest.raises(TypeError):
        DpMobilityReport(test_data, "not a GeoDataFrame", privacy_budget=None)

    # wrong input for privacy_budget
    with pytest.raises(ValueError):
        DpMobilityReport(test_data, test_tessellation, privacy_budget=-1)
    with pytest.raises(TypeError):
        DpMobilityReport(test_data, test_tessellation, privacy_budget="not a number")

    # wrong input for max_trips_per_user
    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data, test_tessellation, max_trips_per_user=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            max_trips_per_user="not an int",
            privacy_budget=None,
        )
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data, test_tessellation, max_trips_per_user=3.1, privacy_budget=None
        )

    # warning if privacy budget but no max trips per user are set
    with pytest.warns(Warning):
        DpMobilityReport(
            test_data, test_tessellation, privacy_budget=1, max_trips_per_user=None
        )

    # wrong analysis_exclusion
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            analysis_exclusion="not a list",
            privacy_budget=None,
        )

    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            analysis_exclusion=[const.OVERVIEW, "wrong input"],
            privacy_budget=None,
        )

    # wrong budget split
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            budget_split="something else than a dict",
            privacy_budget=None,
        )

    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            budget_split={"wrong key": 10},
            privacy_budget=None,
        )

    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            budget_split={const.OD_FLOWS: "not an int"},
            privacy_budget=None,
        )

    # warning with analysis in budget_split and exclude_analysis
    with pytest.warns(Warning):
        DpMobilityReport(
            test_data,
            test_tessellation,
            analysis_exclusion=[const.OD_FLOWS, const.RADIUS_OF_GYRATION],
            budget_split={const.OD_FLOWS: 100},
            privacy_budget=None,
        )

    with pytest.warns(Warning):
        DpMobilityReport(
            test_data,
            test_tessellation,
            analysis_exclusion=[const.OD_ANALYSIS],
            budget_split={const.OD_FLOWS: 100},
            privacy_budget=None,
        )

    # wrong input for max_travel_time
    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data, test_tessellation, max_travel_time=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            max_travel_time="not a number",
            privacy_budget=None,
        )

    # wrong input for bin_range_travel_time
    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data, test_tessellation, bin_range_travel_time=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            bin_range_travel_time="not a number",
            privacy_budget=None,
        )

    # wrong input for max_jump_length
    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data, test_tessellation, max_jump_length=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            max_jump_length="not a number",
            privacy_budget=None,
        )

    # wrong input for bin_range_jump_length
    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data, test_tessellation, bin_range_jump_length=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            bin_range_jump_length="not a number",
            privacy_budget=None,
        )

    # wrong input for max_radius_of_gyration
    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data, test_tessellation, max_radius_of_gyration=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            max_radius_of_gyration="not a number",
            privacy_budget=None,
        )

    # wrong input for bin_range_radius_of_gyration
    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            bin_range_radius_of_gyration=-1,
            privacy_budget=None,
        )
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            bin_range_radius_of_gyration="not a number",
            privacy_budget=None,
        )

    # wrong input for user_privacy
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data, test_tessellation, user_privacy="not a bool", privacy_budget=None
        )

    # wrong input for evalu
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data, test_tessellation, evalu="not a bool", privacy_budget=None
        )

    # wrong input for disable_progress_bar
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            disable_progress_bar="not a bool",
            privacy_budget=None,
        )

    # wrong input for timewindows
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            timewindows="not a list",
            privacy_budget=None,
        )

    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            timewindows=["not an int", 2, 3],
            privacy_budget=None,
        )

    # wrong input for seed
    with pytest.raises(TypeError):
        DpMobilityReport(
            test_data,
            test_tessellation,
            privacy_budget=None,
            seed_sampling="not an int",
        )

    with pytest.raises(ValueError):
        DpMobilityReport(
            test_data, test_tessellation, privacy_budget=None, seed_sampling=-3
        )


def test_report_output(test_data, test_data_sequence, test_tessellation):
    report = DpMobilityReport(test_data, test_tessellation, privacy_budget=None).report
    assert isinstance(report, dict)
    assert list(report.keys()) == [
        const.DS_STATISTICS,
        const.MISSING_VALUES,
        const.TRIPS_OVER_TIME,
        const.TRIPS_PER_WEEKDAY,
        const.TRIPS_PER_HOUR,
        const.VISITS_PER_TILE,
        const.VISITS_PER_TILE_TIMEWINDOW,
        const.OD_FLOWS,
        const.TRAVEL_TIME,
        const.JUMP_LENGTH,
        const.TRIPS_PER_USER,
        const.USER_TIME_DELTA,
        const.RADIUS_OF_GYRATION,
        const.USER_TILE_COUNT,
        const.MOBILITY_ENTROPY,
    ]

    report = DpMobilityReport(test_data, test_tessellation, privacy_budget=1).report
    assert isinstance(report, dict)
    assert list(report.keys()) == [
        const.DS_STATISTICS,
        const.MISSING_VALUES,
        const.TRIPS_OVER_TIME,
        const.TRIPS_PER_WEEKDAY,
        const.TRIPS_PER_HOUR,
        const.VISITS_PER_TILE,
        const.VISITS_PER_TILE_TIMEWINDOW,
        const.OD_FLOWS,
        const.TRAVEL_TIME,
        const.JUMP_LENGTH,
        const.TRIPS_PER_USER,
        const.USER_TIME_DELTA,
        const.RADIUS_OF_GYRATION,
        const.USER_TILE_COUNT,
        const.MOBILITY_ENTROPY,
    ]

    # without tessellation
    report = DpMobilityReport(test_data, privacy_budget=None).report
    assert isinstance(report, dict)
    assert list(report.keys()) == [
        const.DS_STATISTICS,
        const.MISSING_VALUES,
        const.TRIPS_OVER_TIME,
        const.TRIPS_PER_WEEKDAY,
        const.TRIPS_PER_HOUR,
        const.TRIPS_PER_USER,
        const.USER_TIME_DELTA,
        const.RADIUS_OF_GYRATION,
    ]

    # without datetime
    report = DpMobilityReport(
        test_data_sequence,
        test_tessellation,
        privacy_budget=None,
    ).report

    assert list(report.keys()) == [
        const.DS_STATISTICS,
        const.MISSING_VALUES,
        const.VISITS_PER_TILE,
        const.OD_FLOWS,
        const.JUMP_LENGTH,
        const.TRIPS_PER_USER,
        const.RADIUS_OF_GYRATION,
        const.USER_TILE_COUNT,
        const.MOBILITY_ENTROPY,
    ]


def test_analysis_exclusion(test_data, test_tessellation):
    dpmr = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        analysis_exclusion=[const.OVERVIEW],
    )
    assert set(dpmr.analysis_exclusion) == {
        const.DS_STATISTICS,
        const.MISSING_VALUES,
        const.TRIPS_OVER_TIME,
        const.TRIPS_PER_WEEKDAY,
        const.TRIPS_PER_HOUR,
    }
    assert list(dpmr.report.keys()) == [
        const.VISITS_PER_TILE,
        const.VISITS_PER_TILE_TIMEWINDOW,
        const.OD_FLOWS,
        const.TRAVEL_TIME,
        const.JUMP_LENGTH,
        const.TRIPS_PER_USER,
        const.USER_TIME_DELTA,
        const.RADIUS_OF_GYRATION,
        const.USER_TILE_COUNT,
        const.MOBILITY_ENTROPY,
    ]

    dpmr = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        analysis_exclusion=[const.OVERVIEW, const.DS_STATISTICS],
    )
    assert set(dpmr.analysis_exclusion) == {
        const.DS_STATISTICS,
        const.MISSING_VALUES,
        const.TRIPS_OVER_TIME,
        const.TRIPS_PER_WEEKDAY,
        const.TRIPS_PER_HOUR,
    }
    assert list(dpmr.report.keys()) == [
        const.VISITS_PER_TILE,
        const.VISITS_PER_TILE_TIMEWINDOW,
        const.OD_FLOWS,
        const.TRAVEL_TIME,
        const.JUMP_LENGTH,
        const.TRIPS_PER_USER,
        const.USER_TIME_DELTA,
        const.RADIUS_OF_GYRATION,
        const.USER_TILE_COUNT,
        const.MOBILITY_ENTROPY,
    ]

    dpmr = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        analysis_exclusion=[const.JUMP_LENGTH, const.MOBILITY_ENTROPY],
    )
    assert set(dpmr.analysis_exclusion) == {
        const.JUMP_LENGTH,
        const.MOBILITY_ENTROPY,
    }

    assert list(dpmr.report.keys()) == [
        const.DS_STATISTICS,
        const.MISSING_VALUES,
        const.TRIPS_OVER_TIME,
        const.TRIPS_PER_WEEKDAY,
        const.TRIPS_PER_HOUR,
        const.VISITS_PER_TILE,
        const.VISITS_PER_TILE_TIMEWINDOW,
        const.OD_FLOWS,
        const.TRAVEL_TIME,
        const.TRIPS_PER_USER,
        const.USER_TIME_DELTA,
        const.RADIUS_OF_GYRATION,
        const.USER_TILE_COUNT,
    ]


def test_analysis_selection(test_data, test_tessellation):
    dpmr = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        analysis_selection=[const.OVERVIEW],
    )
    assert set(dpmr.analysis_exclusion) == {
        const.VISITS_PER_TILE,
        const.VISITS_PER_TILE_TIMEWINDOW,
        const.OD_FLOWS,
        const.TRAVEL_TIME,
        const.JUMP_LENGTH,
        const.TRIPS_PER_USER,
        const.USER_TIME_DELTA,
        const.RADIUS_OF_GYRATION,
        const.USER_TILE_COUNT,
        const.MOBILITY_ENTROPY,
    }
    assert list(dpmr.report.keys()) == [
        const.DS_STATISTICS,
        const.MISSING_VALUES,
        const.TRIPS_OVER_TIME,
        const.TRIPS_PER_WEEKDAY,
        const.TRIPS_PER_HOUR,
    ]

    dpmr = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        analysis_selection=[const.OVERVIEW, const.DS_STATISTICS],
    )
    assert set(dpmr.analysis_exclusion) == {
        const.VISITS_PER_TILE,
        const.VISITS_PER_TILE_TIMEWINDOW,
        const.OD_FLOWS,
        const.TRAVEL_TIME,
        const.JUMP_LENGTH,
        const.TRIPS_PER_USER,
        const.USER_TIME_DELTA,
        const.RADIUS_OF_GYRATION,
        const.USER_TILE_COUNT,
        const.MOBILITY_ENTROPY,
    }
    assert list(dpmr.report.keys()) == [
        const.DS_STATISTICS,
        const.MISSING_VALUES,
        const.TRIPS_OVER_TIME,
        const.TRIPS_PER_WEEKDAY,
        const.TRIPS_PER_HOUR,
    ]

    dpmr = DpMobilityReport(
        test_data,
        test_tessellation,
        privacy_budget=None,
        analysis_selection=[const.JUMP_LENGTH, const.MOBILITY_ENTROPY],
    )
    assert set(dpmr.analysis_exclusion) == {
        const.DS_STATISTICS,
        const.MISSING_VALUES,
        const.TRIPS_OVER_TIME,
        const.TRIPS_PER_WEEKDAY,
        const.TRIPS_PER_HOUR,
        const.VISITS_PER_TILE,
        const.VISITS_PER_TILE_TIMEWINDOW,
        const.OD_FLOWS,
        const.TRAVEL_TIME,
        const.TRIPS_PER_USER,
        const.USER_TIME_DELTA,
        const.RADIUS_OF_GYRATION,
        const.USER_TILE_COUNT,
    }

    assert list(dpmr.report.keys()) == [
        const.JUMP_LENGTH,
        const.MOBILITY_ENTROPY,
    ]


def test_to_html_file(test_data, test_data_sequence, test_tessellation, tmp_path):

    # DpMobilityReport( # type: ignore
    #     test_data, privacy_budget=None  # type: ignore
    # ).to_file("test1.html")  # type: ignore

    # DpMobilityReport(  # type: ignore
    #     test_data,  # type: ignore
    #     test_tessellation,  # type: ignore
    #     analysis_exclusion=[],  # type: ignore
    #     privacy_budget=1000,  # type: ignore
    #     max_travel_time=30,  # type: ignore
    # ).to_file(
    #     "test2.html"
    # )  # type: ignore

    file_name = tmp_path / "html/test_output1.html"
    file_name.parent.mkdir()
    DpMobilityReport(test_data, test_tessellation, privacy_budget=None).to_file(
        file_name
    )
    assert file_name.is_file()

    file_name = tmp_path / "html/test_output2.html"
    DpMobilityReport(test_data, test_tessellation, privacy_budget=0.1).to_file(
        file_name
    )
    assert file_name.is_file()

    file_name = tmp_path / "html/test_output3.html"
    DpMobilityReport(
        test_data,
        test_tessellation,
        analysis_exclusion=[
            const.RADIUS_OF_GYRATION,
            const.MOBILITY_ENTROPY,
            const.TRIPS_PER_USER,
        ],
        privacy_budget=None,
    ).to_file(file_name)
    assert file_name.is_file()

    # without tessellation
    file_name = tmp_path / "html/test_output4.html"
    DpMobilityReport(test_data).to_file(file_name)
    assert file_name.is_file()

    # without timestamps
    file_name = tmp_path / "html/test_output5.html"
    DpMobilityReport(test_data_sequence).to_file(file_name)
    assert file_name.is_file()

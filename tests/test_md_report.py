import geopandas as gpd
import pandas as pd
import pytest

from dp_mobility_report import constants as const
from dp_mobility_report import md_report


@pytest.fixture
def test_data():
    """Load a test dataset."""
    return pd.read_csv("tests/test_files/test_data.csv")


@pytest.fixture
def test_tessellation():
    """Load a test tessellation."""
    return gpd.read_file("tests/test_files/test_tessellation.geojson")


def test_MobilityDataReport(test_data, test_tessellation):
    """Test instance of MobilityDataReport is created properly with valid input and default values."""
    mob_report = md_report.MobilityDataReport(
        test_data, test_tessellation, privacy_budget=None
    )
    assert isinstance(mob_report, md_report.MobilityDataReport)

    mob_report = md_report.MobilityDataReport(
        test_data, test_tessellation, privacy_budget=0.1
    )
    assert isinstance(mob_report, md_report.MobilityDataReport)
    mob_report = md_report.MobilityDataReport(
        test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user=None
    )
    assert isinstance(mob_report, md_report.MobilityDataReport)

    # only one trip
    mob_report = md_report.MobilityDataReport(
        test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user=1
    )
    assert isinstance(mob_report, md_report.MobilityDataReport)

    # reasonable number of trips
    mob_report = md_report.MobilityDataReport(
        test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user=3
    )
    assert isinstance(mob_report, md_report.MobilityDataReport)

    # more trips than present
    mob_report = md_report.MobilityDataReport(
        test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user=1000
    )
    assert isinstance(mob_report, md_report.MobilityDataReport)


def test_wrong_input_params_MobilityDataReport(test_data, test_tessellation):
    """Test if wrong input parameters are caught correctly."""
    # wrong input for privacy_butget
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            "not a DataFrame", test_tessellation, privacy_budget=None
        )

    # wrong input for tessellation
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data, "not a GeoDataFrame", privacy_budget=None
        )

    # wrong input for privacy_butget
    with pytest.raises(ValueError):
        md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget=-1)
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data, test_tessellation, privacy_budget="not a number"
        )

    # wrong input for max_trips_per_user
    with pytest.raises(ValueError):
        md_report.MobilityDataReport(
            test_data, test_tessellation, max_trips_per_user=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            max_trips_per_user="not an int",
            privacy_budget=None,
        )
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data, test_tessellation, max_trips_per_user=3.1, privacy_budget=None
        )

    # wrong analysis selection
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            analysis_selection=const.ALL,
            privacy_budget=None,
        )
    with pytest.raises(ValueError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            analysis_selection=[const.OVERVIEW, "wrong input"],
            privacy_budget=None,
        )

    # wrong input for max_travel_time
    with pytest.raises(ValueError):
        md_report.MobilityDataReport(
            test_data, test_tessellation, max_travel_time=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            max_travel_time="not a number",
            privacy_budget=None,
        )

    # wrong input for bin_range_travel_time
    with pytest.raises(ValueError):
        md_report.MobilityDataReport(
            test_data, test_tessellation, bin_range_travel_time=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            bin_range_travel_time="not a number",
            privacy_budget=None,
        )

    # wrong input for max_jump_length
    with pytest.raises(ValueError):
        md_report.MobilityDataReport(
            test_data, test_tessellation, max_jump_length=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            max_jump_length="not a number",
            privacy_budget=None,
        )

    # wrong input for bin_range_jump_length
    with pytest.raises(ValueError):
        md_report.MobilityDataReport(
            test_data, test_tessellation, bin_range_jump_length=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            bin_range_jump_length="not a number",
            privacy_budget=None,
        )

    # wrong input for max_radius_of_gyration
    with pytest.raises(ValueError):
        md_report.MobilityDataReport(
            test_data, test_tessellation, max_radius_of_gyration=-1, privacy_budget=None
        )
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            max_radius_of_gyration="not a number",
            privacy_budget=None,
        )

    # wrong input for bin_range_radius_of_gyration
    with pytest.raises(ValueError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            bin_range_radius_of_gyration=-1,
            privacy_budget=None,
        )
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            bin_range_radius_of_gyration="not a number",
            privacy_budget=None,
        )

    # wrong input for user_privacy
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data, test_tessellation, user_privacy="not a bool", privacy_budget=None
        )

    # wrong input for evalu
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data, test_tessellation, evalu="not a bool", privacy_budget=None
        )

    # wrong input for disable_progress_bar
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            disable_progress_bar="not a bool",
            privacy_budget=None,
        )

    # wrong input for timewindows
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            timewindows="not a list",
            privacy_budget=None,
        )

    with pytest.raises(TypeError):
        md_report.MobilityDataReport(
            test_data,
            test_tessellation,
            timewindows=["not an int", 2, 3],
            privacy_budget=None,
        )


def test_report_output(test_data, test_tessellation):
    report = md_report.MobilityDataReport(
        test_data, test_tessellation, privacy_budget=None
    ).report
    assert isinstance(report, dict)
    assert list(report.keys()) == [
        "ds_statistics",
        "missing_values",
        "trips_over_time",
        "trips_per_weekday",
        "trips_per_hour",
        "counts_per_tile",
        "counts_per_tile_timewindow",
        "od_flows",
        "travel_time",
        "jump_length",
        "trips_per_user",
        "user_time_delta",
        "radius_of_gyration",
        "location_entropy",
        "user_tile_count",
        "mobility_entropy",
    ]


def test_to_html_file(test_data, test_tessellation, tmp_path):
    file_name = tmp_path / "html/test_output.html"
    file_name.parent.mkdir()
    md_report.MobilityDataReport(
        test_data, test_tessellation, privacy_budget=None
    ).to_file(file_name)
    assert file_name.is_file()

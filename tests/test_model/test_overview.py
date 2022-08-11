import geopandas as gpd
import pandas as pd
import pytest

from dp_mobility_report import constants as const
from dp_mobility_report.md_report import MobilityDataReport
from dp_mobility_report.model import overview


@pytest.fixture
def test_mdreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return MobilityDataReport(test_data, test_tessellation, privacy_budget=None)


def test_get_dataset_statistics(test_mdreport):
    """Correct dateset statistics without noise."""
    ds_stats = overview.get_dataset_statistics(test_mdreport, None).data
    assert ds_stats["n_records"] == 200
    assert ds_stats["n_trips"] == 100
    assert ds_stats["n_complete_trips"] == 100
    assert ds_stats["n_incomplete_trips"] == 0
    assert ds_stats["n_users"] == 20
    assert ds_stats["n_locations"] == 200


def test_get_missing_values(test_mdreport):
    """Correct missing values without noise."""
    missings = overview.get_missing_values(test_mdreport, None).data
    assert missings[const.UID] == 0
    assert missings[const.TID] == 0
    assert missings[const.DATETIME] == 0
    assert missings[const.LAT] == 0
    assert missings[const.LNG] == 0


def test_get_trips_over_time(test_mdreport):
    """Correct trips over time values without noise."""
    trips_over_time = overview.get_trips_over_time(test_mdreport, None)
    assert trips_over_time.datetime_precision == const.PREC_DATE
    expected_quartiles = pd.Series(
        data={
            "min": "2020-12-13",
            "max": "2020-12-19",
        }
    )
    assert trips_over_time.quartiles.astype(str).equals(expected_quartiles)
    assert trips_over_time.data.trip_count.tolist() == [15, 15, 11, 16, 22, 18, 3]


def test_get_trips_per_weekday(test_mdreport):
    """Correct trips per weekday values without noise."""
    trips_per_weekday = overview.get_trips_per_weekday(test_mdreport, None).data
    assert trips_per_weekday.sum() == 100
    assert trips_per_weekday["Monday"] == 15
    assert trips_per_weekday["Tuesday"] == 11
    assert trips_per_weekday["Wednesday"] == 16
    assert trips_per_weekday["Thursday"] == 22
    assert trips_per_weekday["Friday"] == 18
    assert trips_per_weekday["Saturday"] == 3
    assert trips_per_weekday["Sunday"] == 15

    # test that all days are created even if not present in data
    test_mdreport.df = test_mdreport.df[test_mdreport.df.day_name == "Monday"]
    trips_per_weekday = overview.get_trips_per_weekday(test_mdreport, None).data
    assert trips_per_weekday.index.tolist() == [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]


def test_get_trips_per_hour(test_mdreport):
    """Correct trips per hour values without noise."""
    trips_per_hour = overview.get_trips_per_hour(test_mdreport, None).data
    assert trips_per_hour[const.HOUR].min() == 0
    assert trips_per_hour[const.HOUR].max() == 23
    assert trips_per_hour[const.TIME_CATEGORY].unique().tolist() == [
        "weekday_end",
        "weekday_start",
        "weekend_end",
        "weekend_start",
    ]
    assert trips_per_hour.columns.tolist() == [const.HOUR, const.TIME_CATEGORY, "count"]
    assert len(trips_per_hour) == 73

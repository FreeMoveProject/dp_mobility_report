import geopandas as gpd
import pandas as pd
import pytest

from dp_mobility_report import DpMobilityReport
from dp_mobility_report import constants as const
from dp_mobility_report.model import overview


@pytest.fixture
def test_mreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return DpMobilityReport(test_data, test_tessellation, privacy_budget=None)


def test_get_dataset_statistics(test_mreport):
    """Correct dateset statistics without noise."""
    ds_stats = overview.get_dataset_statistics(test_mreport, None).data
    assert ds_stats["n_records"] == 200
    assert ds_stats["n_trips"] == 100
    assert ds_stats["n_complete_trips"] == 100
    assert ds_stats["n_incomplete_trips"] == 0
    assert ds_stats["n_users"] == 20
    assert ds_stats["n_locations"] == 200


def test_get_missing_values(test_mreport):
    """Correct missing values without noise."""
    missings = overview.get_missing_values(test_mreport, None).data
    assert missings[const.UID] == 0
    assert missings[const.TID] == 0
    assert missings[const.DATETIME] == 0
    assert missings[const.LAT] == 0
    assert missings[const.LNG] == 0


def test_get_trips_over_time(test_mreport):
    """Correct trips over time values without noise."""
    trips_over_time = overview.get_trips_over_time(test_mreport, None)
    assert trips_over_time.datetime_precision == const.PREC_DATE
    expected_quartiles = pd.Series(
        data={
            "min": "2020-12-13",
            "max": "2020-12-19",
        }
    )
    assert trips_over_time.quartiles.astype(str).equals(expected_quartiles)
    assert trips_over_time.data.trip_count.tolist() == [15, 15, 11, 16, 22, 18, 3]


def test_get_trips_per_weekday(test_mreport):
    """Correct trips per weekday values without noise."""
    trips_per_weekday = overview.get_trips_per_weekday(test_mreport, None).data
    assert trips_per_weekday.sum() == 100
    assert trips_per_weekday["Monday"] == 15
    assert trips_per_weekday["Tuesday"] == 11
    assert trips_per_weekday["Wednesday"] == 16
    assert trips_per_weekday["Thursday"] == 22
    assert trips_per_weekday["Friday"] == 18
    assert trips_per_weekday["Saturday"] == 3
    assert trips_per_weekday["Sunday"] == 15


def test_get_trips_per_hour(test_mreport):
    """Correct trips per hour values without noise."""
    trips_per_hour = overview.get_trips_per_hour(test_mreport, None).data
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

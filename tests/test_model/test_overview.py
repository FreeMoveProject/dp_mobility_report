import geopandas as gpd
import pandas as pd
import pytest

from dp_mobility_report import DpMobilityReport
from dp_mobility_report import constants as const
from dp_mobility_report.model import overview


@pytest.fixture
def test_dpmreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return DpMobilityReport(test_data, test_tessellation, privacy_budget=None)


def test_get_dataset_statistics(test_dpmreport):
    """Correct dateset statistics without noise."""
    ds_stats = overview.get_dataset_statistics(test_dpmreport, None).data
    assert ds_stats[const.N_RECORDS] == 200
    assert ds_stats[const.N_TRIPS] == 100
    assert ds_stats[const.N_COMPLETE_TRIPS] == 100
    assert ds_stats[const.N_INCOMPLETE_TRIPS] == 0
    assert ds_stats[const.N_USERS] == 20
    assert ds_stats[const.N_LOCATIONS] == 200


def test_get_missing_values(test_dpmreport):
    """Correct missing values without noise."""
    missings = overview.get_missing_values(test_dpmreport, None).data
    assert missings[const.UID] == 0
    assert missings[const.TID] == 0
    assert missings[const.DATETIME] == 0
    assert missings[const.LAT] == 0
    assert missings[const.LNG] == 0


def test_get_trips_over_time(test_dpmreport):
    """Correct trips over time values without noise."""
    trips_over_time = overview.get_trips_over_time(test_dpmreport, None)
    assert trips_over_time.datetime_precision == const.PREC_DATE
    expected_quartiles = pd.Series(
        data={
            "min": "2020-12-13",
            "max": "2020-12-19",
        }
    )
    assert trips_over_time.quartiles.astype(str).equals(expected_quartiles)
    assert trips_over_time.data.trip_count.tolist() == [15, 15, 11, 16, 22, 18, 3]


def test_get_trips_per_weekday(test_dpmreport):
    """Correct trips per weekday values without noise."""
    trips_per_weekday = overview.get_trips_per_weekday(test_dpmreport, None).data
    assert trips_per_weekday.sum() == 100
    assert trips_per_weekday["Monday"] == 15
    assert trips_per_weekday["Tuesday"] == 11
    assert trips_per_weekday["Wednesday"] == 16
    assert trips_per_weekday["Thursday"] == 22
    assert trips_per_weekday["Friday"] == 18
    assert trips_per_weekday["Saturday"] == 3
    assert trips_per_weekday["Sunday"] == 15

    # test that all days are created even if not present in data
    test_dpmreport._df = test_dpmreport.df[
        test_dpmreport.df[const.DAY_NAME] == "Monday"
    ]
    trips_per_weekday = overview.get_trips_per_weekday(test_dpmreport, None).data
    assert trips_per_weekday.index.tolist() == [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]


def test_get_trips_per_hour(test_dpmreport):
    """Correct trips per hour values without noise."""
    trips_per_hour = overview.get_trips_per_hour(test_dpmreport, None).data
    assert trips_per_hour[const.HOUR].min() == 0
    assert trips_per_hour[const.HOUR].max() == 23
    assert trips_per_hour[const.TIME_CATEGORY].unique().tolist() == [
        "weekday end",
        "weekday start",
        "weekend end",
        "weekend start",
    ]
    assert trips_per_hour.columns.tolist() == [const.HOUR, const.TIME_CATEGORY, "perc"]
    assert len(trips_per_hour) == 96

    # test that all hours are created even if not present in data
    test_dpmreport._df = test_dpmreport.df[test_dpmreport.df[const.HOUR] == 16]
    trips_per_hour = overview.get_trips_per_hour(test_dpmreport, None).data
    assert len(trips_per_hour) == 96
    assert trips_per_hour[const.HOUR].unique().tolist() == list(range(0, 24))

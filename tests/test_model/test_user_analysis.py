import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from dp_mobility_report.md_report import MobilityDataReport
from dp_mobility_report.model import user_analysis


@pytest.fixture
def test_mdreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return MobilityDataReport(test_data, test_tessellation, privacy_budget=None)


def test_get_trips_per_user(test_mdreport):
    trips_per_user = user_analysis.get_trips_per_user(test_mdreport, None)
    assert trips_per_user.data[0].tolist() == [1, 0, 4, 3, 5, 2, 3, 1, 1]
    assert trips_per_user.data[1].round().tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ]
    assert len(trips_per_user.data[0]) == 9
    assert all(np.diff(trips_per_user.data[1]) == 1)
    assert trips_per_user.quartiles.tolist() == [1.0, 3.75, 5.0, 6.25, 9.0]
    assert trips_per_user.n_outliers == 0


def test_get_user_time_delta(test_mdreport):
    user_time_delta = user_analysis.get_user_time_delta(test_mdreport, None)
    assert user_time_delta.quartiles[0].total_seconds() == -2613.0
    assert user_time_delta.quartiles[1].total_seconds() == 26630.0
    assert user_time_delta.quartiles[2].total_seconds() == 63126.5
    assert user_time_delta.quartiles[3].total_seconds() == 127326.25
    assert user_time_delta.quartiles[4].total_seconds() == 347720.0
    assert user_time_delta.n_outliers == 3


def test_get_radius_of_gyration(test_mdreport):
    rog = user_analysis.get_radius_of_gyration(test_mdreport, None)
    assert rog.data[0].tolist() == [0, 0, 0, 0, 2, 0, 3, 7, 5, 3]
    assert rog.data[1].round(3).tolist() == [
        0.0,
        0.457,
        0.914,
        1.370,
        1.827,
        2.284,
        2.741,
        3.197,
        3.654,
        4.111,
        4.568,
    ]
    assert len(rog.data[0]) == 10
    assert all(np.diff(rog.data[1]).round(3) == 0.457)
    assert rog.quartiles.round(3).tolist() == [2.200, 3.325, 3.529, 3.881, 4.568]
    assert rog.n_outliers is None


def test_get_location_entropy(test_mdreport):
    location_entropy = user_analysis.get_location_entropy(test_mdreport, None)
    assert location_entropy.data.round(4).tolist() == [4.0455, 4.0588, 4.0455, 4.0791]
    assert len(location_entropy.data) == 4


def test_get_user_tile_count(test_mdreport):
    user_tile_count = user_analysis.get_user_tile_count(test_mdreport, None)
    assert user_tile_count.data[0].tolist() == [2, 0, 18]
    assert user_tile_count.data[1].round(1).tolist() == [2, 3, 4]
    assert len(user_tile_count.data[0]) == 3
    assert all(np.diff(user_tile_count.data[1]) == 1)
    assert user_tile_count.quartiles.round().tolist() == [2, 4, 4, 4, 4]
    assert user_tile_count.n_outliers is None


def test_get_mobility_entropy(test_mdreport):
    mobility_entropy = user_analysis.get_mobility_entropy(test_mdreport, None)
    assert mobility_entropy.data[0].tolist() == [2, 18]
    assert mobility_entropy.data[1].round(2).tolist() == [
        0.8,
        0.9,
        1.0,
    ]
    assert len(mobility_entropy.data[0]) == 2
    assert all(np.diff(mobility_entropy.data[1]).round(1) == 0.1)
    assert mobility_entropy.quartiles.round(2).tolist() == [
        0.83,
        0.96,
        0.99,
        0.99,
        1.00,
    ]
    assert mobility_entropy.n_outliers == 0

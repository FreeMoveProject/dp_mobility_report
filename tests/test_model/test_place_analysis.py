import geopandas as gpd
import pandas as pd
import pytest

from dp_mobility_report import DpMobilityReport
from dp_mobility_report.model import place_analysis


@pytest.fixture
def test_dpmreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return DpMobilityReport(test_data, test_tessellation, privacy_budget=None)


def test_get_visits_per_tile(test_dpmreport):
    """Correct visits per tile values without noise."""
    visits_per_location = place_analysis.get_visits_per_tile(test_dpmreport, None)
    assert visits_per_location.quartiles[["min", "25%", "50%", "75%", "max"]].round(
        2
    ).tolist() == [49, 49.75, 50, 50, 50]
    assert visits_per_location.data.visits.tolist() == [50, 50, 50, 49]
    assert visits_per_location.n_outliers == 1


def test_get_visits_per_tile_timewindow(test_dpmreport):
    visits_timewindow = place_analysis.get_visits_per_tile_timewindow(
        test_dpmreport, None
    ).data
    assert len(visits_timewindow.columns) == 12
    assert len(visits_timewindow.index) == 4
    assert visits_timewindow.sum().sum() == 100
    assert visits_timewindow[("weekday", "1: 2-6")].round(2).tolist() == [0, 0, 3, 7]
    assert visits_timewindow[("weekend", "4: 14-18")].round(2).tolist() == [0, 0, 2, 0]
    assert visits_timewindow[("weekday", "3: 10-14")].round(2).tolist() == [0, 0, 7, 6]

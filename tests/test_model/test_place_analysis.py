import geopandas as gpd
import pandas as pd
import pytest

from dp_mobility_report.md_report import MobilityDataReport
from dp_mobility_report.model import place_analysis


@pytest.fixture
def test_mdreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return MobilityDataReport(test_data, test_tessellation, privacy_budget=None)


def test_get_visits_per_tile(test_mdreport):
    """Correct visits per tile values without noise."""
    visits_per_location = place_analysis.get_visits_per_tile(test_mdreport, None)
    assert visits_per_location.quartiles.tolist() == [49.0, 49.75, 50.0, 50.0, 50.0]
    assert visits_per_location.data.visit_count.tolist() == [50, 50, 50, 49]
    assert visits_per_location.data.visit_count.sum() == 199
    assert visits_per_location.n_outliers == 1


def test_get_visits_per_tile_timewindow(test_mdreport):
    visits_timewindow = place_analysis.get_visits_per_tile_timewindow(
        test_mdreport, None
    ).data
    assert len(visits_timewindow.columns) == 12
    assert len(visits_timewindow.index) == 4
    assert visits_timewindow.sum().sum() == 100
    assert visits_timewindow[("weekday", "1: 2-6")].tolist() == [0, 0, 3, 7]
    assert visits_timewindow[("weekend", "4: 14-18")].tolist() == [0, 0, 2, 0]
    assert visits_timewindow[("weekday", "3: 10-14")].tolist() == [0, 0, 7, 6]

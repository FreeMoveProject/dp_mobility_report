import pytest

import numpy as np
import pandas as pd
import geopandas as gpd
from dp_mobility_report import md_report
from dp_mobility_report.md_report import MobilityDataReport
from dp_mobility_report.model import od_analysis
from dp_mobility_report import constants as const


@pytest.fixture
def test_mdreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return MobilityDataReport(test_data, test_tessellation, privacy_budget=None)

@pytest.fixture
def test_od_shape():
    """Load a test od shape."""
    return pd.read_csv("tests/test_files/test_od_shape.csv", parse_dates=[const.DATETIME, const.DATETIME_END], 
        dtype={const.UID:str, const.TILE_ID:str, const.TILE_ID_END:str})


def test_get_od_shape(test_mdreport):
    od_shape = od_analysis.get_od_shape(test_mdreport.df, test_mdreport.tessellation)
    assert len(od_shape) == 99
    assert od_shape.columns.tolist() == [const.TID, const.TILE_ID, const.DATETIME, const.LAT, const.LNG, const.TILE_ID_END,
       const.DATETIME_END, const.LAT_END, const.LNG_END]
    assert all(od_shape[const.DATETIME] < od_shape[const.DATETIME_END])
    expected_tid_5_coords = [52.5281, 13.3416, 52.5082, 13.3459]
    actual_tid_5_coords = od_shape[od_shape[const.TID] == 5][[const.LAT, const.LNG, const.LAT_END, const.LNG_END]].values[0].tolist()
    assert actual_tid_5_coords == expected_tid_5_coords

def test_get_od_flows(test_od_shape, test_mdreport):
    od_flows = od_analysis.get_od_flows(test_od_shape, test_mdreport, None).data
    assert od_flows.columns.tolist() == ["origin", "destination", "flow"]
    assert len(od_flows) == 2
    assert od_flows.values[0].tolist() == ["1", "3", 50.0]
    assert od_flows.values[1].tolist() == ["2", "4", 49.0]

def test_get_intra_tile_flows():
    od_flows = pd.DataFrame(data=dict(origin=[1,2], destination=[3,4], flow = [10, 20]))
    assert od_analysis.get_intra_tile_flows(od_flows) == 0
    od_flows = pd.DataFrame(data=dict(origin=[1,2,2,3], destination=[3,4,2,3], flow = [10, 20, 10, 4]))
    assert od_analysis.get_intra_tile_flows(od_flows) == 14

def test_get_travel_time(test_od_shape, test_mdreport):
    travel_time = od_analysis.get_travel_time(test_od_shape, test_mdreport, None)
    assert travel_time.data[0].tolist() == [10, 10, 10, 8, 11, 8, 14, 9, 9, 10]
    assert travel_time.data[1].tolist() == [0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0, 120.0]
    assert len(travel_time.data[0]) == 10
    assert all(np.diff(travel_time.data[1]) == 12)
    assert travel_time.quartiles.tolist() == [2.0, 29.0, 60.0, 86.0, 120.0]
    assert travel_time.n_outliers == 0

    test_mdreport.max_travel_time=60
    test_mdreport.bin_range_travel_time=5
    travel_time = od_analysis.get_travel_time(test_od_shape, test_mdreport, None)
    assert travel_time.data[0].tolist() == [2, 8, 5, 3, 4, 4, 4, 2, 2, 6, 7, 3]
    assert travel_time.data[1].tolist() == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    assert len(travel_time.data[0]) == 12
    assert all(np.diff(travel_time.data[1]) == 5)
    assert travel_time.quartiles.tolist() == [2.0, 13.25, 29.0, 47.0, 60.0]
    assert travel_time.n_outliers == 49

def test_get_jump_length(test_od_shape, test_mdreport):
    jump_length = od_analysis.get_jump_length(test_od_shape, test_mdreport, None)
    assert jump_length.data[0].tolist() == [0, 7, 10, 19, 17, 19, 15, 9, 1, 2]
    assert jump_length.data[1].round(3).tolist() == [0.0, 0.874, 1.748, 2.622, 3.497, 4.371, 5.245, 6.119, 6.993, 7.867, 8.741]
    assert len(jump_length.data[0]) == 10
    assert all(np.diff(jump_length.data[1]).round(3) == 0.874)
    assert jump_length.quartiles.round(3).tolist() == [1.114, 3.039, 4.185, 5.350, 8.741]
    assert jump_length.n_outliers == 0

    test_mdreport.max_jump_length=4
    test_mdreport.bin_range_jump_length=0.5
    jump_length = od_analysis.get_jump_length(test_od_shape, test_mdreport, None)
    assert jump_length.data[0].tolist() == [0, 0, 4, 3, 9, 5, 15, 11]
    assert jump_length.data[1].tolist() == [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    assert len(jump_length.data[0]) == 8
    assert all(np.diff(jump_length.data[1]).round(1) == 0.5)
    assert jump_length.quartiles.round(3).tolist() == [1.114, 2.299, 3.017, 3.470, 3.997]
    assert jump_length.n_outliers == 52

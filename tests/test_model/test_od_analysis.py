#!/usr/bin/env python

import pytest

import numpy as np
import pandas as pd
import geopandas as gpd
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
    assert travel_time.data[0].tolist() == [12, 8, 10, 6, 13, 5, 11, 12]
    assert travel_time.data[1].tolist() == [ 2., 13., 24., 35., 46., 57., 68., 79., 90.]
    assert len(travel_time.data[0]) == 8
    assert all(np.diff(travel_time.data[1]) == 11)
    assert travel_time.quartiles.tolist() == [2.0, 20.0, 48.0, 72.0, 90.0]
    assert travel_time.n_outliers == 22

def test_get_jump_length(test_od_shape, test_mdreport):
    jump_length = od_analysis.get_jump_length(test_od_shape, test_mdreport, None)
    assert jump_length.data[0].tolist() == [11, 18, 25, 22, 17,  4,  2]
    assert jump_length.data[1].round().tolist() == [1114.0, 2203.0, 3293.0, 4383.0, 5472.0, 6562.0, 7652.0, 8741.0]
    assert len(jump_length.data[0]) == 7
    assert all(np.diff(jump_length.data[1].round(-1)) == 1090)
    assert jump_length.quartiles.round().tolist() == [1114.0, 3039.0, 4185.0, 5350.0, 8741.0]
    assert jump_length.n_outliers == 0

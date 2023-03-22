import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from dp_mobility_report import DpMobilityReport
from dp_mobility_report import constants as const
from dp_mobility_report.model import od_analysis


@pytest.fixture
def test_dpmreport():
    """Create a test report."""
    test_data = pd.read_csv("tests/test_files/test_data.csv")
    test_tessellation = gpd.read_file("tests/test_files/test_tessellation.geojson")
    return DpMobilityReport(test_data, test_tessellation, privacy_budget=None)


@pytest.fixture
def test_od_shape():
    """Load a test od shape."""
    return pd.read_csv(
        "tests/test_files/test_od_shape.csv",
        parse_dates=[const.DATETIME, const.DATETIME_END],
        dtype={const.UID: str, const.TILE_ID: str, const.TILE_ID_END: str},
    )


def test_get_od_shape(test_dpmreport):
    od_shape = od_analysis.get_od_shape(test_dpmreport.df)
    assert len(od_shape) == 100
    assert od_shape.columns.tolist() == [
        const.TID,
        const.TILE_ID,
        const.DATETIME,
        const.LAT,
        const.LNG,
        const.TILE_ID_END,
        const.DATETIME_END,
        const.LAT_END,
        const.LNG_END,
    ]
    assert all(od_shape[const.DATETIME] < od_shape[const.DATETIME_END])
    expected_tid_5_coords = [52.5281, 13.3416, 52.5082, 13.3459]
    actual_tid_5_coords = (
        od_shape[od_shape[const.TID] == 5][
            [const.LAT, const.LNG, const.LAT_END, const.LNG_END]
        ]
        .values[0]
        .tolist()
    )
    assert actual_tid_5_coords == expected_tid_5_coords


def test_get_od_flows(test_od_shape, test_dpmreport):
    od_flows = od_analysis.get_od_flows(test_od_shape, test_dpmreport, None).data
    assert od_flows.columns.tolist() == ["origin", "destination", "flow"]
    assert len(od_flows) == 2
    od_flows["flow"] = od_flows["flow"].round()
    assert od_flows.values[0].tolist() == ["1", "3", 50]
    assert od_flows.values[1].tolist() == ["2", "4", 49]


def test_get_intra_tile_flows():
    od_flows = pd.DataFrame(
        data={"origin": [1, 2], "destination": [3, 4], "flow": [10, 20]}
    )
    assert od_analysis.get_intra_tile_flows(od_flows) == 0
    od_flows = pd.DataFrame(
        data={
            "origin": [1, 2, 2, 3],
            "destination": [3, 4, 2, 3],
            "flow": [10, 20, 10, 4],
        }
    )
    assert od_analysis.get_intra_tile_flows(od_flows) == 14


def test_get_travel_time(test_od_shape, test_dpmreport):
    travel_time = od_analysis.get_travel_time(test_od_shape, test_dpmreport, None)
    assert travel_time.data[0].round().tolist() == [
        13.0,
        10.0,
        7.0,
        9.0,
        13.0,
        6.0,
        14.0,
        9.0,
        8.0,
        10.0,
    ]
    assert travel_time.data[1].tolist() == [1, 13, 25, 37, 48, 60, 72, 84, 96, 107, 119]
    assert len(travel_time.data[0]) == 10
    assert travel_time.quartiles.round().tolist() == [2.0, 29.0, 60.0, 86.0, 120.0]

    test_dpmreport.max_travel_time = 60
    test_dpmreport.bin_range_travel_time = 5
    travel_time = od_analysis.get_travel_time(test_od_shape, test_dpmreport, None)
    assert travel_time.data[0].round().tolist() == [
        2.0,
        8.0,
        6.0,
        2.0,
        4.0,
        4.0,
        4.0,
        2.0,
        3.0,
        6.0,
        7.0,
        2.0,
        49.0,
    ]
    assert travel_time.data[1].tolist() == [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        np.Inf,
    ]
    assert len(travel_time.data[0]) == 13
    assert travel_time.quartiles.round().tolist() == [2.0, 29.0, 60.0, 86.0, 120.0]


def test_get_jump_length(test_od_shape, test_dpmreport):
    jump_length = od_analysis.get_jump_length(test_od_shape, test_dpmreport, None)
    assert jump_length.data[0].round().tolist() == [7, 10, 15, 17, 16, 17, 9, 5, 2, 1]
    assert jump_length.data[1].round(3).tolist() == [
        1.113,
        1.876,
        2.639,
        3.402,
        4.165,
        4.928,
        5.69,
        6.453,
        7.216,
        7.979,
        8.742,
    ]
    assert len(jump_length.data[0]) == 10
    assert all(np.diff(jump_length.data[1]).round(3) == 0.763)
    assert jump_length.quartiles.round(3).tolist() == [
        1.114,
        3.039,
        4.185,
        5.350,
        8.741,
    ]

    test_dpmreport.max_jump_length = 4
    test_dpmreport.bin_range_jump_length = 0.5
    jump_length = od_analysis.get_jump_length(test_od_shape, test_dpmreport, None)
    assert jump_length.data[0].round().tolist() == [
        4.0,
        3.0,
        9.0,
        5.0,
        15.0,
        11.0,
        53.0,
    ]
    assert jump_length.data[1].tolist() == [1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, np.Inf]
    assert len(jump_length.data[0]) == 7
    assert jump_length.quartiles.round(3).tolist() == [1.114, 3.039, 4.185, 5.35, 8.741]

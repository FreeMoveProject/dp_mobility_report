#!/usr/bin/env python

import logging
import pytest

import  numpy as np
import pandas as pd
import geopandas as gpd
from dp_mobility_report.model import preprocessing
from dp_mobility_report import constants as const

LOGGER = logging.getLogger(__name__)

@pytest.fixture
def test_data():
    """Load a test dataset."""
    return pd.read_csv("tests/test_files/test_data.csv")

@pytest.fixture
def test_tessellation():
    """Load a test tessellation."""
    return gpd.read_file("tests/test_files/test_tessellation.geojson")

def test_preprocess_tessellation(test_tessellation):
    """Test preprocessing of tessellation: no changes if proper input is given, otherwise correct changes are applied."""
    # test if input and output are the same
    processed_tessellation = preprocessing.preprocess_tessellation(test_tessellation)
    assert test_tessellation.equals(processed_tessellation)
    
    # test is tile id is properly cast
    test_tessellation.tile_id = test_tessellation.tile_id.astype(int)
    assert (type(test_tessellation.tile_id[0]) == np.int64)
    processed_tessellation = preprocessing.preprocess_tessellation(test_tessellation)
    assert (type(processed_tessellation.tile_id[0]) == str)

    # test if tile name is created
    processed_tessellation = preprocessing.preprocess_tessellation(test_tessellation.drop("tile_name", axis = 1))
    assert processed_tessellation.tile_id.equals(processed_tessellation.tile_name)

    # test if crs is transformed if not EPSG:4326
    test_tessellation.to_crs(3857, inplace = True)
    processed_tessellation = preprocessing.preprocess_tessellation(test_tessellation)
    assert processed_tessellation.crs.equals(const.DEFAULT_CRS)
   

def test_raised_errors_in_preprocess_tessellation(test_tessellation):
    """Test if correct errors are raised with wrong tessellation input."""
    with pytest.raises(Exception):
        preprocessing.preprocess_tessellation(test_tessellation.drop("tile_id", axis = 1))
    with pytest.raises(Exception):
        preprocessing.preprocess_tessellation(test_tessellation.drop("geometry", axis = 1))


def test_preprocess_data(test_data, test_tessellation, caplog):
    """Test correct preprocessing of data."""
    processed_data = preprocessing.preprocess_data(test_data, test_tessellation, max_trips_per_user=5, user_privacy=True)
    assert processed_data.columns.tolist() == ['tile_id', 'tile_name', 'tid', 'id', 'uid', 'datetime', 'lat', 'lng', 'hour', 'is_weekend', 'point_type']
    
    # all waypoints are removed
    n_per_tid = processed_data.groupby('tid').count().id
    assert max(n_per_tid) <= 2
    
    # log output if tile id already present
    with caplog.at_level(logging.INFO):
        processed_data = preprocessing.preprocess_data(processed_data, test_tessellation, max_trips_per_user=5, user_privacy=True)
    assert "'tile_id' present in data. No new assignment of points to tessellation." in caplog.text

    
def test_data_validation(test_data):
    """Test if correct errors are raised with wrong data input."""
    with pytest.raises(ValueError):
        preprocessing._validate_columns(test_data.drop("uid", axis = 1))
    with pytest.raises(ValueError):
        preprocessing._validate_columns(test_data.drop("tid", axis = 1))
    with pytest.raises(ValueError):
        preprocessing._validate_columns(test_data.drop("lat", axis = 1))
    with pytest.raises(ValueError):
        preprocessing._validate_columns(test_data.drop("lng", axis = 1))
    with pytest.raises(ValueError):
        preprocessing._validate_columns(test_data.drop("datetime", axis = 1))

    with pytest.raises(TypeError):
        test_data_ = test_data.drop("lat", axis = 1)
        test_data_["lat"] = "not a float"
        preprocessing._validate_columns(test_data_)
    with pytest.raises(TypeError):
        test_data_ = test_data.drop("lng", axis = 1)
        test_data_["lng"] = "not a float"
        preprocessing._validate_columns(test_data_)
    with pytest.raises(TypeError):
        test_data_ = test_data.drop("datetime", axis = 1)
        test_data_["datetime"] = "not in datetime format"
        preprocessing._validate_columns(test_data_)
    


def test_assign_points_to_tessellation(test_data, test_tessellation):
    """All points are assigned to the correct tile."""
    assigned_df = preprocessing.assign_points_to_tessellation(test_data, test_tessellation)
    assert isinstance(assigned_df, pd.DataFrame)
    assert "tile_id" in assigned_df.columns
    assert assigned_df[(round(assigned_df.lat, 4) == 52.5210)  & (round(assigned_df.lng, 4) == 13.3540)].tile_id.iloc[0] == 1
    assert assigned_df[(round(assigned_df.lat, 4) == 52.4727)  & (round(assigned_df.lng, 4) ==13.4474)].tile_id.iloc[0] == 4

def test_sample_trips(test_data):
    # same length, if max_trips_per_user are max
    sampled_data = preprocessing.sample_trips(test_data, test_data.groupby("uid").nunique().tid.max(), True)
    assert len(sampled_data) == len(test_data)

    # no sampling if user_privacy is false
    sampled_data = preprocessing.sample_trips(test_data, 1, False)
    assert len(sampled_data) == len(test_data)

    sampled_data = preprocessing.sample_trips(test_data, 2, True)
    assert sampled_data.groupby("uid").nunique().tid.sum() == 39
    assert sampled_data.groupby("uid").nunique().tid.max() == 2

    # no duplicates drawn from sample
    sampled_data = preprocessing.sample_trips(test_data, 100, True)
    assert len(sampled_data[sampled_data.duplicated()]) == 0

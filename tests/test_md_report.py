#!/usr/bin/env python

import pytest
import pandas as pd
import geopandas as gpd
from dp_mobility_report import constants as const
from dp_mobility_report import md_report



@pytest.fixture
def test_data():
    """Load a test dataset."""
    return pd.read_csv("tests/test_files/test_data.csv")

@pytest.fixture
def test_tessellation():
    """Load a test tessellation."""
    return gpd.read_file("tests/test_files/test_tessellation.geojson")

def test_MobilityDataReport(test_data, test_tessellation):
    """Test instance of MobilityDataReport is created properly with valid input and default values."""
    mob_report = md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget=None)
    assert isinstance(mob_report, md_report.MobilityDataReport)

    mob_report = md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget=0.1)
    assert isinstance(mob_report, md_report.MobilityDataReport)
    mob_report = md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user = None)
    assert isinstance(mob_report, md_report.MobilityDataReport)
    
    # only one trip
    mob_report = md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user = 1)
    assert isinstance(mob_report, md_report.MobilityDataReport)

    # reasonable number of trips
    mob_report = md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user = 3)
    assert isinstance(mob_report, md_report.MobilityDataReport)

    # more trips than present
    mob_report = md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget=0.1, max_trips_per_user = 1000)
    assert isinstance(mob_report, md_report.MobilityDataReport)

def test_wrong_input_params_MobilityDataReport(test_data, test_tessellation):
    """Test if wrong input parameters are caught correctly."""
    # wrong input for privacy_butget
    with pytest.raises(TypeError):
        md_report.MobilityDataReport("not a DataFrame", test_tessellation, privacy_budget=None)

    # wrong input for tessellation
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(test_data, "not a GeoDataFrame", privacy_budget=None)

    # wrong input for privacy_butget
    with pytest.raises(ValueError):
        md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget=-1)
    with pytest.raises(TypeError):
        md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget = "not a number")

    # wrong input for max_trips_per_user
    mob_report = md_report.MobilityDataReport(test_data, test_tessellation, max_trips_per_user = -1, privacy_budget=None)
    assert mob_report.max_trips_per_user == test_data.groupby(const.UID).nunique()[const.TID].max()
    mob_report = md_report.MobilityDataReport(test_data, test_tessellation, max_trips_per_user = "not an int", privacy_budget=None)
    assert mob_report.max_trips_per_user == test_data.groupby(const.UID).nunique()[const.TID].max()
    mob_report = md_report.MobilityDataReport(test_data, test_tessellation, max_trips_per_user = 3.1, privacy_budget=None)
    assert mob_report.max_trips_per_user == test_data.groupby(const.UID).nunique()[const.TID].max()


def test_report_output(test_data, test_tessellation):
    report = md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget=None).report
    assert isinstance(report, dict)
    assert list(report.keys()) == ['ds_statistics', 'missing_values', 'trips_over_time', 'trips_per_weekday', 
        'trips_per_hour', 'counts_per_tile', 'counts_per_tile_timewindow', 'od_flows', 'travel_time', 'jump_length', 
        'traj_per_user', 'user_time_delta', 'radius_gyration', 'location_entropy', 'user_tile_count', 
        'mobility_entropy']


def test_to_html_file(test_data, test_tessellation, tmp_path):
    file_name = tmp_path / "html/test_output.html"
    file_name.parent.mkdir()
    md_report.MobilityDataReport(test_data, test_tessellation, privacy_budget=None).to_file(file_name)
    assert file_name.is_file()
import os

import geopandas as gpd
import pandas as pd

from dp_mobility_report import md_report

import config 

path_data = config.path_data
path_html_output = config.path_html_output

if not os.path.exists(path_html_output):
    os.makedirs(path_html_output)

# MADRID
df = pd.read_csv(os.path.join(path_data, "madrid.csv"))
tessellation = gpd.read_file(os.path.join(path_data, "madrid_tessellation.gpkg"))

report = md_report.MobilityDataReport(
    df,
    tessellation,
    privacy_budget=10,
    analysis_selection=["all"],
    max_trips_per_user=5,
    max_travel_time=90,
    bin_range_travel_time=5,
    max_jump_length=30,
    bin_range_jump_length=3,
    max_radius_of_gyration=18,
    bin_range_radius_of_gyration=1.5,
)
report.to_file(os.path.join(path_html_output, "madrid.html"), top_n_flows=100)

report = md_report.MobilityDataReport(
    df,
    tessellation,
    privacy_budget=None,
    analysis_selection=["all"],
    max_trips_per_user=None,
    max_travel_time=90,
    bin_range_travel_time=5,
    max_jump_length=30,
    bin_range_jump_length=3,
    max_radius_of_gyration=18,
    bin_range_radius_of_gyration=1.5,
)
report.to_file(os.path.join(path_html_output, "madrid_noPrivacy.html"), top_n_flows=300)
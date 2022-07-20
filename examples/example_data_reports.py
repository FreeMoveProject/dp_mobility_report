import sys, os
path = os.path.dirname(__file__)
sys.path.append(os.path.join(path, ".."))

import geopandas as gpd
import pandas as pd

from dp_mobility_report import md_report

path_data = (
    "examples/"
)
path_html_output = "examples/html"

if not os.path.exists(path_html_output):
    os.makedirs(path_html_output)

# GEOLIFE
df = pd.read_csv(os.path.join(path_data, "geolife_dpstar.dat-eps10.0-iteration0.csv"))
tessellation = gpd.read_file(os.path.join(path_data, "geolife_tessellation.gpkg"))
tessellation["tile_name"] = tessellation.tile_id

report = md_report.MobilityDataReport(
    df,
    tessellation,
    privacy_budget=None,
    evalu=True,
    user_privacy = False,
    max_trips_per_user=5,
    max_travel_time=90,
    bin_range_travel_time=5,
    max_jump_length=3000,
    bin_range_jump_length=30,
    max_radius_of_gyration=3000,
    bin_range_radius_of_gyration=15,
    timestamps = False
)
report.to_file(os.path.join(path_html_output, "geolife_dpstar_eps10.html"), top_n_flows=100)

#diff private
"""report = md_report.MobilityDataReport(
    df,
    tessellation,
    privacy_budget=11,
    analysis_selection=["overview", "place_analysis"],
    evalu=True,
    max_trips_per_user=5,
    max_travel_time=90,
    bin_range_travel_time=5,
    max_jump_length=30,
    bin_range_jump_length=3,
    max_radius_of_gyration=18,
    bin_range_radius_of_gyration=1.5,
)
report.to_file(os.path.join(path_html_output, "geolife_eps11.html"), top_n_flows=100)"""


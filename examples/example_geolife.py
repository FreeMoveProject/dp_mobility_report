import os

import config
import geopandas as gpd
import pandas as pd

from dp_mobility_report import DpMobilityReport
from dp_mobility_report import constants as const

# set paths to data and output (either with config file or hardcoded)
path_data = config.path_data
path_html_output = config.path_html_output

if not os.path.exists(path_html_output):
    os.makedirs(path_html_output)

# GEOLIFE
df = pd.read_csv(os.path.join(path_data, "geolife.csv"))
tessellation = gpd.read_file(os.path.join(path_data, "geolife_tessellation.gpkg"))
tessellation["tile_name"] = tessellation.tile_id


report = DpMobilityReport(
    df,
    tessellation,
    privacy_budget=None,
    max_trips_per_user=None,
    max_travel_time=90,
    bin_range_travel_time=5,
    max_jump_length=30,
    bin_range_jump_length=3,
    max_radius_of_gyration=18,
    bin_range_radius_of_gyration=1.5,
)
report.to_file(
    os.path.join(path_html_output, "geolife_noPrivacy.html"), top_n_flows=100
)


report = DpMobilityReport(
    df,
    tessellation,
    privacy_budget=50,
    analysis_selection=["overview", "place_analysis"],
    budget_split={const.VISITS_PER_TILE: 10},
    max_trips_per_user=5,
    max_travel_time=90,
    bin_range_travel_time=5,
    max_jump_length=30,
    bin_range_jump_length=3,
    max_radius_of_gyration=18,
    bin_range_radius_of_gyration=1.5,
)
report.to_file(os.path.join(path_html_output, "geolife.html"), top_n_flows=100)

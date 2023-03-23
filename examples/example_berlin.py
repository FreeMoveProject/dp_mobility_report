import os

import config
import geopandas as gpd
import pandas as pd

from dp_mobility_report import DpMobilityReport

# set paths to data and output (either with config file or hardcoded)
path_data = config.path_data
path_html_output = config.path_html_output

if not os.path.exists(path_html_output):
    os.makedirs(path_html_output)


# BERLIN
df = pd.read_csv(
    os.path.join(path_data, "berlin_w_tile_id.csv"), dtype={"tile_id": str}
)
tessellation = gpd.read_file(os.path.join(path_data, "berlin_tessellation.gpkg"))

report = DpMobilityReport(
    df,
    tessellation,
    privacy_budget=1,
    max_trips_per_user=5,
    max_travel_time=90,
    bin_range_travel_time=5,
    max_jump_length=30,
    bin_range_jump_length=3,
    max_radius_of_gyration=18,
    bin_range_radius_of_gyration=1.5,
    subtitle="Berlin Dataset - privacy",
)
report.to_file(os.path.join(path_html_output, "berlin.html"), top_n_flows=300)

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
    subtitle="Berlin Dataset - no privacy",
)
report.to_file(os.path.join(path_html_output, "berlin_noPrivacy.html"), top_n_flows=300)

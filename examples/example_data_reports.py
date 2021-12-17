import os
import pandas as pd
import geopandas as gpd

from dp_mobility_report import md_report

path_data = "data"
path_html_output = "html"

if not os.path.exists(path_html_output):
    os.makedirs(path_html_output)

# GEOLIFE 
df = pd.read_csv(os.path.join(path_data, "geolife_without_waypoints_df.csv"))
tessellation = gpd.read_file(os.path.join(path_data, "geolife_tessellation.gpkg"))
tessellation["tile_name"] = tessellation.tile_id

report = md_report.MobilityDataReport(
                df, 
                tessellation,
                privacy_budget = None, 
                analysis_selection=["all"],
                max_trips_per_user = None)
report.to_file(os.path.join(path_html_output,"geolife_no_privacy.html"))


# MADRID 
df = pd.read_csv(os.path.join(path_data,"crtm_madrid_df.csv"))
tessellation = gpd.read_file(os.path.join(path_data, "crtm_madrid_tessellation.gpkg"))

report = md_report.MobilityDataReport(
                df, 
                tessellation,
                privacy_budget = None, 
                analysis_selection=["all"],
                max_trips_per_user = None,
                top_x_flows = 200)
report.to_file(os.path.join(path_html_output,"madrid_no_privacy.html"))


# BERLIN 
df = pd.read_csv(os.path.join(path_data, "tapas_with_tileid_10perc_df.csv"), dtype = {'tile_id': float})
# fix wrong tile ids
df['tile_id'].fillna(-99, inplace = True)
df['tile_id'] = df.tile_id.astype(int).astype(str)
tessellation = gpd.read_file(os.path.join(path_data, "tapas_tessellation.gpkg"))


report = md_report.MobilityDataReport(
                df, 
                tessellation,
                privacy_budget = None, 
                analysis_selection=["all"],
                max_trips_per_user = None, 
                top_x_flows = 300)
report.to_file(os.path.join(path_html_output, "berlin_no_privacy.html"))


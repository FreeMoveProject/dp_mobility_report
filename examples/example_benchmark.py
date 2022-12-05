import os

import config
import geopandas as gpd
import pandas as pd

from dp_mobility_report.benchmark.benchmarkreport import BenchmarkReport

path_data = config.path_data
path_html_output = config.path_html_output

if not os.path.exists(path_html_output):
    os.makedirs(path_html_output)

# GEOLIFE
df = pd.read_csv(os.path.join(path_data, "geolife.csv"))
tessellation = gpd.read_file(os.path.join(path_data, "geolife_tessellation.gpkg"))
tessellation["tile_name"] = tessellation.tile_id

benchmarkreport = BenchmarkReport(
    df_base=df,
    tessellation=tessellation,
    privacy_budget_base=None,
    privacy_budget_alternative=15.0,
    max_trips_per_user_base=10,
    max_trips_per_user_alternative=10,
)

measures = benchmarkreport.similarity_measures
print(measures)

print(benchmarkreport.measure_selection)

# measures.to_file(os.path.join(path_html_output, "measures.html"))

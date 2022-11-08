import os

import geopandas as gpd
import pandas as pd

from dp_mobility_report.benchmark.benchmarkreport import BenchmarkReport

path_data = "examples/"
path_html_output = "examples/html"

if not os.path.exists(path_html_output):
    os.makedirs(path_html_output)

# GEOLIFE
df = pd.read_csv(os.path.join(path_data, "geolife.csv"))
tessellation = gpd.read_file(os.path.join(path_data, "geolife_tessellation.gpkg"))
tessellation["tile_name"] = tessellation.tile_id

benchmarkeval = BenchmarkReport(
    df_alternative=df,
    df_base=df,
    tessellation=tessellation,
    privacy_budget_alternative=15.0,
    privacy_budget_base=None,
    user_privacy_alternative=True,
    user_privacy_base=True,
    max_trips_per_user_alternative=10,
    max_trips_per_user_base=10,
)

measures = benchmarkeval.similarity_measures
print(measures)
# measures.to_file(os.path.join(path_html_output, "measures.html"))

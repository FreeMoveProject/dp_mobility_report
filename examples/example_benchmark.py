import os

import config
import geopandas as gpd
import pandas as pd

from dp_mobility_report import BenchmarkReport
from dp_mobility_report import constants as const

path_data = config.path_data
path_html_output = config.path_html_output

if not os.path.exists(path_html_output):
    os.makedirs(path_html_output)

df = pd.read_csv(
    os.path.join(path_data, "berlin_w_tile_id.csv"), dtype={"tile_id": str}
)
tessellation = gpd.read_file(os.path.join(path_data, "berlin_tessellation.gpkg"))

# benchmark
benchmarkreport = BenchmarkReport(
    df_base=df,
    tessellation=tessellation,
    privacy_budget_base=None,
    privacy_budget_alternative=1,
    max_trips_per_user_base=None,
    max_trips_per_user_alternative=5,
    analysis_exclusion=[
        const.USER_TIME_DELTA
    ],  # exclude analyses that you are not interested in, so save privacy budget
    # analysis_inclusion # can be used instead of anaylsis_exclusion
    budget_split_alternative={
        const.VISITS_PER_TILE: 50,
        const.VISITS_PER_TIME_TILE: 300,
        const.OD_FLOWS: 500,
    },  # custom split of the privacy budget: to allocate more budget for certain analyses
    subtitle="Berlin Benchmark report",  # provide a meaningful subtitle for your report readers
)


measures = benchmarkreport.similarity_measures

benchmarkreport.to_file(
    os.path.join(path_html_output, "berlin_benchmark.html"), top_n_flows=100
)

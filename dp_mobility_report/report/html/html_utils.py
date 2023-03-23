import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import geopandas as gpd
import jinja2
import numpy as np
from pandas import Series

from dp_mobility_report import constants as const

if TYPE_CHECKING:
    from dp_mobility_report import BenchmarkReport

# Initializing Jinja
package_loader = jinja2.PackageLoader(
    "dp_mobility_report", "report/html/html_templates"
)
jinja2_env = jinja2.Environment(
    lstrip_blocks=True, trim_blocks=True, loader=package_loader
)


def get_template(template_name: str) -> jinja2.Template:
    return jinja2_env.get_template(template_name)


def render_summary(summary: Series, target_type: Optional[type] = None) -> str:
    summary_list = [
        {"name": "Min.", "value": fmt(summary["min"], target_type)},
        {"name": "Max.", "value": fmt(summary["max"], target_type)},
    ]
    if "25%" in summary:
        summary_list.insert(
            1, {"name": "75%", "value": fmt(summary["75%"], target_type)}
        )
        summary_list.insert(
            1, {"name": "Median", "value": fmt(summary["50%"], target_type)}
        )
        summary_list.insert(
            1, {"name": "25%", "value": fmt(summary["25%"], target_type)}
        )

    if "mean" in summary:
        summary_list.insert(
            0, {"name": "Mean", "value": fmt(summary["mean"], target_type)}
        )

    template_table = jinja2_env.get_template("table.html")
    summary_html = template_table.render(rows=summary_list, align="right-align")
    return summary_html


def render_benchmark_summary(
    summary_base: Series,
    summary_alternative: Series,
    target_type: Optional[type] = None,
) -> str:
    summary_list = [
        {
            "name": "Min.",
            "value": (
                fmt(summary_base["min"], target_type),
                fmt(summary_alternative["min"], target_type),
            ),
        },
        {
            "name": "Max.",
            "value": (
                fmt(summary_base["max"], target_type),
                fmt(summary_alternative["max"], target_type),
            ),
        },
    ]
    if "25%" in summary_base:
        summary_list.insert(
            1,
            {
                "name": "75%",
                "value": (
                    fmt(summary_base["75%"], target_type),
                    fmt(summary_alternative["75%"], target_type),
                ),
            },
        )
        summary_list.insert(
            1,
            {
                "name": "Median",
                "value": (
                    fmt(summary_base["50%"], target_type),
                    fmt(summary_alternative["50%"], target_type),
                ),
            },
        )
        summary_list.insert(
            1,
            {
                "name": "25%",
                "value": (
                    fmt(summary_base["25%"], target_type),
                    fmt(summary_alternative["25%"], target_type),
                ),
            },
        )

    if "mean" in summary_base:
        summary_list.insert(
            0,
            {
                "name": "Mean",
                "value": (
                    fmt(summary_base["mean"], target_type),
                    fmt(summary_alternative["mean"], target_type),
                ),
            },
        )

    template_table = jinja2_env.get_template("table_benchmark.html")
    summary_html = template_table.render(rows=summary_list)
    return summary_html


def render_user_input_info(
    max_value: Optional[Union[int, float]], bin_size: Optional[Union[int, float]]
) -> str:
    return f"""User configuration for histogram chart: <br>
            maximum value: {max_value} <br>
            bin size: {bin_size}"""


def fmt(value: Any, target_type: Optional[type] = None) -> Any:
    if (value is None) or (
        isinstance(value, (float, np.floating, int, np.integer)) and math.isnan(value)
    ):
        return "-"
    if target_type and (value is not None):
        value = target_type(value)
    if isinstance(value, (float, np.floating)):
        if math.isinf(value) or np.isnan(value):
            return "not defined"
        value = round(value, 2)
        value = f"{value:.2f}"
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(
        value, bool
    ):
        value = f"{value:,}"
    return value


def fmt_moe(margin_of_error: Optional[float]) -> str:
    if (margin_of_error is None) or (margin_of_error == 0):
        return "0.0"
    else:
        margin_of_error = round(margin_of_error, 1)
        return f"{margin_of_error:,}"


def fmt_config(value: Union[dict, list]) -> str:
    def _join_tuple_string(strings_tuple: tuple) -> str:
        return ": ".join(strings_tuple)

    analysis_format = {
        const.DS_STATISTICS: "Dataset statistics",
        const.MISSING_VALUES: "Missing values",
        const.TRIPS_OVER_TIME: "Trips over time",
        const.TRIPS_PER_WEEKDAY: "Trips per weekday",
        const.TRIPS_PER_HOUR: "Trips per hour",
        const.VISITS_PER_TILE: "Visits per tile",
        const.VISITS_PER_TIME_TILE: "Visits per time tile",
        const.VISITS_PER_TILE_OUTLIERS: "Visits per tile outliers",
        const.VISITS_PER_TILE_RANKING: "Visits per tile ranking",
        const.VISITS_PER_TILE_QUARTILES: "Visits per tile quartiles",
        const.OD_FLOWS: "OD flows",
        const.OD_FLOWS_RANKING: "OD flows ranking",
        const.OD_FLOWS_QUARTILES: "Od flows quartiles",
        const.TRAVEL_TIME: "Travel time",
        const.TRAVEL_TIME_QUARTILES: "Travel time quartiles",
        const.JUMP_LENGTH: "Jump length",
        const.JUMP_LENGTH_QUARTILES: "Jump length quartiles",
        const.TRIPS_PER_USER: "Trips per user",
        const.TRIPS_PER_USER_QUARTILES: "Trips per user quartiles",
        const.USER_TIME_DELTA: "User time delta",
        const.USER_TIME_DELTA_QUARTILES: "User time delta quartiles",
        const.RADIUS_OF_GYRATION: "Radius of gyration",
        const.RADIUS_OF_GYRATION_QUARTILES: "Radius of gyration quartiles",
        const.USER_TILE_COUNT: "User tile count",
        const.USER_TILE_COUNT_QUARTILES: "User tile count quartiles",
        const.MOBILITY_ENTROPY: "Mobility entropy",
        const.MOBILITY_ENTROPY_QUARTILES: "Mobility entropy quartiles",
    }

    if value == {}:
        return "Evenly distributed"
    if value == []:
        return "None"
    if isinstance(value, dict):
        zipped = zip(
            [analysis_format[s] for s in value.keys()], [str(v) for v in value.values()]
        )
        return ", ".join(list(map(_join_tuple_string, zipped)))
    if isinstance(value, list):
        return ", ".join([analysis_format[s] for s in value])


def render_eps(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    else:
        return round(value, 4)


def get_centroids(tessellation: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    centroids = (
        tessellation.geometry.to_crs(3857).centroid.to_crs(4326).apply(lambda x: x.xy)
    )
    lngs = [c[0].pop() for c in centroids]
    lats = [c[1].pop() for c in centroids]
    return dict(zip(tessellation[const.TILE_ID], zip(lngs, lats)))


def all_available_measures(
    analysis_name: str, benchmarkreport: "BenchmarkReport"
) -> dict:
    available_measures: Dict[str, Union[str, List[str]]] = {}

    if analysis_name in benchmarkreport.smape:
        available_measures[const.SMAPE] = str(fmt(benchmarkreport.smape[analysis_name]))
    if analysis_name in benchmarkreport.jsd:
        available_measures[const.JSD] = str(fmt(benchmarkreport.jsd[analysis_name]))
    if analysis_name in benchmarkreport.kld:
        available_measures[const.KLD] = str(fmt(benchmarkreport.kld[analysis_name]))
    if analysis_name in benchmarkreport.emd:
        available_measures[const.EMD] = str(fmt(benchmarkreport.emd[analysis_name]))
    if analysis_name in benchmarkreport.kt:
        available_measures[const.KT] = [
            f"Top {topn_i}: {fmt(kt_i)}"
            for kt_i, topn_i in zip(
                benchmarkreport.kt[analysis_name], benchmarkreport.top_n_ranking
            )
        ]
    if analysis_name in benchmarkreport.top_n_cov:
        available_measures[const.TOP_N_COV] = [
            f"Top {topn_i}: {fmt(topcov_i * 100)}%"
            for topcov_i, topn_i in zip(
                benchmarkreport.top_n_cov[analysis_name], benchmarkreport.top_n_ranking
            )
        ]

    return available_measures

from typing import Any, Optional, Union, TYPE_CHECKING

import geopandas as gpd
import jinja2
import numpy as np
from pandas import Series
import math

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


def render_summary(summary: Series) -> str:
    summary_list = [
        {"name": "Min.", "value": fmt(summary["min"])},
        {"name": "Max.", "value": fmt(summary["max"])},
    ]
    if "25%" in summary:
        summary_list.insert(1, {"name": "75%", "value": fmt(summary["75%"])})
        summary_list.insert(1, {"name": "Median", "value": fmt(summary["50%"])})
        summary_list.insert(1, {"name": "25%", "value": fmt(summary["25%"])})

    if "mean" in summary:
        summary_list.insert(0, {"name": "Mean", "value": fmt(summary["mean"])})

    template_table = jinja2_env.get_template("table.html")
    summary_html = template_table.render(rows=summary_list)
    return summary_html


def render_benchmark_summary(
    summary_base: Series, summary_alternative: Series, target_type: Optional[type] = None
) -> str:
    summary_list = [
        {
            "name": "Min.",
            "value": (fmt(summary_base["min"], target_type), fmt(summary_alternative["min"], target_type)),
        },
        {
            "name": "Max.",
            "value": (fmt(summary_base["max"], target_type), fmt(summary_alternative["max"], target_type)),
        },
    ]
    if "25%" in summary_base:
        summary_list.insert(
            1,
            {
                "name": "75%",
                "value": (fmt(summary_base["75%"], target_type), fmt(summary_alternative["75%"], target_type)),
            },
        )
        summary_list.insert(
            1,
            {
                "name": "Median",
                "value": (fmt(summary_base["50%"], target_type), fmt(summary_alternative["50%"], target_type)),
            },
        )
        summary_list.insert(
            1,
            {
                "name": "25%",
                "value": (fmt(summary_base["25%"], target_type), fmt(summary_alternative["25%"], target_type)),
            },
        )

    if "mean" in summary_base:
        summary_list.insert(
            0,
            {
                "name": "Mean",
                "value": (fmt(summary_base["mean"], target_type), fmt(summary_alternative["mean"], target_type)),
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


def render_moe_info(margin_of_error: int) -> str:
    return """To provide privacy, quartile values are not necessarily the true values but, e.g., instead of the true maximum value the 
        second or third highest value is displayed.
        This is achieved by the so-called exponential mechanism, where a value is drawn based on probabilites defined by the privacy budget. 
        Generally, a value closer to the true value has a higher chance of being drawn."""
    # The true quartile values lie with a <b>95% chance within ± {margin_of_error} records</b> away from the true values.""" # TODO: margin_of_error reveals true record count?


def fmt(value: Any, target_type: Optional[type] = None) -> Any:
    if target_type and value:
        value = target_type(value)
    if isinstance(value, (float, np.floating)):
        if math.isinf(value) or np.isnan(value):
            return "not defined"
        value = round(value, 2)
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(
        value, bool
    ):
        value = f"{value:,}"
    return value

#TODO
def fmt_moe(margin_of_error: Optional[float]) -> float:
    if (margin_of_error is None) or (margin_of_error == 0):
        return 0
    return round(margin_of_error, 1)


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

def all_available_measures(analysis_name: str, benchmarkreport: "BenchmarkReport") -> dict:
    available_measures = {}

    if analysis_name in benchmarkreport.re.keys():
        available_measures[const.RE] = str(fmt(benchmarkreport.re[analysis_name]))
    if analysis_name in benchmarkreport.jsd.keys():
        available_measures[const.JSD] = str(fmt(benchmarkreport.jsd[analysis_name]))
    if analysis_name in benchmarkreport.kld.keys():
        available_measures[const.KLD] = str(fmt(benchmarkreport.kld[analysis_name]))
    if analysis_name in benchmarkreport.emd.keys():
        available_measures[const.EMD] = str(fmt(benchmarkreport.emd[analysis_name]))
    if analysis_name in benchmarkreport.smape.keys():
        available_measures[const.SMAPE] = str(fmt(benchmarkreport.smape[analysis_name]))
    if analysis_name in benchmarkreport.kt.keys():
        available_measures[const.KT] = str(fmt(benchmarkreport.kt[analysis_name]))
    if analysis_name in benchmarkreport.top_n_cov.keys():
        available_measures[const.TOP_N] = str(fmt(benchmarkreport.top_n_cov[analysis_name]))

    return available_measures
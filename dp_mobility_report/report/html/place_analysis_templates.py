import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import DfSection
from dp_mobility_report.report.html.html_utils import (
    fmt_moe,
    get_template,
    render_eps,
    render_summary,
)

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from dp_mobility_report.visualization import plot, v_utils


def render_place_analysis(
    dpmreport: "DpMobilityReport",
    tessellation: GeoDataFrame,
    temp_map_folder: Path,
    output_filename: str,
) -> str:
    THRESHOLD = 0.2  # 20%
    args: dict = {}
    report = dpmreport.report

    args[
        "privacy_info"
    ] = f"""Tiles below a certain threshold are grayed out: 
        Due to the applied noise, tiles with a low visit count are likely to contain a high percentage of noise. 
        For usability reasons, such unrealistic values are grayed out. 
        More specifically: The threshold is set so that values for tiles with a 5% chance (or higher) of deviating more than {round(THRESHOLD * 100)} percentage points from the estimated value are not shown."""
    args["output_filename"] = output_filename

    if const.VISITS_PER_TILE not in dpmreport.analysis_exclusion:
        args["visits_per_tile_eps"] = render_eps(
            report[const.VISITS_PER_TILE].privacy_budget
        )
        args["visits_per_tile_moe"] = fmt_moe(
            report[const.VISITS_PER_TILE].margin_of_error_laplace
        )

        args["points_outside_tessellation_info"] = render_points_outside_tess(
            report[const.VISITS_PER_TILE]
        )
        args["visits_per_tile_legend"] = render_visits_per_tile(
            report[const.VISITS_PER_TILE], tessellation, THRESHOLD, temp_map_folder
        )
        quartiles = report[const.VISITS_PER_TILE].quartiles.round()

        args["visits_per_tile_summary_table"] = render_summary(
            quartiles.astype(int),
            "Distribution of visits per tile",  # extrapolate visits from dp record count
        )
        args["visits_per_tile_cumsum_linechart"] = render_visits_per_tile_cumsum(
            report[const.VISITS_PER_TILE]
        )
        args["most_freq_tiles_ranking"] = render_most_freq_tiles_ranking(
            report[const.VISITS_PER_TILE],
        )

    if const.VISITS_PER_TILE_TIMEWINDOW not in dpmreport.analysis_exclusion:
        args["visits_per_tile_timewindow_eps"] = render_eps(
            report[const.VISITS_PER_TILE_TIMEWINDOW].privacy_budget
        )
        args["visits_per_tile_timewindow_moe"] = fmt_moe(
            report[const.VISITS_PER_TILE_TIMEWINDOW].margin_of_error_laplace
        )

        args["visits_per_tile_time_map"] = render_visits_per_tile_timewindow(
            report[const.VISITS_PER_TILE_TIMEWINDOW], tessellation, THRESHOLD
        )

    template_structure = get_template("place_analysis_segment.html")

    return template_structure.render(args)


def render_points_outside_tess(visits_per_tile: DfSection) -> str:
    return f"""{round(visits_per_tile.n_outliers)} ({round(visits_per_tile.n_outliers / (visits_per_tile.data["visits"].sum()
 + visits_per_tile.n_outliers) * 100, 2)}%) points are outside the given tessellation 
    (95% confidence interval ± {round(visits_per_tile.margin_of_error_laplace)})."""


def render_visits_per_tile(
    visits_per_tile: DfSection,
    tessellation: GeoDataFrame,
    threshold: float,
    temp_map_folder: Path,
) -> str:

    # merge count and tessellation
    counts_per_tile_gdf = pd.merge(
        tessellation,
        visits_per_tile.data[[const.TILE_ID, "visits"]],
        how="left",
        left_on=const.TILE_ID,
        right_on=const.TILE_ID,
    )

    # filter visit counts above error threshold
    moe_deviation = (
        visits_per_tile.margin_of_error_laplace / counts_per_tile_gdf["visits"]
    )

    counts_per_tile_gdf.loc[moe_deviation > threshold, "visits"] = None
    map, legend = plot.choropleth_map(
        counts_per_tile_gdf,
        "visits",
        scale_title="number of visits",
        aliases=["Tile ID", "Tile Name", "number of visits"],
    )

    map.save(os.path.join(temp_map_folder, "visits_per_tile_map.html"))

    legend_html = v_utils.fig_to_html(legend)
    plt.close()
    return legend_html


def render_visits_per_tile_cumsum(counts_per_tile: DfSection) -> str:
    df_cumsum = counts_per_tile.cumsum

    chart = plot.linechart(
        df_cumsum,
        "n",
        "cum_perc",
        "Number of tiles",
        "Cumulated sum of visits per tile",
        add_diagonal=True,
    )
    html = v_utils.fig_to_html(chart)
    plt.close()
    return html


def render_most_freq_tiles_ranking(visits_per_tile: DfSection, top_x: int = 10) -> str:
    topx_tiles = visits_per_tile.data.nlargest(top_x, "visits")
    topx_tiles["rank"] = list(range(1, len(topx_tiles) + 1))
    labels = (
        topx_tiles["rank"].astype(str)
        + ": "
        + topx_tiles[const.TILE_NAME]
        + "(Id: "
        + topx_tiles[const.TILE_ID]
        + ")"
    )

    ranking = plot.ranking(
        round(topx_tiles.visits),
        "number of visits per tile",
        y_labels=labels,
        margin_of_error=visits_per_tile.margin_of_error_laplace,
    )
    html_ranking = v_utils.fig_to_html(ranking)
    plt.close()
    return html_ranking


def render_visits_per_tile_timewindow(
    counts_per_tile_timewindow: DfSection, tessellation: GeoDataFrame, threshold: float
) -> str:
    data = counts_per_tile_timewindow.data
    moe_perc_per_tile_timewindow = (
        counts_per_tile_timewindow.margin_of_error_laplace / data
    )

    if data is None:
        return None

    data[moe_perc_per_tile_timewindow > threshold] = None

    output_html = ""
    if "weekday" in data.columns:
        output_html += "<h4>Weekday</h4>"
        output_html += _create_timewindow_segment(data.loc[:, "weekday"], tessellation)

    if "weekend" in data.columns:
        output_html += "<h4>Weekend</h4>"
        output_html += _create_timewindow_segment(data.loc[:, "weekend"], tessellation)
    plt.close()
    return output_html


def _create_timewindow_segment(df: pd.DataFrame, tessellation: GeoDataFrame) -> str:
    visits_choropleth = plot.multi_choropleth_map(df, tessellation)

    tile_means = df.mean(axis=1)
    dev_from_avg = df.div(tile_means, axis=0)
    deviation_choropleth = plot.multi_choropleth_map(
        dev_from_avg, tessellation, diverging_cmap=True
    )
    return f"""<h4>Number of visits</h4>
        {v_utils.fig_to_html_as_png(visits_choropleth)}
        <h4>Deviation from tile average</h4>
        <div><p>The average of each cell 
        over all time windows equals 1 (100% of average traffic). 
        A value of < 1 (> 1) means that a tile is visited less (more) frequently in this time window than it is on average.</p></div>
        {v_utils.fig_to_html_as_png(deviation_choropleth)}"""  # svg might get too large

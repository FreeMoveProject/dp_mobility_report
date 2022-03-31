from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import Section
from dp_mobility_report.report.html.html_utils import fmt, get_template, render_summary
from dp_mobility_report.visualization import plot, v_utils


def render_place_analysis(report: dict, tessellation: GeoDataFrame) -> str:
    points_outside_tessellation_info = ""
    privacy_info = "Unrealistic values: Tiles with a 5% chance of deviating more than 10 percentage points from the estimated value are grayed out in the map view."
    counts_per_tile_map = ""
    counts_per_tile_legend = ""
    counts_per_tile_summary_table = ""
    counts_per_tile_cumsum_linechart = ""
    most_freq_tiles_ranking = ""
    counts_per_tile_time_map = ""

    if (const.COUNTS_PER_TILE in report) and (
        report[const.COUNTS_PER_TILE].data is not None
    ):
        points_outside_tessellation_info = render_points_outside_tess(
            report[const.COUNTS_PER_TILE]
        )
        counts_per_tile_map, counts_per_tile_legend = render_counts_per_tile(
            report[const.COUNTS_PER_TILE], tessellation
        )
        counts_per_tile_summary_table = render_summary(
            report[const.COUNTS_PER_TILE].quartiles, "Distribution of visits per tile"
        )
        counts_per_tile_cumsum_linechart = render_counts_per_tile_cumsum(
            report[const.COUNTS_PER_TILE]
        )
        most_freq_tiles_ranking = render_most_freq_tiles_ranking(
            report[const.COUNTS_PER_TILE]
        )

    if (const.COUNTS_PER_TILE_TIMEWINDOW in report) and (
        report[const.COUNTS_PER_TILE_TIMEWINDOW] is not None
    ):
        counts_per_tile_time_map = render_counts_per_tile_timewindow(
            report[const.COUNTS_PER_TILE_TIMEWINDOW], tessellation
        )

    template_structure = get_template("place_analysis_segment.html")

    return template_structure.render(
        points_outside_tessellation_info=points_outside_tessellation_info,
        privacy_info=privacy_info,
        counts_per_tile_map=counts_per_tile_map,
        counts_per_tile_legend=counts_per_tile_legend,
        counts_per_tile_summary_table=counts_per_tile_summary_table,
        counts_per_tile_cumsum_linechart=counts_per_tile_cumsum_linechart,
        most_freq_tiles_ranking=most_freq_tiles_ranking,
        counts_per_tile_time_map=counts_per_tile_time_map,
    )


def render_points_outside_tess(counts_per_tile: Section) -> str:
    return f"{counts_per_tile.n_outliers} points are outside the given tessellation (95% confidence interval ± {round(counts_per_tile.margin_of_error_laplace)})."


def render_counts_per_tile(
    counts_per_tile: Section, tessellation: GeoDataFrame, threshold: float = 0.1
) -> Tuple[str, str]:
    data = counts_per_tile.data
    # merge count and tessellation
    counts_per_tile_gdf = pd.merge(
        tessellation,
        data[[const.TILE_ID, "visit_count"]],
        how="left",
        left_on=const.TILE_ID,
        right_on=const.TILE_ID,
    )

    # filter visit counts above error threshold
    moe_deviation = (
        counts_per_tile.margin_of_error_laplace / counts_per_tile_gdf["visit_count"]
    )
    counts_per_tile_gdf.loc[moe_deviation > threshold, "visit_count"] = None
    map, legend = plot.choropleth_map(
        counts_per_tile_gdf, "visit_count", scale_title="Number of visits"
    )
    html = map.get_root().render()
    legend_html = v_utils.fig_to_html(legend)
    plt.close()
    return html, legend_html


def render_counts_per_tile_cumsum(counts_per_tile: Section) -> str:
    df_cumsum = counts_per_tile.cumsum_simulations

    chart = plot.linechart(
        df_cumsum,
        "n",
        "cum_perc",
        "Number of tiles",
        "Cumulated sum of visits per tile",
        simulations=df_cumsum.columns[2:52],
        add_diagonal=True,
    )
    html = v_utils.fig_to_html(chart)
    plt.close()
    return html


def render_most_freq_tiles_ranking(counts_per_tile: Section, top_x: int = 10) -> str:
    topx_tiles = counts_per_tile.data.nlargest(top_x, "visit_count")
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
        topx_tiles.visit_count,
        "Number of visits per tile",
        y_labels=labels,
        margin_of_error=counts_per_tile.margin_of_error_laplace,
    )
    html_ranking = v_utils.fig_to_html(ranking)
    plt.close()
    return html_ranking


def render_counts_per_tile_timewindow(
    counts_per_tile_timewindow: Section, tessellation: GeoDataFrame, threshold=0.1
) -> str:
    data = counts_per_tile_timewindow.data
    if data is None:
        return None

    moe_counts_per_tile_timewindow = (
        counts_per_tile_timewindow.margin_of_error_laplace / data
    )
    data[moe_counts_per_tile_timewindow > threshold] = None

    output_html = ""
    if "weekday" in data.columns:
        absolute_weekday = plot.multi_choropleth_map(
            data.loc[:, "weekday"], tessellation
        )

        tile_means = data.loc[:, "weekday"].mean(axis=1)
        dev_from_avg = data.loc[:, "weekday"].div(tile_means, axis=0)
        relative_weekday = plot.multi_choropleth_map(dev_from_avg, tessellation)
        output_html += (
            "<h4>Weekday: absolute count</h4>"
            + v_utils.fig_to_html_as_png(absolute_weekday)  # svg might get too large
            + "<h4>Weekday: deviation from average</h4>"
            + v_utils.fig_to_html_as_png(relative_weekday)  # svg might get too large
        )

    if "weekend" in data.columns:
        absolute_weekend = plot.multi_choropleth_map(
            data.loc[:, "weekend"], tessellation
        )

        tile_means = data.loc[:, "weekend"].mean(axis=1)
        dev_from_avg = data.loc[:, "weekend"].div(tile_means, axis=0)
        relative_weekend = plot.multi_choropleth_map(dev_from_avg, tessellation)
        output_html += (
            "<h4>Weekend: absolute count</h4>"
            + v_utils.fig_to_html_as_png(absolute_weekend)
            + "<h4>Weekend: deviation from average</h4>"
            + v_utils.fig_to_html_as_png(relative_weekend)
        )
    plt.close()
    return output_html

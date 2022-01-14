from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from dp_mobility_report import constants as const
from dp_mobility_report.report.html.html_utils import get_template, render_summary
from dp_mobility_report.visualization import plot, v_utils


def render_place_analysis(report: dict, tessellation: GeoDataFrame) -> str:
    points_outside_tessellation_info = ""
    counts_per_tile_map = ""
    counts_per_tile_legend = ""
    counts_per_tile_summary_table = ""
    counts_per_tile_cumsum_linechart = ""
    most_freq_tiles_ranking = ""
    counts_per_tile_time_map = ""

    if const.COUNTS_PER_TILE in report:
        points_outside_tessellation_info = render_points_outside_tess(
            report[const.COUNTS_PER_TILE].n_outliers
        )

    if const.COUNTS_PER_TILE in report:
        counts_per_tile_map, counts_per_tile_legend = render_counts_per_tile(
            report[const.COUNTS_PER_TILE].data, tessellation
        )
        counts_per_tile_summary_table = render_summary(
            report[const.COUNTS_PER_TILE].quartiles
        )
        counts_per_tile_cumsum_linechart = render_counts_per_tile_cumsum(
            report[const.COUNTS_PER_TILE].data
        )
        most_freq_tiles_ranking = render_most_freq_tiles_ranking(
            report[const.COUNTS_PER_TILE].data
        )

    if const.COUNTS_PER_TILE_TIMEWINDOW in report:
        counts_per_tile_time_map = render_counts_per_tile_timewindow(
            report[const.COUNTS_PER_TILE_TIMEWINDOW].data, tessellation
        )

    template_structure = get_template("place_analysis_segment.html")

    return template_structure.render(
        points_outside_tessellation_info=points_outside_tessellation_info,
        counts_per_tile_map=counts_per_tile_map,
        counts_per_tile_legend=counts_per_tile_legend,
        counts_per_tile_summary_table=counts_per_tile_summary_table,
        counts_per_tile_cumsum_linechart=counts_per_tile_cumsum_linechart,
        most_freq_tiles_ranking=most_freq_tiles_ranking,
        counts_per_tile_time_map=counts_per_tile_time_map,
    )


def render_points_outside_tess(points_outside_tessellation: int) -> str:
    return "Points outside the given tessellation: " + str(points_outside_tessellation)


def render_counts_per_tile(
    counts_per_tile: pd.DataFrame, tessellation: GeoDataFrame
) -> Tuple[str, str]:
    # merge count and tessellation
    counts_per_tile_gdf = pd.merge(
        tessellation,
        counts_per_tile[[const.TILE_ID, "visit_count"]],
        how="left",
        left_on=const.TILE_ID,
        right_on=const.TILE_ID,
    )
    map, legend = plot.choropleth_map(
        counts_per_tile_gdf, "visit_count", scale_title="Number of visits"
    )
    html = map.get_root().render()
    legend_html = v_utils.fig_to_html(legend)
    plt.close()
    return html, legend_html


def render_counts_per_tile_cumsum(counts_per_tile: pd.DataFrame) -> str:
    df_cumsum = pd.DataFrame()
    df_cumsum["cum_perc"] = round(
        counts_per_tile.visit_count.sort_values(ascending=False).cumsum()
        / sum(counts_per_tile.visit_count),
        2,
    )
    df_cumsum["n"] = np.arange(1, len(counts_per_tile) + 1)
    df_cumsum.reset_index(drop=True, inplace=True)
    chart = plot.linechart(
        df_cumsum,
        "n",
        "cum_perc",
        "Number of tiles",
        "Cumulated sum of counts per tile",
        add_diagonal=True,
    )
    html = v_utils.fig_to_html(chart)
    plt.close()
    return html


def render_most_freq_tiles_ranking(
    counts_per_tile: pd.DataFrame, top_n: int = 10
) -> str:
    topx_tiles = counts_per_tile.nlargest(top_n, "visit_count")
    topx_tiles["rank"] = list(range(1, len(topx_tiles) + 1))

    topx_tiles_list = []
    for _, row in topx_tiles.iterrows():
        topx_tiles_list.append(
            {
                "name": row["rank"],
                "value": str(row[const.TILE_NAME])
                + " (Id: "
                + str(row[const.TILE_ID])
                + "): "
                + str(row["visit_count"]),
            }
        )
    template_table = get_template("table.html")
    tile_ranking_html = template_table.render(
        name="Ranking most frequently visited tiles", rows=topx_tiles_list
    )

    return tile_ranking_html


def render_counts_per_tile_timewindow(
    counts_per_tile_timewindow: pd.DataFrame, tessellation: GeoDataFrame
) -> str:
    output_html = ""
    if "weekday" in counts_per_tile_timewindow.columns:
        absolute_weekday = plot.multi_choropleth_map(
            counts_per_tile_timewindow.loc[:, "weekday"], tessellation
        )

        tile_means = counts_per_tile_timewindow.loc[:, "weekday"].mean(axis=1)
        dev_from_avg = counts_per_tile_timewindow.loc[:, "weekday"].div(
            tile_means, axis=0
        )
        relative_weekday = plot.multi_choropleth_map(dev_from_avg, tessellation)
        output_html += (
            "<h4>Weekday: absolute count</h4>"
            + v_utils.fig_to_html(absolute_weekday)
            + "<h4>Weekday: deviation from average</h4>"
            + v_utils.fig_to_html(relative_weekday)
        )

    if "weekend" in counts_per_tile_timewindow.columns:
        absolute_weekend = plot.multi_choropleth_map(
            counts_per_tile_timewindow.loc[:, "weekend"], tessellation
        )

        tile_means = counts_per_tile_timewindow.loc[:, "weekend"].mean(axis=1)
        dev_from_avg = counts_per_tile_timewindow.loc[:, "weekend"].div(
            tile_means, axis=0
        )
        relative_weekend = plot.multi_choropleth_map(dev_from_avg, tessellation)
        output_html += (
            "<h4>Weekend: absolute count</h4>"
            + v_utils.fig_to_html(absolute_weekend)
            + "<h4>Weekend: deviation from average</h4>"
            + v_utils.fig_to_html(relative_weekend)
        )
    plt.close()
    return output_html

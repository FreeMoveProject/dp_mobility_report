import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import Section
from dp_mobility_report.report.html.html_utils import get_template, render_summary
from dp_mobility_report.visualization import plot, v_utils


def render_place_analysis(
    report: dict,
    tessellation: GeoDataFrame,
    temp_map_folder: Path,
    output_filename: str,
) -> str:
    THRESHOLD = 0.2  # 20%
    points_outside_tessellation_info = ""
    privacy_info = f"""Tiles below a certain threshold are grayed out: 
        Due to the applied noise, tiles with a low visit count are likely to contain a high percentage of noise. 
        For usability reasons, such unrealistic values are grayed out. 
        More specifically: The threshold is set so that values for tiles with a 5% chance of deviating more than {round(THRESHOLD * 100)} percentage points from the estimated value are not shown."""
    visits_per_tile_legend = ""
    visits_per_tile_summary_table = ""
    visits_per_tile_cumsum_linechart = ""
    most_freq_tiles_ranking = ""
    counts_per_tile_time_map = ""

    if (const.VISITS_PER_TILE in report) and (
        report[const.VISITS_PER_TILE].data is not None
    ):
        points_outside_tessellation_info = render_points_outside_tess(
            report[const.VISITS_PER_TILE]
        )
        visits_per_tile_legend = render_visits_per_tile(
            report[const.VISITS_PER_TILE], tessellation, THRESHOLD, temp_map_folder
        )
        quartiles = report[const.VISITS_PER_TILE].quartiles.round()

        visits_per_tile_summary_table = render_summary(
            quartiles.astype(int),
            "Distribution of visits per tile",  # extrapolate visits from dp record count
        )
        visits_per_tile_cumsum_linechart = render_counts_per_tile_cumsum(
            report[const.VISITS_PER_TILE]
        )
        most_freq_tiles_ranking = render_most_freq_tiles_ranking(
            report[const.VISITS_PER_TILE],
        )

    if (const.VISITS_PER_TILE_TIMEWINDOW in report) and (
        report[const.VISITS_PER_TILE_TIMEWINDOW] is not None
    ):
        counts_per_tile_time_map = render_counts_per_tile_timewindow(
            report[const.VISITS_PER_TILE_TIMEWINDOW], tessellation, THRESHOLD
        )

    template_structure = get_template("place_analysis_segment.html")

    return template_structure.render(
        points_outside_tessellation_info=points_outside_tessellation_info,
        privacy_info=privacy_info,
        output_filename=output_filename,
        visits_per_tile_legend=visits_per_tile_legend,
        counts_per_tile_summary_table=visits_per_tile_summary_table,
        counts_per_tile_cumsum_linechart=visits_per_tile_cumsum_linechart,
        most_freq_tiles_ranking=most_freq_tiles_ranking,
        counts_per_tile_time_map=counts_per_tile_time_map,
    )


def render_points_outside_tess(visits_per_tile: Section) -> str:
    return f"""{round(visits_per_tile.n_outliers)} ({round(visits_per_tile.n_outliers, 2)}%) of points are outside the given tessellation 
    (95% confidence interval ± {round(visits_per_tile.margin_of_error_laplace)})."""


def render_visits_per_tile(
    perc_per_tile: Section,
    tessellation: GeoDataFrame,
    threshold: float,
    temp_map_folder: Path,
) -> str:

    # merge count and tessellation
    counts_per_tile_gdf = pd.merge(
        tessellation,
        perc_per_tile.data[[const.TILE_ID, "visits"]],
        how="left",
        left_on=const.TILE_ID,
        right_on=const.TILE_ID,
    )

    # filter visit counts above error threshold
    moe_deviation = (
        perc_per_tile.margin_of_error_laplace / counts_per_tile_gdf["visits"]
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


def render_counts_per_tile_cumsum(counts_per_tile: Section) -> str:
    df_cumsum = counts_per_tile.cumsum_simulations

    chart = plot.linechart(
        df_cumsum,
        "n",
        "cum_perc",
        "Number of tiles",
        "Cumulated sum of visits per tile",
        # simulations=df_cumsum.columns[2:52],
        add_diagonal=True,
    )
    html = v_utils.fig_to_html(chart)
    plt.close()
    return html


def render_most_freq_tiles_ranking(perc_per_tile: Section, top_x: int = 10) -> str:
    topx_tiles = perc_per_tile.data.nlargest(top_x, "visits")
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
        margin_of_error=perc_per_tile.margin_of_error_laplace,
    )
    html_ranking = v_utils.fig_to_html(ranking)
    plt.close()
    return html_ranking


def render_counts_per_tile_timewindow(
    counts_per_tile_timewindow: Section, tessellation: GeoDataFrame, threshold: int
) -> str:
    data = counts_per_tile_timewindow.data
    moe_counts_per_tile_timewindow = (
        counts_per_tile_timewindow.margin_of_error_laplace / data
    )

    if data is None:
        return None

    data[moe_counts_per_tile_timewindow > threshold] = None

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

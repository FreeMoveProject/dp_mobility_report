import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from dp_mobility_report import BenchmarkReport

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import DfSection
from dp_mobility_report.report.html.html_utils import (
    fmt_moe,
    get_template,
    render_eps,
    render_summary,
    render_benchmark_summary,
    fmt,
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

def render_benchmark_place_analysis(
    report_base: dict,
    report_alternative: dict,
    tessellation: GeoDataFrame,
    temp_map_folder: Path,
    output_filename: str,
    analysis_exclusion: list,
    benchmark:"BenchmarkReport",

) -> str:
    THRESHOLD = 0.2  # 20%
    points_outside_tessellation_info = ""
    privacy_info = f"""Tiles below a certain threshold are grayed out: 
        Due to the applied noise, tiles with a low visit count are likely to contain a high percentage of noise. 
        For usability reasons, such unrealistic values are grayed out. 
        More specifically: The threshold is set so that values for tiles with a 5% chance (or higher) of deviating more than {round(THRESHOLD * 100)} percentage points from the estimated value are not shown."""
    visits_per_tile_eps = None
    visits_per_tile_moe = None
    visits_per_tile_legend = ""
    visits_per_tile_summary_table = ""
    visits_per_tile_cumsum_linechart = ""
    most_freq_tiles_ranking = ""
    visits_per_time_tile_eps = None
    visits_per_time_tile_moe = None
    visits_per_tile_time_map = ""

    if const.VISITS_PER_TILE not in analysis_exclusion:
        visits_per_tile_eps = (render_eps(report_base.report[const.VISITS_PER_TILE].privacy_budget), render_eps(report_alternative.report[const.VISITS_PER_TILE].privacy_budget))
        visits_per_tile_moe = (fmt_moe(
            report_base.report[const.VISITS_PER_TILE].margin_of_error_laplace), fmt_moe(report_alternative.report[const.VISITS_PER_TILE].margin_of_error_laplace)
        )

        points_outside_tessellation_info_base = render_points_outside_tess(
            report_base.report[const.VISITS_PER_TILE]
        )
        points_outside_tessellation_info_alternative = render_points_outside_tess(
            report_base.report[const.VISITS_PER_TILE]
        )
        visits_per_tile_legend = render_benchmark_visits_per_tile(
            report_base.report[const.VISITS_PER_TILE], report_alternative.report[const.VISITS_PER_TILE], tessellation, THRESHOLD, temp_map_folder
        )
        quartiles_base=report_base.report[const.VISITS_PER_TILE].quartiles.round()
        quartiles_alternative=report_alternative.report[const.VISITS_PER_TILE].quartiles.round()
        visits_per_tile_summary_table = render_benchmark_summary(
            quartiles_base.astype(int), 
            quartiles_alternative.astype(int),
            "Distribution of visits per tile",  # extrapolate visits from dp record count
        )
        visits_per_tile_cumsum_linechart = render_visits_per_tile_cumsum(
            report_base.report[const.VISITS_PER_TILE], report_alternative.report[const.VISITS_PER_TILE]
        )
        most_freq_tiles_ranking = render_most_freq_tiles_ranking_benchmark(
            report_base.report[const.VISITS_PER_TILE], report_alternative.report[const.VISITS_PER_TILE]
        )
        visits_per_tile_measure=(const.format[benchmark.measure_selection[const.VISITS_PER_TILE]], fmt(benchmark.similarity_measures[const.VISITS_PER_TILE]))
        visits_per_time_tile_measure=(const.format[benchmark.measure_selection[const.VISITS_PER_TIME_TILE]], fmt(benchmark.similarity_measures[const.VISITS_PER_TIME_TILE]))

    if const.VISITS_PER_TIME_TILE not in analysis_exclusion:
        visits_per_time_tile_eps = (render_eps(
            report_base.report[const.VISITS_PER_TIME_TILE].privacy_budget), render_eps(report_alternative.report[const.VISITS_PER_TIME_TILE].privacy_budget)
        )
        visits_per_time_tile_moe = (
            fmt_moe(report_base.report[const.VISITS_PER_TIME_TILE].margin_of_error_laplace), 
            fmt_moe(report_alternative.report[const.VISITS_PER_TIME_TILE].margin_of_error_laplace)
        )

        visits_per_tile_time_map = render_visits_per_time_tile_benchmark(
            report_base.report[const.VISITS_PER_TIME_TILE], report_alternative.report[const.VISITS_PER_TIME_TILE], tessellation, THRESHOLD
        )

    template_structure = get_template("place_analysis_segment_benchmark.html")

    return template_structure.render(
        output_filename=output_filename,
        points_outside_tessellation_info_base=points_outside_tessellation_info_base,
        points_outside_tessellation_info_alternative=points_outside_tessellation_info_alternative,
        privacy_info=privacy_info,
        visits_per_tile_eps=visits_per_tile_eps,
        visits_per_tile_moe=visits_per_tile_moe,
        visits_per_tile_legend=visits_per_tile_legend,
        visits_per_tile_summary_table=visits_per_tile_summary_table,
        visits_per_tile_cumsum_linechart=visits_per_tile_cumsum_linechart,
        most_freq_tiles_ranking=most_freq_tiles_ranking,
        visits_per_time_tile_eps=visits_per_time_tile_eps,
        visits_per_time_tile_moe=visits_per_time_tile_moe,
        visits_per_tile_time_map=visits_per_tile_time_map,
        visits_per_tile_measure=visits_per_tile_measure,
        visits_per_time_tile_measure=visits_per_time_tile_measure,
    )

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

def render_benchmark_visits_per_tile(
    visits_per_tile_base: DfSection,
    visits_per_tile_alternative: DfSection,
    tessellation: GeoDataFrame,
    threshold: float,
    temp_map_folder: Path,
) -> str:

    visits_per_tile_base_sorted=visits_per_tile_base.data.sort_values('tile_id')
    visits_per_tile_alternative_sorted=visits_per_tile_alternative.data.sort_values('tile_id')
    relative_base = visits_per_tile_base_sorted['visits']/visits_per_tile_base_sorted['visits'].sum()
    relative_alternative = visits_per_tile_alternative_sorted['visits']/visits_per_tile_alternative_sorted['visits'].sum()
    deviation_from_base = pd.DataFrame({const.TILE_ID: visits_per_tile_base_sorted[const.TILE_ID],
                                        'deviation': relative_alternative-relative_base})

    # merge count and tessellation
    counts_per_tile_gdf = pd.merge(
        tessellation,
        deviation_from_base,
        how="left",
        left_on=const.TILE_ID,
        right_on=const.TILE_ID,
    )

    # # filter visit counts above error threshold
    # moe_deviation = (
    #     visits_per_tile.margin_of_error_laplace / counts_per_tile_gdf["deviation"]
    # )

    # counts_per_tile_gdf.loc[moe_deviation > threshold, "visits"] = None

    map, legend = plot.choropleth_map(
        counts_per_tile_gdf,
        "deviation",
        scale_title="deviation of relative counts from base \n deviation = alternative - base",
        aliases=["Tile ID", "Tile Name", "deviation"],
        diverging_cmap=True
    )

    map.save(os.path.join(temp_map_folder, "visits_per_tile_map.html"))

    legend_html = v_utils.fig_to_html(legend)
    plt.close()
    return legend_html

def render_visits_per_tile_cumsum(counts_per_tile: DfSection, counts_per_tile_alternative: Optional[DfSection]=None) -> str:
    df_cumsum = counts_per_tile.cumsum
    if counts_per_tile_alternative:
        df_cumsum_alternative = counts_per_tile_alternative.cumsum 
    else: 
        df_cumsum_alternative = None

    chart = plot.linechart_new(
        data=df_cumsum,
        x="n",
        y="cum_perc",
        data_alternative=df_cumsum_alternative,
        x_axis_label="Number of tiles",
        y_axis_label="Cumulated sum of visits per tile",
        add_diagonal=False,
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

def render_most_freq_tiles_ranking_benchmark(visits_per_tile_base: DfSection, visits_per_tile_alternative: DfSection, top_x: int = 10) -> str:
    topx_tiles_base = visits_per_tile_base.data.nlargest(top_x, "visits")
    topx_tiles_base['visits'] = topx_tiles_base['visits']/topx_tiles_base['visits'].sum()*100
    topx_tiles_alternative = visits_per_tile_alternative.data.nlargest(top_x, "visits")
    topx_tiles_alternative['visits']=topx_tiles_alternative['visits']/topx_tiles_alternative['visits'].sum()*100
    topx_tiles_merged = topx_tiles_base.merge(topx_tiles_alternative, on=['tile_id', 'tile_name'], how='outer', suffixes=('_base','_alternative'))
    topx_tiles_merged.sort_values(by=['visits_base', 'visits_alternative'])
    topx_tiles_merged["rank"] = list(range(1, len(topx_tiles_merged) + 1))
    labels = (
        topx_tiles_merged["rank"].astype(str)
        + ": "
        + topx_tiles_merged[const.TILE_NAME]
        + "(Id: "
        + topx_tiles_merged[const.TILE_ID]
        + ")"
    )

    ranking = plot.ranking(
        x=topx_tiles_merged.visits_base,
        x_axis_label="percentage of visits per tile",
        y_labels=labels,
        x_alternative=topx_tiles_merged.visits_alternative,
        margin_of_error=visits_per_tile_base.margin_of_error_laplace,
        #margin_of_error_alternative=visits_per_tile_alternative.margin_of_error_laplace
    )
    html_ranking = v_utils.fig_to_html(ranking)
    plt.close()
    return html_ranking

def render_visits_per_time_tile(
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

def render_visits_per_time_tile_benchmark(
    counts_per_tile_timewindow: DfSection, counts_per_tile_timewindow_alternative: DfSection, tessellation: GeoDataFrame, threshold: float
) -> str:
    data = counts_per_tile_timewindow.data
    data_alternative = counts_per_tile_timewindow_alternative.data

    # moe_perc_per_tile_timewindow_base = (
    #     counts_per_tile_timewindow.margin_of_error_laplace / data
    # )
    # moe_perc_per_tile_timewindow_alternative = (
    #     counts_per_tile_timewindow_alternative.margin_of_error_laplace / data_alternative
    # )
    if data is None:
        return None
    if data_alternative is None:
        return None

    #data[moe_perc_per_tile_timewindow > threshold] = None

    output_html = ""
    if "weekday" in data.columns:
        weekday_base = data.loc[:, "weekday"]/data.loc[:, "weekday"].sum().sum()
        weekday_alternative = data_alternative.loc[:, "weekday"]/data_alternative.loc[:, "weekday"].sum().sum()
        output_html += "<h4>Weekday</h4>"
        output_html += _create_timewindow_segment_benchmark(weekday_base-weekday_alternative, tessellation)

    if "weekend" in data.columns:
        weekend_base = data.loc[:, "weekend"]/data.loc[:, "weekend"].sum().sum()
        weekend_alternative = data_alternative.loc[:, "weekend"]/data_alternative.loc[:, "weekend"].sum().sum()
        output_html += "<h4>Weekend</h4>"
        output_html += _create_timewindow_segment_benchmark(weekend_base-weekend_alternative, tessellation)
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

def _create_timewindow_segment_benchmark(df: pd.DataFrame, tessellation: GeoDataFrame) -> str:
    visits_choropleth = plot.multi_choropleth_map(df, tessellation, diverging_cmap=True)
    return f"""<h4>Deviation from base</h4>
        {v_utils.fig_to_html_as_png(visits_choropleth)}"""  # svg might get too large

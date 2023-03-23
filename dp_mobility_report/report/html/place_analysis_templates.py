import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame

if TYPE_CHECKING:
    from dp_mobility_report import BenchmarkReport, DpMobilityReport

import folium

from dp_mobility_report import constants as const
from dp_mobility_report.benchmark.similarity_measures import symmetric_perc_error
from dp_mobility_report.model.section import DfSection
from dp_mobility_report.report.html.html_utils import (
    all_available_measures,
    fmt_moe,
    get_template,
    render_benchmark_summary,
    render_eps,
    render_summary,
)
from dp_mobility_report.visualization import plot, v_utils


def render_place_analysis(
    dpmreport: "DpMobilityReport",
    temp_map_folder: Path,
    output_filename: str,
) -> str:
    THRESHOLD = 0.2  # 20%
    args: dict = {}
    report = dpmreport.report
    tessellation = dpmreport.tessellation

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
            quartiles, int  # extrapolate visits from dp record count
        )
        args["visits_per_tile_cumsum_linechart"] = render_visits_per_tile_cumsum(
            report[const.VISITS_PER_TILE],
            diagonal=True,
        )
        args["most_freq_tiles_ranking"] = render_most_freq_tiles_ranking(
            report[const.VISITS_PER_TILE],
        )

    if const.VISITS_PER_TIME_TILE not in dpmreport.analysis_exclusion:
        args["visits_per_tile_timewindow_eps"] = render_eps(
            report[const.VISITS_PER_TIME_TILE].privacy_budget
        )
        args["visits_per_tile_timewindow_moe"] = fmt_moe(
            report[const.VISITS_PER_TIME_TILE].margin_of_error_laplace
        )

        args["visits_per_tile_time_map"] = render_visits_per_time_tile(
            report[const.VISITS_PER_TIME_TILE], tessellation, THRESHOLD
        )
        args["visits_per_tile_time_info"] = "User configuration of timewindows: " + str(
            [
                f"{first} - {second}"
                for first, second in zip(
                    dpmreport.timewindows[0:-1], dpmreport.timewindows[1:]
                )
            ]
        )

    template_structure = get_template("place_analysis_segment.html")

    return template_structure.render(args)


def render_benchmark_place_analysis(
    benchmark: "BenchmarkReport",
    temp_map_folder: Path,
    output_filename: str,
) -> str:
    args: dict = {}
    report_base = benchmark.report_base.report
    report_alternative = benchmark.report_alternative.report
    tessellation = benchmark.report_base.tessellation
    template_measures = get_template("similarity_measures.html")

    args["output_filename"] = output_filename

    if const.VISITS_PER_TILE not in benchmark.analysis_exclusion:
        args["visits_per_tile_eps"] = (
            render_eps(report_base[const.VISITS_PER_TILE].privacy_budget),
            render_eps(report_alternative[const.VISITS_PER_TILE].privacy_budget),
        )
        # change moe unit to relative counts
        rel_visits_per_tile_moe_base = (
            report_base[const.VISITS_PER_TILE].margin_of_error_laplace
            / report_base[const.VISITS_PER_TILE].data["visits"].sum()
            if report_base[const.VISITS_PER_TILE].data["visits"].sum() > 0
            else report_base[const.VISITS_PER_TILE].margin_of_error_laplace
        )
        rel_visits_per_tile_moe_alternative = (
            report_alternative[const.VISITS_PER_TILE].margin_of_error_laplace
            / report_alternative[const.VISITS_PER_TILE].data["visits"].sum()
            if report_alternative[const.VISITS_PER_TILE].data["visits"].sum()
            else report_alternative[const.VISITS_PER_TILE].margin_of_error_laplace
        )
        args["visits_per_tile_moe"] = (
            fmt_moe(rel_visits_per_tile_moe_base),
            fmt_moe(rel_visits_per_tile_moe_alternative),
        )

        args["points_outside_tessellation_info_base"] = render_points_outside_tess(
            report_base[const.VISITS_PER_TILE]
        )
        args[
            "points_outside_tessellation_info_alternative"
        ] = render_points_outside_tess(report_alternative[const.VISITS_PER_TILE])
        args["visits_per_tile_legend"] = render_benchmark_visits_per_tile(
            report_base[const.VISITS_PER_TILE],
            report_alternative[const.VISITS_PER_TILE],
            tessellation,
            temp_map_folder,
        )
        quartiles_base = report_base[const.VISITS_PER_TILE].quartiles.round()
        quartiles_alternative = report_alternative[
            const.VISITS_PER_TILE
        ].quartiles.round()
        args["visits_per_tile_summary_table"] = render_benchmark_summary(
            quartiles_base,
            quartiles_alternative,
            target_type=int,
        )
        args["visits_per_tile_cumsum_linechart"] = render_visits_per_tile_cumsum(
            report_base[const.VISITS_PER_TILE],
            report_alternative[const.VISITS_PER_TILE],
            diagonal=False,
        )
        args["most_freq_tiles_ranking"] = render_most_freq_tiles_ranking_benchmark(
            report_base[const.VISITS_PER_TILE],
            report_alternative[const.VISITS_PER_TILE],
            rel_visits_per_tile_moe_base,
            rel_visits_per_tile_moe_alternative,
        )
        args["visits_per_tile_measure"] = template_measures.render(
            all_available_measures(const.VISITS_PER_TILE, benchmark)
        )

        args["visits_per_tile_summary_measure"] = template_measures.render(
            all_available_measures(const.VISITS_PER_TILE_QUARTILES, benchmark)
        )

        args["visits_per_tile_ranking_measure"] = template_measures.render(
            {
                **all_available_measures(const.VISITS_PER_TILE_RANKING, benchmark),
                **{"top_n_object": "locations"},
            }
        )

    if const.VISITS_PER_TIME_TILE not in benchmark.analysis_exclusion:
        args["visits_per_tile_timewindow_eps"] = (
            render_eps(report_base[const.VISITS_PER_TIME_TILE].privacy_budget),
            render_eps(report_alternative[const.VISITS_PER_TIME_TILE].privacy_budget),
        )
        args["visits_per_tile_timewindow_moe"] = (
            fmt_moe(report_base[const.VISITS_PER_TIME_TILE].margin_of_error_laplace),
            fmt_moe(
                report_alternative[const.VISITS_PER_TIME_TILE].margin_of_error_laplace
            ),
        )

        args["visits_per_tile_time_map"] = render_visits_per_time_tile_benchmark(
            report_base[const.VISITS_PER_TIME_TILE],
            report_alternative[const.VISITS_PER_TIME_TILE],
            tessellation,
        )
        args["visits_per_tile_time_info"] = "User configuration of timewindows: " + str(
            [
                f"{first} - {second}"
                for first, second in zip(
                    benchmark.report_base.timewindows[0:-1],
                    benchmark.report_base.timewindows[1:],
                )
            ]
        )

        args["visits_per_time_tile_measure"] = template_measures.render(
            all_available_measures(const.VISITS_PER_TIME_TILE, benchmark)
        )

    template_structure = get_template("place_analysis_segment_benchmark.html")

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
        cmap=const.BASE_CMAP,
    )

    map.save(os.path.join(temp_map_folder, "visits_per_tile_map.html"))

    legend_html = v_utils.fig_to_html(legend)
    plt.close()
    return legend_html


def render_benchmark_visits_per_tile(
    visits_per_tile_base: DfSection,
    visits_per_tile_alternative: DfSection,
    tessellation: GeoDataFrame,
    temp_map_folder: Path,
) -> List[str]:

    visits_per_tile_base_sorted = visits_per_tile_base.data.sort_values(
        "tile_id"
    ).reset_index()
    visits_per_tile_alternative_sorted = visits_per_tile_alternative.data.sort_values(
        "tile_id"
    ).reset_index()
    relative_base = (
        visits_per_tile_base_sorted["visits"]
        / visits_per_tile_base_sorted["visits"].sum()
    )
    relative_alternative = (
        visits_per_tile_alternative_sorted["visits"]
        / visits_per_tile_alternative_sorted["visits"].sum()
    )
    deviation_from_base = pd.DataFrame(
        {
            const.TILE_ID: visits_per_tile_base_sorted[const.TILE_ID],
            "relative_base": relative_base,
            "relative_alternative": relative_alternative,
        }
    )

    deviation_from_base["deviation"] = deviation_from_base.apply(
        lambda x: symmetric_perc_error(
            x["relative_alternative"], x["relative_base"], keep_direction=True
        ),
        axis=1,
    )

    # merge count and tessellation
    counts_per_tile_gdf = pd.merge(
        tessellation,
        deviation_from_base,
        how="left",
        left_on=const.TILE_ID,
        right_on=const.TILE_ID,
    )

    map, legend = plot.choropleth_map(
        counts_per_tile_gdf,
        "deviation",
        scale_title="deviation of relative counts per tile from base",
        aliases=["Tile ID", "Tile Name", "deviation"],
        is_cmap_diverging=True,
        min_scale=-2,
        max_scale=2,
        layer_name="Deviation",
    )
    min_scale = min(
        counts_per_tile_gdf["relative_base"].min(),
        counts_per_tile_gdf["relative_alternative"].min(),
    )
    max_scale = max(
        counts_per_tile_gdf["relative_base"].max(),
        counts_per_tile_gdf["relative_alternative"].max(),
    )

    map, legend_base = plot.choropleth_map(
        counts_per_tile_gdf,
        "relative_base",
        scale_title="relative counts per tile",
        aliases=["Tile ID", "Tile Name", "relative counts"],
        is_cmap_diverging=False,
        map=map,
        layer_name="Relative visits base",
        show=False,
        min_scale=min_scale,
        max_scale=max_scale,
        cmap=const.BASE_CMAP,
    )
    map, legend_alternative = plot.choropleth_map(
        counts_per_tile_gdf,
        "relative_alternative",
        scale_title="relative counts per tile",
        aliases=["Tile ID", "Tile Name", "relative counts"],
        is_cmap_diverging=False,
        map=map,
        layer_name="Relative visits alternative",
        show=False,
        min_scale=min_scale,
        max_scale=max_scale,
        cmap=const.ALT_CMAP,
    )

    folium.LayerControl(collapsed=False).add_to(map)

    map.save(os.path.join(temp_map_folder, "visits_per_tile_map.html"))

    legend_html_deviation = v_utils.fig_to_html(legend)
    legend_html_base = v_utils.fig_to_html(legend_base)
    legend_html_alternative = v_utils.fig_to_html(legend_alternative)
    plt.close()
    return [
        v_utils.resize_width(legend_html_deviation, 100),
        v_utils.resize_width(legend_html_base, 100),
        v_utils.resize_width(legend_html_alternative, 100),
    ]


def render_visits_per_tile_cumsum(
    counts_per_tile: DfSection,
    counts_per_tile_alternative: Optional[DfSection] = None,
    diagonal: bool = False,
) -> str:
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
        y_axis_label="Cumulated sum of relative visits per tile",
        add_diagonal=diagonal,
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
        + " (Id: "
        + topx_tiles[const.TILE_ID]
        + ")"
    )

    ranking = plot.ranking(
        round(topx_tiles.visits),
        "number of visits per tile",
        y_labels=labels,
        margin_of_error=visits_per_tile.margin_of_error_laplace,
        figsize=(8, max(6, min(len(labels) * 0.5, 8))),
    )
    html_ranking = v_utils.fig_to_html(ranking)
    plt.close(ranking)
    return html_ranking


def render_most_freq_tiles_ranking_benchmark(
    visits_per_tile_base: DfSection,
    visits_per_tile_alternative: DfSection,
    rel_visits_per_tile_moe_base: float,
    rel_visits_per_tile_moe_alternative: float,
    top_x: int = 10,
) -> str:
    topx_tiles_base = visits_per_tile_base.data.nlargest(top_x, "visits")
    topx_tiles_base["visits"] = (
        topx_tiles_base["visits"] / topx_tiles_base["visits"].sum() * 100
    )
    topx_tiles_alternative = visits_per_tile_alternative.data.nlargest(top_x, "visits")
    topx_tiles_alternative["visits"] = (
        topx_tiles_alternative["visits"] / topx_tiles_alternative["visits"].sum() * 100
    )
    topx_tiles_merged = topx_tiles_base.merge(
        topx_tiles_alternative,
        on=["tile_id", "tile_name"],
        how="outer",
        suffixes=("_base", "_alternative"),
    )
    topx_tiles_merged.sort_values(by=["visits_base", "visits_alternative"])
    topx_tiles_merged["rank"] = list(range(1, len(topx_tiles_merged) + 1))
    labels = (
        topx_tiles_merged["rank"].astype(str)
        + ": "
        + topx_tiles_merged[const.TILE_NAME]
        + " (Id: "
        + topx_tiles_merged[const.TILE_ID]
        + ")"
    )

    ranking = plot.ranking(
        x=topx_tiles_merged.visits_base,
        x_axis_label="percentage of visits per tile",
        y_labels=labels,
        x_alternative=topx_tiles_merged.visits_alternative,
        margin_of_error=rel_visits_per_tile_moe_base,
        margin_of_error_alternative=rel_visits_per_tile_moe_alternative,
        figsize=(8, max(6, min(len(labels) * 0.5, 8))),
    )
    html_ranking = v_utils.fig_to_html(ranking)
    plt.close(ranking)
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
    counts_per_tile_timewindow: DfSection,
    counts_per_tile_timewindow_alternative: DfSection,
    tessellation: GeoDataFrame,
) -> str:
    data = counts_per_tile_timewindow.data
    data_alternative = counts_per_tile_timewindow_alternative.data

    if data is None:
        return None
    if data_alternative is None:
        return None

    output_html = ""
    if "weekday" in data.columns:
        weekday_base = data.loc[:, "weekday"] / data.loc[:, "weekday"].sum().sum()
        weekday_alternative = (
            data_alternative.loc[:, "weekday"]
            / data_alternative.loc[:, "weekday"].sum().sum()
        )

        deviation_values = [
            [
                symmetric_perc_error(alt, base, keep_direction=True)
                for alt, base in zip(alt_array, base_array)
            ]
            for alt_array, base_array in zip(
                weekday_alternative.values, weekday_base.values
            )
        ]
        deviation = pd.DataFrame(deviation_values)
        deviation.set_index(weekday_base.index, inplace=True)
        deviation.columns = weekday_base.columns

        output_html += "<h4>Weekday</h4>"
        output_html += _create_timewindow_segment_benchmark(deviation, tessellation)

    if "weekend" in data.columns:
        weekend_base = data.loc[:, "weekend"] / data.loc[:, "weekend"].sum().sum()
        weekend_alternative = (
            data_alternative.loc[:, "weekend"]
            / data_alternative.loc[:, "weekend"].sum().sum()
        )
        deviation = [
            [
                symmetric_perc_error(alt, base, keep_direction=True)
                for alt, base in zip(alt_array, base_array)
            ]
            for alt_array, base_array in zip(
                weekend_alternative.values, weekend_base.values
            )
        ]
        deviation = pd.DataFrame(deviation)
        deviation.index = weekend_base.index
        deviation.columns = weekend_base.columns

        output_html += "<h4>Weekend</h4>"
        output_html += _create_timewindow_segment_benchmark(deviation, tessellation)
    plt.close()
    return output_html


def _create_timewindow_segment(df: pd.DataFrame, tessellation: GeoDataFrame) -> str:
    visits_choropleth = plot.multi_choropleth_map(df, tessellation)

    tile_means = df.mean(axis=1)
    dev_from_avg = df.div(tile_means, axis=0)
    deviation_choropleth = plot.multi_choropleth_map(
        dev_from_avg, tessellation, is_cmap_diverging=True, min_scale=0, vcenter=1
    )
    html = f"""<h4>Number of visits</h4>
        {v_utils.fig_to_html_as_png(visits_choropleth)}
        <h4>Deviation from tile average</h4>
        <div><p>The average of each tile 
        over all time windows equals 1 (100% of average traffic). 
        A value of < 1 (> 1) means that a tile is visited less (more) frequently in this time window than it is on average.</p></div>
        {v_utils.fig_to_html_as_png(deviation_choropleth)}"""  # svg might get too large
    plt.close(visits_choropleth)
    plt.close(deviation_choropleth)
    return html


def _create_timewindow_segment_benchmark(
    df_deviation: pd.DataFrame, tessellation: GeoDataFrame
) -> str:
    visits_choropleth = plot.multi_choropleth_map(
        df_deviation, tessellation, is_cmap_diverging=True, min_scale=-2, max_scale=2
    )
    html = f"""<h4>Deviation from base</h4>
        {v_utils.fig_to_html_as_png(visits_choropleth)}"""  # svg might get too large
    plt.close(visits_choropleth)
    return html

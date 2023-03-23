import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport, BenchmarkReport

import folium

from dp_mobility_report import constants as const
from dp_mobility_report.benchmark.similarity_measures import symmetric_perc_error
from dp_mobility_report.model import od_analysis
from dp_mobility_report.model.section import DfSection, TupleSection
from dp_mobility_report.report.html.html_utils import (
    all_available_measures,
    fmt_moe,
    get_template,
    render_benchmark_summary,
    render_eps,
    render_summary,
    render_user_input_info,
)
from dp_mobility_report.visualization import plot, v_utils


def render_od_analysis(
    dpmreport: "DpMobilityReport",
    top_n_flows: int,
    temp_map_folder: Path,
    output_filename: str,
) -> str:
    THRESHOLD = 0.2  # 20 %
    args: dict = {}
    report = dpmreport.report

    args[
        "user_config_info"
    ] = f"User configuration: display max. top {top_n_flows} OD connections on map"
    args["output_filename"] = output_filename

    if const.OD_FLOWS not in dpmreport.analysis_exclusion:
        args["od_eps"] = render_eps(report[const.OD_FLOWS].privacy_budget)
        args["od_moe"] = fmt_moe(report[const.OD_FLOWS].margin_of_error_laplace)

        args["od_legend"] = render_origin_destination_flows(
            report[const.OD_FLOWS],
            dpmreport.tessellation,
            top_n_flows,
            THRESHOLD,
            temp_map_folder,
        )
        args["intra_tile_flows_info"] = render_intra_tile_flows(
            report[const.OD_FLOWS], len(dpmreport.tessellation)
        )
        quartiles = report[const.OD_FLOWS].quartiles.round()
        args["flows_summary_table"] = render_summary(quartiles, int)
        args["flows_cumsum_linechart"] = render_flows_cumsum(
            report[const.OD_FLOWS], diagonal=True
        )
        args["most_freq_flows_ranking"] = render_most_freq_flows_ranking(
            report[const.OD_FLOWS], dpmreport.tessellation
        )

    if const.TRAVEL_TIME not in dpmreport.analysis_exclusion:
        args["travel_time_eps"] = render_eps(report[const.TRAVEL_TIME].privacy_budget)
        args["travel_time_moe"] = fmt_moe(
            report[const.TRAVEL_TIME].margin_of_error_laplace
        )

        args["travel_time_hist_info"] = render_user_input_info(
            dpmreport.max_travel_time, dpmreport.bin_range_travel_time
        )
        args["travel_time_hist"] = render_travel_time_hist(report[const.TRAVEL_TIME])
        args["travel_time_summary_table"] = render_summary(
            report[const.TRAVEL_TIME].quartiles
        )

    if const.JUMP_LENGTH not in dpmreport.analysis_exclusion:
        args["jump_length_eps"] = render_eps(report[const.JUMP_LENGTH].privacy_budget)
        args["jump_length_moe"] = fmt_moe(
            report[const.JUMP_LENGTH].margin_of_error_laplace
        )

        args["jump_length_hist_info"] = render_user_input_info(
            dpmreport.max_jump_length, dpmreport.bin_range_jump_length
        )
        args["jump_length_hist"] = render_jump_length_hist(report[const.JUMP_LENGTH])
        args["jump_length_summary_table"] = render_summary(
            report[const.JUMP_LENGTH].quartiles
        )

    template_structure = get_template("od_analysis_segment.html")
    return template_structure.render(args)


def render_benchmark_od_analysis(
    benchmark: "BenchmarkReport",
    top_n_flows: int,
    temp_map_folder: Path,
    output_filename: str,
) -> str:
    report_base = benchmark.report_base.report
    report_alternative = benchmark.report_alternative.report
    tessellation = benchmark.report_base.tessellation
    args: dict = {}
    template_measures = get_template("similarity_measures.html")

    args[
        "user_config_info"
    ] = f"User configuration: display max. top {top_n_flows} OD connections on map"
    args["output_filename"] = output_filename

    if const.OD_FLOWS not in benchmark.analysis_exclusion:
        args["od_eps"] = (
            render_eps(report_base[const.OD_FLOWS].privacy_budget),
            render_eps(report_alternative[const.OD_FLOWS].privacy_budget),
        )
        args["od_moe"] = (
            fmt_moe(report_base[const.OD_FLOWS].margin_of_error_laplace),
            fmt_moe(report_alternative[const.OD_FLOWS].margin_of_error_laplace),
        )

        args["od_legend"] = render_benchmark_origin_destination_flows(
            report_base[const.OD_FLOWS],
            report_alternative[const.OD_FLOWS],
            tessellation,
            top_n_flows,
            temp_map_folder,
        )

        quartiles_base = report_base[const.OD_FLOWS].quartiles.round()
        quartiles_alternative = report_alternative[const.OD_FLOWS].quartiles.round()
        args["flows_summary_table"] = render_benchmark_summary(
            quartiles_base, quartiles_alternative, target_type=int
        )
        args["flows_cumsum_linechart"] = render_flows_cumsum(
            report_base[const.OD_FLOWS],
            report_alternative[const.OD_FLOWS],
        )
        args["most_freq_flows_ranking"] = render_most_freq_flows_ranking_benchmark(
            report_base[const.OD_FLOWS],
            report_alternative[const.OD_FLOWS],
            tessellation,
        )
        args["flows_measure"] = template_measures.render(
            all_available_measures(const.OD_FLOWS, benchmark)
        )

        args["flows_summary_measure"] = template_measures.render(
            all_available_measures(const.OD_FLOWS_QUARTILES, benchmark)
        )

        args["od_flows_ranking_measure"] = template_measures.render(
            {
                **all_available_measures(const.OD_FLOWS_RANKING, benchmark),
                **{"top_n_object": "flows"},
            }
        )

    if const.TRAVEL_TIME not in benchmark.analysis_exclusion:
        args["travel_time_eps"] = (
            render_eps(report_base[const.TRAVEL_TIME].privacy_budget),
            render_eps(report_alternative[const.TRAVEL_TIME].privacy_budget),
        )
        args["travel_time_moe"] = (
            fmt_moe(report_base[const.TRAVEL_TIME].margin_of_error_laplace),
            fmt_moe(report_alternative[const.TRAVEL_TIME].margin_of_error_laplace),
        )

        args["travel_time_hist_info"] = render_user_input_info(
            benchmark.report_base.max_travel_time,
            benchmark.report_base.bin_range_travel_time,
        )
        args["travel_time_hist"] = render_travel_time_hist(
            report_base[const.TRAVEL_TIME], report_alternative[const.TRAVEL_TIME]
        )
        args["travel_time_summary_table"] = render_benchmark_summary(
            report_base[const.TRAVEL_TIME].quartiles,
            report_alternative[const.TRAVEL_TIME].quartiles,
            target_type=float,
        )
        args["travel_time_measure"] = template_measures.render(
            all_available_measures(const.TRAVEL_TIME, benchmark)
        )
        args["travel_time_summary_measure"] = template_measures.render(
            all_available_measures(const.TRAVEL_TIME_QUARTILES, benchmark)
        )

    if const.JUMP_LENGTH not in benchmark.analysis_exclusion:
        args["jump_length_eps"] = (
            render_eps(report_base[const.JUMP_LENGTH].privacy_budget),
            render_eps(report_alternative[const.JUMP_LENGTH].privacy_budget),
        )
        args["jump_length_moe"] = (
            fmt_moe(report_base[const.JUMP_LENGTH].margin_of_error_laplace),
            fmt_moe(report_alternative[const.JUMP_LENGTH].margin_of_error_laplace),
        )

        args["jump_length_hist_info"] = render_user_input_info(
            benchmark.report_base.max_jump_length,
            benchmark.report_base.bin_range_jump_length,
        )
        args["jump_length_hist"] = render_jump_length_hist(
            report_base[const.JUMP_LENGTH], report_alternative[const.JUMP_LENGTH]
        )
        args["jump_length_summary_table"] = render_benchmark_summary(
            report_base[const.JUMP_LENGTH].quartiles,
            report_alternative[const.JUMP_LENGTH].quartiles,
            target_type=float,
        )
        args["jump_length_measure"] = template_measures.render(
            all_available_measures(const.JUMP_LENGTH, benchmark)
        )
        args["jump_length_summary_measure"] = template_measures.render(
            all_available_measures(const.JUMP_LENGTH_QUARTILES, benchmark)
        )

    template_structure = get_template("od_analysis_segment_benchmark.html")
    return template_structure.render(args)


def render_origin_destination_flows(
    od_flows: DfSection,
    tessellation: GeoDataFrame,
    top_n_flows: int,
    threshold: float,
    temp_map_folder: Path,
) -> str:

    data = od_flows.data.copy()
    moe_deviation = od_flows.margin_of_error_laplace / data[const.FLOW]
    data.loc[moe_deviation > threshold, const.FLOW] = None

    # create intra_tile basemap
    intra_tile_flows = data[data[const.ORIGIN] == data[const.DESTINATION]]
    tessellation_intra_flows = pd.merge(
        tessellation,
        intra_tile_flows,
        how="left",
        left_on=const.TILE_ID,
        right_on=const.ORIGIN,
    )

    map, intra_tile_legend = plot.choropleth_map(
        tessellation_intra_flows,
        const.FLOW,
        "intra-tile flows",
        cmap=const.BASE_CMAP,
    )  # get innerflows as color for choropleth

    # create od flows map
    top_n_flows = top_n_flows if top_n_flows <= len(data) else len(data)
    flows = data[
        (data[const.ORIGIN] != data[const.DESTINATION]) & data[const.FLOW].notna()
    ].nlargest(top_n_flows, const.FLOW)
    # only plot lines if there are any flows between tiles
    if not flows[const.FLOW].isnull().all():
        map = plot.flows(map, flows, tessellation)

    map.save(os.path.join(temp_map_folder, "od_map.html"))

    html_legend = v_utils.fig_to_html(intra_tile_legend)
    plt.close()
    return html_legend


def merge_innerflows(
    base_flows: pd.DataFrame, alternative_flows: pd.DataFrame
) -> pd.DataFrame:
    flows = base_flows.merge(
        alternative_flows,
        how="outer",
        on=["origin", "destination"],
        suffixes=["_alt", "_base"],
    )
    flows["flow_base"] = flows["flow_base"] / np.sum(flows["flow_base"])
    flows["flow_alt"] = flows["flow_alt"] / np.sum(flows["flow_alt"])
    flows["flow_base"] = flows["flow_base"].fillna(0)
    flows["flow_alt"] = flows["flow_alt"].fillna(0)
    flows["deviation"] = flows.apply(
        lambda x: symmetric_perc_error(
            x["flow_alt"], x["flow_base"], keep_direction=True
        ),
        axis=1,
    )
    flows.drop(["flow_alt", "flow_base"], axis=1, inplace=True)
    return flows


def render_benchmark_origin_destination_flows(
    od_flows_base: DfSection,
    od_flows_alternative: DfSection,
    tessellation: GeoDataFrame,
    top_n_flows: int,
    temp_map_folder: Path,
) -> str:
    data_base = od_flows_base.data.copy()
    data_alternative = od_flows_alternative.data.copy()

    top_n_flows_base = top_n_flows if top_n_flows <= len(data_base) else len(data_base)
    top_n_flows_alternative = (
        top_n_flows if top_n_flows <= len(data_alternative) else len(data_alternative)
    )

    innerflows = merge_innerflows(
        data_alternative[data_alternative.origin == data_alternative.destination],
        data_base[data_base.origin == data_base.destination],
    )

    tessellation_intra_flows = pd.merge(
        tessellation,
        innerflows,
        how="left",
        left_on=const.TILE_ID,
        right_on="origin",
    )

    tessellation_intra_flows.deviation.fillna(0, inplace=True)

    map, intra_tile_legend = plot.choropleth_map(
        tessellation_intra_flows,
        "deviation",
        "deviation from base intra-tile flows",
        layer_name="Intra-tile flows",
        is_cmap_diverging=True,
        min_scale=-2,
        max_scale=2,
    )  # get innerflows as color for choropleth

    flows_base = data_base[
        (data_base[const.ORIGIN] != data_base[const.DESTINATION])
        & data_base[const.FLOW].notna()
    ].nlargest(top_n_flows_base, const.FLOW)

    flows_alternative = data_alternative[
        (data_alternative[const.ORIGIN] != data_alternative[const.DESTINATION])
        & data_alternative[const.FLOW].notna()
    ].nlargest(top_n_flows_alternative, const.FLOW)

    if not flows_base.flow.isnull().all():
        map = plot.flows(
            map,
            flows_base,
            tessellation,
            flow_color=const.DARK_BLUE,
            marker_color=const.LIGHT_BLUE,
            layer_name="OD flows base",
        )

    if not flows_alternative.flow.isnull().all():
        map = plot.flows(
            map,
            flows_alternative,
            tessellation,
            flow_color=const.ORANGE,
            marker_color=const.LIGHT_ORANGE,
            layer_name="OD flows alternative",
        )

    folium.LayerControl(collapsed=False).add_to(map)

    map.save(os.path.join(temp_map_folder, "od_map.html"))

    html_legend = v_utils.fig_to_html(intra_tile_legend)
    plt.close()
    return html_legend


def render_intra_tile_flows(od_flows: DfSection, n_tiles: int) -> str:
    od_sum = od_flows.data["flow"].sum()
    if od_sum == 0:
        intra_tile_flows_perc = 0
        moe_info = 0
    else:
        intra_tile_flows_perc = round(
            od_analysis.get_intra_tile_flows(od_flows.data) / od_sum * 100, 2
        )
        # TODO: is this actually a correct estimation? -> the more noise the more non-intra tile flows are overestimated...
        moe_info = round(n_tiles * od_flows.margin_of_error_laplace / od_sum * 100, 2)
    ci_interval_info = (
        f"(95% confidence interval ± {moe_info} percentage points)"
        if od_flows.margin_of_error_laplace is not None
        else ""
    )

    return f"{intra_tile_flows_perc}% of flows start and end within the same cell {ci_interval_info}."


def render_flows_cumsum(
    od_flows: DfSection,
    od_flows_alternative: Optional[DfSection] = None,
    diagonal: bool = False,
) -> str:
    df_cumsum = od_flows.cumsum
    if od_flows_alternative:
        df_cumsum_alternative = od_flows_alternative.cumsum
    else:
        df_cumsum_alternative = None

    chart = plot.linechart_new(
        data=df_cumsum,
        x="n",
        y="cum_perc",
        data_alternative=df_cumsum_alternative,
        x_axis_label="Number of OD tile pairs",
        y_axis_label="Cumulated sum of relative OD flow counts",
        add_diagonal=diagonal,
    )
    html = v_utils.fig_to_html(chart)
    plt.close(chart)
    return html


def render_most_freq_flows_ranking(
    od_flows: DfSection, tessellation: GeoDataFrame, top_x: int = 10
) -> str:
    topx_flows = od_flows.data.nlargest(top_x, "flow")

    # if no intra-cell flows should be shown
    # topx_flows = od_flows.data[(od_flows.data.origin != od_flows.data.destination)].nlargest(top_x, "flow") # type: ignore

    topx_flows["rank"] = list(range(1, len(topx_flows) + 1))
    topx_flows = topx_flows.merge(
        tessellation[[const.TILE_ID, const.TILE_NAME]],
        how="left",
        left_on=const.ORIGIN,
        right_on=const.TILE_ID,
    )
    topx_flows = topx_flows.merge(
        tessellation[[const.TILE_ID, const.TILE_NAME]],
        how="left",
        left_on=const.DESTINATION,
        right_on=const.TILE_ID,
        suffixes=("_origin", "_destination"),
    )
    labels = (
        topx_flows["rank"].astype(str)
        + ": "
        + topx_flows[f"{const.TILE_NAME}_origin"]
        + " - "
        + topx_flows[f"{const.TILE_NAME}_destination"]
    )

    ranking = plot.ranking(
        topx_flows.flow,
        "Number of flows per OD pair",
        y_labels=labels,
        margin_of_error=od_flows.margin_of_error_laplace,
        figsize=(8, max(6, min(len(labels) * 0.5, 8))),
    )
    html_ranking = v_utils.fig_to_html(ranking)
    plt.close(ranking)
    return html_ranking


def render_most_freq_flows_ranking_benchmark(
    od_flows_base: DfSection,
    od_flows_alternative: DfSection,
    tessellation: GeoDataFrame,
    top_x: int = 10,
) -> str:

    topx_flows_base = od_flows_base.data.nlargest(top_x, "flow")
    topx_flows_base["flow"] = (
        (topx_flows_base["flow"] / od_flows_base.data["flow"].sum() * 100)
        if od_flows_base.data["flow"].sum() > 0
        else topx_flows_base["flow"]
    )
    topx_flows_alternative = od_flows_alternative.data.nlargest(top_x, "flow")
    topx_flows_alternative["flow"] = (
        (topx_flows_alternative["flow"] / od_flows_alternative.data["flow"].sum() * 100)
        if od_flows_alternative.data["flow"].sum()
        else topx_flows_alternative["flow"]
    )

    topx_flows_merged = topx_flows_base.merge(
        topx_flows_alternative,
        on=["origin", "destination"],
        suffixes=("_base", "_alternative"),
        how="outer",
    )
    topx_flows_merged.sort_values(by=["flow_base", "flow_alternative"])
    # if no intra-cell flows should be shown
    # topx_flows = od_flows.data[(od_flows.data.origin != od_flows.data.destination)].nlargest(top_x, "flow") # type: ignore

    topx_flows_merged["rank"] = list(range(1, len(topx_flows_merged) + 1))
    topx_flows_merged = topx_flows_merged.merge(
        tessellation[[const.TILE_ID, const.TILE_NAME]],
        how="left",
        left_on="origin",
        right_on=const.TILE_ID,
    )
    topx_flows_merged = topx_flows_merged.merge(
        tessellation[[const.TILE_ID, const.TILE_NAME]],
        how="left",
        left_on="destination",
        right_on=const.TILE_ID,
        suffixes=("_origin", "_destination"),
    )
    labels = (
        topx_flows_merged["rank"].astype(str)
        + ": "
        + topx_flows_merged[f"{const.TILE_NAME}_origin"]
        + " - "
        + topx_flows_merged[f"{const.TILE_NAME}_destination"]
    )
    moe_base = (
        od_flows_base.margin_of_error_laplace / od_flows_base.data["flow"].sum()
        if od_flows_base.data["flow"].sum() > 0
        else od_flows_base.margin_of_error_laplace
    )
    moe_alterative = (
        od_flows_alternative.margin_of_error_laplace
        / od_flows_alternative.data["flow"].sum()
        if od_flows_alternative.data["flow"].sum() > 0
        else od_flows_alternative.margin_of_error_laplace
    )
    ranking = plot.ranking(
        x=topx_flows_merged.flow_base,
        x_alternative=topx_flows_merged.flow_alternative,
        x_axis_label="percentage of flows per OD pair",
        y_labels=labels,
        margin_of_error=moe_base,
        margin_of_error_alternative=moe_alterative,
        figsize=(8, max(6, min(len(labels) * 0.5, 8))),
    )
    html_ranking = v_utils.fig_to_html(ranking)
    plt.close(ranking)
    return html_ranking


def render_travel_time_hist(
    travel_time_hist: TupleSection,
    travel_time_hist_alternative: Optional[TupleSection] = None,
) -> str:
    if travel_time_hist_alternative:
        alternative_data = travel_time_hist_alternative.data
        alternative_moe = travel_time_hist_alternative.margin_of_error_laplace
    else:
        alternative_data = None
        alternative_moe = None
    hist = plot.histogram(
        hist=travel_time_hist.data,
        hist_alternative=alternative_data,
        x_axis_label="travel time (min.)",
        y_axis_label="% of trips",
        x_axis_type=int,
        margin_of_error=travel_time_hist.margin_of_error_laplace,
        margin_of_error_alternative=alternative_moe,
        figsize=(max(6, min(len(travel_time_hist.data[1]) * 0.5, 10)), 6),
    )
    html_hist = v_utils.fig_to_html(hist)
    plt.close()
    return html_hist


def render_jump_length_hist(
    jump_length_hist: TupleSection,
    jump_length_hist_alternative: Optional[TupleSection] = None,
) -> str:
    if jump_length_hist_alternative:
        alternative_data = jump_length_hist_alternative.data
        alternative_moe = jump_length_hist_alternative.margin_of_error_laplace
    else:
        alternative_data = None
        alternative_moe = None

    hist = plot.histogram(
        hist=jump_length_hist.data,
        hist_alternative=alternative_data,
        x_axis_label="jump length (kilometers)",
        y_axis_label="% of trips",
        x_axis_type=float,
        margin_of_error=jump_length_hist.margin_of_error_laplace,
        margin_of_error_alternative=alternative_moe,
        figsize=(max(6, min(len(jump_length_hist.data[1]) * 0.5, 10)), 6),
    )
    html_hist = v_utils.fig_to_html(hist)
    plt.close()
    return html_hist

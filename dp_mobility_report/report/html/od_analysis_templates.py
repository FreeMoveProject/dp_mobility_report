import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from geopandas.geodataframe import GeoDataFrame

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from dp_mobility_report import constants as const
from dp_mobility_report.model import od_analysis
from dp_mobility_report.model.section import DfSection, TupleSection
from dp_mobility_report.report.html.html_utils import (
    fmt_moe,
    get_template,
    render_eps,
    render_moe_info,
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
        "privacy_info"
    ] = f"""Intra-tile flows below a certain threshold are grayed out: 
        Due to the applied noise, tiles with a low intra-tile flow count are likely to contain a high percentage of noise. 
        For usability reasons, such unrealistic values are grayed out. 
        More specifically: The threshold is set so that values for tiles with a 5% chance (or higher) of deviating more than {round(THRESHOLD * 100)} percentage points from the estimated value are not shown."""
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
        args["flows_summary_table"] = render_summary(
            quartiles.astype(int),
            "Distribution of flows per OD pair",
        )
        args["flows_cumsum_linechart"] = render_flows_cumsum(report[const.OD_FLOWS])
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
        args["travel_time_moe_info"] = render_moe_info(
            report[const.TRAVEL_TIME].margin_of_error_expmech
        )
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
        args["jump_length_moe_info"] = render_moe_info(
            report[const.JUMP_LENGTH].margin_of_error_expmech
        )
        args["jump_length_summary_table"] = render_summary(
            report[const.JUMP_LENGTH].quartiles
        )

    template_structure = get_template("od_analysis_segment.html")
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
    tessellation_innerflow = pd.merge(
        tessellation,
        intra_tile_flows,
        how="left",
        left_on=const.TILE_ID,
        right_on=const.ORIGIN,
    )

    intra_tile_basemap, intra_tile_legend = plot.choropleth_map(
        tessellation_innerflow, const.FLOW, "intra-tile flows"
    )  # get innerflows as color for choropleth

    # create od flows map
    top_n_flows = top_n_flows if top_n_flows <= len(data) else len(data)
    flows = data[
        (data[const.ORIGIN] != data[const.DESTINATION]) & data[const.FLOW].notna()
    ].nlargest(top_n_flows, const.FLOW)
    # only plot lines if there are any flows between tiles
    if not flows[const.FLOW].isnull().all():
        intra_tile_basemap = plot.flows(intra_tile_basemap, flows, tessellation)

    intra_tile_basemap.save(os.path.join(temp_map_folder, "od_map.html"))

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


def render_flows_cumsum(od_flows: DfSection) -> str:
    df_cumsum = od_flows.cumsum

    chart = plot.linechart(
        df_cumsum,
        "n",
        "cum_perc",
        "Number of OD tile pairs",
        "Cumulated sum of flows between OD pairs",
        add_diagonal=True,
    )
    html = v_utils.fig_to_html(chart)
    plt.close()
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
        + " - \n"
        + topx_flows[f"{const.TILE_NAME}_destination"]
    )

    ranking = plot.ranking(
        topx_flows.flow,
        "Number of flows per OD pair",
        y_labels=labels,
        margin_of_error=od_flows.margin_of_error_laplace,
    )
    html_ranking = v_utils.fig_to_html(ranking)
    plt.close()
    return html_ranking


def render_travel_time_hist(travel_time_hist: TupleSection) -> str:
    hist = plot.histogram(
        travel_time_hist.data,
        x_axis_label="travel time (min.)",
        y_axis_label="% of trips",
        x_axis_type=int,
        margin_of_error=travel_time_hist.margin_of_error_laplace,
    )
    html_hist = v_utils.fig_to_html(hist)
    plt.close()
    return html_hist


def render_jump_length_hist(jump_length_hist: TupleSection) -> str:
    hist = plot.histogram(
        jump_length_hist.data,
        x_axis_label="jump length (kilometers)",
        y_axis_label="% of trips",
        x_axis_type=float,
        margin_of_error=jump_length_hist.margin_of_error_laplace,
    )
    html_hist = v_utils.fig_to_html(hist)
    plt.close()
    return html_hist

from typing import TYPE_CHECKING, Tuple
from scipy.stats import laplace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skmob
from geopandas.geodataframe import GeoDataFrame

if TYPE_CHECKING:
    from dp_mobility_report.md_report import MobilityDataReport

from dp_mobility_report.model.section import Section
from dp_mobility_report import constants as const
from dp_mobility_report.model import od_analysis
from dp_mobility_report.report.html.html_utils import (
    fmt,
    get_template,
    render_outlier_info,
    render_summary,
)
from dp_mobility_report.visualization import plot, v_utils


def render_od_analysis(mdreport: "MobilityDataReport", top_n_flows: int) -> str:
    privacy_info = "Unrealistic values: OD connections with a 5% chance of deviating more than 10 percentage points from the estimated value are removed in the map view."
    od_map = ""
    od_legend = ""
    intra_tile_flows_info = ""
    flows_summary_table = ""
    flows_cumsum_linechart = ""
    most_freq_flows_ranking = ""
    outlier_count_travel_time_info = ""
    travel_time_hist = ""
    travel_time_summary_table = ""
    outlier_count_jump_length_info = ""
    jump_length_hist = ""
    jump_length_summary_table = ""

    report = mdreport.report

    if const.OD_FLOWS in report and report[const.OD_FLOWS].data is not None:
        od_map, od_legend = render_origin_destination_flows(
            report[const.OD_FLOWS].data.copy(), mdreport.tessellation, top_n_flows
        )
        intra_tile_flows_info = render_intra_tile_flows(report[const.OD_FLOWS].data)
        flows_summary_table = render_summary(
            report[const.OD_FLOWS].data.flow.describe()
        )
        flows_cumsum_linechart = render_flows_cumsum(report[const.OD_FLOWS].data)
        most_freq_flows_ranking = render_most_freq_flows_ranking(
            report[const.OD_FLOWS], mdreport.tessellation
        )

    if const.TRAVEL_TIME in report and report[const.TRAVEL_TIME].data is not None:
        outlier_count_travel_time_info = render_outlier_info(
            report[const.TRAVEL_TIME].n_outliers,
            mdreport.max_travel_time,
        )
        travel_time_hist = render_travel_time_hist(report[const.TRAVEL_TIME])
        travel_time_summary_table = render_summary(report[const.TRAVEL_TIME].quartiles)

    if const.JUMP_LENGTH in report and report[const.JUMP_LENGTH].data is not None:
        outlier_count_jump_length_info = render_outlier_info(
            report[const.JUMP_LENGTH].n_outliers,
            mdreport.max_jump_length,
        )
        jump_length_hist = render_jump_length_hist(report[const.JUMP_LENGTH])
        jump_length_summary_table = render_summary(report[const.JUMP_LENGTH].quartiles)

    template_structure = get_template("od_analysis_segment.html")
    return template_structure.render(
        privacy_info=privacy_info,
        od_map=od_map,
        od_legend=od_legend,
        intra_tile_flows_info=intra_tile_flows_info,
        flows_summary_table=flows_summary_table,
        flows_cumsum_linechart=flows_cumsum_linechart,
        most_freq_flows_ranking=most_freq_flows_ranking,
        outlier_count_travel_time_info=outlier_count_travel_time_info,
        travel_time_hist=travel_time_hist,
        travel_time_summary_table=travel_time_summary_table,
        outlier_count_jump_length_info=outlier_count_jump_length_info,
        jump_length_hist=jump_length_hist,
        jump_length_summary_table=jump_length_summary_table,
    )


def render_origin_destination_flows(
    od_flows: pd.DataFrame, tessellation: GeoDataFrame, top_n_flows: int, threshold: float = 0.1
) -> Tuple[str, str]:

    od_flows.loc[od_flows.moe_deviation > threshold, "flow"] = None
    top_n_flows = top_n_flows if top_n_flows <= len(od_flows) else len(od_flows)
    innerflow = od_flows[od_flows.origin == od_flows.destination]

    tessellation_innerflow = pd.merge(
        tessellation,
        innerflow,
        how="left",
        left_on=const.TILE_ID,
        right_on="origin",
    )

    fdf = skmob.FlowDataFrame(
        od_flows, tessellation=tessellation_innerflow, tile_id=const.TILE_ID
    )
    
    #tessellation_innerflow.loc[tessellation_innerflow.flow.isna(), "flow"] = 0
    innerflow_chropleth, innerflow_legend = plot.choropleth_map(
        tessellation_innerflow, "flow", "Number of intra-tile flows"
    )  # get innerflows as color for choropleth

    od_map = (
        fdf[fdf.origin != fdf.destination]
        .nlargest(top_n_flows, "flow")
        .plot_flows(flow_color="red", map_f=innerflow_chropleth)
    )
    html = od_map.get_root().render()
    html_legend = v_utils.fig_to_html(innerflow_legend)
    plt.close()
    return html, html_legend


def render_intra_tile_flows(od_flows: pd.DataFrame) -> str:
    flow_count = od_flows.flow.sum()
    intra_tile_flows = od_analysis.get_intra_tile_flows(od_flows)
    return (
        str(intra_tile_flows)
        + " ("
        + str(fmt(intra_tile_flows / flow_count * 100))
        + " %)"
        + " of flows start and end within the same cell."
    )


def render_flows_cumsum(od_flows: pd.DataFrame) -> str:
    df_cumsum = pd.DataFrame()
    df_cumsum["cum_perc"] = round(
        od_flows.flow.sort_values(ascending=False).cumsum() / sum(od_flows.flow), 2
    )
    df_cumsum["n"] = np.arange(1, len(od_flows) + 1)
    df_cumsum.reset_index(drop=True, inplace=True)
    chart = plot.linechart(
        df_cumsum,
        "n",
        "cum_perc",
        "Number of OD tile pairs",
        "Cumulated sum of flows between OD tile pairs",
        add_diagonal=True,
    )
    html = v_utils.fig_to_html(chart)
    plt.close()
    return html


def render_most_freq_flows_ranking(
    od_flows: Section, tessellation: GeoDataFrame, top_x: int = 10
) -> str:
    topx_flows = od_flows.data.nlargest(top_x, "flow")
    topx_flows["rank"] = list(range(1, len(topx_flows) + 1))
    topx_flows = topx_flows.merge(
        tessellation[[const.TILE_ID, const.TILE_NAME]],
        how="left",
        left_on="origin",
        right_on=const.TILE_ID,
    )
    topx_flows = topx_flows.merge(
        tessellation[[const.TILE_ID, const.TILE_NAME]],
        how="left",
        left_on="destination",
        right_on=const.TILE_ID,
        suffixes=("_origin", "_destination"),
    )

    topx_flows_list = []
    for _, row in topx_flows.iterrows():
        topx_flows_list.append(
            {
                "rank": row["rank"],
                "name": str(row[f"{const.TILE_NAME}_origin"])
                + " - "
                + str(row[f"{const.TILE_NAME}_origin"]),
                "lower_bound": str(fmt(round(row["flow"] - od_flows.margin_of_error))),
                "estimate": str(fmt(row["flow"])),
                "upper_bound": str(fmt(round(row["flow"] + od_flows.margin_of_error))),
            }
        )
    template_table = get_template("table_conf_interval.html")
    tile_ranking_html = template_table.render(
        name="Ranking most frequent OD connections", rows=topx_flows_list
    )
    return tile_ranking_html


def render_travel_time_hist(travel_time_hist: Section) -> str:
    hist = plot.histogram(
        travel_time_hist.data, x_axis_label="travel time (min.)", x_axis_type=int, 
        margin_of_error = travel_time_hist.margin_of_error
    )
    html_hist = v_utils.fig_to_html(hist)
    plt.close()
    return html_hist


def render_jump_length_hist(jump_length_hist: Section) -> str:
    hist = plot.histogram(
        jump_length_hist.data, x_axis_label="jump length (kilometers)", x_axis_type=float,
        margin_of_error=jump_length_hist.margin_of_error
    )
    html_hist = v_utils.fig_to_html(hist)
    plt.close()
    return html_hist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skmob

from dp_mobility_report import md_report
from dp_mobility_report.model import od_analysis
from dp_mobility_report.report.html.utils import (
    fmt,
    get_template,
    render_outlier_info,
    render_summary,
)
from dp_mobility_report.visualization import plot, utils


def render_od_analysis(mdreport):
    od_map = None
    intra_tile_flows_info = None
    flows_summary_table = None
    flows_cumsum_linechart = None
    most_freq_flows_ranking = None
    outlier_count_travel_time_info = None
    travel_time_hist = None
    travel_time_summary_table = None
    outlier_count_jump_length_info = None
    jump_length_hist = None
    jump_length_summary_table = None

    report = mdreport.report

    if "od_flows" in report:
        od_map = render_origin_destination_flows(report["od_flows"].data, mdreport)
        intra_tile_flows_info = render_intra_tile_flows(report["od_flows"].data)
        flows_summary_table = render_summary(report["od_flows"].data.flow.describe())
        flows_cumsum_linechart = render_flows_cumsum(report["od_flows"].data)
        most_freq_flows_ranking = render_most_freq_flows_ranking(
            report["od_flows"].data, mdreport.tessellation
        )

    if "travel_time" in report:
        outlier_count_travel_time_info = render_outlier_info(
            report["travel_time"].n_outliers,
            mdreport.max_travel_time,
        )
        travel_time_hist = render_travel_time_hist(report["travel_time"].data)
        travel_time_summary_table = render_summary(
            report["travel_time"].quartiles
        )

    if "jump_length" in report:
        outlier_count_jump_length_info = render_outlier_info(
            report["jump_length"].n_outliers,
            mdreport.max_jump_length,
        )
        jump_length_hist = render_jump_length_hist(report["jump_length"].data)
        jump_length_summary_table = render_summary(
            report["jump_length"].quartiles
        )
    template_structure = get_template("od_analysis_segment.html")
    return template_structure.render(
        od_map=od_map,
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


### render od analysis functions


def render_origin_destination_flows(od_flows, mdreport):
    n_flows = (
        mdreport.top_x_flows if mdreport.top_x_flows <= len(od_flows) else len(od_flows)
    )
    innerflow = od_flows[od_flows.origin == od_flows.destination]

    tessellation_innerflow = pd.merge(
        mdreport.tessellation,
        innerflow,
        how="left",
        left_on="tile_id",
        right_on="origin",
    )

    fdf = skmob.FlowDataFrame(
        od_flows, tessellation=tessellation_innerflow, tile_id="tile_id"
    )
    tessellation_innerflow.loc[tessellation_innerflow.flow.isna(), "flow"] = 0
    innerflow_chropleth = plot.choropleth_map(
        tessellation_innerflow, "flow", "Number of intra-tile flows"
    )  # get innerflows as color for choropleth

    od_map = (
        fdf[fdf.origin != fdf.destination]
        .nlargest(n_flows, "flow")
        .plot_flows(flow_color="red", map_f=innerflow_chropleth)
    )
    html = od_map.get_root().render()
    plt.close()
    return html


def render_intra_tile_flows(od_flows):
    flow_count = od_flows.flow.sum()
    intra_tile_flows = od_analysis.get_intra_tile_flows(od_flows)
    return (
        str(intra_tile_flows)
        + " ("
        + str(fmt(intra_tile_flows / flow_count * 100))
        + " %)"
        + " of flows start and end within the same cell."
    )


def render_flows_cumsum(od_flows):
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
    html = utils.fig_to_html(chart)
    plt.close()
    return html


# TODO: decide on topx
def render_most_freq_flows_ranking(od_flows, tessellation, top_x=10):
    topx_flows = od_flows.nlargest(top_x, "flow")
    topx_flows["rank"] = list(range(1, len(topx_flows) + 1))
    topx_flows = topx_flows.merge(
        tessellation[["tile_id", "tile_name"]],
        how="left",
        left_on="origin",
        right_on="tile_id",
    )
    topx_flows = topx_flows.merge(
        tessellation[["tile_id", "tile_name"]],
        how="left",
        left_on="destination",
        right_on="tile_id",
        suffixes=("_origin", "_destination"),
    )

    topx_flows_list = []
    for _, row in topx_flows.iterrows():
        topx_flows_list.append(
            {
                "name": row["rank"],
                "value": str(row["tile_name_origin"])
                + " - "
                + str(row["tile_name_destination"])
                + ": "
                + str(row["flow"]),
            }
        )
    template_table = get_template("table.html")
    tile_ranking_html = template_table.render(
        name="Ranking most frequent OD connections", rows=topx_flows_list
    )
    return tile_ranking_html


def render_travel_time_hist(travel_time_hist):
    hist = plot.histogram(
        travel_time_hist, x_axis_label="travel time (min.)", x_axis_type=int
    )
    html_hist = utils.fig_to_html(hist)
    plt.close()
    return html_hist


def render_jump_length_hist(jump_length_hist):
    hist = plot.histogram(
        jump_length_hist, x_axis_label="jump length (meters)", x_axis_type=int
    )
    html_hist = utils.fig_to_html(hist)
    plt.close()
    return html_hist

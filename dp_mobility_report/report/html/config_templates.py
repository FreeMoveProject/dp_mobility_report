from typing import TYPE_CHECKING

import pandas as pd

import dp_mobility_report.constants as const
from dp_mobility_report.report.html.html_utils import fmt, get_template

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport


def render_config(dpmreport: "DpMobilityReport") -> str:

    args: dict = {}

    args["config_table"] = render_config_table(dpmreport)
    args["privacy_info"] = render_privacy_info(dpmreport.privacy_budget is not None)
    if (dpmreport.tessellation is None):
        args["tessellation_info"] = "<br> <br>No tessellation has been provided. All analyses based on the tessellation have been excluded."

    if not pd.core.dtypes.common.is_datetime64_dtype(dpmreport.df[const.DATETIME]):
        args["timestamp_info"] = "<br> <br>Dataframe does not contain timestamps. All analyses based on timestamps have been excluded."

    if  max(dpmreport.df[const.TID].value_counts()) == 1:
        args["od_info"] = "<br> <br>All trips in the dataset only contain a single record, therefore, all origin-destination analyses (OD Flows, travel time, jump length) have been excluded."

    template_structure = get_template("config_segment.html")
    return template_structure.render(args)


def render_config_table(dpmreport: "DpMobilityReport") -> str:

    config_list = [
        {"name": "Max. trips per user", "value": fmt(dpmreport.max_trips_per_user)},
        {"name": "Privacy budget", "value": fmt(dpmreport.privacy_budget)},
        {"name": "User privacy", "value": fmt(dpmreport.user_privacy)},
        {"name": "Excluded analyses", "value": fmt(dpmreport.analysis_exclusion)},
        {"name": "Budget split", "value": fmt(dpmreport.budget_split)},
        {"name": "Evaluation dev. mode", "value": fmt(dpmreport.evalu)},
    ]

    # create html from template
    template_table = get_template("table.html")
    dataset_stats_html = template_table.render(rows=config_list)
    return dataset_stats_html


def render_privacy_info(with_privacy: bool) -> str:
    if with_privacy:
        return "Noise has been added to provide differential privacy. The 95%-confidence interval gives an intuition about the reliability of the noisy results."
    else:
        return "No noise has been added."

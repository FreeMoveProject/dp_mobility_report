from typing import TYPE_CHECKING

import pandas as pd

import dp_mobility_report.constants as const
from dp_mobility_report.model.preprocessing import has_points_inside_tessellation
from dp_mobility_report.report.html.html_utils import fmt, fmt_config, get_template

if TYPE_CHECKING:
    from dp_mobility_report import BenchmarkReport, DpMobilityReport


def render_config(dpmreport: "DpMobilityReport") -> str:

    args: dict = {}

    args["config_table"] = render_config_table(dpmreport)
    args["privacy_info"] = render_privacy_info(dpmreport.privacy_budget is not None)
    if dpmreport.tessellation is None:
        args[
            "tessellation_info"
        ] = "No tessellation has been provided. All analyses based on the tessellation have been excluded."
    elif not has_points_inside_tessellation(dpmreport.df, dpmreport.tessellation):
        args[
            "tessellation_info"
        ] = "No records are within the given tessellation. All analyses based on the tessellation have been excluded."

    if not pd.core.dtypes.common.is_datetime64_dtype(dpmreport.df[const.DATETIME]):
        args[
            "timestamp_info"
        ] = "Dataframe does not contain timestamps. All analyses based on timestamps have been excluded."

    if max(dpmreport.df[const.TID].value_counts()) == 1:
        args[
            "od_info"
        ] = "All trips in the dataset only contain a single record, therefore, all origin-destination analyses (OD Flows, travel time, jump length) have been excluded."

    template_structure = get_template("config_segment.html")
    return template_structure.render(args)


def render_benchmark_config(benchmarkreport: "BenchmarkReport") -> str:

    args: dict = {}

    args["config_table"] = render_benchmark_config_table(benchmarkreport)

    if benchmarkreport.report_base.tessellation is None:
        args[
            "tessellation_info"
        ] = "No tessellation has been provided. All analyses based on the tessellation have been excluded."
    elif (
        not has_points_inside_tessellation(
            benchmarkreport.report_base.df, benchmarkreport.report_base.tessellation
        )
    ) | (
        not has_points_inside_tessellation(
            benchmarkreport.report_alternative.df,
            benchmarkreport.report_alternative.tessellation,
        )
    ):
        args[
            "tessellation_info"
        ] = "No records are within the given tessellation. All analyses based on the tessellation have been excluded."

    if (
        not pd.core.dtypes.common.is_datetime64_dtype(
            benchmarkreport.report_base.df[const.DATETIME]
        )
    ) or (
        not pd.core.dtypes.common.is_datetime64_dtype(
            benchmarkreport.report_alternative.df[const.DATETIME]
        )
    ):
        args[
            "timestamp_info"
        ] = "At least one of the datasets does not contain timestamps. All analyses based on timestamps have been excluded."

    if (max(benchmarkreport.report_base.df[const.TID].value_counts()) == 1) or (
        max(benchmarkreport.report_alternative.df[const.TID].value_counts()) == 1
    ):
        args[
            "od_info"
        ] = "All trips in at least one of the datasets only contain a single record, therefore, all origin-destination analyses (OD Flows, travel time, jump length) have been excluded."

    template_structure = get_template("config_segment.html")
    return template_structure.render(args)


def render_similarity_info() -> str:

    template_structure = get_template("similarity_info.html")
    return template_structure.render()


def render_dp_info() -> str:

    template_structure = get_template("dp_info.html")
    return template_structure.render()


def render_config_table(dpmreport: "DpMobilityReport") -> str:

    config_list = [
        {"name": "Max. trips per user", "value": fmt(dpmreport.max_trips_per_user)},
        {"name": "Privacy budget", "value": fmt(dpmreport.privacy_budget)},
        {"name": "User privacy", "value": fmt(dpmreport.user_privacy)},
        {"name": "Budget split", "value": fmt_config(dpmreport.budget_split)},
        {"name": "Evaluation dev. mode", "value": fmt(dpmreport.evalu)},
        {
            "name": "Excluded analyses",
            "value": fmt_config(dpmreport.analysis_exclusion),
        },
    ]

    # create html from template
    template_table = get_template("table.html")
    dataset_stats_html = template_table.render(rows=config_list, align="left-align")
    return dataset_stats_html


def render_benchmark_config_table(benchmarkreport: "BenchmarkReport") -> str:

    config_list = [
        {
            "name": "Max. trips per user",
            "value": (
                fmt(benchmarkreport.report_base.max_trips_per_user),
                fmt(benchmarkreport.report_alternative.max_trips_per_user),
            ),
        },
        {
            "name": "Privacy budget",
            "value": (
                fmt(benchmarkreport.report_base.privacy_budget),
                fmt(benchmarkreport.report_alternative.privacy_budget),
            ),
        },
        {
            "name": "User privacy",
            "value": (
                fmt(benchmarkreport.report_base.user_privacy),
                fmt(benchmarkreport.report_alternative.privacy_budget),
            ),
        },
        {
            "name": "Budget split",
            "value": (
                fmt_config(benchmarkreport.report_base.budget_split),
                fmt_config(benchmarkreport.report_alternative.budget_split),
            ),
        },
        {
            "name": "Evaluation dev. mode",
            "value": (
                fmt(benchmarkreport.report_base.evalu),
                fmt(benchmarkreport.report_alternative.evalu),
            ),
        },
        {
            "name": "Excluded analyses",
            "value": (fmt_config(benchmarkreport.analysis_exclusion),),
        },
    ]

    # create html from template
    template_table = get_template("table_benchmark.html")
    dataset_stats_html = template_table.render(name="Configuration", rows=config_list)
    return dataset_stats_html


def render_privacy_info(with_privacy: bool) -> str:
    if with_privacy:
        return "Noise has been added to provide differential privacy. The 95%-confidence interval gives an intuition about the reliability of the noisy results."
    else:
        return "No noise has been added."

from typing import TYPE_CHECKING

import pandas as pd

import dp_mobility_report.constants as const
from dp_mobility_report.report.html.html_utils import fmt, get_template

if TYPE_CHECKING:
    from dp_mobility_report import BenchmarkReport, DpMobilityReport


def render_config(dpmreport: "DpMobilityReport") -> str:

    args: dict = {}

    args["config_table"] = render_config_table(dpmreport)
    args["privacy_info"] = render_privacy_info(dpmreport.privacy_budget is not None)
    args["tessellation_info"] = (
        ""
        if (dpmreport.tessellation is not None)
        else "<br> <br>No tessellation has been provided. All analyses based on the tessellation have been excluded."
    )
    args["timestamp_info"] = (
        ""
        if pd.core.dtypes.common.is_datetime64_dtype(dpmreport.df[const.DATETIME])
        else "<br> <br>Dataframe does not contain timestamps. All analyses based on timestamps have been excluded."
    )
    template_structure = get_template("config_segment.html")
    return template_structure.render(args)


def render_benchmark_config(benchmarkreport: "BenchmarkReport") -> str:

    args: dict = {}

    args["config_table"] = render_benchmark_config_table(benchmarkreport)
    # args["privacy_info"] = render_privacy_info(benchmarkreport.privacy_budget is not None)
    # args["tessellation_info"] = (
    #     ""
    #     if (benchmarkreport.tessellation is not None)
    #     else "<br> <br>No tessellation has been provided. All analyses based on the tessellation have been excluded."
    # )
    # args["timestamp_info"] = (
    #     ""
    #     if pd.core.dtypes.common.is_datetime64_dtype(benchmarkreport.df[const.DATETIME])
    #     else "<br> <br>Dataframe does not contain timestamps. All analyses based on timestamps have been excluded."
    # )

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
                fmt(benchmarkreport.report_base.budget_split),
                fmt(benchmarkreport.report_alternative.budget_split),
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
            "value": (
                fmt(benchmarkreport.analysis_exclusion),
                fmt(benchmarkreport.analysis_exclusion),
            ),
        },  # TODO verbundene zelle
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

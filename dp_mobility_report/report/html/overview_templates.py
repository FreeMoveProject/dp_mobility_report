from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import DfSection, DictSection, SeriesSection
from dp_mobility_report.report.html.html_utils import (
    all_available_measures,
    fmt,
    fmt_moe,
    get_template,
    render_benchmark_summary,
    render_eps,
    render_summary,
)

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from dp_mobility_report.visualization import plot, v_utils

if TYPE_CHECKING:
    from dp_mobility_report import BenchmarkReport


def render_overview(dpmreport: "DpMobilityReport") -> str:
    args: dict = {}
    report = dpmreport.report

    if const.DS_STATISTICS not in dpmreport.analysis_exclusion:
        args["dataset_stats_table"] = render_dataset_statistics(
            report[const.DS_STATISTICS]
        )
        args["dataset_stats_eps"] = (
            render_eps(report[const.DS_STATISTICS].privacy_budget),
        )

    if const.MISSING_VALUES not in dpmreport.analysis_exclusion:
        args["missing_values_table"] = render_missing_values(
            report[const.MISSING_VALUES]
        )
        args["missing_values_eps"] = (
            render_eps(report[const.MISSING_VALUES].privacy_budget),
        )

    if const.TRIPS_OVER_TIME not in dpmreport.analysis_exclusion:
        args["trips_over_time_eps"] = render_eps(
            report[const.TRIPS_OVER_TIME].privacy_budget
        )
        args["trips_over_time_moe"] = fmt_moe(
            report[const.TRIPS_OVER_TIME].margin_of_error_laplace
        )
        args["trips_over_time_info"] = render_trips_over_time_info(
            report[const.TRIPS_OVER_TIME].datetime_precision
        )
        args["trips_over_time_linechart"] = render_trips_over_time(
            report[const.TRIPS_OVER_TIME]
        )
        args["trips_over_time_summary_table"] = render_summary(
            report[const.TRIPS_OVER_TIME].quartiles
        )

    if const.TRIPS_PER_WEEKDAY not in dpmreport.analysis_exclusion:
        args["trips_per_weekday_eps"] = render_eps(
            report[const.TRIPS_PER_WEEKDAY].privacy_budget
        )
        args["trips_per_weekday_moe"] = fmt_moe(
            report[const.TRIPS_PER_WEEKDAY].margin_of_error_laplace
        )

        args["trips_per_weekday_barchart"] = render_trips_per_weekday(
            report[const.TRIPS_PER_WEEKDAY]
        )

    if const.TRIPS_PER_HOUR not in dpmreport.analysis_exclusion:
        args["trips_per_hour_eps"] = render_eps(
            report[const.TRIPS_PER_HOUR].privacy_budget
        )
        args["trips_per_hour_moe"] = fmt_moe(
            report[const.TRIPS_PER_HOUR].margin_of_error_laplace
        )

        args["trips_per_hour_linechart"] = render_trips_per_hour(
            report[const.TRIPS_PER_HOUR].data, margin_of_error=None
        )

    template_structure = get_template("overview_segment.html")
    return template_structure.render(args)


def render_benchmark_overview(benchmark: "BenchmarkReport") -> str:

    args: dict = {}
    report_base = benchmark.report_base.report
    report_alternative = benchmark.report_alternative.report
    template_measures = get_template("similarity_measures.html")

    if const.DS_STATISTICS not in benchmark.analysis_exclusion:
        args["dataset_stats_table"] = render_benchmark_dataset_statistics(
            report_base[const.DS_STATISTICS],
            report_alternative[const.DS_STATISTICS],
            benchmark.smape,
        )
        args["dataset_stats_eps"] = (
            render_eps(report_base[const.DS_STATISTICS].privacy_budget),
            render_eps(report_alternative[const.DS_STATISTICS].privacy_budget),
        )

    if const.MISSING_VALUES not in benchmark.analysis_exclusion:
        args["missing_values_table"] = render_benchmark_missing_values(
            report_base[const.MISSING_VALUES],
            report_alternative[const.MISSING_VALUES],
            benchmark.smape,
        )
        args["missing_values_eps"] = (
            render_eps(report_base[const.MISSING_VALUES].privacy_budget),
            render_eps(report_alternative[const.MISSING_VALUES].privacy_budget),
        )

    if const.TRIPS_OVER_TIME not in benchmark.analysis_exclusion:
        args["trips_over_time_eps"] = (
            render_eps(report_base[const.TRIPS_OVER_TIME].privacy_budget),
            render_eps(report_alternative[const.TRIPS_OVER_TIME].privacy_budget),
        )
        args["trips_over_time_moe"] = (
            fmt_moe(report_base[const.TRIPS_OVER_TIME].margin_of_error_laplace),
            fmt_moe(report_alternative[const.TRIPS_OVER_TIME].margin_of_error_laplace),
        )
        args["trips_over_time_info"] = render_trips_over_time_info(
            report_base[const.TRIPS_OVER_TIME].datetime_precision
        )
        args["trips_over_time_linechart"] = render_benchmark_trips_over_time(
            report_base[const.TRIPS_OVER_TIME],
            report_alternative[const.TRIPS_OVER_TIME],
        )
        args["trips_over_time_summary_table"] = render_benchmark_summary(
            report_base[const.TRIPS_OVER_TIME].quartiles,
            report_alternative[const.TRIPS_OVER_TIME].quartiles,
        )
        args["trips_over_time_measure"] = template_measures.render(
            all_available_measures(const.TRIPS_OVER_TIME, benchmark)
        )

    if const.TRIPS_PER_WEEKDAY not in benchmark.analysis_exclusion:
        args["trips_per_weekday_eps"] = (
            render_eps(report_base[const.TRIPS_PER_WEEKDAY].privacy_budget),
            render_eps(report_alternative[const.TRIPS_PER_WEEKDAY].privacy_budget),
        )
        args["trips_per_weekday_moe"] = (
            fmt_moe(report_base[const.TRIPS_PER_WEEKDAY].margin_of_error_laplace),
            fmt_moe(
                report_alternative[const.TRIPS_PER_WEEKDAY].margin_of_error_laplace
            ),
        )

        args["trips_per_weekday_barchart"] = render_benchmark_trips_per_weekday(
            report_base[const.TRIPS_PER_WEEKDAY],
            report_alternative[const.TRIPS_PER_WEEKDAY],
        )
        args["trips_per_weekday_measure"] = template_measures.render(
            all_available_measures(const.TRIPS_PER_WEEKDAY, benchmark)
        )

    if const.TRIPS_PER_HOUR not in benchmark.analysis_exclusion:
        args["trips_per_hour_eps"] = (
            render_eps(report_base[const.TRIPS_PER_HOUR].privacy_budget),
            render_eps(report_alternative[const.TRIPS_PER_HOUR].privacy_budget),
        )
        args["trips_per_hour_moe"] = (
            fmt_moe(report_base[const.TRIPS_PER_HOUR].margin_of_error_laplace),
            fmt_moe(report_alternative[const.TRIPS_PER_HOUR].margin_of_error_laplace),
        )

        dataset = np.append(
            np.repeat("base", len(report_base[const.TRIPS_PER_HOUR].data)),
            np.repeat(
                "alternative", len(report_alternative[const.TRIPS_PER_HOUR].data)
            ),
        )
        combined_trips_per_hour = pd.concat(
            [
                report_base[const.TRIPS_PER_HOUR].data,
                report_alternative[const.TRIPS_PER_HOUR].data,
            ]
        )
        combined_trips_per_hour["dataset"] = dataset
        args["trips_per_hour_linechart"] = render_trips_per_hour(
            combined_trips_per_hour, margin_of_error=None, style="dataset"
        )
        args["trips_per_hour_measure"] = template_measures.render(
            all_available_measures(const.TRIPS_PER_HOUR, benchmark)
        )

    template_structure = get_template("overview_segment_benchmark.html")
    return template_structure.render(args)


def render_dataset_statistics(dataset_statistics: DictSection) -> str:
    moe = dataset_statistics.margin_of_errors_laplace
    data = dataset_statistics.data
    dataset_stats_list = [
        {
            "name": "Number of records",
            "estimate": fmt(data[const.N_RECORDS], target_type=int),
            "margin_of_error": fmt_moe(moe[const.N_RECORDS]),
        },
        {
            "name": "Number of distinct trips",
            "estimate": fmt(data[const.N_TRIPS], target_type=int),
            "margin_of_error": fmt_moe(moe[const.N_TRIPS]),
        },
        {
            "name": "Number of complete trips (start and and point)",
            "estimate": fmt(data[const.N_COMPLETE_TRIPS], target_type=int),
            "margin_of_error": fmt_moe(moe[const.N_COMPLETE_TRIPS]),
        },
        {
            "name": "Number of incomplete trips (single point)",
            "estimate": fmt(data[const.N_INCOMPLETE_TRIPS], target_type=int),
            "margin_of_error": fmt_moe(moe[const.N_INCOMPLETE_TRIPS]),
        },
        {
            "name": "Number of distinct users",
            "estimate": fmt(data[const.N_USERS], target_type=int),
            "margin_of_error": fmt_moe(moe[const.N_USERS]),
        },
        {
            "name": "Number of distinct locations (lat & lon combination)",
            "estimate": fmt(data[const.N_LOCATIONS], target_type=int),
            "margin_of_error": fmt_moe(moe[const.N_LOCATIONS]),
        },
    ]

    # create html from template
    template_table = get_template("table_conf_interval.html")
    dataset_stats_html = template_table.render(
        rows=dataset_stats_list,
    )
    return dataset_stats_html


def render_benchmark_dataset_statistics(
    dataset_statistics_base: DictSection,
    dataset_statistics_alternative: DictSection,
    smape: dict,
) -> str:
    moe_base = dataset_statistics_base.margin_of_errors_laplace
    moe_alternative = dataset_statistics_alternative.margin_of_errors_laplace
    data_base = dataset_statistics_base.data
    data_alternative = dataset_statistics_alternative.data

    dataset_stats_list = [
        {
            "name": "Number of records",
            "estimate": (
                fmt(data_base[const.N_RECORDS], target_type=int),
                fmt(data_alternative[const.N_RECORDS], target_type=int),
            ),
            "margin_of_error": (
                fmt_moe(moe_base[const.N_RECORDS]),
                fmt_moe(moe_alternative[const.N_RECORDS]),
            ),
            const.SMAPE: fmt(smape[const.N_RECORDS], target_type=float),
        },
        {
            "name": "Number of distinct trips",
            "estimate": (
                fmt(data_base[const.N_TRIPS], target_type=int),
                fmt(data_alternative[const.N_TRIPS], target_type=int),
            ),
            "margin_of_error": (
                fmt_moe(moe_base[const.N_TRIPS]),
                fmt_moe(moe_alternative[const.N_TRIPS]),
            ),
            const.SMAPE: fmt(smape[const.N_TRIPS], target_type=float),
        },
        {
            "name": "Number of complete trips (start and and point)",
            "estimate": (
                fmt(data_base[const.N_COMPLETE_TRIPS], target_type=int),
                fmt(data_alternative[const.N_COMPLETE_TRIPS], target_type=int),
            ),
            "margin_of_error": (
                fmt_moe(moe_base[const.N_COMPLETE_TRIPS]),
                fmt_moe(moe_alternative[const.N_COMPLETE_TRIPS]),
            ),
            const.SMAPE: fmt(smape[const.N_COMPLETE_TRIPS], target_type=float),
        },
        {
            "name": "Number of incomplete trips (single point)",
            "estimate": (
                fmt(data_base[const.N_INCOMPLETE_TRIPS], target_type=int),
                fmt(data_alternative[const.N_INCOMPLETE_TRIPS], target_type=int),
            ),
            "margin_of_error": (
                fmt_moe(moe_base[const.N_INCOMPLETE_TRIPS]),
                fmt_moe(moe_alternative[const.N_INCOMPLETE_TRIPS]),
            ),
            const.SMAPE: fmt(smape[const.N_INCOMPLETE_TRIPS], target_type=float),
        },
        {
            "name": "Number of distinct users",
            "estimate": (
                fmt(data_base[const.N_USERS], target_type=int),
                fmt(data_alternative[const.N_USERS], target_type=int),
            ),
            "margin_of_error": (
                fmt_moe(moe_base[const.N_USERS]),
                fmt_moe(moe_alternative[const.N_USERS]),
            ),
            const.SMAPE: fmt(smape[const.N_USERS], target_type=float),
        },
        {
            "name": "Number of distinct locations (lat & lon combination)",
            "estimate": (
                fmt(data_base[const.N_LOCATIONS], target_type=int),
                fmt(data_alternative[const.N_LOCATIONS], target_type=int),
            ),
            "margin_of_error": (
                fmt_moe(moe_base[const.N_LOCATIONS]),
                fmt_moe(moe_alternative[const.N_LOCATIONS]),
            ),
            const.SMAPE: fmt(smape[const.N_LOCATIONS], target_type=float),
        },
    ]

    # create html from template
    template_table = get_template("table_conf_interval_benchmark.html")
    dataset_stats_html = template_table.render(
        rows=dataset_stats_list,
    )
    return dataset_stats_html


def render_missing_values(missing_values: DictSection) -> str:
    moe = round(missing_values.margin_of_error_laplace, 1)
    data = missing_values.data
    missing_values_list = [
        {
            "name": "User ID (uid)",
            "estimate": fmt(data[const.UID], target_type=int),
            "margin_of_error": fmt_moe(moe),
        },
        {
            "name": "Trip ID (tid)",
            "estimate": fmt(data[const.TID], target_type=int),
            "margin_of_error": fmt_moe(moe),
        },
        {
            "name": "Timestamp (datetime)",
            "estimate": fmt(data[const.DATETIME], target_type=int),
            "margin_of_error": fmt_moe(moe),
        },
        {
            "name": "Latitude (lat)",
            "estimate": fmt(data[const.LAT], target_type=int),
            "margin_of_error": fmt_moe(moe),
        },
        {
            "name": "Longitude (lng)",
            "estimate": fmt(data[const.LNG], target_type=int),
            "margin_of_error": fmt_moe(moe),
        },
    ]

    template_table = get_template("table_conf_interval.html")
    missing_values_html = template_table.render(
        privacy_budget=render_eps(missing_values.privacy_budget),
        rows=missing_values_list,
    )
    return missing_values_html


def render_benchmark_missing_values(
    missing_values_base: DictSection,
    missing_values_alternative: DictSection,
    smape: dict,
) -> str:
    moe_base = round(missing_values_base.margin_of_error_laplace, 1)
    moe_alternative = round(missing_values_alternative.margin_of_error_laplace, 1)
    data_base = missing_values_base.data
    data_alternative = missing_values_alternative.data

    missing_values_list = [
        {
            "name": "User ID (uid)",
            "estimate": (
                fmt(data_base[const.UID], target_type=int),
                fmt(data_alternative[const.UID], target_type=int),
            ),
            "margin_of_error": (fmt_moe(moe_base), fmt_moe(moe_alternative)),
            const.SMAPE: fmt(smape[const.UID], target_type=float),
        },
        {
            "name": "Trip ID (tid)",
            "estimate": (
                fmt(data_base[const.TID], target_type=int),
                fmt(data_alternative[const.TID], target_type=int),
            ),
            "margin_of_error": (fmt_moe(moe_base), fmt_moe(moe_alternative)),
            const.SMAPE: fmt(smape[const.TID], target_type=float),
        },
        {
            "name": "Timestamp (datetime)",
            "estimate": (
                fmt(data_base[const.DATETIME], target_type=int),
                fmt(data_alternative[const.DATETIME], target_type=int),
            ),
            "margin_of_error": (fmt_moe(moe_base), fmt_moe(moe_alternative)),
            const.SMAPE: fmt(smape[const.DATETIME], target_type=float),
        },
        {
            "name": "Latitude (lat)",
            "estimate": (
                fmt(data_base[const.LAT], target_type=int),
                fmt(data_alternative[const.LAT], target_type=int),
            ),
            "margin_of_error": (fmt_moe(moe_base), fmt_moe(moe_alternative)),
            const.SMAPE: fmt(smape[const.LAT], target_type=float),
        },
        {
            "name": "Longitude (lng)",
            "estimate": (
                fmt(data_base[const.LNG], target_type=int),
                fmt(data_alternative[const.LNG], target_type=int),
            ),
            "margin_of_error": (fmt_moe(moe_base), fmt_moe(moe_alternative)),
            const.SMAPE: fmt(smape[const.LNG], target_type=float),
        },
    ]

    template_table = get_template("table_conf_interval_benchmark.html")
    missing_values_html = template_table.render(
        rows=missing_values_list,
    )

    return missing_values_html


def render_trips_over_time_info(datetime_precision: str) -> str:
    return f"Timestamps have been aggregated by {datetime_precision}."


def render_trips_over_time(trips_over_time: DfSection) -> str:
    if len(trips_over_time.data) <= 14:
        chart = plot.barchart(
            x=trips_over_time.data[const.DATETIME].to_numpy(),
            y=trips_over_time.data["trips"].to_numpy(),
            margin_of_error=trips_over_time.margin_of_error_laplace,
            x_axis_label="Date",
            y_axis_label="% of trips",
            rotate_label=True,
            figsize=(max(5, min(9, len(trips_over_time.data.values))), 6),
        )
        html = v_utils.fig_to_html(chart)
    else:
        chart = plot.linechart(
            data=trips_over_time.data,
            x=const.DATETIME,
            y="trips",
            x_axis_label="Date",
            y_axis_label="% of trips",
            margin_of_error=trips_over_time.margin_of_error_laplace,
            rotate_label=True,
            figsize=(9, 6),
        )
        html = v_utils.fig_to_html(chart)
    plt.close(chart)
    return html


def render_benchmark_trips_over_time(
    trips_over_time: DfSection, trips_over_time_alternative: DfSection
) -> str:
    if len(trips_over_time.data) <= 14:
        chart = plot.barchart(
            x=trips_over_time.data[const.DATETIME].to_numpy(),
            y=trips_over_time.data["trips"].to_numpy(),
            y_alternative=trips_over_time_alternative.data["trips"].to_numpy(),
            margin_of_error=trips_over_time.margin_of_error_laplace,
            margin_of_error_alternative=trips_over_time_alternative.margin_of_error_laplace,
            x_axis_label="Date",
            y_axis_label="% of trips",
            rotate_label=True,
            figsize=(max(5, min(9, len(trips_over_time.data.values))), 6),
        )
        html = v_utils.fig_to_html(chart)

    else:
        chart = plot.linechart_new(
            data=trips_over_time.data,
            x=const.DATETIME,
            data_alternative=trips_over_time_alternative.data,
            y="trips",
            x_axis_label="Date",
            y_axis_label="% of trips",
            margin_of_error=trips_over_time.margin_of_error_laplace,
            margin_of_error_alternative=trips_over_time_alternative.margin_of_error_laplace,
            rotate_label=True,
            figsize=(9, 6),
        )
        html = v_utils.fig_to_html(chart)
    plt.close(chart)
    return html


def render_trips_per_weekday(trips_per_weekday: SeriesSection) -> str:
    chart = plot.barchart(
        x=trips_per_weekday.data.index.to_numpy(),
        y=trips_per_weekday.data.values,
        margin_of_error=trips_per_weekday.margin_of_error_laplace,
        x_axis_label="Weekday",
        y_axis_label="% of trips",
        rotate_label=True,
        figsize=(max(5, min(9, len(trips_per_weekday.data.values))), 6),
    )
    plt.close(chart)
    return v_utils.fig_to_html(chart)


def render_benchmark_trips_per_weekday(
    trips_per_weekday: SeriesSection, trips_per_weekday_alternative: SeriesSection
) -> str:
    chart = plot.barchart(
        x=trips_per_weekday.data.index.to_numpy(),
        y=trips_per_weekday.data.values,
        y_alternative=trips_per_weekday_alternative.data.values,
        margin_of_error=trips_per_weekday.margin_of_error_laplace,
        margin_of_error_alternative=trips_per_weekday_alternative.margin_of_error_laplace,
        x_axis_label="Weekday",
        y_axis_label="% of trips",
        rotate_label=True,
        figsize=(max(5, min(9, len(trips_per_weekday.data.values))), 6),
    )
    plt.close(chart)
    return v_utils.fig_to_html(chart)


def render_trips_per_hour(
    trips_per_hour: pd.DataFrame,
    margin_of_error: Optional[float],
    style: Optional[str] = None,
) -> str:
    chart = plot.multi_linechart(
        data=trips_per_hour,
        x=const.HOUR,
        y="perc",
        style=style,
        color=const.TIME_CATEGORY,
        x_axis_label="Hour of day",
        y_axis_label="% of trips",
        hue_order=["weekday start", "weekday end", "weekend start", "weekend end"],
        margin_of_error=margin_of_error,
        figsize=(10, 6),
    )
    html = v_utils.fig_to_html(chart)
    plt.close(chart)
    return html

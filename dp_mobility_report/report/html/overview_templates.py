from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import DfSection, DictSection, SeriesSection
from dp_mobility_report.report.html.html_utils import (
    fmt,
    fmt_moe,
    get_template,
    render_eps,
    render_moe_info,
    render_summary,
    render_benchmark_summary
)

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from dp_mobility_report.visualization import plot, v_utils
from typing import TYPE_CHECKING, Optional
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from dp_mobility_report import BenchmarkReport


def render_overview(dpmreport: "DpMobilityReport") -> str:
    args: dict = {}
    report = dpmreport.report

    if const.DS_STATISTICS not in dpmreport.analysis_exclusion:
        args["dataset_stats_table"] = render_dataset_statistics(
            report[const.DS_STATISTICS]
        )

    if const.MISSING_VALUES not in dpmreport.analysis_exclusion:
        args["missing_values_table"] = render_missing_values(
            report[const.MISSING_VALUES]
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
        trips_over_time_linechart = render_trips_over_time(
            report[const.TRIPS_OVER_TIME]
        )
        args["trips_over_time_moe_info"] = render_moe_info(
            report[const.TRIPS_OVER_TIME].margin_of_error_expmech
        )
        args["trips_over_time_summary_table"] = render_summary(
            report[const.TRIPS_OVER_TIME].quartiles
        )

    if const.TRIPS_PER_WEEKDAY not in dpmreport.analysis_exclusion:
        args["trips_per_weekday_eps"] = render_eps(
            report[const.TRIPS_PER_WEEKDAY].privacy_budget
        )
        args["trips_per_weekday_moe"] = fmt_moe(
            report[const.TRIPS_PER_HOUR].margin_of_error_laplace
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
        trips_per_hour_linechart = render_trips_per_hour(report[const.TRIPS_PER_HOUR])

    template_structure = get_template("overview_segment.html")
    return template_structure.render(args)


def render_benchmark_overview(report_base: dict, report_alternative:dict, analysis_exclusion, benchmark: "BenchmarkReport") -> str:
    #agrs={}
    dataset_stats_table = ""
    missing_values_table = ""
    trips_over_time_eps = None
    trips_over_time_moe = None
    trips_over_time_info = ""
    trips_over_time_linechart = ""
    trips_over_time_moe_info = ""
    trips_over_time_summary_table = ""
    trips_over_time_measure = ""
    trips_per_weekday_eps = None
    trips_per_weekday_moe = None
    trips_per_weekday_barchart = ""
    trips_per_weekday_measure = ""
    trips_per_hour_eps = None
    trips_per_hour_moe = None
    trips_per_hour_linechart = ""
    trips_per_hour_measure = ""

    if const.DS_STATISTICS not in analysis_exclusion:
        dataset_stats_table = render_benchmark_dataset_statistics(report_base.report[const.DS_STATISTICS], report_alternative.report[const.DS_STATISTICS], benchmark.re)

    if const.MISSING_VALUES not in analysis_exclusion:
        missing_values_table = render_benchmark_missing_values(report_base.report[const.MISSING_VALUES], report_alternative.report[const.MISSING_VALUES], benchmark.re)

    if const.TRIPS_OVER_TIME not in analysis_exclusion:
        trips_over_time_eps = (render_eps(report_base.report[const.TRIPS_OVER_TIME].privacy_budget),render_eps(report_alternative.report[const.TRIPS_OVER_TIME].privacy_budget))
        trips_over_time_moe = (fmt_moe(report_base.report[const.TRIPS_OVER_TIME].margin_of_error_laplace),
                                fmt_moe(report_alternative.report[const.TRIPS_OVER_TIME].margin_of_error_laplace))
        trips_over_time_info = render_trips_over_time_info(report_base.report[const.TRIPS_OVER_TIME].datetime_precision)
        trips_over_time_linechart = render_benchmark_trips_over_time(
            report_base.report[const.TRIPS_OVER_TIME], report_alternative.report[const.TRIPS_OVER_TIME]
        )
        trips_over_time_moe_info = render_moe_info(
            report_base.report[const.TRIPS_OVER_TIME].margin_of_error_expmech
        )
        trips_over_time_summary_table = render_benchmark_summary(
            report_base.report[const.TRIPS_OVER_TIME].quartiles, report_alternative.report[const.TRIPS_OVER_TIME].quartiles
        )
        trips_over_time_measure = (const.format[benchmark.measure_selection[const.TRIPS_OVER_TIME]], fmt(benchmark.similarity_measures[const.TRIPS_OVER_TIME]))

    if const.TRIPS_PER_WEEKDAY not in analysis_exclusion:
        trips_per_weekday_eps = (render_eps(report_base.report[const.TRIPS_PER_WEEKDAY].privacy_budget), 
                                    render_eps(report_alternative.report[const.TRIPS_PER_WEEKDAY].privacy_budget))
        trips_per_weekday_moe = (fmt_moe(report_base.report[const.TRIPS_PER_WEEKDAY].margin_of_error_laplace), 
                                    fmt_moe(report_alternative.report[const.TRIPS_PER_WEEKDAY].margin_of_error_laplace))

        trips_per_weekday_barchart = render_benchmark_trips_per_weekday(
            report_base.report[const.TRIPS_PER_WEEKDAY], report_alternative.report[const.TRIPS_PER_WEEKDAY]
            )
        trips_per_weekday_measure = (const.format[benchmark.measure_selection[const.TRIPS_PER_WEEKDAY]], fmt(benchmark.similarity_measures[const.TRIPS_PER_WEEKDAY]))

    if const.TRIPS_PER_HOUR not in analysis_exclusion:
        trips_per_hour_eps = (render_eps(report_base.report[const.TRIPS_PER_HOUR].privacy_budget), 
                                render_eps(report_alternative.report[const.TRIPS_PER_HOUR].privacy_budget))
        trips_per_hour_moe = (fmt_moe(report_base.report[const.TRIPS_PER_HOUR].margin_of_error_laplace), 
                                fmt_moe(report_alternative.report[const.TRIPS_PER_HOUR].margin_of_error_laplace))
        
        dataset = np.append(np.repeat("base", len(report_base.report[const.TRIPS_PER_HOUR].data)),
                            np.repeat("alternative", len(report_alternative.report[const.TRIPS_PER_HOUR].data)))
        combined_trips_per_hour = pd.concat([report_base.report[const.TRIPS_PER_HOUR].data, report_alternative.report[const.TRIPS_PER_HOUR].data])
        combined_trips_per_hour['dataset'] = dataset
        trips_per_hour_linechart = render_trips_per_hour(combined_trips_per_hour, margin_of_error=None, style="dataset")
        trips_per_hour_measure = (const.format[benchmark.measure_selection[const.TRIPS_PER_HOUR]], fmt(benchmark.similarity_measures[const.TRIPS_PER_HOUR]))

    template_structure = get_template("overview_segment_benchmark.html")
    return template_structure.render(
        dataset_stats_table=dataset_stats_table,
        missing_values_table=missing_values_table,
        trips_over_time_eps=trips_over_time_eps,
        trips_over_time_moe=trips_over_time_moe,
        trips_over_time_info=trips_over_time_info,
        trips_over_time_linechart=trips_over_time_linechart,
        trips_over_time_measure=trips_over_time_measure,
        trips_over_time_moe_info=trips_over_time_moe_info,
        trips_over_time_summary_table=trips_over_time_summary_table,
        trips_per_weekday_eps=trips_per_weekday_eps,
        trips_per_weekday_moe=trips_per_weekday_moe,
        trips_per_weekday_barchart=trips_per_weekday_barchart,
        trips_per_weekday_measure=trips_per_weekday_measure,
        trips_per_hour_eps=trips_per_hour_eps,
        trips_per_hour_moe=trips_per_hour_moe,
        trips_per_hour_linechart=trips_per_hour_linechart,
        trips_per_hour_measure=trips_per_hour_measure
    )


def render_dataset_statistics(dataset_statistics: DictSection) -> str:
    moe = dataset_statistics.margin_of_errors_laplace
    data = dataset_statistics.data
    dataset_stats_list = [
        {
            "name": "Number of records",
            "estimate": fmt(data[const.N_RECORDS]),
            "margin_of_error": fmt_moe(moe[const.N_RECORDS]),
        },
        {
            "name": "Distinct trips",
            "estimate": fmt(data[const.N_TRIPS]),
            "margin_of_error": fmt_moe(moe[const.N_TRIPS]),
        },
        {
            "name": "Number of complete trips (start and and point)",
            "estimate": fmt(data[const.N_COMPLETE_TRIPS]),
            "margin_of_error": fmt_moe(moe[const.N_COMPLETE_TRIPS]),
        },
        {
            "name": "Number of incomplete trips (single point)",
            "estimate": fmt(data[const.N_INCOMPLETE_TRIPS]),
            "margin_of_error": fmt_moe(moe[const.N_INCOMPLETE_TRIPS]),
        },
        {
            "name": "Distinct users",
            "estimate": fmt(data[const.N_USERS]),
            "margin_of_error": fmt_moe(moe[const.N_USERS]),
        },
        {
            "name": "Distinct locations (lat & lon combination)",
            "estimate": fmt(data[const.N_LOCATIONS]),
            "margin_of_error": fmt_moe(moe[const.N_LOCATIONS]),
        },
    ]

    # create html from template
    template_table = get_template("table_conf_interval.html")
    dataset_stats_html = template_table.render(
        name="Dataset statistics",
        privacy_budget=render_eps(dataset_statistics.privacy_budget),
        rows=dataset_stats_list,
    )
    return dataset_stats_html


def render_benchmark_dataset_statistics(dataset_statistics_base: DictSection, dataset_statistics_alternative: DictSection, re:dict) -> str:
    moe_base = dataset_statistics_base.margin_of_errors_laplace
    moe_alternative = dataset_statistics_alternative.margin_of_errors_laplace
    data_base = dataset_statistics_base.data
    data_alternative = dataset_statistics_alternative.data

    dataset_stats_list = [
        {
            "name": "Number of records",
            "estimate": (fmt(data_base[const.N_RECORDS]), fmt(data_alternative[const.N_RECORDS])),
            "margin_of_error": (fmt_moe(moe_base[const.N_RECORDS]), fmt_moe(moe_alternative[const.N_RECORDS])),
            "relative_error": re['n_records']
        },
        {
            "name": "Distinct trips",
            "estimate": (fmt(data_base[const.N_TRIPS]), fmt(data_alternative[const.N_TRIPS])),
            "margin_of_error": (fmt_moe(moe_base[const.N_TRIPS]), fmt_moe(moe_alternative[const.N_TRIPS])),
            "relative_error": re['n_trips']

        },
        {
            "name": "Number of complete trips (start and and point)",
            "estimate": (fmt(data_base[const.N_COMPLETE_TRIPS]), fmt(data_alternative[const.N_COMPLETE_TRIPS])),
            "margin_of_error": (fmt_moe(moe_base[const.N_COMPLETE_TRIPS]), fmt_moe(moe_alternative[const.N_COMPLETE_TRIPS])),
            "relative_error": re['n_complete_trips']
        },
        {
            "name": "Number of incomplete trips (single point)",
            "estimate": (fmt(data_base[const.N_INCOMPLETE_TRIPS]), fmt(data_alternative[const.N_INCOMPLETE_TRIPS])),
            "margin_of_error": (fmt_moe(moe_base[const.N_INCOMPLETE_TRIPS]), fmt_moe(data_alternative[const.N_INCOMPLETE_TRIPS])),
            "relative_error": re['n_incomplete_trips']
        },
        {
            "name": "Distinct users",
            "estimate": (fmt(data_base[const.N_USERS]), fmt(data_alternative[const.N_USERS])),
            "margin_of_error": (fmt_moe(moe_base[const.N_USERS]),fmt_moe(moe_alternative[const.N_USERS])),
            "relative_error": re['n_users']
        },
        {
            "name": "Distinct locations (lat & lon combination)",
            "estimate": (fmt(data_base[const.N_LOCATIONS]), fmt(data_alternative[const.N_LOCATIONS])),
            "margin_of_error": (fmt_moe(moe_base[const.N_LOCATIONS]),fmt_moe(moe_alternative[const.N_LOCATIONS])),
            "relative_error": re['n_locations']
        },
    ]

    # create html from template
    template_table = get_template("table_conf_interval_benchmark.html")
    dataset_stats_html = template_table.render(
        name="Dataset statistics",
        privacy_budget_base=render_eps(dataset_statistics_base.privacy_budget),
        privacy_budget_alternative=render_eps(dataset_statistics_alternative.privacy_budget),
        rows=dataset_stats_list,
    )
    return dataset_stats_html


def render_missing_values(missing_values: DictSection) -> str:
    moe = round(missing_values.margin_of_error_laplace, 1)
    data = missing_values.data
    missing_values_list = [
        {
            "name": "User ID (uid)",
            "estimate": fmt(data[const.UID]),
            "margin_of_error": fmt_moe(moe),
        },
        {
            "name": "Trip ID (tid)",
            "estimate": fmt(data[const.TID]),
            "margin_of_error": fmt_moe(moe),
        },
        {
            "name": "Timestamp (datetime)",
            "estimate": fmt(data[const.DATETIME]),
            "margin_of_error": fmt_moe(moe),
        },
        {
            "name": "Latitude (lat)",
            "estimate": fmt(data[const.LAT]),
            "margin_of_error": fmt_moe(moe),
        },
        {
            "name": "Longitude (lng)",
            "estimate": fmt(data[const.LNG]),
            "margin_of_error": fmt_moe(moe),
        },
    ]

    template_table = get_template("table_conf_interval.html")
    missing_values_html = template_table.render(
        name="Missing values",
        privacy_budget=render_eps(missing_values.privacy_budget),
        rows=missing_values_list,
    )
    return missing_values_html

def render_benchmark_missing_values(missing_values_base: DictSection, missing_values_alternative: DictSection, re:dict) -> str:
    moe_base = round(missing_values_base.margin_of_error_laplace, 1)
    moe_alternative = round(missing_values_alternative.margin_of_error_laplace, 1)
    data_base = missing_values_base.data
    data_alternative = missing_values_alternative.data

    missing_values_list = [
        {
            "name": "User ID (uid)",
            "estimate": (fmt(data_base[const.UID]), fmt(data_alternative[const.UID])),
            "margin_of_error": (fmt_moe(moe_base),fmt_moe(moe_alternative)),
            "relative_error": re['uid']

        },
        {
            "name": "Trip ID (tid)",
            "estimate": (fmt(data_base[const.TID]), fmt(data_alternative[const.TID])),
            "margin_of_error": (fmt_moe(moe_base),fmt_moe(moe_alternative)),
            "relative_error": re['tid']

        },
        {
            "name": "Timestamp (datetime)",
            "estimate": (fmt(data_base[const.DATETIME]), fmt(data_alternative[const.DATETIME])),
            "margin_of_error": (fmt_moe(moe_base),fmt_moe(moe_alternative)),
            "relative_error": re['datetime']
        },
        {
            "name": "Latitude (lat)",
            "estimate": (fmt(data_base[const.LAT]), fmt(data_alternative[const.LAT])),
            "margin_of_error": (fmt_moe(moe_base),fmt_moe(moe_alternative)),
            "relative_error": re['lat']
        },
        {
            "name": "Longitude (lng)",
            "estimate": (fmt(data_base[const.LNG]), fmt(data_alternative[const.LNG])),
            "margin_of_error": (fmt_moe(moe_base),fmt_moe(moe_alternative)),
            "relative_error": re['lng']
        },
    ]

    template_table = get_template("table_conf_interval_benchmark.html")
    missing_values_html = template_table.render(
        name="Missing values",
        privacy_budget_base=render_eps(missing_values_base.privacy_budget),
        privacy_budget_alternative=render_eps(missing_values_alternative.privacy_budget),
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
            margin_of_error=fmt_moe(trips_over_time.margin_of_error_laplace),
            x_axis_label="Date",
            y_axis_label="% of trips",
            rotate_label=True,
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
        )
        html = v_utils.fig_to_html(chart)
    plt.close()
    return html

def render_benchmark_trips_over_time(trips_over_time: DfSection, trips_over_time_alternative: DfSection) -> str:
    if len(trips_over_time.data) <= 14:
        chart = plot.barchart(
            x=trips_over_time.data[const.DATETIME].to_numpy(),
            #x_alternative=trips_over_time_alternative.data[const.DATETIME].to_numpy(),
            y=trips_over_time.data["trips"].to_numpy(),
            y_alternative=trips_over_time_alternative.data["trips"].to_numpy(),
            margin_of_error=fmt_moe(trips_over_time.margin_of_error_laplace),
            margin_of_error_alternative=fmt_moe(trips_over_time_alternative.margin_of_error_laplace),
            x_axis_label="Date",
            y_axis_label="% of trips",
            rotate_label=True,
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
        )
        html = v_utils.fig_to_html(chart)
    plt.close()
    return html

def render_trips_per_weekday(trips_per_weekday: SeriesSection) -> str:
    chart = plot.barchart(
        x=trips_per_weekday.data.index.to_numpy(),
        y=trips_per_weekday.data.values,
        margin_of_error=trips_per_weekday.margin_of_error_laplace,
        x_axis_label="Weekday",
        y_axis_label="% of trips",
        rotate_label=True,
    )
    plt.close()
    return v_utils.fig_to_html(chart)

def render_benchmark_trips_per_weekday(trips_per_weekday: SeriesSection, trips_per_weekday_alternative: SeriesSection) -> str:
    chart = plot.barchart(
        x=trips_per_weekday.data.index.to_numpy(),
        y=trips_per_weekday.data.values,
        y_alternative=trips_per_weekday_alternative.data.values,
        margin_of_error=trips_per_weekday.margin_of_error_laplace,
        margin_of_error_alternative = trips_per_weekday_alternative.margin_of_error_laplace,
        x_axis_label="Weekday",
        y_axis_label="% of trips",
        rotate_label=True,
    )
    plt.close()
    return v_utils.fig_to_html(chart)


def render_trips_per_hour(trips_per_hour: pd.DataFrame, margin_of_error: Optional[float], style: Optional[str]=None) -> str:
    chart = plot.multi_linechart(
        data=trips_per_hour,
        x=const.HOUR,
        y="perc",
        style=style,
        color=const.TIME_CATEGORY,
        x_axis_label="Hour of day",
        y_axis_label="% of trips",
        hue_order=["weekday_start", "weekday_end", "weekend_start", "weekend_end"],
        margin_of_error=margin_of_error,
    )
    html = v_utils.fig_to_html(chart)
    plt.close()
    return html

# def render_benchmark_trips_per_hour(trips_per_hour: DfSection) -> str:
#     chart = plot.multi_linechart(
#         data=trips_per_hour.data,
#         x=const.HOUR,
#         y="perc",
#         style="Dataset",
#         color=const.TIME_CATEGORY,
#         x_axis_label="Hour of day",
#         y_axis_label="% of trips",
#         hue_order=["weekday_start", "weekday_end", "weekend_start", "weekend_end"],
#         margin_of_error=trips_per_hour.margin_of_error_laplace,
#     )
#     html = v_utils.fig_to_html(chart)
#     plt.close()
#     return html
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
)

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from dp_mobility_report.visualization import plot, v_utils


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
        args["trips_over_time_linechart"] = render_trips_over_time(
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
        args["trips_per_hour_linechart"] = render_trips_per_hour(
            report[const.TRIPS_PER_HOUR]
        )

    template_structure = get_template("overview_segment.html")
    return template_structure.render(args)


def render_dataset_statistics(dataset_statistics: DictSection) -> str:
    moe = dataset_statistics.margin_of_errors_laplace
    data = dataset_statistics.data
    dataset_stats_list = [
        {
            "name": "Number of records",
            "estimate": fmt(data["n_records"]),
            "margin_of_error": fmt_moe(moe["records"]),
        },
        {
            "name": "Distinct trips",
            "estimate": fmt(data["n_trips"]),
            "margin_of_error": fmt_moe(moe["trips"]),
        },
        {
            "name": "Number of complete trips (start and and point)",
            "estimate": fmt(data["n_complete_trips"]),
            "margin_of_error": fmt_moe(moe["complete_trips"]),
        },
        {
            "name": "Number of incomplete trips (single point)",
            "estimate": fmt(data["n_incomplete_trips"]),
            "margin_of_error": fmt_moe(moe["incomplete_trips"]),
        },
        {
            "name": "Distinct users",
            "estimate": fmt(data["n_users"]),
            "margin_of_error": fmt_moe(moe["users"]),
        },
        {
            "name": "Distinct locations (lat & lon combination)",
            "estimate": fmt(data["n_locations"]),
            "margin_of_error": fmt_moe(moe["locations"]),
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


def render_trips_per_hour(trips_per_hour: DfSection) -> str:
    chart = plot.multi_linechart(
        data=trips_per_hour.data,
        x=const.HOUR,
        y="perc",
        color=const.TIME_CATEGORY,
        x_axis_label="Hour of day",
        y_axis_label="% of trips",
        hue_order=["weekday_start", "weekday_end", "weekend_start", "weekend_end"],
        margin_of_error=trips_per_hour.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(chart)
    plt.close()
    return html

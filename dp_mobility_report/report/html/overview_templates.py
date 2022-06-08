import matplotlib.pyplot as plt

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import Section
from dp_mobility_report.report.html.html_utils import (
    fmt,
    get_template,
    render_moe_info,
    render_summary,
)
from dp_mobility_report.visualization import plot, v_utils


def render_overview(report: dict) -> str:
    dataset_stats_table = ""
    missing_values_table = ""
    trips_over_time_linechart = ""
    trips_over_time_summary_table = ""
    trips_over_time_moe_info = ""
    trips_per_weekday_barchart = ""
    trips_per_hour_linechart = ""

    if const.DS_STATISTICS in report and report[const.DS_STATISTICS].data is not None:
        dataset_stats_table = render_dataset_statistics(report[const.DS_STATISTICS])

    if const.MISSING_VALUES in report and report[const.MISSING_VALUES].data is not None:
        missing_values_table = render_missing_values(report[const.MISSING_VALUES])

    if (
        const.TRIPS_OVER_TIME in report
        and report[const.TRIPS_OVER_TIME].data is not None
    ):
        trips_over_time_info = render_trips_over_time_info(
            report[const.TRIPS_OVER_TIME].datetime_precision
        )
        trips_over_time_linechart = render_trips_over_time(
            report[const.TRIPS_OVER_TIME]
        )
        trips_over_time_summary_table = render_summary(
            report[const.TRIPS_OVER_TIME].quartiles
        )
        trips_over_time_moe_info = render_moe_info(
            report[const.TRIPS_OVER_TIME].margin_of_error_expmech
        )

    if (
        const.TRIPS_PER_WEEKDAY in report
        and report[const.TRIPS_PER_WEEKDAY].data is not None
    ):
        trips_per_weekday_barchart = render_trips_per_weekday(
            report[const.TRIPS_PER_WEEKDAY]
        )

    if const.TRIPS_PER_HOUR in report and report[const.TRIPS_PER_HOUR].data is not None:
        trips_per_hour_linechart = render_trips_per_hour(report[const.TRIPS_PER_HOUR])

    template_structure = get_template("overview_segment.html")
    return template_structure.render(
        dataset_stats_table=dataset_stats_table,
        missing_values_table=missing_values_table,
        trips_over_time_info=trips_over_time_info,
        trips_over_time_linechart=trips_over_time_linechart,
        trips_over_time_moe_info=trips_over_time_moe_info,
        trips_over_time_summary_table=trips_over_time_summary_table,
        trips_per_weekday_barchart=trips_per_weekday_barchart,
        trips_per_hour_linechart=trips_per_hour_linechart,
    )


def render_dataset_statistics(dataset_statistics: Section) -> str:
    ci = dataset_statistics.conf_interval
    data = dataset_statistics.data
    dataset_stats_list = [
        {
            "name": "Number of records",
            "lower_limit": fmt(ci["ci95_records"][0]),
            "estimate": fmt(data["n_records"]),
            "upper_limit": fmt(ci["ci95_records"][1]),
        },
        {
            "name": "Distinct trips",
            "lower_limit": fmt(ci["ci95_trips"][0]),
            "estimate": fmt(data["n_trips"]),
            "upper_limit": fmt(ci["ci95_trips"][1]),
        },
        {
            "name": "Number of complete trips (start and and point)",
            "lower_limit": fmt(ci["ci95_complete_trips"][0]),
            "estimate": fmt(data["n_complete_trips"]),
            "upper_limit": fmt(ci["ci95_complete_trips"][1]),
        },
        {
            "name": "Number of incomplete trips (single point)",
            "lower_limit": fmt(ci["ci95_incomplete_trips"][0]),
            "estimate": fmt(data["n_incomplete_trips"]),
            "upper_limit": fmt(ci["ci95_incomplete_trips"][1]),
        },
        {
            "name": "Distinct users",
            "lower_limit": fmt(ci["ci95_users"][0]),
            "estimate": fmt(data["n_users"]),
            "upper_limit": fmt(ci["ci95_users"][1]),
        },
        {
            "name": "Distinct locations (lat & lon combination)",
            "lower_limit": fmt(ci["ci95_locations"][0]),
            "estimate": fmt(data["n_locations"]),
            "upper_limit": fmt(ci["ci95_locations"][1]),
        },
    ]

    # create html from template
    template_table = get_template("table_conf_interval.html")
    dataset_stats_html = template_table.render(
        name="Dataset statistics", rows=dataset_stats_list
    )
    return dataset_stats_html


def render_missing_values(missing_values: Section) -> str:
    ci = missing_values.conf_interval
    data = missing_values.data
    missing_values_list = [
        {
            "name": "User ID (uid)",
            "lower_limit": fmt(ci["ci95_" + const.UID][0]),
            "estimate": fmt(data[const.UID]),
            "upper_limit": fmt(ci["ci95_" + const.UID][1]),
        },
        {
            "name": "Trip ID (tid)",
            "lower_limit": fmt(ci["ci95_" + const.TID][0]),
            "estimate": fmt(data[const.TID]),
            "upper_limit": fmt(ci["ci95_" + const.TID][1]),
        },
        {
            "name": "Timestamp (datetime)",
            "lower_limit": fmt(ci["ci95_" + const.DATETIME][0]),
            "estimate": fmt(data[const.DATETIME]),
            "upper_limit": fmt(ci["ci95_" + const.DATETIME][1]),
        },
        {
            "name": "Latitude (lat)",
            "lower_limit": fmt(ci["ci95_" + const.LAT][0]),
            "estimate": fmt(data[const.LAT]),
            "upper_limit": fmt(ci["ci95_" + const.LAT][1]),
        },
        {
            "name": "Longitude (lng)",
            "lower_limit": fmt(ci["ci95_" + const.LNG][0]),
            "estimate": fmt(data[const.LNG]),
            "upper_limit": fmt(ci["ci95_" + const.LNG][1]),
        },
    ]

    template_table = get_template("table_conf_interval.html")
    missing_values_html = template_table.render(
        name="Missing values", rows=missing_values_list
    )
    return missing_values_html


def render_trips_over_time_info(datetime_precision: str) -> str:
    return f"Timestamps have been aggregated by {datetime_precision}."


def render_trips_over_time(trips_over_time: Section) -> str:
    if len(trips_over_time.data) <= 14:
        chart = plot.barchart(
            x=trips_over_time.data[const.DATETIME].to_numpy(),
            y=trips_over_time.data["trip_count"].to_numpy(),
            margin_of_error=trips_over_time.margin_of_error_laplace,
            x_axis_label="Date",
            y_axis_label="% of trips",
            rotate_label=True,
        )
        html = v_utils.fig_to_html(chart)
    else:
        chart = plot.linechart(
            data=trips_over_time.data,
            x=const.DATETIME,
            y="trip_count",
            x_axis_label="Date",
            y_axis_label="% of trips",
            margin_of_error=trips_over_time.margin_of_error_laplace,
            rotate_label=True,
        )
        html = v_utils.fig_to_html(chart)
    plt.close()
    return html


def render_trips_per_weekday(trips_per_weekday: Section) -> str:
    chart = plot.barchart(
        x=trips_per_weekday.data.index.to_numpy(),
        y=trips_per_weekday.data.values,
        margin_of_error=trips_per_weekday.margin_of_error_laplace,
        x_axis_label="Weekday",
        y_axis_label="% of trips per weekday",
        rotate_label=True,
    )
    plt.close()
    return v_utils.fig_to_html(chart)


def render_trips_per_hour(trips_per_hour: Section) -> str:
    chart = plot.multi_linechart(
        data=trips_per_hour.data,
        x=const.HOUR,
        y="count",
        color=const.TIME_CATEGORY,
        x_axis_label="Hour of day",
        y_axis_label="% of trips",
        hue_order=["weekday_start", "weekday_end", "weekend_start", "weekend_end"],
        margin_of_error=trips_per_hour.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(chart)
    plt.close()
    return html

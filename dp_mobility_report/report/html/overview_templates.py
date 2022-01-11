import matplotlib.pyplot as plt
from pandas import DataFrame, Series

from dp_mobility_report import constants as const
from dp_mobility_report.report.html.html_utils import fmt, get_template, render_summary
from dp_mobility_report.visualization import plot, v_utils


def render_overview(report: dict) -> str:
    dataset_stats_table = None
    missing_values_table = None
    trips_over_time_linechart = None
    trips_over_time_linechart = None
    trips_over_time_summary_table = None
    trips_per_weekday_barchart = None
    trips_per_hour_linechart = None

    if const.DS_STATISTICS in report:
        dataset_stats_table = render_dataset_statistics(
            report[const.DS_STATISTICS].data
        )

    if const.MISSING_VALUES in report:
        missing_values_table = render_missing_values(report[const.MISSING_VALUES].data)

    if const.TRIPS_OVER_TIME in report:
        trips_over_time_linechart = render_trips_over_time(
            report[const.TRIPS_OVER_TIME].data
        )
        trips_over_time_summary_table = render_summary(
            report[const.TRIPS_OVER_TIME].quartiles
        )

    if const.TRIPS_PER_WEEKDAY in report:
        trips_per_weekday_barchart = render_trips_per_weekday(
            report[const.TRIPS_PER_WEEKDAY].data
        )

    if const.TRIPS_PER_HOUR in report:
        trips_per_hour_linechart = render_trips_per_hour(
            report[const.TRIPS_PER_HOUR].data
        )

    template_structure = get_template("overview_segment.html")
    return template_structure.render(
        dataset_stats_table=dataset_stats_table,
        missing_values_table=missing_values_table,
        trips_over_time_linechart=trips_over_time_linechart,
        trips_over_time_summary_table=trips_over_time_summary_table,
        trips_per_weekday_barchart=trips_per_weekday_barchart,
        trips_per_hour_linechart=trips_per_hour_linechart,
    )


def render_dataset_statistics(dataset_statistics: dict) -> str:

    dataset_stats_list = [
        {"name": "Number of records", "value": fmt(dataset_statistics["n_records"])},
        {"name": "Distinct trips", "value": fmt(dataset_statistics["n_trips"])},
        {
            "name": "Number of complete trips (start and and point)",
            "value": fmt(dataset_statistics["n_complete_trips"]),
        },
        {
            "name": "Number of incomplete trips (single point)",
            "value": fmt(dataset_statistics["n_incomplete_trips"]),
        },
        {"name": "Distinct users", "value": fmt(dataset_statistics["n_users"])},
        {
            "name": "Distinct locations (lat & lon combination)",
            "value": fmt(dataset_statistics["n_locations"]),
        },
    ]

    # create html from template
    template_table = get_template("table.html")
    dataset_stats_html = template_table.render(
        name="Dataset statistics", rows=dataset_stats_list
    )
    return dataset_stats_html


def render_missing_values(missing_values: dict) -> str:
    missing_values_list = [
        {"name": "User ID (uid)", "value": fmt(missing_values[const.UID])},
        {"name": "Trip ID (tid)", "value": fmt(missing_values[const.TID])},
        {"name": "Timestamp (datetime)", "value": fmt(missing_values[const.DATETIME])},
        {"name": "Latitude (lat)", "value": fmt(missing_values[const.LAT])},
        {"name": "Longitude (lng)", "value": fmt(missing_values[const.LNG])},
    ]

    template_table = get_template("table.html")
    missing_values_html = template_table.render(
        name="Missing values", rows=missing_values_list
    )
    return missing_values_html


def render_trips_over_time(trips_over_time: DataFrame) -> str:
    if len(trips_over_time) <= 20:
        chart = plot.barchart(
            x=trips_over_time[const.DATETIME].to_numpy(),
            y=trips_over_time["trip_count"].to_numpy(),
            x_axis_label="Date",
            y_axis_label="Frequency",
            rotate_label=True,
        )
        html = v_utils.fig_to_html(chart)
    else:
        chart = plot.linechart(
            trips_over_time, const.DATETIME, "trip_count", "Date", "Frequency"
        )
        html = v_utils.fig_to_html(chart)
    plt.close()
    return html


def render_trips_per_weekday(trips_per_weekday: Series) -> str:
    chart = plot.barchart(
        x=trips_per_weekday.index.to_numpy(),
        y=trips_per_weekday.values,
        x_axis_label="Weekday",
        y_axis_label="Average trips per weekday",
        rotate_label=True,
        order_x=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
    )
    plt.close()
    return v_utils.fig_to_html(chart)


def render_trips_per_hour(trips_per_hour: DataFrame) -> str:
    chart = plot.multi_linechart(
        trips_per_hour,
        const.HOUR,
        "count",
        const.TIME_CATEGORY,
        hue_order=["weekday_start", "weekday_end", "weekend_start", "weekend_end"],
    )
    html = v_utils.fig_to_html(chart)
    plt.close()
    return html

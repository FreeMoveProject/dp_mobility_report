from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import TupleSection
from dp_mobility_report.report.html.html_utils import (
    get_template,
    render_eps,
    render_moe_info,
    render_summary,
    render_user_input_info,
)
from dp_mobility_report.visualization import plot, v_utils


def render_user_analysis(dpmreport: "DpMobilityReport") -> str:
    trips_per_user_info = f"Trips per user are limited according to the configured maximum of trips per user: {dpmreport.max_trips_per_user}"
    trips_per_user_eps = None
    trips_per_user_hist = ""
    trips_per_user_summary_table = ""
    trips_per_user_moe_info = ""
    time_between_traj_eps = None
    overlapping_trips_info = ""
    time_between_traj_hist = ""
    time_between_traj_summary_table = ""
    time_between_traj_moe_info = ""
    radius_of_gyration_eps = None
    radius_of_gyration_hist_info = ""
    radius_of_gyration_hist = ""
    radius_of_gyration_summary_table = ""
    radius_of_gyration_moe_info = ""
    distinct_tiles_user_eps = None
    distinct_tiles_user_hist = ""
    distinct_tiles_user_summary_table = ""
    distinct_tiles_moe_info = ""
    mobility_entropy_eps = None
    mobility_entropy_hist = ""
    mobility_entropy_summary_table = ""
    mobility_entropy_moe_info = ""

    report = dpmreport.report

    if (const.TRIPS_PER_USER in report) and (
        report[const.TRIPS_PER_USER].data is not None
    ):
        trips_per_user_eps = render_eps(report[const.TRIPS_PER_USER].privacy_budget)
        trips_per_user_hist = render_trips_per_user(report[const.TRIPS_PER_USER])
        trips_per_user_summary_table = render_summary(
            report[const.TRIPS_PER_USER].quartiles
        )
        trips_per_user_moe_info = render_moe_info(
            report[const.TRIPS_PER_USER].margin_of_error_expmech
        )

    if (
        (const.USER_TIME_DELTA in report)
        and (report[const.USER_TIME_DELTA] is not None)
        and (report[const.USER_TIME_DELTA].quartiles is not None)
    ):
        time_between_traj_eps = render_eps(report[const.USER_TIME_DELTA].privacy_budget)
        overlapping_trips_info = render_overlapping_trips(report[const.USER_TIME_DELTA])
        time_between_traj_hist = render_time_between_traj(report[const.USER_TIME_DELTA])
        time_between_traj_summary_table = render_summary(
            report[const.USER_TIME_DELTA].quartiles
        )
        time_between_traj_moe_info = render_moe_info(
            report[const.USER_TIME_DELTA].margin_of_error_expmech
        )

    if (const.RADIUS_OF_GYRATION in report) and (
        report[const.RADIUS_OF_GYRATION].data is not None
    ):
        radius_of_gyration_eps = render_eps(
            report[const.RADIUS_OF_GYRATION].privacy_budget
        )
        radius_of_gyration_hist_info = render_user_input_info(
            dpmreport.max_radius_of_gyration, dpmreport.bin_range_radius_of_gyration
        )
        radius_of_gyration_hist = render_radius_of_gyration(
            report[const.RADIUS_OF_GYRATION]
        )
        radius_of_gyration_summary_table = render_summary(
            report[const.RADIUS_OF_GYRATION].quartiles
        )
        radius_of_gyration_moe_info = render_moe_info(
            report[const.RADIUS_OF_GYRATION].margin_of_error_expmech
        )

    if (const.USER_TILE_COUNT in report) and (
        report[const.USER_TILE_COUNT].data is not None
    ):
        distinct_tiles_user_eps = render_eps(
            report[const.USER_TILE_COUNT].privacy_budget
        )
        distinct_tiles_user_hist = render_distinct_tiles_user(
            report[const.USER_TILE_COUNT]
        )
        distinct_tiles_user_summary_table = render_summary(
            report[const.USER_TILE_COUNT].quartiles
        )
        distinct_tiles_moe_info = render_moe_info(
            report[const.USER_TILE_COUNT].margin_of_error_expmech
        )

    if (const.MOBILITY_ENTROPY in report) and (
        report[const.MOBILITY_ENTROPY].data is not None
    ):
        mobility_entropy_eps = render_eps(report[const.MOBILITY_ENTROPY].privacy_budget)
        mobility_entropy_hist = render_mobility_entropy(report[const.MOBILITY_ENTROPY])
        mobility_entropy_summary_table = render_summary(
            report[const.MOBILITY_ENTROPY].quartiles
        )
        mobility_entropy_moe_info = render_moe_info(
            report[const.MOBILITY_ENTROPY].margin_of_error_expmech
        )

    template_structure = get_template("user_analysis_segment.html")

    return template_structure.render(
        trips_per_user_eps=trips_per_user_eps,
        trips_per_user_info=trips_per_user_info,
        trips_per_user_hist=trips_per_user_hist,
        trips_per_user_summary_table=trips_per_user_summary_table,
        trips_per_user_moe_info=trips_per_user_moe_info,
        time_between_traj_eps=time_between_traj_eps,
        overlapping_trips_info=overlapping_trips_info,
        time_between_traj_hist=time_between_traj_hist,
        time_between_traj_summary_table=time_between_traj_summary_table,
        time_between_traj_moe_info=time_between_traj_moe_info,
        radius_of_gyration_eps=radius_of_gyration_eps,
        radius_of_gyration_hist_info=radius_of_gyration_hist_info,
        radius_of_gyration_hist=radius_of_gyration_hist,
        radius_of_gyration_summary_table=radius_of_gyration_summary_table,
        radius_of_gyration_moe_info=radius_of_gyration_moe_info,
        distinct_tiles_user_eps=distinct_tiles_user_eps,
        distinct_tiles_user_hist=distinct_tiles_user_hist,
        distinct_tiles_user_summary_table=distinct_tiles_user_summary_table,
        distinct_tiles_moe_info=distinct_tiles_moe_info,
        mobility_entropy_eps=mobility_entropy_eps,
        mobility_entropy_hist=mobility_entropy_hist,
        mobility_entropy_summary_table=mobility_entropy_summary_table,
        mobility_entropy_moe_info=mobility_entropy_moe_info,
    )


def render_trips_per_user(trips_per_user_hist: TupleSection) -> str:
    hist = plot.histogram(
        trips_per_user_hist.data,
        x_axis_label="Number of trips per user",
        y_axis_label="% of users",
        x_axis_type=int,
        margin_of_error=trips_per_user_hist.margin_of_error_laplace,
    )
    return v_utils.fig_to_html(hist)


def render_time_between_traj(time_between_traj_hist: TupleSection) -> str:
    hist = plot.histogram(
        time_between_traj_hist.data,
        x_axis_label="Hours between consecutive trips",
        y_axis_label="% of trips",
        x_axis_type=float,
        ndigits_x_label=1,
        margin_of_error=time_between_traj_hist.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_overlapping_trips(n_traj_overlaps: TupleSection) -> str:
    if n_traj_overlaps.n_outliers is None:
        return ""
    ci_interval_info = (
        f"(95% confidence interval ± {round(n_traj_overlaps.margin_of_error_laplace)})"
        if n_traj_overlaps.margin_of_error_laplace is not None
        else ""
    )

    return f"""<h4>Plausibility check: overlapping user trips</h4>
    <p>
        There are overlapping trips in the dataset: The negative minimum time delta implies that there is a trip of a user that starts before
        the previous one has ended. This might be an indication of a faulty dataset.
    </p>
    <p>
    There are {n_traj_overlaps.n_outliers}  cases where the start time of the following trip precedes the previous end time {ci_interval_info}.
    </p>
    """


def render_radius_of_gyration(radius_of_gyration_hist: TupleSection) -> str:
    hist = plot.histogram(
        radius_of_gyration_hist.data,
        x_axis_label="radius of gyration (in km)",
        y_axis_label="% of users",
        x_axis_type=float,
        margin_of_error=radius_of_gyration_hist.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_distinct_tiles_user(user_tile_count_hist: TupleSection) -> str:
    hist = plot.histogram(
        user_tile_count_hist.data,
        x_axis_label="number of distinct tiles a user has visited",
        y_axis_label="% of users",
        x_axis_type=int,
        margin_of_error=user_tile_count_hist.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_mobility_entropy(mobility_entropy: TupleSection) -> str:
    hist = plot.histogram(
        (mobility_entropy.data[0], mobility_entropy.data[1].round(2)),
        x_axis_label="mobility entropy",
        y_axis_label="% of users",
        margin_of_error=mobility_entropy.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html

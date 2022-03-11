from typing import TYPE_CHECKING, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame

if TYPE_CHECKING:
    from dp_mobility_report.md_report import MobilityDataReport

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import Section
from dp_mobility_report.report.html.html_utils import (
    get_template,
    render_moe_info,
    render_outlier_info,
    render_summary,
)
from dp_mobility_report.visualization import plot, v_utils


def render_user_analysis(mdreport: "MobilityDataReport") -> str:
    trips_per_user_hist = ""
    trips_per_user_summary_table = ""
    trips_per_user_moe_info = ""
    overlapping_trips_info = ""
    time_between_traj_summary_table = ""
    time_between_traj_moe_info =""
    outlier_count_radius_of_gyration_info = ""
    radius_of_gyration_hist = ""
    radius_of_gyration_summary_table = ""
    radius_of_gyration_moe_info = ""
    location_entropy_map = ""
    location_entropy_legend = ""
    distinct_tiles_user_hist = ""
    distinct_tiles_user_summary_table = ""
    distinct_tiles_moe_info =""
    mobility_entropy_hist = ""
    mobility_entropy_summary_table = ""
    mobility_entropy_moe_info = ""

    report = mdreport.report

    if (const.TRIPS_PER_USER in report) and (
        report[const.TRIPS_PER_USER].data is not None
    ):
        trips_per_user_hist = render_trips_per_user(report[const.TRIPS_PER_USER])
        trips_per_user_summary_table = render_summary(
            report[const.TRIPS_PER_USER].quartiles
        )
        trips_per_user_moe_info = render_moe_info(report[const.TRIPS_PER_USER].margin_of_error_expmech)

    if (
        (const.USER_TIME_DELTA in report)
        and (report[const.USER_TIME_DELTA] is not None)
        and (report[const.USER_TIME_DELTA].quartiles is not None)
    ):
        time_between_traj_summary_table = render_summary(
            report[const.USER_TIME_DELTA].quartiles
        )
        time_between_traj_moe_info = render_moe_info(report[const.USER_TIME_DELTA].margin_of_error_expmech)
        overlapping_trips_info = render_overlapping_trips(report[const.USER_TIME_DELTA])

    if (const.RADIUS_OF_GYRATION in report) and (
        report[const.RADIUS_OF_GYRATION].data is not None
    ):
        radius_of_gyration_hist = render_radius_of_gyration(
            report[const.RADIUS_OF_GYRATION]
        )
        radius_of_gyration_summary_table = render_summary(
            report[const.RADIUS_OF_GYRATION].quartiles
        )
        radius_of_gyration_moe_info = render_moe_info(report[const.RADIUS_OF_GYRATION].margin_of_error_expmech)

        if mdreport.max_radius_of_gyration is not None:
            outlier_count_radius_of_gyration_info = render_outlier_info(
                report[const.RADIUS_OF_GYRATION].n_outliers,
                report[const.RADIUS_OF_GYRATION].margin_of_error,
                mdreport.max_jump_length,
            )

    if (const.LOCATION_ENTROPY in report) and (
        report[const.LOCATION_ENTROPY].data is not None
    ):
        location_entropy_map, location_entropy_legend = render_location_entropy(
            report[const.LOCATION_ENTROPY], mdreport.tessellation
        )

    if (const.USER_TILE_COUNT in report) and (
        report[const.USER_TILE_COUNT].data is not None
    ):
        distinct_tiles_user_hist = render_distinct_tiles_user(
            report[const.USER_TILE_COUNT]
        )
        distinct_tiles_user_summary_table = render_summary(
            report[const.USER_TILE_COUNT].quartiles
        )
        distinct_tiles_moe_info=render_moe_info(report[const.USER_TILE_COUNT].margin_of_error_expmech)


    if (const.MOBILITY_ENTROPY in report) and (
        report[const.MOBILITY_ENTROPY].data is not None
    ):
        mobility_entropy_hist = render_mobility_entropy(report[const.MOBILITY_ENTROPY])
        mobility_entropy_summary_table = render_summary(
            report[const.MOBILITY_ENTROPY].quartiles
        )
        mobility_entropy_moe_info=render_moe_info(report[const.MOBILITY_ENTROPY].margin_of_error_expmech)

    template_structure = get_template("user_analysis_segment.html")

    return template_structure.render(
        trips_per_user_hist=trips_per_user_hist,
        trips_per_user_summary_table=trips_per_user_summary_table,
        trips_per_user_moe_info=trips_per_user_moe_info,
        overlapping_trips_info=overlapping_trips_info,
        time_between_traj_summary_table=time_between_traj_summary_table,
        time_between_traj_moe_info=time_between_traj_moe_info,
        outlier_count_radius_of_gyration_info=outlier_count_radius_of_gyration_info,
        radius_of_gyration_hist=radius_of_gyration_hist,
        radius_of_gyration_summary_table=radius_of_gyration_summary_table,
        radius_of_gyration_moe_info=radius_of_gyration_moe_info,
        location_entropy_map=location_entropy_map,
        location_entropy_legend=location_entropy_legend,
        distinct_tiles_user_hist=distinct_tiles_user_hist,
        distinct_tiles_user_summary_table=distinct_tiles_user_summary_table,
        distinct_tiles_moe_info=distinct_tiles_moe_info,
        mobility_entropy_hist=mobility_entropy_hist,
        mobility_entropy_summary_table=mobility_entropy_summary_table,
        mobility_entropy_moe_info=mobility_entropy_moe_info
        )


def render_trips_per_user(trips_per_user_hist: Section) -> str:
    hist = plot.histogram(
        trips_per_user_hist.data,
        x_axis_label="Number of trips per user",
        x_axis_type=int,
        margin_of_error=trips_per_user_hist.margin_of_error_laplace,
    )
    return v_utils.fig_to_html(hist)


def render_overlapping_trips(n_traj_overlaps: Section) -> str:
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


def render_radius_of_gyration(radius_of_gyration_hist: Section) -> str:
    hist = plot.histogram(
        radius_of_gyration_hist.data,
        x_axis_label="radius of gyration",
        x_axis_type=float,
        margin_of_error=radius_of_gyration_hist.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_location_entropy(
    location_entropy: Section, tessellation: GeoDataFrame, threshold: float = 0.1
) -> Tuple[str, str]:
    # 0: all trips by a single user
    # large: evenly distributed over different users (2^x possible different users)
    data = location_entropy.data

    location_entropy_gdf = pd.merge(
        tessellation,
        data,
        how="left",
        left_on="tile_id",
        right_on="tile_id",
    )
    moe_deviation = (
        location_entropy.margin_of_error_laplace / location_entropy_gdf[const.LOCATION_ENTROPY]
    )
    location_entropy_gdf.loc[moe_deviation > threshold, const.LOCATION_ENTROPY] = None
    location_entropy_map, location_entropy_legend = plot.choropleth_map(
        location_entropy_gdf,
        const.LOCATION_ENTROPY,
        "Location entropy (0: all trips by a single user - large: users visit tile evenly)",
        min_scale=0,
    )
    html = location_entropy_map.get_root().render()
    legend = v_utils.fig_to_html(location_entropy_legend)
    plt.close()
    return html, legend


def render_distinct_tiles_user(user_tile_count_hist: Section) -> str:
    hist = plot.histogram(
        user_tile_count_hist.data,
        x_axis_label="number of distinct tiles a user has visited",
        x_axis_type=int,
        margin_of_error=user_tile_count_hist.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_mobility_entropy(mobility_entropy: Section) -> str:
    hist = plot.histogram(
        (mobility_entropy.data[0], mobility_entropy.data[1].round(2)),
        x_axis_label="mobility entropy",
        margin_of_error=mobility_entropy.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html

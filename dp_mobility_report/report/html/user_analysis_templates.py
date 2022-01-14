from typing import TYPE_CHECKING, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame

if TYPE_CHECKING:
    from dp_mobility_report.md_report import MobilityDataReport

from dp_mobility_report import constants as const
from dp_mobility_report.report.html.html_utils import (
    get_template,
    render_outlier_info,
    render_summary,
)
from dp_mobility_report.visualization import plot, v_utils


def render_user_analysis(mdreport: "MobilityDataReport") -> str:
    trips_per_user_summary_table = ""
    trips_per_user_hist = ""
    overlapping_trips_info = ""
    time_between_traj_summary_table = ""
    outlier_count_radius_of_gyration_info = ""
    radius_of_gyration_summary_table = ""
    radius_of_gyration_hist = ""
    location_entropy_map = ""
    location_entropy_legend = ""
    distinct_tiles_user_summary_table = ""
    distinct_tiles_user_hist = ""
    mobility_entropy_summary_table = ""
    mobility_entropy_hist = ""
    real_entropy_summary_table = ""
    real_entropy_hist = ""

    report = mdreport.report

    if const.TRIPS_PER_USER in report:
        trips_per_user_summary_table = render_summary(
            report[const.TRIPS_PER_USER].quartiles
        )

    if const.TRIPS_PER_USER in report:
        trips_per_user_hist = render_trips_per_user(report[const.TRIPS_PER_USER].data)

    if (const.USER_TIME_DELTA in report) & (report[const.USER_TIME_DELTA] is not None):
        overlapping_trips_info = render_overlapping_trips(
            report[const.USER_TIME_DELTA].n_outliers
        )
        time_between_traj_summary_table = render_summary(
            report[const.USER_TIME_DELTA].quartiles
        )

    if const.RADIUS_OF_GYRATION in report:
        outlier_count_radius_of_gyration_info = render_outlier_info(
            report[const.RADIUS_OF_GYRATION].n_outliers,
            mdreport.max_jump_length,
        )
        radius_of_gyration_summary_table = render_summary(
            report[const.RADIUS_OF_GYRATION].quartiles
        )
        radius_of_gyration_hist = render_radius_of_gyration(
            report[const.RADIUS_OF_GYRATION].data
        )

    if const.LOCATION_ENTROPY in report:
        location_entropy_map, location_entropy_legend = render_location_entropy(
            report[const.LOCATION_ENTROPY].data, mdreport.tessellation
        )

    if const.USER_TILE_COUNT in report:
        distinct_tiles_user_summary_table = render_summary(
            report[const.USER_TILE_COUNT].quartiles
        )
        distinct_tiles_user_hist = render_distinct_tiles_user(
            report[const.USER_TILE_COUNT].data
        )

    if const.MOBILITY_ENTROPY in report:
        mobility_entropy_summary_table = render_summary(
            report[const.MOBILITY_ENTROPY].quartiles
        )
        mobility_entropy_hist = render_mobility_entropy(
            report[const.MOBILITY_ENTROPY].data
        )

    template_structure = get_template("user_analysis_segment.html")

    return template_structure.render(
        trips_per_user_summary_table=trips_per_user_summary_table,
        trips_per_user_hist=trips_per_user_hist,
        overlapping_trips_info=overlapping_trips_info,
        time_between_traj_summary_table=time_between_traj_summary_table,
        outlier_count_radius_of_gyration_info=outlier_count_radius_of_gyration_info,
        radius_of_gyration_summary_table=radius_of_gyration_summary_table,
        radius_of_gyration_hist=radius_of_gyration_hist,
        location_entropy_map=location_entropy_map,
        location_entropy_legend=location_entropy_legend,
        distinct_tiles_user_hist=distinct_tiles_user_hist,
        distinct_tiles_user_summary_table=distinct_tiles_user_summary_table,
        mobility_entropy_hist=mobility_entropy_hist,
        mobility_entropy_summary_table=mobility_entropy_summary_table,
        real_entropy_hist=real_entropy_hist,
        real_entropy_summary_table=real_entropy_summary_table,
    )


def render_trips_per_user(trips_per_user_hist: Tuple) -> str:
    hist = plot.histogram(
        trips_per_user_hist,
        x_axis_label="number of trips per user",
        x_axis_type=int,
    )
    return v_utils.fig_to_html(hist)


def render_overlapping_trips(n_traj_overlaps: int) -> str:
    return (
        "There are "
        + str(n_traj_overlaps)
        + " cases where the start time of the following trajectory precedes the previous end time."
    )


def render_radius_of_gyration(radius_of_gyration_hist: Tuple) -> str:
    hist = plot.histogram(
        radius_of_gyration_hist, x_axis_label="radius of gyration", x_axis_type=float
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_location_entropy(
    location_entropy: pd.Series, tessellation: GeoDataFrame
) -> Tuple[str, str]:
    # 0: all trips by a single user
    # large: evenly distributed over different users (2^x possible different users)
    location_entropy_gdf = pd.merge(
        tessellation,
        location_entropy,
        how="left",
        left_on="tile_id",
        right_on="tile_id",
    )
    location_entropy_map, location_entropy_legend =  plot.choropleth_map(
            location_entropy_gdf,
            const.LOCATION_ENTROPY,
            "Location entropy (0: all trips by a single user - large: users visit tile evenly)",
            min_scale=0,
        )
    html = (
        location_entropy_map
        .get_root()
        .render()
    )
    legend = v_utils.fig_to_html(location_entropy_legend)
    plt.close()
    return html, legend


def render_distinct_tiles_user(user_tile_count_hist: Tuple) -> str:
    hist = plot.histogram(
        user_tile_count_hist,
        x_axis_label="number of distinct tiles a user has visited",
        x_axis_type=int,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_mobility_entropy(mobility_entropy: Tuple) -> str:
    hist = plot.histogram(
        (mobility_entropy[0], mobility_entropy[1].round(2)),
        x_axis_label="mobility entropy",
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html

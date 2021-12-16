import pandas as pd

from dp_mobility_report.report.html.utils import (
    render_summary,
    render_outlier_info,
    get_template,
)
from dp_mobility_report.visualization import plot, utils


def render_user_analysis(mdreport):
    traj_per_user_summary_table = None
    traj_per_user_hist = None
    overlapping_trips_info = None
    time_between_traj_summary_table = None
    outlier_count_radius_gyration_info = None
    radius_gyration_summary_table = None
    radius_gyration_hist = None
    location_entropy_map = None
    distinct_tiles_user_summary_table = None
    distinct_tiles_user_hist = None
    uncorrelated_entropy_summary_table = None
    uncorrelated_entropy_hist = None
    real_entropy_summary_table = None
    real_entropy_hist = None

    report = mdreport.report

    if "traj_per_user_section" in report:
        traj_per_user_summary_table = render_summary(
            report["traj_per_user_section"].quartiles
        )

    if "traj_per_user_section" in report:
        traj_per_user_hist = render_traj_per_user(report["traj_per_user_section"].data)

    if ("user_time_delta_section" in report) & (
        report["user_time_delta_section"] is not None
    ):
        overlapping_trips_info = render_overlapping_trips(
            report["user_time_delta_section"].n_outliers
        )
        time_between_traj_summary_table = render_summary(
            report["user_time_delta_section"].quartiles
        )

    if "radius_gyration_section" in report:
        outlier_count_radius_gyration_info = render_outlier_info(
            report["radius_gyration_section"].n_outliers, mdreport.max_jump_length,
        )
        radius_gyration_summary_table = render_summary(
            report["radius_gyration_section"].quartiles
        )
        radius_gyration_hist = render_radius_gyration(
            report["radius_gyration_section"].data
        )

    if "location_entropy_section" in report:
        location_entropy_map = render_location_entropy(
            report["location_entropy_section"], mdreport.tessellation
        )

    if "user_tile_count_section" in report:
        distinct_tiles_user_summary_table = render_summary(
            report["user_tile_count_section"].quartiles
        )
        distinct_tiles_user_hist = render_distinct_tiles_user(
            report["user_tile_count_section"].data
        )

    if "uncorrelated_entropy_section" in report:
        uncorrelated_entropy_summary_table = render_summary(
            report["uncorrelated_entropy_section"].quartiles
        )
        uncorrelated_entropy_hist = render_uncorrelated_entropy(
            report["uncorrelated_entropy_section"].data
        )

    if "real_entropy" in report:
        real_entropy_summary_table = render_summary(
            report["real_entropy_section"].quartiles
        )
        real_entropy_hist = render_real_entropy(report["real_entropy_section"].data)

    template_structure = get_template("user_analysis_segment.html")

    return template_structure.render(
        traj_per_user_summary_table=traj_per_user_summary_table,
        traj_per_user_hist=traj_per_user_hist,
        overlapping_trips_info=overlapping_trips_info,
        time_between_traj_summary_table=time_between_traj_summary_table,
        outlier_count_radius_gyration_info=outlier_count_radius_gyration_info,
        radius_gyration_summary_table=radius_gyration_summary_table,
        radius_gyration_hist=radius_gyration_hist,
        location_entropy_map=location_entropy_map,
        distinct_tiles_user_hist=distinct_tiles_user_hist,
        distinct_tiles_user_summary_table=distinct_tiles_user_summary_table,
        uncorrelated_entropy_hist=uncorrelated_entropy_hist,
        uncorrelated_entropy_summary_table=uncorrelated_entropy_summary_table,
        real_entropy_hist=real_entropy_hist,
        real_entropy_summary_table=real_entropy_summary_table,
    )


### render user analysis functions
def render_traj_per_user(traj_per_user_hist):
    hist = plot.histogram(
        traj_per_user_hist,
        x_axis_label="number of trajectories per user",
        x_axis_type=int,
    )
    return utils.fig_to_html(hist)


def render_overlapping_trips(n_traj_overlaps):
    return (
        "There are "
        + str(n_traj_overlaps)
        + " cases where the start time of the following trajectory precedes the previous end time."
    )


def render_radius_gyration(radius_gyration_hist):
    hist = plot.histogram(
        radius_gyration_hist, x_axis_label="radius of gyration", x_axis_type=int
    )
    return utils.fig_to_html(hist)


def render_location_entropy(location_entropy, tessellation):
    # 0: all trips by a single user
    # large: evenly distributed over different users (2^x possible different users)
    location_entropy_gdf = pd.merge(
        tessellation,
        location_entropy,
        how="left",
        left_on="tile_id",
        right_on="tile_id",
    )
    return (
        plot.choropleth_map(
            location_entropy_gdf,
            "location_entropy",
            "Location entropy (0: all trips by a single user - large: users visit tile evenly",
            min_scale=0,
        )
        .get_root()
        .render()
    )


def render_distinct_tiles_user(user_tile_count_hist):
    hist = plot.histogram(
        user_tile_count_hist,
        x_axis_label="number of distinct tiles a user has visited",
        x_axis_type=int,
    )
    return utils.fig_to_html(hist)


def render_uncorrelated_entropy(uncorrelated_entropy):
    hist = plot.histogram(
        (uncorrelated_entropy[0], uncorrelated_entropy[1].round(2)),
        x_axis_label="uncorrelated entropy",
    )
    return utils.fig_to_html(hist)


def render_real_entropy(real_entropy):
    hist = plot.histogram(
        (real_entropy[0], real_entropy[1].round(2)), x_axis_label="real entropy"
    )
    return utils.fig_to_html(hist)

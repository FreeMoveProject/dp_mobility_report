from datetime import timedelta
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport, BenchmarkReport

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import TupleSection
from dp_mobility_report.report.html.html_utils import (
    all_available_measures,
    fmt_moe,
    get_template,
    render_benchmark_summary,
    render_eps,
    render_summary,
    render_user_input_info,
)
from dp_mobility_report.visualization import plot, v_utils


def render_user_analysis(dpmreport: "DpMobilityReport") -> str:

    args: dict = {}
    report = dpmreport.report

    args[
        "trips_per_user_info"
    ] = f"Trips per user are limited according to the configured maximum of trips per user: {dpmreport.max_trips_per_user}"

    if const.TRIPS_PER_USER not in dpmreport.analysis_exclusion:
        args["trips_per_user_eps"] = render_eps(
            report[const.TRIPS_PER_USER].privacy_budget
        )
        args["trips_per_user_moe"] = fmt_moe(
            report[const.TRIPS_PER_USER].margin_of_error_laplace
        )
        args["trips_per_user_hist"] = render_trips_per_user(
            report[const.TRIPS_PER_USER]
        )
        args["trips_per_user_summary_table"] = render_summary(
            report[const.TRIPS_PER_USER].quartiles
        )

    if const.USER_TIME_DELTA not in dpmreport.analysis_exclusion:
        args["time_between_traj_eps"] = render_eps(
            report[const.USER_TIME_DELTA].privacy_budget
        )
        args["time_between_traj_moe"] = fmt_moe(
            report[const.USER_TIME_DELTA].margin_of_error_laplace
        )
        if report[const.USER_TIME_DELTA].quartiles["min"] < timedelta(seconds=0):
            args[
                "plausi_check_info"
            ] = """<strong>Plausibility check</strong>: 
            There are overlapping trips in the dataset.
            The negative minimum time delta implies that there is a trip of a user that starts before the previous one has ended. 
            This might be an indication of an error in the dataset."""
        args["time_between_traj_hist_info"] = render_user_input_info(
            dpmreport.max_user_time_delta, dpmreport.bin_range_user_time_delta
        )
        args["time_between_traj_hist"] = render_time_between_traj(
            report[const.USER_TIME_DELTA]
        )
        args["time_between_traj_summary_table"] = render_summary(
            report[const.USER_TIME_DELTA].quartiles
        )

    if const.RADIUS_OF_GYRATION not in dpmreport.analysis_exclusion:
        args["radius_of_gyration_eps"] = render_eps(
            report[const.RADIUS_OF_GYRATION].privacy_budget
        )
        args["radius_of_gyration_moe"] = fmt_moe(
            report[const.RADIUS_OF_GYRATION].margin_of_error_laplace
        )
        args["radius_of_gyration_hist_info"] = render_user_input_info(
            dpmreport.max_radius_of_gyration, dpmreport.bin_range_radius_of_gyration
        )
        args["radius_of_gyration_hist"] = render_radius_of_gyration(
            report[const.RADIUS_OF_GYRATION]
        )
        args["radius_of_gyration_summary_table"] = render_summary(
            report[const.RADIUS_OF_GYRATION].quartiles
        )

    if const.USER_TILE_COUNT not in dpmreport.analysis_exclusion:
        args["distinct_tiles_user_eps"] = render_eps(
            report[const.USER_TILE_COUNT].privacy_budget
        )
        args["distinct_tiles_user_moe"] = fmt_moe(
            report[const.USER_TILE_COUNT].margin_of_error_laplace
        )
        args["distinct_tiles_user_hist_info"] = render_user_input_info(
            dpmreport.max_user_tile_count, dpmreport.bin_range_user_tile_count
        )
        args["distinct_tiles_user_hist"] = render_distinct_tiles_user(
            report[const.USER_TILE_COUNT]
        )
        args["distinct_tiles_user_summary_table"] = render_summary(
            report[const.USER_TILE_COUNT].quartiles
        )

    if const.MOBILITY_ENTROPY not in dpmreport.analysis_exclusion:
        args["mobility_entropy_eps"] = render_eps(
            report[const.MOBILITY_ENTROPY].privacy_budget
        )
        args["mobility_entropy_moe"] = fmt_moe(
            report[const.MOBILITY_ENTROPY].margin_of_error_laplace
        )
        args["mobility_entropy_hist"] = render_mobility_entropy(
            report[const.MOBILITY_ENTROPY]
        )
        args["mobility_entropy_summary_table"] = render_summary(
            report[const.MOBILITY_ENTROPY].quartiles
        )

    template_structure = get_template("user_analysis_segment.html")

    return template_structure.render(args)


def render_benchmark_user_analysis(benchmark: "BenchmarkReport") -> str:

    args: dict = {}
    report_base = benchmark.report_base.report
    report_alternative = benchmark.report_alternative.report
    template_measures = get_template("similarity_measures.html")

    args[
        "trips_per_user_info"
    ] = f"Trips per user are limited according to the configured maximum of trips per user: {benchmark.report_base.max_trips_per_user} (base); {benchmark.report_alternative.max_trips_per_user} (alternative)."

    if const.TRIPS_PER_USER not in benchmark.analysis_exclusion:
        args["trips_per_user_eps"] = (
            render_eps(report_base[const.TRIPS_PER_USER].privacy_budget),
            render_eps(report_alternative[const.TRIPS_PER_USER].privacy_budget),
        )
        args["trips_per_user_moe"] = (
            fmt_moe(report_base[const.TRIPS_PER_USER].margin_of_error_laplace),
            fmt_moe(report_alternative[const.TRIPS_PER_USER].margin_of_error_laplace),
        )
        # args["trips_per_user_hist"] = render_trips_per_user(report_base[const.TRIPS_PER_USER],report_alternative[const.TRIPS_PER_USER]) #TODO bins need to be the same
        args["trips_per_user_summary_table"] = render_benchmark_summary(
            report_base[const.TRIPS_PER_USER].quartiles,
            report_alternative[const.TRIPS_PER_USER].quartiles,
            target_type=int,
        )
        args["trips_per_user_measure"] = template_measures.render(
            all_available_measures(const.TRIPS_PER_USER, benchmark)
        )

        args["trips_per_user_summary_measure"] = template_measures.render(
            all_available_measures(const.TRIPS_PER_USER_QUARTILES, benchmark)
        )

    if const.USER_TIME_DELTA not in benchmark.analysis_exclusion:
        args["time_between_traj_eps"] = (
            render_eps(report_base[const.USER_TIME_DELTA].privacy_budget),
            render_eps(report_alternative[const.USER_TIME_DELTA].privacy_budget),
        )
        args["time_between_traj_moe"] = (
            fmt_moe(report_base[const.USER_TIME_DELTA].margin_of_error_laplace),
            fmt_moe(report_alternative[const.USER_TIME_DELTA].margin_of_error_laplace),
        )
        args["time_between_traj_hist_info"] = render_user_input_info(
            benchmark.report_base.max_user_time_delta,
            benchmark.report_base.bin_range_user_time_delta,
        )
        args["time_between_traj_hist"] = render_time_between_traj(
            report_base[const.USER_TIME_DELTA],
            report_alternative[const.USER_TIME_DELTA],
        )
        args["time_between_traj_summary_table"] = render_benchmark_summary(
            report_base[const.USER_TIME_DELTA].quartiles,
            report_alternative[const.USER_TIME_DELTA].quartiles,
        )
        args["time_between_traj_measure"] = template_measures.render(
            all_available_measures(const.USER_TIME_DELTA, benchmark)
        )

        args["time_between_traj_summary_measure"] = template_measures.render(
            all_available_measures(const.USER_TIME_DELTA_QUARTILES, benchmark)
        )

    if const.RADIUS_OF_GYRATION not in benchmark.analysis_exclusion:
        args["radius_of_gyration_eps"] = (
            render_eps(report_base[const.RADIUS_OF_GYRATION].privacy_budget),
            render_eps(report_alternative[const.RADIUS_OF_GYRATION].privacy_budget),
        )
        args["radius_of_gyration_moe"] = (
            fmt_moe(report_base[const.RADIUS_OF_GYRATION].margin_of_error_laplace),
            fmt_moe(
                report_alternative[const.RADIUS_OF_GYRATION].margin_of_error_laplace
            ),
        )
        args["radius_of_gyration_hist_info"] = render_user_input_info(
            benchmark.report_base.max_radius_of_gyration,
            benchmark.report_base.bin_range_radius_of_gyration,
        )
        args["radius_of_gyration_hist"] = render_radius_of_gyration(
            report_base[const.RADIUS_OF_GYRATION],
            report_alternative[const.RADIUS_OF_GYRATION],
        )
        args["radius_of_gyration_summary_table"] = render_benchmark_summary(
            report_base[const.RADIUS_OF_GYRATION].quartiles,
            report_alternative[const.RADIUS_OF_GYRATION].quartiles,
            target_type=float,
        )
        args["radius_of_gyration_measure"] = template_measures.render(
            all_available_measures(const.RADIUS_OF_GYRATION, benchmark)
        )

        args["radius_of_gyration_summary_measure"] = template_measures.render(
            all_available_measures(const.RADIUS_OF_GYRATION_QUARTILES, benchmark)
        )

    if const.USER_TILE_COUNT not in benchmark.analysis_exclusion:
        args["distinct_tiles_user_eps"] = (
            render_eps(report_base[const.USER_TILE_COUNT].privacy_budget),
            render_eps(report_alternative[const.USER_TILE_COUNT].privacy_budget),
        )
        args["distinct_tiles_user_moe"] = (
            fmt_moe(report_base[const.USER_TILE_COUNT].margin_of_error_laplace),
            fmt_moe(report_alternative[const.USER_TILE_COUNT].margin_of_error_laplace),
        )
        args["distinct_tiles_user_hist_info"] = render_user_input_info(
            benchmark.report_base.max_user_tile_count,
            benchmark.report_base.bin_range_user_tile_count,
        )
        args["distinct_tiles_user_hist"] = render_distinct_tiles_user(
            report_base[const.USER_TILE_COUNT],
            report_alternative[const.USER_TILE_COUNT],
        )
        args["distinct_tiles_user_summary_table"] = render_benchmark_summary(
            report_base[const.USER_TILE_COUNT].quartiles,
            report_alternative[const.USER_TILE_COUNT].quartiles,
            target_type=int,
        )
        args["distinct_tiles_measure"] = template_measures.render(
            all_available_measures(const.USER_TILE_COUNT, benchmark)
        )

        args["distinct_tiles_summary_measure"] = template_measures.render(
            all_available_measures(const.USER_TILE_COUNT_QUARTILES, benchmark)
        )

    if const.MOBILITY_ENTROPY not in benchmark.analysis_exclusion:
        args["mobility_entropy_eps"] = (
            render_eps(report_base[const.MOBILITY_ENTROPY].privacy_budget),
            render_eps(report_alternative[const.MOBILITY_ENTROPY].privacy_budget),
        )
        args["mobility_entropy_moe"] = (
            fmt_moe(report_base[const.MOBILITY_ENTROPY].margin_of_error_laplace),
            fmt_moe(report_alternative[const.MOBILITY_ENTROPY].margin_of_error_laplace),
        )
        args["mobility_entropy_hist"] = render_mobility_entropy(
            report_base[const.MOBILITY_ENTROPY],
            report_alternative[const.MOBILITY_ENTROPY],
        )
        args["mobility_entropy_summary_table"] = render_benchmark_summary(
            report_base[const.MOBILITY_ENTROPY].quartiles,
            report_alternative[const.MOBILITY_ENTROPY].quartiles,
            target_type=float,
        )
        args["mobility_entropy_measure"] = template_measures.render(
            all_available_measures(const.MOBILITY_ENTROPY, benchmark)
        )

        args["mobility_entropy_summary_measure"] = template_measures.render(
            all_available_measures(const.MOBILITY_ENTROPY_QUARTILES, benchmark)
        )

    template_structure = get_template("user_analysis_segment_benchmark.html")

    return template_structure.render(args)


def render_trips_per_user(
    trips_per_user_hist: TupleSection,
    trips_per_user_hist_alternative: Optional[TupleSection] = None,
) -> str:
    if trips_per_user_hist_alternative:
        alternative_data = trips_per_user_hist_alternative.data
        alternative_moe = trips_per_user_hist_alternative.margin_of_error_laplace
    else:
        alternative_data = None
        alternative_moe = None
    hist = plot.histogram(
        hist=trips_per_user_hist.data,
        hist_alternative=alternative_data,
        x_axis_label="Number of trips per user",
        y_axis_label="% of users",
        x_axis_type=int,
        margin_of_error=trips_per_user_hist.margin_of_error_laplace,
        margin_of_error_alternative=alternative_moe,
        figsize=(max(7, min(len(trips_per_user_hist.data[1]) * 0.5, 11)), 6),
    )
    return v_utils.fig_to_html(hist)


def render_time_between_traj(
    time_between_traj_hist: TupleSection,
    time_between_traj_hist_alternative: Optional[TupleSection] = None,
) -> str:
    if time_between_traj_hist_alternative:
        alternative_data = time_between_traj_hist_alternative.data
        alternative_moe = time_between_traj_hist_alternative.margin_of_error_laplace
    else:
        alternative_data = None
        alternative_moe = None
    hist = plot.histogram(
        hist=time_between_traj_hist.data,
        hist_alternative=alternative_data,
        x_axis_label="Hours between consecutive trips",
        y_axis_label="% of trips",
        x_axis_type=float,
        ndigits_x_label=1,
        margin_of_error=time_between_traj_hist.margin_of_error_laplace,
        margin_of_error_alternative=alternative_moe,
        figsize=(max(7, min(len(time_between_traj_hist.data[1]) * 0.5, 11)), 6),
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_radius_of_gyration(
    radius_of_gyration_hist: TupleSection,
    radius_of_gyration_hist_alternative: Optional[TupleSection] = None,
) -> str:
    if radius_of_gyration_hist_alternative:
        alternative_data = radius_of_gyration_hist_alternative.data
        alternative_error = radius_of_gyration_hist_alternative.margin_of_error_laplace
    else:
        alternative_data = None
        alternative_error = None
    hist = plot.histogram(
        hist=radius_of_gyration_hist.data,
        hist_alternative=alternative_data,
        x_axis_label="radius of gyration (in km)",
        y_axis_label="% of users",
        x_axis_type=float,
        margin_of_error=radius_of_gyration_hist.margin_of_error_laplace,
        margin_of_error_alternative=alternative_error,
        figsize=(max(7, min(len(radius_of_gyration_hist.data[1]) * 0.5, 11)), 6),
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_distinct_tiles_user(
    user_tile_count_hist: TupleSection,
    user_tile_count_hist_alternative: Optional[TupleSection] = None,
) -> str:
    if user_tile_count_hist_alternative:
        alternative_data = user_tile_count_hist_alternative.data
        alternative_error = user_tile_count_hist_alternative.margin_of_error_laplace
    else:
        alternative_data = None
        alternative_error = None
    hist = plot.histogram(
        hist=user_tile_count_hist.data,
        hist_alternative=alternative_data,
        x_axis_label="number of distinct tiles a user has visited",
        y_axis_label="% of users",
        x_axis_type=int,
        margin_of_error=user_tile_count_hist.margin_of_error_laplace,
        margin_of_error_alternative=alternative_error,
        figsize=(max(7, min(len(user_tile_count_hist.data[1]) * 0.5, 11)), 6),
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_mobility_entropy(
    mobility_entropy: TupleSection,
    mobility_entropy_alternative: Optional[TupleSection] = None,
) -> str:
    if mobility_entropy_alternative:
        alternative_data = (
            mobility_entropy_alternative.data[0],
            mobility_entropy_alternative.data[1].round(2),
        )
        alternative_error = mobility_entropy_alternative.margin_of_error_laplace
    else:
        alternative_data = None
        alternative_error = None
    hist = plot.histogram(
        hist=(mobility_entropy.data[0], mobility_entropy.data[1].round(2)),
        hist_alternative=alternative_data,
        min_value=mobility_entropy.quartiles["min"],
        x_axis_label="mobility entropy",
        y_axis_label="% of users",
        margin_of_error=mobility_entropy.margin_of_error_laplace,
        margin_of_error_alternative=alternative_error,
        figsize=(max(7, min(len(mobility_entropy.data[1]) * 0.5, 11)), 6),
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html

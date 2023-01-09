from datetime import timedelta
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport, BenchmarkReport

from dp_mobility_report import constants as const
from dp_mobility_report.model.section import TupleSection
from dp_mobility_report.report.html.html_utils import (
    get_template,
    fmt,
    render_eps,
    render_moe_info,
    render_summary,
    render_benchmark_summary,
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

    args[
        "trips_per_user_info"
    ] = f"Trips per user are limited according to the configured maximum of trips per user: {dpmreport.max_trips_per_user}"

    if const.TRIPS_PER_USER not in dpmreport.analysis_exclusion:
        args["trips_per_user_eps"] = render_eps(
            report[const.TRIPS_PER_USER].privacy_budget
        )
        args["trips_per_user_hist"] = render_trips_per_user(
            report[const.TRIPS_PER_USER]
        )
        args["trips_per_user_summary_table"] = render_summary(
            report[const.TRIPS_PER_USER].quartiles
        )
        args["trips_per_user_moe_info"] = render_moe_info(
            report[const.TRIPS_PER_USER].margin_of_error_expmech
        )

    if const.USER_TIME_DELTA not in dpmreport.analysis_exclusion:
        args["time_between_traj_eps"] = render_eps(
            report[const.USER_TIME_DELTA].privacy_budget
        )
        if report[const.USER_TIME_DELTA].quartiles["min"] < timedelta(seconds=0):
            args[
                "plausi_check_info"
            ] = """<strong>Plausibility check</strong>: 
            There are overlapping trips in the dataset.
            The negative minimum time delta implies that there is a trip of a user that starts before the previous one has ended. 
            This might be an indication of an error in the dataset."""
        args["time_between_traj_hist"] = render_time_between_traj(
            report[const.USER_TIME_DELTA]
        )
        args["time_between_traj_summary_table"] = render_summary(
            report[const.USER_TIME_DELTA].quartiles
        )
        args["time_between_traj_moe_info"] = render_moe_info(
            report[const.USER_TIME_DELTA].margin_of_error_expmech
        )

    if const.RADIUS_OF_GYRATION not in dpmreport.analysis_exclusion:
        args["radius_of_gyration_eps"] = render_eps(
            report[const.RADIUS_OF_GYRATION].privacy_budget
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
        args["radius_of_gyration_moe_info"] = render_moe_info(
            report[const.RADIUS_OF_GYRATION].margin_of_error_expmech
        )

    if const.USER_TILE_COUNT not in dpmreport.analysis_exclusion:
        args["distinct_tiles_user_eps"] = render_eps(
            report[const.USER_TILE_COUNT].privacy_budget
        )
        args["distinct_tiles_user_hist"] = render_distinct_tiles_user(
            report[const.USER_TILE_COUNT]
        )
        args["distinct_tiles_user_summary_table"] = render_summary(
            report[const.USER_TILE_COUNT].quartiles
        )
        args["distinct_tiles_moe_info"] = render_moe_info(
            report[const.USER_TILE_COUNT].margin_of_error_expmech
        )

    if const.MOBILITY_ENTROPY not in dpmreport.analysis_exclusion:
        args["mobility_entropy_eps"] = render_eps(
            report[const.MOBILITY_ENTROPY].privacy_budget
        )
        args["mobility_entropy_hist"] = render_mobility_entropy(
            report[const.MOBILITY_ENTROPY]
        )
        args["mobility_entropy_summary_table"] = render_summary(
            report[const.MOBILITY_ENTROPY].quartiles
        )
        args["mobility_entropy_moe_info"] = render_moe_info(
            report[const.MOBILITY_ENTROPY].margin_of_error_expmech
        )

    template_structure = get_template("user_analysis_segment.html")

    return template_structure.render(args)


def render_benchmark_user_analysis(
    report_base: "DpMobilityReport", 
    report_alternative: "DpMobilityReport", 
    analysis_exclusion, 
    benchmark:"BenchmarkReport") -> str:
    
    trips_per_user_info = f"Trips per user are limited according to the configured maximum of trips per user: {report_base.max_trips_per_user} (base); {report_alternative.max_trips_per_user} (alternative)."
    trips_per_user_eps = None
    trips_per_user_hist = ""
    trips_per_user_summary_table = ""
    trips_per_user_moe_info = ""
    trips_per_user_measure = ""
    time_between_traj_eps = None
    overlapping_trips_info = ""
    time_between_traj_hist = ""
    time_between_traj_summary_table = ""
    time_between_traj_moe_info = ""
    time_between_traj_measure = ""
    radius_of_gyration_eps = None
    radius_of_gyration_hist_info = ""
    radius_of_gyration_hist = ""
    radius_of_gyration_summary_table = ""
    radius_of_gyration_moe_info = ""
    radius_of_gyration_measure = ""
    distinct_tiles_user_eps = None
    distinct_tiles_user_hist = ""
    distinct_tiles_user_summary_table = ""
    distinct_tiles_moe_info = ""
    distinct_tiles_measure = ""
    mobility_entropy_eps = None
    mobility_entropy_hist = ""
    mobility_entropy_summary_table = ""
    mobility_entropy_moe_info = ""
    mobility_entropy_measure = ""

    report_base = report_base.report
    report_alternative = report_alternative.report

    if const.TRIPS_PER_USER not in analysis_exclusion:
        trips_per_user_eps = (render_eps(report_base[const.TRIPS_PER_USER].privacy_budget),render_eps(report_alternative[const.TRIPS_PER_USER].privacy_budget))

        #trips_per_user_hist = render_trips_per_user(report_base[const.TRIPS_PER_USER],report_alternative[const.TRIPS_PER_USER]) #TODO bins need to be the same
        trips_per_user_summary_table = render_benchmark_summary(
            report_base[const.TRIPS_PER_USER].quartiles, report_alternative[const.TRIPS_PER_USER].quartiles)
        trips_per_user_moe_info = render_moe_info(
            report_base[const.TRIPS_PER_USER].margin_of_error_expmech
        )
        trips_per_user_measure=(const.format[benchmark.measure_selection[const.TRIPS_PER_USER]], fmt(benchmark.similarity_measures[const.TRIPS_PER_USER]))
    
    if const.USER_TIME_DELTA not in analysis_exclusion:
        time_between_traj_eps = (render_eps(report_base[const.USER_TIME_DELTA].privacy_budget), render_eps(report_alternative[const.USER_TIME_DELTA].privacy_budget))
        time_between_traj_hist = render_time_between_traj(report_base[const.USER_TIME_DELTA], report_alternative[const.USER_TIME_DELTA])
        time_between_traj_summary_table = render_benchmark_summary(
            report_base[const.USER_TIME_DELTA].quartiles, report_alternative[const.USER_TIME_DELTA].quartiles
        )
        time_between_traj_moe_info = render_moe_info(
            report_base[const.USER_TIME_DELTA].margin_of_error_expmech
        )
        time_between_traj_measure=(const.format[benchmark.measure_selection[const.USER_TIME_DELTA]], fmt(benchmark.similarity_measures[const.USER_TIME_DELTA]))

    if const.RADIUS_OF_GYRATION not in analysis_exclusion:
        radius_of_gyration_eps = (render_eps(
            report_base[const.RADIUS_OF_GYRATION].privacy_budget), render_eps(report_alternative[const.RADIUS_OF_GYRATION].privacy_budget))
        
        radius_of_gyration_hist_info = render_user_input_info(
            benchmark.report_base.max_radius_of_gyration, benchmark.report_base.bin_range_radius_of_gyration
        )
        radius_of_gyration_hist = render_radius_of_gyration(
            report_base[const.RADIUS_OF_GYRATION], report_alternative[const.RADIUS_OF_GYRATION]
        )
        radius_of_gyration_summary_table = render_benchmark_summary(
            report_base[const.RADIUS_OF_GYRATION].quartiles, report_alternative[const.RADIUS_OF_GYRATION].quartiles
        )
        radius_of_gyration_moe_info = render_moe_info(
            report_base[const.RADIUS_OF_GYRATION].margin_of_error_expmech
        )
        radius_of_gyration_measure=(const.format[benchmark.measure_selection[const.RADIUS_OF_GYRATION]], fmt(benchmark.similarity_measures[const.RADIUS_OF_GYRATION]))

    if const.USER_TILE_COUNT not in analysis_exclusion:
        distinct_tiles_user_eps = (render_eps(report_base[const.USER_TILE_COUNT].privacy_budget), render_eps(report_alternative[const.USER_TILE_COUNT].privacy_budget))

        distinct_tiles_user_hist = render_distinct_tiles_user(
            report_base[const.USER_TILE_COUNT], report_alternative[const.USER_TILE_COUNT]
        )
        distinct_tiles_user_summary_table = render_benchmark_summary(
            report_base[const.USER_TILE_COUNT].quartiles, report_alternative[const.USER_TILE_COUNT].quartiles
        )
        distinct_tiles_moe_info = render_moe_info(
            report_base[const.USER_TILE_COUNT].margin_of_error_expmech
        )
        distinct_tiles_measure=(const.format[benchmark.measure_selection[const.USER_TILE_COUNT]], fmt(benchmark.similarity_measures[const.USER_TILE_COUNT]))


    if const.MOBILITY_ENTROPY not in analysis_exclusion:
        mobility_entropy_eps = (render_eps(report_base[const.MOBILITY_ENTROPY].privacy_budget),render_eps(report_alternative[const.MOBILITY_ENTROPY].privacy_budget))
        mobility_entropy_hist = render_mobility_entropy(report_base[const.MOBILITY_ENTROPY], report_alternative[const.MOBILITY_ENTROPY])
        mobility_entropy_summary_table = render_benchmark_summary(
            report_base[const.MOBILITY_ENTROPY].quartiles, report_alternative[const.MOBILITY_ENTROPY].quartiles
        )
        mobility_entropy_moe_info = render_moe_info(
            report_base[const.MOBILITY_ENTROPY].margin_of_error_expmech
        )
        mobility_entropy_measure=(const.format[benchmark.measure_selection[const.MOBILITY_ENTROPY]], fmt(benchmark.similarity_measures[const.MOBILITY_ENTROPY]))
    template_structure = get_template("user_analysis_segment_benchmark.html")

    return template_structure.render(
        trips_per_user_eps=trips_per_user_eps,
        trips_per_user_info=trips_per_user_info,
        trips_per_user_hist=trips_per_user_hist,
        trips_per_user_summary_table=trips_per_user_summary_table,
        trips_per_user_moe_info=trips_per_user_moe_info,
        trips_per_user_measure=trips_per_user_measure,
        time_between_traj_eps=time_between_traj_eps,
        #overlapping_trips_info=overlapping_trips_info,
        time_between_traj_hist=time_between_traj_hist,
        time_between_traj_summary_table=time_between_traj_summary_table,
        time_between_traj_moe_info=time_between_traj_moe_info,
        time_between_traj_measure=time_between_traj_measure,
        radius_of_gyration_eps=radius_of_gyration_eps,
        radius_of_gyration_hist_info=radius_of_gyration_hist_info,
        radius_of_gyration_hist=radius_of_gyration_hist,
        radius_of_gyration_summary_table=radius_of_gyration_summary_table,
        radius_of_gyration_moe_info=radius_of_gyration_moe_info,
        radius_of_gyration_measure=radius_of_gyration_measure,
        distinct_tiles_user_eps=distinct_tiles_user_eps,
        distinct_tiles_user_hist=distinct_tiles_user_hist,
        distinct_tiles_user_summary_table=distinct_tiles_user_summary_table,
        distinct_tiles_moe_info=distinct_tiles_moe_info,
        distinct_tiles_measure=distinct_tiles_measure,
        mobility_entropy_eps=mobility_entropy_eps,
        mobility_entropy_hist=mobility_entropy_hist,
        mobility_entropy_summary_table=mobility_entropy_summary_table,
        mobility_entropy_moe_info=mobility_entropy_moe_info,
        mobility_entropy_measure=mobility_entropy_measure
    )


def render_trips_per_user(trips_per_user_hist: TupleSection, trips_per_user_hist_alternative: Optional[TupleSection]=None) -> str:
    if trips_per_user_hist_alternative:
        alternative_data = trips_per_user_hist_alternative.data
    else:
        alternative_data = None
    hist = plot.histogram(
        hist=trips_per_user_hist.data,
        hist_alternative=trips_per_user_hist_alternative.data,
        x_axis_label="Number of trips per user",
        y_axis_label="% of users",
        x_axis_type=int,
        margin_of_error=trips_per_user_hist.margin_of_error_laplace,
        margin_of_error_alternative=trips_per_user_hist_alternative.margin_of_error_laplace,
    )
    return v_utils.fig_to_html(hist)


def render_time_between_traj(time_between_traj_hist: TupleSection, time_between_traj_hist_alternative: Optional[TupleSection]=None) -> str:
    hist = plot.histogram(
        hist=time_between_traj_hist.data,
        hist_alternative=time_between_traj_hist_alternative.data,
        x_axis_label="Hours between consecutive trips",
        y_axis_label="% of trips",
        x_axis_type=float,
        ndigits_x_label=1,
        margin_of_error=time_between_traj_hist.margin_of_error_laplace,
        margin_of_error_alternative=time_between_traj_hist_alternative.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_radius_of_gyration(radius_of_gyration_hist: TupleSection, radius_of_gyration_hist_alternative: Optional[TupleSection]=None) -> str:
    hist = plot.histogram(
        hist=radius_of_gyration_hist.data,
        hist_alternative=radius_of_gyration_hist_alternative.data,
        x_axis_label="radius of gyration (in km)",
        y_axis_label="% of users",
        x_axis_type=float,
        margin_of_error=radius_of_gyration_hist.margin_of_error_laplace,
        margin_of_error_alternative=radius_of_gyration_hist_alternative.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_distinct_tiles_user(user_tile_count_hist: TupleSection, user_tile_count_hist_alternative: Optional[TupleSection]=None) -> str:
    hist = plot.histogram(
        hist=user_tile_count_hist.data,
        hist_alternative=user_tile_count_hist_alternative.data,
        x_axis_label="number of distinct tiles a user has visited",
        y_axis_label="% of users",
        x_axis_type=int,
        margin_of_error=user_tile_count_hist.margin_of_error_laplace,
        margin_of_error_alternative=user_tile_count_hist_alternative.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html


def render_mobility_entropy(mobility_entropy: TupleSection, mobility_entropy_alternative: Optional[TupleSection]=None) -> str:
    hist = plot.histogram(
        hist=(mobility_entropy.data[0], mobility_entropy.data[1].round(2)),
        hist_alternative=(mobility_entropy_alternative.data[0], mobility_entropy_alternative.data[1].round(2)),
        min_value=mobility_entropy.quartiles["min"],
        x_axis_label="mobility entropy",
        y_axis_label="% of users",
        margin_of_error=mobility_entropy.margin_of_error_laplace,
        margin_of_error_alternative=mobility_entropy_alternative.margin_of_error_laplace,
    )
    html = v_utils.fig_to_html(hist)
    plt.close()
    return html

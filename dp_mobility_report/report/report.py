from tqdm.auto import tqdm

from dp_mobility_report import constants as const
from dp_mobility_report.model import (
    od_analysis,
    overview,
    place_analysis,
    user_analysis,
)


def report_elements(mdreport):

    report = {}
    is_all_analyses = const.ALL in mdreport.analysis_selection

    with tqdm(  # progress bar
        total=4, desc="Create report", disable=mdreport.disable_progress_bar
    ) as pbar:
        if is_all_analyses | (const.OVERVIEW in mdreport.analysis_selection):
            report = {**report, **add_overview_elements(mdreport)}
        pbar.update()

        if is_all_analyses | (const.PLACE_ANALYSIS in mdreport.analysis_selection):
            report = {**report, **add_place_analysis_elements(mdreport)}
        pbar.update()

        if is_all_analyses | (const.OD_ANALYSIS in mdreport.analysis_selection):
            _od_shape = od_analysis.get_od_shape(mdreport.df, mdreport.tessellation)
            report = {**report, **add_od_analysis_elements(mdreport, _od_shape)}
        pbar.update()

        if is_all_analyses | (const.USER_ANALYSIS in mdreport.analysis_selection):
            report = {**report, **add_user_analysis_elements(mdreport)}
        pbar.update()

    return report


def add_overview_elements(mdreport):
    if mdreport.privacy_budget is None or mdreport.evalu is True:
        epsilon = mdreport.privacy_budget
    else:
        epsilon = mdreport.privacy_budget / 6
    return {
        "ds_statistics": overview.get_dataset_statistics(mdreport, epsilon),
        "missing_values": overview.get_missing_values(mdreport, epsilon),
        "trips_over_time": overview.get_trips_over_time(mdreport, epsilon),
        "trips_per_weekday": overview.get_trips_per_weekday(mdreport, epsilon),
        "trips_per_hour": overview.get_trips_per_hour(mdreport, epsilon),
    }


def add_place_analysis_elements(mdreport):
    if mdreport.privacy_budget is None or mdreport.evalu is True:
        epsilon = mdreport.privacy_budget
    else:
        epsilon = mdreport.privacy_budget / 2
    return {
        "counts_per_tile": place_analysis.get_visits_per_tile(mdreport, epsilon),
        "counts_per_tile_timewindow": place_analysis.get_visits_per_tile_timewindow(
            mdreport, epsilon
        ),
    }


def add_od_analysis_elements(mdreport, _od_shape):
    if mdreport.privacy_budget is None or mdreport.evalu is True:
        epsilon = mdreport.privacy_budget
    else:
        epsilon = mdreport.privacy_budget / 3
    return {
        "od_flows": od_analysis.get_od_flows(_od_shape, mdreport, epsilon),
        "travel_time": od_analysis.get_travel_time(_od_shape, mdreport, epsilon),
        "jump_length": od_analysis.get_jump_length(_od_shape, mdreport, epsilon),
    }


def add_user_analysis_elements(mdreport):
    if mdreport.privacy_budget is None or mdreport.evalu is True:
        epsilon = mdreport.privacy_budget
    else:
        epsilon = mdreport.privacy_budget / 7
    return {
        "trips_per_user": user_analysis.get_trips_per_user(mdreport, epsilon),
        "user_time_delta": user_analysis.get_user_time_delta(mdreport, epsilon),
        "radius_of_gyration": user_analysis.get_radius_of_gyration(mdreport, epsilon),
        "location_entropy": user_analysis.get_location_entropy(mdreport, epsilon),
        "user_tile_count": user_analysis.get_user_tile_count(mdreport, epsilon),
        "mobility_entropy": user_analysis.get_mobility_entropy(mdreport, epsilon),
    }

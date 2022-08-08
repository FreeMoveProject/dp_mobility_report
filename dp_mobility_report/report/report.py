from typing import TYPE_CHECKING, Optional, Tuple

from pandas import DataFrame
from tqdm.auto import tqdm

from dp_mobility_report.model.section import Section

if TYPE_CHECKING:
    from dp_mobility_report.md_report import MobilityDataReport

from dp_mobility_report import constants as const
from dp_mobility_report.model import (
    od_analysis,
    overview,
    place_analysis,
    user_analysis,
)


def get_analysis_elements_info(analysis_selection: list, budget_split: dict) -> Tuple:
    is_all_analyses = const.ALL in analysis_selection

    if is_all_analyses:
        element_count = (
            len(const.OVERVIEW_ELEMENTS)
            + len(const.PLACE_ELEMENTS)
            + len(const.OD_ELEMENTS)
            + len(const.USER_ELEMENTS)
        )

    else:
        element_count = (
            len(const.OVERVIEW_ELEMENTS)
            if (const.OVERVIEW in analysis_selection)
            else 0
        )
        element_count += (
            len(const.PLACE_ELEMENTS)
            if (const.PLACE_ANALYSIS in analysis_selection)
            else 0
        )
        element_count += (
            len(const.OD_ELEMENTS) if (const.OD_ANALYSIS in analysis_selection) else 0
        )
        element_count += (
            len(const.USER_ELEMENTS)
            if (const.USER_ANALYSIS in analysis_selection)
            else 0
        )

    # TODO: warning, if elements in budget split but not in selected elements
    # TODO: dont consider elements in budget split that are not selected as elements
    budget_split_sum = (
        element_count - len(budget_split.keys()) + sum(budget_split.values())
    )

    return budget_split_sum, is_all_analyses


def report_elements(mdreport: "MobilityDataReport") -> dict:

    report: dict = {}
    budget_split_sum, is_all_analyses = get_analysis_elements_info(
        mdreport.analysis_selection, mdreport.budget_split
    )
    record_count = None
    trip_count = None

    # get privacy budget for each report element
    if mdreport.privacy_budget is None or mdreport.evalu:
        eps_factor = mdreport.privacy_budget
    else:
        eps_factor = (
            mdreport.privacy_budget / budget_split_sum
            if budget_split_sum > 0
            else mdreport.privacy_budget
        )

    # TODO: multiply each element with its budget split
    with tqdm(  # progress bar
        total=4, desc="Create report", disable=mdreport.disable_progress_bar
    ) as pbar:
        if is_all_analyses | (const.OVERVIEW in mdreport.analysis_selection):
            report = {**report, **add_overview_elements(mdreport, eps_factor)}
            record_count = report[const.DS_STATISTICS].data["n_records"]
            trip_count = report[const.DS_STATISTICS].data["n_trips"]
        pbar.update()

        if is_all_analyses | (const.PLACE_ANALYSIS in mdreport.analysis_selection):
            report = {
                **report,
                **add_place_analysis_elements(mdreport, eps_factor, record_count),
            }
        pbar.update()

        if is_all_analyses | (const.OD_ANALYSIS in mdreport.analysis_selection):
            _od_shape = od_analysis.get_od_shape(mdreport.df, mdreport.tessellation)
            report = {
                **report,
                **add_od_analysis_elements(mdreport, _od_shape, eps_factor, trip_count),
            }
        pbar.update()

        if is_all_analyses | (const.USER_ANALYSIS in mdreport.analysis_selection):
            report = {**report, **add_user_analysis_elements(mdreport, eps_factor)}
        pbar.update()

    return report


def _get_eps(eps_factor: float, analysis_name: str, budget_split: dict) -> float:
    if eps_factor is None:
        return None
    elif analysis_name not in budget_split:
        # analysis gets default factor of 1
        return eps_factor
    else:
        # else epsilon is multiplied by the respective configured factor
        return budget_split[analysis_name] * eps_factor


def add_overview_elements(mdreport: "MobilityDataReport", eps_factor: float) -> dict:
    return {
        const.DS_STATISTICS: overview.get_dataset_statistics(
            mdreport, _get_eps(eps_factor, const.DS_STATISTICS, mdreport.budget_split)
        )
        if const.DS_STATISTICS in const.OVERVIEW_ELEMENTS
        else Section(),
        const.MISSING_VALUES: overview.get_missing_values(
            mdreport, _get_eps(eps_factor, const.MISSING_VALUES, mdreport.budget_split))
        if const.MISSING_VALUES in const.OVERVIEW_ELEMENTS
        else Section(),
        const.TRIPS_OVER_TIME: overview.get_trips_over_time(
            mdreport, _get_eps(eps_factor, const.TRIPS_OVER_TIME, mdreport.budget_split))
        if const.TRIPS_OVER_TIME in const.OVERVIEW_ELEMENTS
        else Section(),
        const.TRIPS_PER_WEEKDAY: overview.get_trips_per_weekday(
            mdreport, _get_eps(eps_factor, const.TRIPS_PER_WEEKDAY, mdreport.budget_split))
        if const.TRIPS_PER_WEEKDAY in const.OVERVIEW_ELEMENTS
        else Section(),
        const.TRIPS_PER_HOUR: overview.get_trips_per_hour(
            mdreport, _get_eps(eps_factor, const.TRIPS_PER_HOUR, mdreport.budget_split))
        if const.TRIPS_PER_HOUR in const.OVERVIEW_ELEMENTS
        else Section(),
    }


def add_place_analysis_elements(
    mdreport: "MobilityDataReport", eps_factor: float, record_count: Optional[int]
) -> dict:
    return {
        const.VISITS_PER_TILE: place_analysis.get_visits_per_tile(
            mdreport, _get_eps(eps_factor, const.VISITS_PER_TILE, mdreport.budget_split), record_count
        )
        if const.VISITS_PER_TILE in const.PLACE_ELEMENTS
        else Section(),
        const.VISITS_PER_TILE_TIMEWINDOW: place_analysis.get_visits_per_tile_timewindow(
            mdreport, _get_eps(eps_factor, const.VISITS_PER_TILE_TIMEWINDOW, mdreport.budget_split), record_count
        )
        if const.VISITS_PER_TILE_TIMEWINDOW in const.PLACE_ELEMENTS
        else Section(),
    }


def add_od_analysis_elements(
    mdreport: "MobilityDataReport",
    _od_shape: DataFrame,
    eps_factor: float,
    trip_count: Optional[int],
) -> dict:
    return {
        const.OD_FLOWS: od_analysis.get_od_flows(
            _od_shape, mdreport, _get_eps(eps_factor, const.OD_FLOWS, mdreport.budget_split), trip_count
        )
        if const.OD_FLOWS in const.OD_ELEMENTS
        else Section(),
        const.TRAVEL_TIME: od_analysis.get_travel_time(
            _od_shape, mdreport, _get_eps(eps_factor, const.TRAVEL_TIME, mdreport.budget_split))
        if const.TRAVEL_TIME in const.OD_ELEMENTS
        else Section(),
        const.JUMP_LENGTH: od_analysis.get_jump_length(
            _od_shape, mdreport, _get_eps(eps_factor, const.JUMP_LENGTH, mdreport.budget_split))
        if const.JUMP_LENGTH in const.OD_ELEMENTS
        else Section(),
    }


def add_user_analysis_elements(
    mdreport: "MobilityDataReport", eps_factor: float
) -> dict:
    return {
        const.TRIPS_PER_USER: user_analysis.get_trips_per_user(
            mdreport, _get_eps(eps_factor, const.TRIPS_PER_USER, mdreport.budget_split))
        if const.TRIPS_PER_USER in const.USER_ELEMENTS
        else Section(),
        const.USER_TIME_DELTA: user_analysis.get_user_time_delta(
            mdreport, _get_eps(eps_factor, const.USER_TIME_DELTA, mdreport.budget_split))
        if const.USER_TIME_DELTA in const.USER_ELEMENTS
        else Section(),
        const.RADIUS_OF_GYRATION: user_analysis.get_radius_of_gyration(
            mdreport, _get_eps(eps_factor, const.RADIUS_OF_GYRATION, mdreport.budget_split)
        )
        if const.RADIUS_OF_GYRATION in const.USER_ELEMENTS
        else Section(),
        const.USER_TILE_COUNT: user_analysis.get_user_tile_count(
            mdreport, _get_eps(eps_factor, const.USER_TILE_COUNT, mdreport.budget_split))
        if const.USER_TILE_COUNT in const.USER_ELEMENTS
        else Section(),
        const.MOBILITY_ENTROPY: user_analysis.get_mobility_entropy(
            mdreport, _get_eps(eps_factor, const.MOBILITY_ENTROPY, mdreport.budget_split))
        if const.MOBILITY_ENTROPY in const.USER_ELEMENTS
        else Section(),
    }

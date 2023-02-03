from typing import TYPE_CHECKING

from pandas import DataFrame
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from dp_mobility_report import constants as const
from dp_mobility_report.model import (
    od_analysis,
    overview,
    place_analysis,
    user_analysis,
)


def report_elements(dpmreport: "DpMobilityReport") -> dict:

    report: dict = {}

    # budget sum consists of number of elements weighted with 1 (no special budget assigned)
    # plus budget for each custom assigned analysis budget
    budget_split_sum = (
        len(const.ELEMENTS)
        - len(dpmreport.analysis_exclusion)
        - len(dpmreport.budget_split.keys())
        + sum(dpmreport.budget_split.values())
    )

    # get privacy budget for each report element
    if dpmreport.privacy_budget is None or dpmreport.evalu:
        eps_factor = dpmreport.privacy_budget
    else:
        eps_factor = (
            dpmreport.privacy_budget / budget_split_sum
            if budget_split_sum > 0
            else dpmreport.privacy_budget
        )

    with tqdm(  # progress bar
        total=4, desc="Create report", disable=dpmreport.disable_progress_bar
    ) as pbar:

        report = {**report, **add_overview_elements(dpmreport, eps_factor)}
        pbar.update()

        report = {
            **report,
            **add_place_analysis_elements(dpmreport, eps_factor),
        }
        pbar.update()

        if not set(const.OD_ELEMENTS).issubset(dpmreport.analysis_exclusion):
            _od_shape = od_analysis.get_od_shape(dpmreport.df)
            report = {
                **report,
                **add_od_analysis_elements(dpmreport, _od_shape, eps_factor),
            }
        pbar.update()

        report = {**report, **add_user_analysis_elements(dpmreport, eps_factor)}
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


def add_overview_elements(dpmreport: "DpMobilityReport", eps_factor: float) -> dict:
    overview_elements: dict = {}

    if (const.DS_STATISTICS in const.OVERVIEW_ELEMENTS) and (
        const.DS_STATISTICS not in dpmreport.analysis_exclusion
    ):
        overview_elements[const.DS_STATISTICS] = overview.get_dataset_statistics(
            dpmreport, _get_eps(eps_factor, const.DS_STATISTICS, dpmreport.budget_split)
        )

    if (const.MISSING_VALUES in const.OVERVIEW_ELEMENTS) and (
        const.MISSING_VALUES not in dpmreport.analysis_exclusion
    ):
        overview_elements[const.MISSING_VALUES] = overview.get_missing_values(
            dpmreport,
            _get_eps(eps_factor, const.MISSING_VALUES, dpmreport.budget_split),
        )

    if (const.TRIPS_OVER_TIME in const.OVERVIEW_ELEMENTS) and (
        const.TRIPS_OVER_TIME not in dpmreport.analysis_exclusion
    ):
        overview_elements[const.TRIPS_OVER_TIME] = overview.get_trips_over_time(
            dpmreport,
            _get_eps(eps_factor, const.TRIPS_OVER_TIME, dpmreport.budget_split),
        )

    if (const.TRIPS_PER_WEEKDAY in const.TRIPS_PER_WEEKDAY) and (
        const.TRIPS_PER_WEEKDAY not in dpmreport.analysis_exclusion
    ):
        overview_elements[const.TRIPS_PER_WEEKDAY] = overview.get_trips_per_weekday(
            dpmreport,
            _get_eps(eps_factor, const.TRIPS_PER_WEEKDAY, dpmreport.budget_split),
        )

    if (const.TRIPS_PER_HOUR in const.OVERVIEW_ELEMENTS) and (
        const.TRIPS_PER_HOUR not in dpmreport.analysis_exclusion
    ):
        overview_elements[const.TRIPS_PER_HOUR] = overview.get_trips_per_hour(
            dpmreport,
            _get_eps(eps_factor, const.TRIPS_PER_HOUR, dpmreport.budget_split),
        )
    return overview_elements


def add_place_analysis_elements(
    dpmreport: "DpMobilityReport",
    eps_factor: float,
) -> dict:
    place_analysis_elements: dict = {}

    if (const.VISITS_PER_TILE in const.PLACE_ELEMENTS) and (
        const.VISITS_PER_TILE not in dpmreport.analysis_exclusion
    ):
        place_analysis_elements[
            const.VISITS_PER_TILE
        ] = place_analysis.get_visits_per_tile(
            dpmreport,
            _get_eps(eps_factor, const.VISITS_PER_TILE, dpmreport.budget_split),
        )

    if (const.VISITS_PER_TIME_TILE in const.PLACE_ELEMENTS) and (
        const.VISITS_PER_TIME_TILE not in dpmreport.analysis_exclusion
    ):
        place_analysis_elements[
            const.VISITS_PER_TIME_TILE
        ] = place_analysis.get_visits_per_time_tile(
            dpmreport,
            _get_eps(eps_factor, const.VISITS_PER_TIME_TILE, dpmreport.budget_split),
        )
    return place_analysis_elements


def add_od_analysis_elements(
    dpmreport: "DpMobilityReport",
    _od_shape: DataFrame,
    eps_factor: float,
) -> dict:
    od_analysis_elements: dict = {}

    if (const.OD_FLOWS in const.OD_ELEMENTS) and (
        const.OD_FLOWS not in dpmreport.analysis_exclusion
    ):
        od_analysis_elements[const.OD_FLOWS] = od_analysis.get_od_flows(
            _od_shape,
            dpmreport,
            _get_eps(eps_factor, const.OD_FLOWS, dpmreport.budget_split),
        )

    if (const.TRAVEL_TIME in const.OD_ELEMENTS) and (
        const.TRAVEL_TIME not in dpmreport.analysis_exclusion
    ):
        od_analysis_elements[const.TRAVEL_TIME] = od_analysis.get_travel_time(
            _od_shape,
            dpmreport,
            _get_eps(eps_factor, const.TRAVEL_TIME, dpmreport.budget_split),
        )

    if (const.JUMP_LENGTH in const.OD_ELEMENTS) and (
        const.JUMP_LENGTH not in dpmreport.analysis_exclusion
    ):
        od_analysis_elements[const.JUMP_LENGTH] = od_analysis.get_jump_length(
            _od_shape,
            dpmreport,
            _get_eps(eps_factor, const.JUMP_LENGTH, dpmreport.budget_split),
        )

    return od_analysis_elements


def add_user_analysis_elements(
    dpmreport: "DpMobilityReport", eps_factor: float
) -> dict:
    user_analysis_elements = {}
    if (const.TRIPS_PER_USER in const.USER_ELEMENTS) and (
        const.TRIPS_PER_USER not in dpmreport.analysis_exclusion
    ):
        user_analysis_elements[const.TRIPS_PER_USER] = user_analysis.get_trips_per_user(
            dpmreport,
            _get_eps(eps_factor, const.TRIPS_PER_USER, dpmreport.budget_split),
        )

    if (const.USER_TIME_DELTA in const.USER_ELEMENTS) and (
        const.USER_TIME_DELTA not in dpmreport.analysis_exclusion
    ):
        user_analysis_elements[
            const.USER_TIME_DELTA
        ] = user_analysis.get_user_time_delta(
            dpmreport,
            _get_eps(eps_factor, const.USER_TIME_DELTA, dpmreport.budget_split),
        )

    if (const.RADIUS_OF_GYRATION in const.USER_ELEMENTS) and (
        const.RADIUS_OF_GYRATION not in dpmreport.analysis_exclusion
    ):
        user_analysis_elements[
            const.RADIUS_OF_GYRATION
        ] = user_analysis.get_radius_of_gyration(
            dpmreport,
            _get_eps(eps_factor, const.RADIUS_OF_GYRATION, dpmreport.budget_split),
        )

    if (const.USER_TILE_COUNT in const.USER_ELEMENTS) and (
        const.USER_TILE_COUNT not in dpmreport.analysis_exclusion
    ):
        user_analysis_elements[
            const.USER_TILE_COUNT
        ] = user_analysis.get_user_tile_count(
            dpmreport,
            _get_eps(eps_factor, const.USER_TILE_COUNT, dpmreport.budget_split),
        )

    if (const.MOBILITY_ENTROPY in const.USER_ELEMENTS) and (
        const.MOBILITY_ENTROPY not in dpmreport.analysis_exclusion
    ):
        user_analysis_elements[
            const.MOBILITY_ENTROPY
        ] = user_analysis.get_mobility_entropy(
            dpmreport,
            _get_eps(eps_factor, const.MOBILITY_ENTROPY, dpmreport.budget_split),
        )
    return user_analysis_elements

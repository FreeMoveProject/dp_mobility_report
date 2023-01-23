import dp_mobility_report.constants as const


def default_measure_selection() -> dict:
    return {
        const.DS_STATISTICS: const.SMAPE,
        const.MISSING_VALUES: const.SMAPE,
        const.TRIPS_OVER_TIME: const.JSD,
        const.TRIPS_PER_WEEKDAY: const.JSD,
        const.TRIPS_PER_HOUR: const.JSD,
        const.VISITS_PER_TILE: const.EMD,
        const.VISITS_PER_TILE_QUARTILES: const.SMAPE,
        const.VISITS_PER_TILE_OUTLIERS: const.SMAPE,
        const.VISITS_PER_TILE_RANKING: const.KT,
        const.VISITS_PER_TIME_TILE: const.EMD,
        const.OD_FLOWS: const.JSD,
        const.OD_FLOWS_RANKING: const.KT,
        const.OD_FLOWS_QUARTILES: const.SMAPE,
        const.TRAVEL_TIME: const.JSD,
        const.TRAVEL_TIME_QUARTILES: const.SMAPE,
        const.JUMP_LENGTH: const.JSD,
        const.JUMP_LENGTH_QUARTILES: const.SMAPE,
        const.TRIPS_PER_USER: const.EMD,  # TODO: implement JSD
        const.TRIPS_PER_USER_QUARTILES: const.SMAPE,
        const.USER_TIME_DELTA: const.JSD,
        const.USER_TIME_DELTA_QUARTILES: const.SMAPE,
        const.RADIUS_OF_GYRATION: const.JSD,
        const.RADIUS_OF_GYRATION_QUARTILES: const.SMAPE,
        const.USER_TILE_COUNT: const.JSD,
        const.USER_TILE_COUNT_QUARTILES: const.SMAPE,
        const.MOBILITY_ENTROPY: const.JSD,
        const.MOBILITY_ENTROPY_QUARTILES: const.SMAPE,
    }


def remove_excluded_analyses_from_measure_selection(
    measure_selection: dict, analysis_exclusion: list
) -> dict:
    def _is_included(x: str) -> bool:
        return not any([x.startswith(analysis) for analysis in analysis_exclusion])

    included_measures = {
        key: value for key, value in measure_selection.items() if _is_included(key)
    }
    return included_measures

from typing import List, Optional, Union

import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame

from dp_mobility_report import DpMobilityReport
from dp_mobility_report import constants as const
from dp_mobility_report.benchmark import preprocessing
from dp_mobility_report.benchmark.similarity_measures import (
    compute_similarity_measures,
    get_selected_measures,
)

# TODO: measure selection


class BenchmarkReport:
    """Generate two (differentially private) mobility reports from one or two mobility datasets. The report will be generated as an HTML file, using the `.to_file()` method.
        Can be based on two datasets (`df_base` and `df_alternative`) or one dataset (`df_base`)) with different privacy settings.
        The arguments df, privacy_budget, user_privacy and max_trips_per_user can differ for the two datasets. The other arguments are the same for both reports.

    Args:
        df_base: `DataFrame` containing the baseline mobility data, see argument `df` of `DpMobilityReport`.
        tessellation: Geopandas `GeoDataFrame` containing the tessellation for spatial aggregations. Expected columns: `tile_id`. If tessellation is not provided in the expected default CRS EPSG:4326 it will automatically be transformed. If no tessellation is provided, all analyses based on the tessellation will automatically be removed.
        df_alternative: `DataFrame` containing the alternative mobility data to be compared against the baseline dataset, see argument `df` of `DpMobilityReport`. If `None`, `df_base` is used for both reports.
        privacy_budget_base: privacy_budget for the differentially private base report. Defaults to `None`, i.e., no privacy guarantee is provided.
        privacy_budget_alternative: privacy_budget for the differentially private alternative report. Defaults to `None`, i.e., no privacy guarantee is provided.
        measure_selection: TODO
        user_privacy_base: Whether item-level or user-level privacy is applied for the base report. Defaults to `True` (user-level privacy).
        user_privacy_alternative: Whether item-level or user-level privacy is applied for the alternative report. Defaults to `True` (user-level privacy).
        max_trips_per_user_base: maximum number of trips a user shall contribute to the data. Dataset will be sampled accordingly. Defaults to `None`, i.e., all trips included.
        max_trips_per_user_alternative: maximum number of trips a user shall contribute to the data. Dataset will be sampled accordingly. Defaults to `None`, i.e., all trips included.
        analysis_selection: Select only needed analyses, see argument `analysis_selection` of `DpMobilityReport`.
        analysis_exclusion: Ignored, if `analysis_selection' is set! Exclude analyses that are not needed, see argument `analysis_exclusion` of `DpMobilityReport`.
        budget_split_base: `dict`to customize how much privacy budget is assigned to which analysis. See argument `budget_split` of `DpMobilityReport`.
        budget_split_alternative: `dict`to customize how much privacy budget is assigned to which analysis. See argument `budget_split` of `DpMobilityReport`.
        timewindows: List of hours as `int` that define the timewindows for the spatial analysis for single time windows. Defaults to [2, 6, 10, 14, 18, 22].
        max_travel_time: Upper bound for travel time histogram. If `None` is given, no upper bound is set. Defaults to `None`.
        bin_range_travel_time: The range a single histogram bin spans for travel time (e.g., 5 for 5 min bins). If `None` is given, the histogram bins will be determined automatically. Defaults to `None`.
        max_jump_length: Upper bound for jump length histogram. If `None` is given, no upper bound is set. Defaults to `None`.
        bin_range_jump_length: The range a single histogram bin spans for jump length (e.g., 1 for 1 km bins). If `None` is given, the histogram bins will be determined automatically. Defaults to `None`.
        max_radius_of_gyration: Upper bound for radius of gyration histogram. If `None` is given, no upper bound is set. Defaults to `None`.
        bin_range_radius_of_gyration: The range a single histogram bin spans for the radius of gyration (e.g., 1 for 1 km bins). If `None` is given, the histogram bins will be determined automatically. Defaults to `None`.
        disable_progress_bar: Whether progress bars should be shown. Defaults to `False`.
        seed_sampling: Provide seed for down-sampling of dataset (according to `max_trips_per_user`) so that the sampling is reproducible. Defaults to `None`, i.e., no seed.
        evalu (bool, optional): Parameter only needed for development and evaluation purposes. Defaults to `False`."""

    report_alternative: DpMobilityReport
    report_base: DpMobilityReport

    similarity_measures: dict

    def __init__(
        self,        
        df_base: DataFrame,        
        tessellation: Optional[GeoDataFrame] = None,
        df_alternative: Optional[DataFrame] = None,
        privacy_budget_base: Optional[Union[int, float]] = None,
        privacy_budget_alternative: Optional[Union[int, float]] = None,
        measure_selection: Union[dict, str] = const.JSD,  # TODO: set default
        user_privacy_base: bool = True,
        user_privacy_alternative: bool = True,
        max_trips_per_user_base: Optional[int] = None,
        max_trips_per_user_alternative: Optional[int] = None,
        analysis_selection: Optional[List[str]] = None,
        analysis_exclusion: Optional[List[str]] = None,
        budget_split_base: dict = {},
        budget_split_alternative: dict = {},
        timewindows: Union[List[int], np.ndarray] = [2, 6, 10, 14, 18, 22],
        max_travel_time: int = 90,
        bin_range_travel_time: int = 5,
        max_jump_length: Union[int, float] = 10,
        bin_range_jump_length: Union[int, float] = 1,
        max_radius_of_gyration: Union[int, float] = 5,
        bin_range_radius_of_gyration: Union[int, float] = 0.5,
        disable_progress_bar: bool = False,
        seed_sampling: int = None,
        evalu: bool = False,
    ) -> None:

        self.report_alternative = DpMobilityReport(
            df_alternative,
            tessellation,
            privacy_budget_alternative,
            user_privacy_alternative,
            max_trips_per_user_alternative,
            analysis_selection,
            analysis_exclusion,
            budget_split_alternative,
            timewindows,
            max_travel_time,
            bin_range_travel_time,
            max_jump_length,
            bin_range_jump_length,
            max_radius_of_gyration,
            bin_range_radius_of_gyration,
            disable_progress_bar,
            seed_sampling,
            evalu,
        )
        self.report_alternative.report

        self.report_base = DpMobilityReport(
            df_base,
            tessellation,
            privacy_budget_base,
            user_privacy_base,
            max_trips_per_user_base,
            analysis_selection,
            analysis_exclusion,
            budget_split_base,
            timewindows,
            max_travel_time,
            bin_range_travel_time,
            max_jump_length,
            bin_range_jump_length,
            max_radius_of_gyration,
            bin_range_radius_of_gyration,
            disable_progress_bar,
            seed_sampling,
            evalu,
        )
        self.report_base.report

        self.report_base._report, self.report_alternative._report = preprocessing.unify_histogram_bins(self.report_base.report, self.report_alternative.report)

        self.analysis_exclusion = preprocessing.combine_analysis_exclusion(
            self.report_alternative.analysis_exclusion,
            self.report_base.analysis_exclusion,
        )
        if measure_selection is None:
            self.measure_selection = default_measure_selection()
        else:
            self.measure_selection = preprocessing.validate_measure_selection(measure_selection)
        self.re, self.kld, self.jsd, self.emd, self.smape = compute_similarity_measures(
            self.analysis_exclusion,
            self.report_alternative.report,
            self.report_base.report,
            self.report_alternative.tessellation,
        )
        self.similarity_measures = get_selected_measures(self)

    # TODO: html file for comparison
    def to_file(self, output_file):
        pass


def default_measure_selection() -> dict: 
    return {
        const.DS_STATISTICS: const.RE,
        const.MISSING_VALUES: const.RE,
        const.TRIPS_OVER_TIME: const.JSD,
        const.TRIPS_PER_WEEKDAY: const.JSD,
        const.TRIPS_PER_HOUR: const.JSD,
        const.VISITS_PER_TILE: const.EMD,
        const.VISITS_PER_TILE_OUTLIERS: const.RE,
        const.VISITS_PER_TILE_TIMEWINDOW: const.EMD,
        const.OD_FLOWS: const.JSD,
        const.TRAVEL_TIME: const.JSD,
        const.TRAVEL_TIME_QUARTILES: const.SMAPE,
        const.JUMP_LENGTH: const.JSD,
        const.JUMP_LENGTH_QUARTILES: const.SMAPE,
        const.TRIPS_PER_USER: const.EMD,
        const.TRIPS_PER_USER_QUARTILES: const.SMAPE, 
        const.USER_TIME_DELTA_QUARTILES: const.SMAPE,
        const.RADIUS_OF_GYRATION: const.JSD,
        const.RADIUS_OF_GYRATION_QUARTILES: const.SMAPE,
        #const.USER_TILE_COUNT: const.EMD, 
        const.USER_TILE_COUNT_QUARTILES: const.SMAPE,
        const.MOBILITY_ENTROPY: const.JSD,
        const.MOBILITY_ENTROPY_QUARTILES: const.SMAPE
    }



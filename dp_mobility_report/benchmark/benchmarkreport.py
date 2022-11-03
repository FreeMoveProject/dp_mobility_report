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

# TODO: naming of "proposal" and "benchmark"; args description
# TODO: measure selection
# TODO: replace names of analyses with constants


class BenchmarkReport:
    """Generate two (differentially private) mobility reports from one or two mobility datasets. The report will be generated as an HTML file, using the `.to_file()` method.
        Can be based on two datasets or one dataset with differnt privacy budgets.
        The arguments df, privacy_budget, user_privacy and max_trips_per_user can differ for the two datasets. The other arguments are the same for both reports.

    Args:
        df_proposal: `DataFrame` containing the mobility data. Expected columns: User ID `uid`, trip ID `tid`, timestamp `datetime`, latitude `lat` and longitude `lng` in CRS EPSG:4326.
        df_benchmark: `DataFrame` containing the mobility data. Expected columns: User ID `uid`, trip ID `tid`, timestamp `datetime`, latitude `lat` and longitude `lng` in CRS EPSG:4326. Baseline
        privacy_budget_proposal: privacy_budget for the differentially private report
        privacy_budget_benchmark: privacy_budget for the differentially private report
        user_privacy_: Whether item-level or user-level privacy is applied. Defaults to True (user-level privacy).
        user_privacy_2: Whether item-level or user-level privacy is applied. Defaults to True (user-level privacy).
        max_trips_per_user_1: maximum number of trips a user shall contribute to the data. Dataset will be sampled accordingly.
        max_trips_per_user_2: maximum number of trips a user shall contribute to the data. Dataset will be sampled accordingly.
        tessellation: Geopandas `GeoDataFrame` containing the tessellation for spatial aggregations. Expected columns: `tile_id`. If tessellation is not provided in the expected default CRS EPSG:4326 it will automatically be transformed.
        analysis_selection: Select only needed analyses. A selection reduces computation time and leaves more privacy budget for higher accuracy of other analyses. Options are `overview`, `place_analysis`, `od_analysis`, `user_analysis` and `all`. Defaults to [`all`].
        timewindows: List of hours as `int` that define the timewindows for the spatial analysis for single time windows. Defaults to [2, 6, 10, 14, 18, 22].
        max_travel_time: Upper bound for travel time histogram. If None is given, no upper bound is set. Defaults to None.
        bin_range_travel_time: The range a single histogram bin spans for travel time (e.g., 5 for 5 min bins). If None is given, the histogram bins will be determined automatically. Defaults to None.
        max_jump_length: Upper bound for jump length histogram. If None is given, no upper bound is set. Defaults to None.
        bin_range_jump_length: The range a single histogram bin spans for jump length (e.g., 1 for 1 km bins). If None is given, the histogram bins will be determined automatically. Defaults to None.
        max_radius_of_gyration: Upper bound for radius of gyration histogram. If None is given, no upper bound is set. Defaults to None.
        bin_range_radius_of_gyration: The range a single histogram bin spans for the radius of gyration (e.g., 1 for 1 km bins). If None is given, the histogram bins will be determined automatically. Defaults to None.
        disable_progress_bar: Whether progress bars should be shown. Defaults to False.
        evalu (bool, optional): Parameter only needed for development and evaluation purposes. Defaults to False."""

    report_proposal: DpMobilityReport
    report_benchmark: DpMobilityReport

    similarity_measures: dict

    def __init__(
        self,
        df_proposal: DataFrame,
        df_benchmark: DataFrame,
        tessellation: GeoDataFrame,
        privacy_budget_proposal: Optional[Union[int, float]],
        privacy_budget_benchmark: Optional[Union[int, float]],
        measure_selection: Union[dict, str] = const.JSD,  # TODO: set default
        user_privacy_proposal: bool = True,
        user_privacy_benchmark: bool = True,
        max_trips_per_user_proposal: Optional[int] = None,
        max_trips_per_user_benchmark: Optional[int] = None,
        analysis_selection: Optional[List[str]] = None,
        analysis_exclusion: Optional[List[str]] = None,
        budget_split_proposal: dict = {},
        budget_split_benchmark: dict = {},
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

        self.report_proposal = DpMobilityReport(
            df_proposal,
            tessellation,
            privacy_budget_proposal,
            user_privacy_proposal,
            max_trips_per_user_proposal,
            analysis_selection,
            analysis_exclusion,
            budget_split_proposal,
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
        self.report_benchmark = DpMobilityReport(
            df_benchmark,
            tessellation,
            privacy_budget_benchmark,
            user_privacy_benchmark,
            max_trips_per_user_benchmark,
            analysis_selection,
            analysis_exclusion,
            budget_split_benchmark,
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
        self.analysis_exclusion = preprocessing.combine_analysis_exclusion(
            self.report_proposal.analysis_exclusion,
            self.report_benchmark.analysis_exclusion,
        )

        # self.measure_selection = preprocessing.validate_measure_selection(measure_selection)
        self.re, self.kld, self.jsd, self.emd, self.smape = compute_similarity_measures(
            self.analysis_exclusion,
            self.report_proposal.report,
            self.report_benchmark.report,
            self.report_proposal.tessellation,
        )
        self.similarity_measures = get_selected_measures(self)

    # TODO: html file for comparison
    def to_file(self, output_file):
        pass

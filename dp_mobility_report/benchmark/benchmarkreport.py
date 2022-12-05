from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from geopandas import GeoDataFrame
from pandas import DataFrame

from dp_mobility_report.benchmark import b_utils, preprocessing
from dp_mobility_report.benchmark.similarity_measures import (
    compute_similarity_measures,
    get_selected_measures,
)
from dp_mobility_report.dpmreport import DpMobilityReport


class BenchmarkReport:
    """Evaluate the similarity of two (differentially private) mobility reports from one or two mobility datasets.
        This can be based on two datasets (`df_base` and `df_alternative`) or one dataset (`df_base`)) with different privacy settings.
        The arguments `df`, `privacy_budget`, `user_privacy`, `max_trips_per_user` and `budget_split` can differ for the two datasets set with the according ending `_base` and `_alternative`. The other arguments are the same for both reports.
        For the evaluation, similarity measures (namely the relative error (RE), Jensen-Shannon divergence (JSD), Kullback-Leibler divergence (KLD), symmetric mean absolute percentage error (SMAPE) and the earth mover's distance (EMD)) are computed to quantify the statistical similarity for each analysis.
        The evaluation, i.e., benchmark report, will be generated as an HTML file, using the `.to_file()` method.

    Args:
        df_base: `DataFrame` containing the baseline mobility data, see argument `df` of `DpMobilityReport`.
        tessellation: Geopandas `GeoDataFrame` containing the tessellation for spatial aggregations. Expected columns: `tile_id`. If tessellation is not provided in the expected default CRS EPSG:4326 it will automatically be transformed. If no tessellation is provided, all analyses based on the tessellation will automatically be removed.
        df_alternative: `DataFrame` containing the alternative mobility data to be compared against the baseline dataset, see argument `df` of `DpMobilityReport`. If `None`, `df_base` is used for both reports.
        privacy_budget_base: privacy_budget for the differentially private base report. Defaults to `None`, i.e., no privacy guarantee is provided.
        privacy_budget_alternative: privacy_budget for the differentially private alternative report. Defaults to `None`, i.e., no privacy guarantee is provided.
        measure_selection: Select similarity measure for each analysis. If `None`, the default from `default_measure_selection()` will be used.
        user_privacy_base: Whether item-level or user-level privacy is applied for the base report. Defaults to `True` (user-level privacy).
        user_privacy_alternative: Whether item-level or user-level privacy is applied for the alternative report. Defaults to `True` (user-level privacy).
        max_trips_per_user_base: maximum number of trips a user shall contribute to the data. Dataset will be sampled accordingly. Defaults to `None`, i.e., all trips included.
        max_trips_per_user_alternative: maximum number of trips a user shall contribute to the data. Dataset will be sampled accordingly. Defaults to `None`, i.e., all trips included.
        analysis_selection: Select only needed analyses, see argument `analysis_selection` of `DpMobilityReport`.
        analysis_exclusion: Ignored, if `analysis_selection' is set! Exclude analyses that are not needed, see argument `analysis_exclusion` of `DpMobilityReport`.
        budget_split_base: `dict`to customize how much privacy budget is assigned to which analysis. See argument `budget_split` of `DpMobilityReport`.
        budget_split_alternative: `dict`to customize how much privacy budget is assigned to which analysis. See argument `budget_split` of `DpMobilityReport`.
        timewindows: List of hours as `int` that define the timewindows for the spatial analysis for single time windows. Defaults to [2, 6, 10, 14, 18, 22].
        max_travel_time: Upper bound for travel time histogram. Defaults to 120 (mins).
        bin_range_travel_time: The range a single histogram bin spans for travel time (e.g., 5 for 5 min bins). Defaults to 5 (min).
        max_jump_length: Upper bound for jump length histogram. Defaults to 10 (km).
        bin_range_jump_length: The range a single histogram bin spans for jump length (e.g., 1 for 1 km bins). Defaults to 1 (km).
        max_radius_of_gyration: Upper bound for radius of gyration histogram. Defaults to 5 (km).
        bin_range_radius_of_gyration: The range a single histogram bin spans for the radius of gyration (e.g., 1 for 1 km bins). Defaults to 0.5 (km).
        max_user_tile_count: Upper bound for distinct tiles per user histogram. If ``None`` is given, no upper bound is set. Defaults to ``None``.
        bin_range_user_tile_count: The range a single histogram bin spans for the distinct tiles per user histogram. If ``None`` is given, the histogram bins will be determined automatically. Defaults to ``None``.
        max_user_time_delta:  Upper bound for user time delta histogram. Defaults to 48 (hours).
        bin_range_user_time_delta: The range a single histogram bin spans for user time delta (e.g., 1 for 1 hour bins). Defaults to 4 (hours).
        disable_progress_bar: Whether progress bars should be shown. Defaults to `False`.
        seed_sampling: Provide seed for down-sampling of dataset (according to `max_trips_per_user`) so that the sampling is reproducible. Defaults to `None`, i.e., no seed.
        evalu (bool, optional): Parameter only needed for development and evaluation purposes. Defaults to `False`."""

    _report_base: DpMobilityReport
    _report_alternative: DpMobilityReport
    _re: dict
    _jsd: dict
    _kld: dict
    _emd: dict
    _smape: dict
    _measure_selection: dict
    _similarity_measures: dict = {}

    def __init__(
        self,
        df_base: DataFrame,
        tessellation: Optional[GeoDataFrame] = None,
        df_alternative: Optional[DataFrame] = None,
        privacy_budget_base: Optional[Union[int, float]] = None,
        privacy_budget_alternative: Optional[Union[int, float]] = None,
        measure_selection: dict = None,
        user_privacy_base: bool = True,
        user_privacy_alternative: bool = True,
        max_trips_per_user_base: Optional[int] = None,
        max_trips_per_user_alternative: Optional[int] = None,
        analysis_selection: Optional[List[str]] = None,
        analysis_exclusion: Optional[List[str]] = None,
        budget_split_base: dict = {},
        budget_split_alternative: dict = {},
        timewindows: Union[List[int], np.ndarray] = [2, 6, 10, 14, 18, 22],
        max_travel_time: int = 120,
        bin_range_travel_time: int = 5,
        max_jump_length: Union[int, float] = 10,
        bin_range_jump_length: Union[int, float] = 1,
        max_radius_of_gyration: Union[int, float] = 5,
        bin_range_radius_of_gyration: Union[int, float] = 0.5,
        max_user_tile_count: Optional[int] = 10,
        bin_range_user_tile_count: Optional[int] = 1,
        max_user_time_delta: Optional[Union[int, float]] = 48,
        bin_range_user_time_delta: Optional[Union[int, float]] = 4,
        disable_progress_bar: bool = False,
        seed_sampling: int = None,
        evalu: bool = False,
    ) -> None:

        self._report_base = DpMobilityReport(
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
            max_user_tile_count,
            bin_range_user_tile_count,
            max_user_time_delta,
            bin_range_user_time_delta,
            disable_progress_bar,
            seed_sampling,
            evalu,
        )
        self.report_base.report

        if df_alternative is None:
            df_alternative = df_base

        self._report_alternative = DpMobilityReport(
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
            max_user_tile_count,
            bin_range_user_tile_count,
            max_user_time_delta,
            bin_range_user_time_delta,
            disable_progress_bar,
            seed_sampling,
            evalu,
        )
        self.report_alternative.report

        self.analysis_exclusion = preprocessing.combine_analysis_exclusion(
            self.report_alternative.analysis_exclusion,
            self.report_base.analysis_exclusion,
        )

        (
            self.report_base._report,
            self.report_alternative._report,
        ) = preprocessing.unify_histogram_bins(
            self.report_base.report,
            self.report_alternative.report,
            self.analysis_exclusion,
        )

        if measure_selection is None:
            self._measure_selection = b_utils.default_measure_selection()
        else:
            self._measure_selection = preprocessing.validate_measure_selection(
                measure_selection, self.analysis_exclusion
            )

        (
            self._re,
            self._kld,
            self._jsd,
            self._emd,
            self._smape,
        ) = compute_similarity_measures(
            self.analysis_exclusion,
            self.report_alternative.report,
            self.report_base.report,
            self.report_base.tessellation,
        )

    @property
    def report_base(self) -> "DpMobilityReport":
        """The base DpMobilityReport"""
        return self._report_base

    @property
    def report_alternative(self) -> "DpMobilityReport":
        """The alternative DpMobilityReport"""
        return self._report_alternative

    @property
    def similarity_measures(self) -> dict:
        """Similarity measures according to `measure_selection`."""
        if not self._similarity_measures:
            self._similarity_measures = get_selected_measures(self)

        return self._similarity_measures

    @property
    def measure_selection(self) -> dict:
        """The specified selected similarity measure for each analysis."""
        return self._measure_selection

    @property
    def re(self) -> dict:
        """The relative error between base and alternative of all selected analyses, where applicable."""
        return self._re

    @property
    def kld(self) -> dict:
        """The Kullback-Leibler divergence between base and alternative of all selected analyses, where applicable."""
        return self._kld

    @property
    def jsd(self) -> dict:
        """The Jensen-Shannon divergence between base and alternative of all selected analyses, where applicable."""
        return self._jsd

    @property
    def emd(self) -> dict:
        """The earth mover's distance between base and alternative of all selected analyses, where applicable."""
        return self._emd

    @property
    def smape(self) -> dict:
        """The symmetric mean absolute percentage error between base and alternative of all selected analyses, where applicable."""
        return self._smape

    # TODO: html file for comparison
    def to_file(self, output_file: Union[str, Path]) -> None:
        pass

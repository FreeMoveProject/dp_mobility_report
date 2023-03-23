import os
import warnings
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pandarallel import pandarallel
from pandas import DataFrame
from tqdm.auto import tqdm

from dp_mobility_report import constants as const
from dp_mobility_report.model import preprocessing
from dp_mobility_report.report import report
from dp_mobility_report.report.html.templates import (
    create_html_assets,
    create_maps_folder,
    render_html,
)


class DpMobilityReport:
    """Generate a (differentially private) mobility report from a mobility dataset. The report will be generated as an HTML file, using the ``.to_file()`` method.

    Args:
        df: ``DataFrame`` containing the mobility data. Expected columns: User ID ``uid``, trip ID ``tid``, timestamp ``datetime`` (or ``int`` to indicate sequence position, if dataset only consists of sequences without timestamps), latitude ``lat`` and longitude ``lng`` in CRS EPSG:4326.
        tessellation: Geopandas ``GeoDataFrame`` containing the tessellation for spatial aggregations. Expected columns: ``tile_id``. If tessellation is not provided in the expected default CRS EPSG:4326, it will automatically be transformed. If no tessellation is provided, all analyses based on the tessellation will automatically be removed.
        privacy_budget: privacy_budget for the differentially private report. Defaults to ``None``, i.e., no privacy guarantee is provided.
        user_privacy: Whether item-level or user-level privacy is applied. Defaults to ``True`` (user-level privacy).
        max_trips_per_user: Maximum number of trips a user is allowed to contribute to the data. Dataset will be sampled accordingly. Defaults to ``None``, i.e., all trips are used. This implies that the actual maximum number of trips per user will be used according to the data, though this violates user-level Differential Privacy.
        analysis_selection: Select only needed analyses. A selection reduces computation time and leaves more privacy budget
        for higher accuracy of other analyses. ``analysis_selection`` takes a list of all analyses to be included. Alternatively, a list of analyses to be excluded can be set with ``analysis_exclusion``.
        Either entire segments can be included: ``const.OVERVIEW``, ``const.PLACE_ANALYSIS``, ``const.OD_ANALYSIS``, ``const.USER_ANALYSIS``
        or any single analysis can be included: ``const.DS_STATISTICS``, ``const.MISSING_VALUES``, ``const.TRIPS_OVER_TIME``, ``const.TRIPS_PER_WEEKDAY``, ``const.TRIPS_PER_HOUR``, ``const.VISITS_PER_TILE``, ``const.VISITS_PER_TIME_TILE``, ``const.OD_FLOWS``, ``const.TRAVEL_TIME``, ``const.JUMP_LENGTH``, ``const.TRIPS_PER_USER``, ``const.USER_TIME_DELTA``, ``const.RADIUS_OF_GYRATION``, ``const.USER_TILE_COUNT``, ``const.MOBILITY_ENTROPY``
        Default is None, i.e., all analyses are included.
        analysis_exclusion: Ignored, if ``analysis_selection`` is set! ``analysis_exclusion`` takes a list of all analyses to be excluded.
        Either entire segments can be excluded: ``const.OVERVIEW``, ``const.PLACE_ANALYSIS``, ``const.OD_ANALYSIS``, ``const.USER_ANALYSIS``
        or any single analysis can be excluded: ``const.DS_STATISTICS``, ``const.MISSING_VALUES``, ``const.TRIPS_OVER_TIME``, ``const.TRIPS_PER_WEEKDAY``, ``const.TRIPS_PER_HOUR``, ``const.VISITS_PER_TILE``, ``const.VISITS_PER_TIME_TILE``, ``const.OD_FLOWS``, ``const.TRAVEL_TIME``, ``const.JUMP_LENGTH``, ``const.TRIPS_PER_USER``, ``const.USER_TIME_DELTA``, ``const.RADIUS_OF_GYRATION``, ``const.USER_TILE_COUNT``, ``const.MOBILITY_ENTROPY``
        budget_split: ``dict`` to customize how much privacy budget is assigned to which analysis. Each key needs to be named according to an analysis and the value needs to be an integer indicating the weight for the privacy budget.
        If no weight is assigned, a default weight of 1 is set.
        For example, if ``budget_split = {const.VISITS_PER_TILE: 10}``, then the privacy budget for ``visits_per_tile`` is 10 times higher than for every other analysis, which all get a default weight of 1.
        Possible ``dict`` keys (all analyses): ``const.DS_STATISTICS``, ``const.MISSING_VALUES``, ``const.TRIPS_OVER_TIME``, ``const.TRIPS_PER_WEEKDAY``, ``const.TRIPS_PER_HOUR``, ``const.VISITS_PER_TILE``, ``const.VISITS_PER_TIME_TILE``, ``const.OD_FLOWS``, ``const.TRAVEL_TIME``, ``const.JUMP_LENGTH``, ``const.TRIPS_PER_USER``, ``const.USER_TIME_DELTA``, ``const.RADIUS_OF_GYRATION``, ``const.USER_TILE_COUNT``, ``const.MOBILITY_ENTROPY``
        timewindows: List of hours as ``int`` that define the timewindows for the spatial analysis for single time windows. Defaults to ``[2, 6, 10, 14, 18, 22]``.
        max_travel_time: Upper bound for travel time histogram. If ``None`` is given, no upper bound is set. Defaults to ``None``.
        bin_range_travel_time: The range a single histogram bin spans for travel time (e.g., 5 for 5 min bins). If ``None`` is given, the histogram bins will be determined automatically. Defaults to ``None``.
        max_jump_length: Upper bound for jump length histogram. If ``None`` is given, no upper bound is set. Defaults to ``None``.
        bin_range_jump_length: The range a single histogram bin spans for jump length (e.g., 1 for 1 km bins). If ``None`` is given, the histogram bins will be determined automatically. Defaults to ``None``.
        max_radius_of_gyration: Upper bound for radius of gyration histogram. If ``None`` is given, no upper bound is set. Defaults to ``None``.
        bin_range_radius_of_gyration: The range a single histogram bin spans for the radius of gyration (e.g., 1 for 1 km bins). If ``None`` is given, the histogram bins will be determined automatically. Defaults to ``None``.
        max_user_tile_count: Upper bound for distinct tiles per user histogram. If ``None`` is given, no upper bound is set. Defaults to ``None``.
        bin_range_user_tile_count: The range a single histogram bin spans for the distinct tiles per user histogram. If ``None`` is given, the histogram bins will be determined automatically. Defaults to ``None``.
        max_user_time_delta:  Upper bound for user time delta histogram. If ``None`` is given, no upper bound is set. Defaults to ``None``.
        bin_range_user_time_delta: The range a single histogram bin spans for user time delta (e.g., 1 for 1 hour bins). If ``None`` is given, the histogram bins will be determined automatically. Defaults to ``None``.
        subtitle: Custom subtitle that appears at the top of the HTML report. Defaults to ``None``.
        disable_progress_bar: Whether progress bars should be shown. Defaults to ``False``.
        seed_sampling: Provide seed for down-sampling of dataset (according to ``max_trips_per_user``) so that the sampling is reproducible. Defaults to ``None``, i.e., no seed.
        evalu: Parameter only needed for development and evaluation purposes. Defaults to ``False``."""

    _report: dict = {}
    _html: str = ""
    _df: DataFrame
    _tessellation: Optional[GeoDataFrame]
    _privacy_budget: Optional[Union[int, float]]
    _max_trips_per_user: int
    _analysis_exclusion: list
    _budget_split: dict

    def __init__(
        self,
        df: DataFrame,
        tessellation: Optional[GeoDataFrame] = None,
        privacy_budget: Optional[Union[int, float]] = None,
        user_privacy: bool = True,
        max_trips_per_user: Optional[int] = None,
        analysis_selection: Optional[List[str]] = None,
        analysis_exclusion: Optional[List[str]] = None,
        budget_split: dict = {},
        timewindows: Union[List[int], np.ndarray] = [2, 6, 10, 14, 18, 22],
        max_travel_time: Optional[int] = None,
        bin_range_travel_time: Optional[int] = None,
        max_jump_length: Optional[Union[int, float]] = None,
        bin_range_jump_length: Optional[Union[int, float]] = None,
        max_radius_of_gyration: Optional[Union[int, float]] = None,
        bin_range_radius_of_gyration: Optional[Union[int, float]] = None,
        max_user_tile_count: Optional[int] = None,
        bin_range_user_tile_count: Optional[int] = None,
        max_user_time_delta: Optional[Union[int, float]] = None,
        bin_range_user_time_delta: Optional[Union[int, float]] = None,
        subtitle: str = None,
        disable_progress_bar: bool = False,
        seed_sampling: int = None,
        evalu: bool = False,
    ) -> None:
        preprocessing.validate_input(
            df,
            tessellation,
            privacy_budget,
            max_trips_per_user,
            analysis_selection,
            analysis_exclusion,
            budget_split,
            disable_progress_bar,
            evalu,
            user_privacy,
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
            seed_sampling,
        )

        (
            analysis_selection,
            analysis_exclusion,
        ) = preprocessing.validate_inclusion_exclusion(
            analysis_selection,
            analysis_exclusion,
        )

        self.user_privacy = user_privacy
        with tqdm(  # progress bar
            total=2, desc="Preprocess data", disable=disable_progress_bar
        ) as pbar:
            self._tessellation = (
                None
                if tessellation is None
                else preprocessing.preprocess_tessellation(tessellation)
            )
            pbar.update()

            self._max_trips_per_user = (
                max_trips_per_user
                if max_trips_per_user is not None
                else df.groupby(const.UID).nunique()[const.TID].max()
            )

            if user_privacy:
                self.count_sensitivity_base = self._max_trips_per_user
            else:
                self.count_sensitivity_base = 1
            self._df = preprocessing.preprocess_data(
                df.copy(),  # copy, to not overwrite users instance of df
                self.tessellation,
                self.max_trips_per_user,
                self.user_privacy,
                seed_sampling,
            )
            pbar.update()

        self._privacy_budget = None if privacy_budget is None else float(privacy_budget)
        self.max_travel_time = max_travel_time
        timewindows.sort()
        self.timewindows = (
            np.array(timewindows) if isinstance(timewindows, list) else timewindows
        )
        self.max_jump_length = max_jump_length
        self.bin_range_jump_length = bin_range_jump_length
        self.bin_range_travel_time = bin_range_travel_time
        self.max_radius_of_gyration = max_radius_of_gyration
        self.bin_range_radius_of_gyration = bin_range_radius_of_gyration
        self.max_user_tile_count = max_user_tile_count
        self.bin_range_user_tile_count = bin_range_user_tile_count
        self.max_user_time_delta = max_user_time_delta
        self.bin_range_user_time_delta = bin_range_user_time_delta
        self._analysis_exclusion = preprocessing.clean_analysis_exclusion(
            analysis_selection,
            analysis_exclusion,
            has_tessellation=(tessellation is not None),
            has_points_inside_tessellation=preprocessing.has_points_inside_tessellation(
                self.df, self.tessellation
            ),
            has_timestamps=pd.core.dtypes.common.is_datetime64_dtype(
                self.df[const.DATETIME]
            ),
            has_od_flows=max(self.df[const.TID].value_counts())
            > 1,  # are there trips with more than a single record?
            has_consecutive_user_trips=max(
                self.df.groupby(const.UID).nunique()[const.TID]
            )
            > 1,
        )
        self._budget_split = preprocessing.clean_budget_split(
            budget_split, self._analysis_exclusion
        )
        self.evalu = evalu
        self.disable_progress_bar = disable_progress_bar
        self.subtitle = subtitle

        # initialize parallel processing
        pandarallel.initialize(verbose=0)

    @property
    def report(self) -> dict:
        """A dictionary with all report elements (i.e., analyses)."""
        if not self._report:
            self._report = report.report_elements(self)
        return self._report

    @property
    def df(self) -> DataFrame:
        """DataFrame containing the processed input mobility data of the report."""
        return self._df

    @property
    def max_trips_per_user(self) -> int:
        """Maximum number of trips per user as specified in the parameters. If ``None`` was given, this equals the actual maximum according to the data."""
        return self._max_trips_per_user

    @property
    def budget_split(self) -> dict:
        """Budget split as specified in the parameters."""
        return self._budget_split

    @property
    def analysis_exclusion(self) -> list:
        """List of analyses that have been excluded from the report and similarity measures. If analysis selection was provided as a parameter, they are inverted to this ``analysis_exclusion`` parameter."""
        return self._analysis_exclusion

    @property
    def privacy_budget(self) -> Union[int, float]:
        """Privacy budget as specified in the parameters."""
        return self._privacy_budget

    @property
    def tessellation(self) -> GeoDataFrame:
        """Processed tessellation."""
        return self._tessellation

    def to_file(
        self,
        output_file: Union[str, Path],
        disable_progress_bar: Optional[bool] = None,
        top_n_flows: int = 100,
    ) -> None:
        """Write the report to a file.
        By default a name is generated.

        Args:
            output_file: The name or the path of the file to store the ``html`` output.
            disable_progress_bar: if ``False``, no progress bar is shown.
            top_n_flows: Determines how many of the top ``n`` origin-destination flows are displayed. Defaults to 100.
        """
        if disable_progress_bar is None:
            disable_progress_bar = self.disable_progress_bar

        if not isinstance(output_file, Path):
            output_file = Path(str(output_file))

        else:
            if output_file.suffix != ".html":
                suffix = output_file.suffix
                output_file = output_file.with_suffix(".html")
                warnings.warn(
                    f"Extension {suffix} not supported. For now we assume .html was intended. "
                    f"To remove this warning, please use .html or .json."
                )

        output_dir = Path(os.path.splitext(output_file)[0])
        filename = Path(os.path.basename(output_file)).stem

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        create_html_assets(output_dir)

        # create report if not created yet (to display progress bar in correct order)
        self.report

        # render html
        data, temp_map_folder = render_html(
            self, filename, top_n_flows, disable_progress_bar
        )

        create_maps_folder(temp_map_folder, output_dir)

        # clean up temp folder
        rmtree(temp_map_folder, ignore_errors=True)

        output_file.write_text(data, encoding="utf-8")

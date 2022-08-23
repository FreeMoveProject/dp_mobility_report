import os
import warnings
from pathlib import Path
from shutil import rmtree
from typing import Any, List, Optional, Union

import numpy as np
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
    """Generate a (differentially private) mobility report from a mobility dataset. The report will be generated as an HTML file, using the `.to_html()` method.

    Args:
        df: `DataFrame` containing the mobility data. Expected columns: User ID `uid`, trip ID `tid`, timestamp `datetime`, latitude `lat` and longitude `lng` in CRS EPSG:4326.
        tessellation: Geopandas `GeoDataFrame` containing the tessellation for spatial aggregations. Expected columns: `tile_id`. If tessellation is not provided in the expected default CRS EPSG:4326 it will automatically be transformed.
        privacy_budget: privacy_budget for the differentially private report
        user_privacy: Whether item-level or user-level privacy is applied. Defaults to True (user-level privacy).
        max_trips_per_user: maximum number of trips a user shall contribute to the data. Dataset will be sampled accordingly.
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

    _report: dict = {}
    _html: str = ""

    def __init__(
        self,
        df: DataFrame,
        tessellation: GeoDataFrame,
        privacy_budget: Optional[Union[int, float]],
        user_privacy: bool = True,
        max_trips_per_user: Optional[int] = None,
        analysis_selection: List[str] = [const.ALL],
        timewindows: Union[List[int], np.ndarray] = [2, 6, 10, 14, 18, 22],
        max_travel_time: Optional[int] = None,
        bin_range_travel_time: Optional[int] = None,
        max_jump_length: Optional[Union[int, float]] = None,
        bin_range_jump_length: Optional[Union[int, float]] = None,
        max_radius_of_gyration: Optional[Union[int, float]] = None,
        bin_range_radius_of_gyration: Optional[Union[int, float]] = None,
        disable_progress_bar: bool = False,
        evalu: bool = False,
    ) -> None:
        _validate_input(
            df,
            tessellation,
            privacy_budget,
            max_trips_per_user,
            analysis_selection,
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
        )

        self.user_privacy = user_privacy
        with tqdm(  # progress bar
            total=2, desc="Preprocess data", disable=disable_progress_bar
        ) as pbar:
            self.tessellation = preprocessing.preprocess_tessellation(tessellation)
            pbar.update()

            self.max_trips_per_user = (
                max_trips_per_user
                if max_trips_per_user is not None
                else df.groupby(const.UID).nunique()[const.TID].max()
            )

            if not user_privacy:
                self.max_trips_per_user = 1
            self.df = preprocessing.preprocess_data(
                df.copy(),  # copy, to not overwrite users instance of df
                tessellation,
                self.max_trips_per_user,
                self.user_privacy,
            )
            pbar.update()

        self.privacy_budget = None if privacy_budget is None else float(privacy_budget)
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
        self.analysis_selection = analysis_selection
        self.evalu = evalu
        self.disable_progress_bar = disable_progress_bar

        # initialize parallel processing
        pandarallel.initialize(verbose=0)

    @property
    def report(self) -> dict:
        """Generate all report elements within a `dict`.

        Returns:
            A dictionary with all report elements.
        """
        if not self._report:
            self._report = report.report_elements(self)
        return self._report

    def to_file(
        self,
        output_file: Union[str, Path],
        disable_progress_bar: Optional[bool] = None,
        top_n_flows: int = 100,
    ) -> None:
        """Write the report to a file.
        By default a name is generated.

        Args:
            output_file: The name or the path of the file to store the `html` output.
            disable_progress_bar: if `False`, no progress bar is shown.
            top_n_flows: Determines how many of the top `n` origin-destination flows are displayed. Defaults to 100.
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

        with tqdm(  # progress bar
            total=1, desc="Create HTML Output", disable=disable_progress_bar
        ) as pbar:
            data, temp_map_folder = render_html(self, filename, top_n_flows)
            pbar.update()

        create_maps_folder(temp_map_folder, output_dir)

        # clean up temp folder
        rmtree(temp_map_folder, ignore_errors=True)

        output_file.write_text(data, encoding="utf-8")


def _validate_input(
    df: DataFrame,
    tessellation: GeoDataFrame,
    privacy_budget: Optional[Union[int, float]],
    max_trips_per_user: Optional[int],
    analysis_selection: List[str],
    disable_progress_bar: bool,
    evalu: bool,
    user_privacy: bool,
    timewindows: Union[List[int], np.ndarray],
    max_travel_time: Optional[int],
    bin_range_travel_time: Optional[int],
    max_jump_length: Optional[Union[int, float]],
    bin_range_jump_length: Optional[Union[int, float]],
    max_radius_of_gyration: Optional[Union[int, float]],
    bin_range_radius_of_gyration: Optional[Union[int, float]],
) -> None:
    if not isinstance(df, DataFrame):
        raise TypeError("'df' is not a Pandas DataFrame.")

    if not isinstance(tessellation, GeoDataFrame):
        raise TypeError("'tessellation' is not a Geopandas GeoDataFrame.")

    if not ((max_trips_per_user is None) or isinstance(max_trips_per_user, int)):
        raise TypeError("'max_trips_per_user' is not numeric.")
    if (max_trips_per_user is not None) and (max_trips_per_user < 1):
        raise ValueError("'max_trips_per_user' has to be greater 0.")

    if not ((analysis_selection is None) or isinstance(analysis_selection, list)):
        raise TypeError("'analysis_selection' is not a list.")
    if not set(analysis_selection).issubset(const.ANALYSIS_SELECTION):
        raise ValueError(
            f"Unknown analysis selection {analysis_selection}. Only elements from {const.ANALYSIS_SELECTION} are valid inputs."
        )

    if not isinstance(timewindows, (list, np.ndarray)):
        raise TypeError("'timewindows' is not a list or a numpy array.")

    timewindows = (
        np.array(timewindows) if isinstance(timewindows, list) else timewindows
    )
    if not all([np.issubdtype(item, int) for item in timewindows]):
        raise TypeError("not all items of 'timewindows' are integers.")

    if privacy_budget is not None:
        _validate_numeric_greater_zero(
            privacy_budget, f"{privacy_budget=}".split("=")[0]
        )
    _validate_numeric_greater_zero(max_travel_time, f"{max_travel_time=}".split("=")[0])
    _validate_numeric_greater_zero(
        bin_range_travel_time, f"{bin_range_travel_time=}".split("=")[0]
    )
    _validate_numeric_greater_zero(max_jump_length, f"{max_jump_length=}".split("=")[0])
    _validate_numeric_greater_zero(
        bin_range_jump_length, f"{bin_range_jump_length=}".split("=")[0]
    )
    _validate_numeric_greater_zero(
        max_radius_of_gyration, f"{max_radius_of_gyration=}".split("=")[0]
    )
    _validate_numeric_greater_zero(
        bin_range_radius_of_gyration, f"{bin_range_radius_of_gyration=}".split("=")[0]
    )
    _validate_bool(user_privacy, f"{user_privacy=}".split("=")[0])
    _validate_bool(evalu, f"{user_privacy=}".split("=")[0])
    _validate_bool(disable_progress_bar, f"{user_privacy=}".split("=")[0])


def _validate_numeric_greater_zero(var: Any, name: str) -> None:
    if not ((var is None) or isinstance(var, (int, float))):
        raise TypeError(f"{name} is not numeric.")
    if (var is not None) and (var <= 0):
        raise ValueError(f"'{name}' has to be greater 0.")


def _validate_bool(var: Any, name: str) -> None:
    if not isinstance(var, bool):
        raise TypeError(f"'{name}' is not type boolean.")

import warnings
import logging

from pathlib import Path
from typing import Type, Union

from pandarallel import pandarallel
from tqdm.auto import tqdm

from pandas import DataFrame
from geopandas import GeoDataFrame

from dp_mobility_report import constants as const
from dp_mobility_report.model import preprocessing
from dp_mobility_report.report import report
from dp_mobility_report.report.html.templates import create_html_assets, render_html


class MobilityDataReport:
    """Generate a (differentially private) mobility report from a dataset stored as
    a pandas `DataFrame`. Expected columns: User ID `uid`, Trip ID `tid`, Timestamp `datetime`, 
    Latitude and Longitude in CRS EPSG:4326 `lat` and `lng`.
       The report will be generated as an HTML file, using the `.to_html()` method.
    """

    _report = None
    _html = None

    def __init__(
        self,
        df,
        tessellation,
        privacy_budget,
        #extra_var=None,
        max_trips_per_user=None,
        timewindows = [2,6,10,14,18,22],
        max_travel_time=90,
        bin_size_travel_time=10,
        max_jump_length=15000,
        bin_size_jump_length=1000,
        max_radius_of_gyration=10000,
        bin_size_radius_of_gyration=1000,
        top_x_flows=100,
        analysis_selection=["all"],
        disable_progress_bar=False,
        evalu=False,
        user_privacy=True,
    ) -> None:
        """Generate a (differentially private) mobility report from a dataset stored as
    a pandas `DataFrame`. 
        Args:
            df (DataFrame): Pandas DataFrame containing the mobility data. Expected columns: User ID `uid`, Trip ID `tid`, Timestamp `datetime`, 
            Latitude and Longitude in CRS EPSG:4326 `lat` and `lng`.
            tessellation(GeoDataFrame): Geopandas GeoDataFrame containing the tessellation for spatial aggregations. If tessellation is not provided
            in the expected default CRS EPSG:4326 it will automatically be transformed.
            privacy_budget (float): privacy_budget for the differentially private report.
            max_trips_per_user(int): maximum number of trips a user shall contribute to the data. Dataset will be sampled accordingly.
            analysis_selection (list, optional): Select only needed analyses. A selection reduces compuation time and leaves more privacy budget 
            for higher accuracy of other analyses.
            Options are `overview`, `place_analysis`, `od_analysis`, `user_analysis` and `all`. Defaults to ["all"].
        """
        # if (extra_var is None) | (extra_var in df.columns):
        #     self.extra_var = extra_var
        # else:
        #     self.extra_var = None
        #     warnings.warn(
        #         f"{extra_var} does not exist in the DataFrame. Therefore, it will be ignored."
        #     )
        
        # check input
        if (not isinstance(df, DataFrame)):
            raise TypeError("'df' is not a Pandas DataFrame.")

        if (not isinstance(tessellation, GeoDataFrame)):
            raise TypeError("'tessellation' is not a Geopandas GeoDataFrame.")

        if (max_trips_per_user is not None) and (not isinstance(max_trips_per_user, int) or (max_trips_per_user < 1)):
            max_trips_per_user = None
            logging.warning("'max_trips_per_user' is not an integer greater 0. It is set to default 'None'")

        if not ((privacy_budget is None) or isinstance(privacy_budget, int) or isinstance(privacy_budget, float)):
            raise TypeError("'privacy_budget' is not a numeric value.")

        if (privacy_budget is not None) and (privacy_budget <= 0):
            raise ValueError("'privacy_budget' is not greater 0.")

        if (not isinstance(timewindows, list)):
            raise TypeError("'timewindows' is not a list.")
        timewindows.sort()

        if not isinstance(user_privacy, bool):
            raise ValueError("'user_privacy' is not type boolean.")

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
                df.copy(), #copy, to not overwrite users instance of df
                tessellation,
            #   self.extra_var,
                self.max_trips_per_user,
                self.user_privacy,
            )
            pbar.update()

        self.privacy_budget = privacy_budget
        self.max_travel_time = max_travel_time
        self.timewindows = timewindows
        self.max_jump_length = max_jump_length
        self.bin_size_jump_length = bin_size_jump_length
        self.bin_size_travel_time = bin_size_travel_time
        self.max_radius_of_gyration = max_radius_of_gyration
        self.bin_size_radius_of_gyration = bin_size_radius_of_gyration
        self.top_x_flows = top_x_flows
        self.analysis_selection = analysis_selection
        self.evalu = evalu
        self.disable_progress_bar = disable_progress_bar
        
        # initialize parallel processing
        pandarallel.initialize(verbose=0)

    @property
    def report(self) -> dict:
        """Generate all report elements.
        Returns:
            A dictionary with all report elements.
        """
        if self._report is None:
            self._report = report.report_elements(self)
        return self._report

    @property
    def html(self) -> str:
        if self._html is None:
            self._html = self._render_html()
        return self._html

    def _render_html(self) -> str:
        html = render_html(self)
        return html

    def to_html(self) -> str:
        """Generate and return complete template as lengthy string
            for using with frameworks.
        Returns:
            HTML output as string.
        """
        return self.html

    def to_file(
        self, output_file: Union[str, Path], disable_progress_bar=None
    ) -> None:
        """Write the report to a file.
        By default a name is generated.
        Args:
            output_file: The name or the path of the file to generate including
            the extension (.html, .json).
            disable_progress_bar: if False, no progress bar is shown.
        """
        if disable_progress_bar is None:
            disable_progress_bar = self.disable_progress_bar

        if not isinstance(output_file, Path):
            output_file = Path(str(output_file))

        if output_file.suffix == ".json":
            data = self.to_json()

        else:
            if output_file.suffix != ".html":
                suffix = output_file.suffix
                output_file = output_file.with_suffix(".html")
                warnings.warn(
                    f"Extension {suffix} not supported. For now we assume .html was intended. "
                    f"To remove this warning, please use .html or .json."
                )

            # TODO: implement create_html_assets

            create_html_assets(output_file)
            with tqdm(  # progress bar
                total=1, desc="Create HTML Output", disable=disable_progress_bar
            ) as pbar:
                data = self.to_html()
                pbar.update()

        output_file.write_text(data, encoding="utf-8")


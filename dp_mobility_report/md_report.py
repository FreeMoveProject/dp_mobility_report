import warnings
from pathlib import Path
from typing import Union

from pandarallel import pandarallel
from tqdm.auto import tqdm

from dp_mobility_report.model import preprocessing
from dp_mobility_report.report import report
from dp_mobility_report.report.html.templates import create_html_assets, render_html


class MobilityDataReport:
    """Generate a mobility data report from a Dataset stored as
    a pandas `DataFrame` in the specified format [...].
       Used as is, it will output its content as an HTML report in a Jupyter notebook.
    """

    _report = None
    _html = None

    def __init__(
        self,
        df,
        tessellation,
        privacy_budget,
        extra_var=None,
        max_trips_per_user=None,
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
        """Generate a mobility data report from a Dataset stored
            as a pandas `DataFrame` in the specified format [...].
           Used as is, it will output its content as an HTML report in a Jupyter notebook

        Args:
            df (DataFrame): DataFrame in defined format.
            privacy_budget (float): privacy_budget
            analysis_selection (list, optional): Select only certain analyses,
            to reduce the used privacy budget. Options are ... . Defaults to ["all"].
        """
        if (extra_var is None) | (extra_var in df.columns):
            self.extra_var = extra_var
        else:
            self.extra_var = None
            warnings.warn(
                f"{extra_var} does not exist in the DataFrame. Therefore, it will be ignored."
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
                else df.groupby("uid").nunique().tid.max()
            )
            if not user_privacy:
                self.max_trips_per_user = 1
            self.df = preprocessing.preprocess_data(
                df.copy(),
                tessellation,
                self.extra_var,
                self.max_trips_per_user,
                self.user_privacy,
            )
            pbar.update()

        self.privacy_budget = privacy_budget
        self.max_travel_time = max_travel_time
        self.max_jump_length = max_jump_length
        self.bin_size_jump_length = bin_size_jump_length
        self.bin_size_travel_time = bin_size_travel_time
        self.max_radius_of_gyration = max_radius_of_gyration
        self.bin_size_radius_of_gyration = bin_size_radius_of_gyration
        self.top_x_flows = top_x_flows
        self.analysis_selection = analysis_selection
        self.evalu = evalu

    @property
    def report(self) -> dict:
        # initialize parallel processing
        pandarallel.initialize(verbose=0)

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
            Profiling report html including wrapper.
        """
        return self.html

    def to_file(
        self, output_file: Union[str, Path], disable_progress_bar=False
    ) -> None:
        """Write the report to a file.
        By default a name is generated.
        Args:
            output_file: The name or the path of the file to generate including
            the extension (.html, .json).
            silent: if False, opens the file in the default browser or download
            it in a Google Colab environment
        """
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

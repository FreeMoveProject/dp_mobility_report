import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import pkg_resources

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from dp_mobility_report import constants as const
from dp_mobility_report.report.html import (
    config_templates,
    html_utils,
    od_analysis_templates,
    overview_templates,
    place_analysis_templates,
    user_analysis_templates,
)


def render_html(
    mreport: "DpMobilityReport", output_filename: str, top_n_flows: int = 100
) -> Tuple[str, Path]:
    template_structure = html_utils.get_template("structure.html")
    temp_map_folder = Path(os.path.join(tempfile.gettempdir(), "maps"))

    # remove any old temp files in case there exist any
    shutil.rmtree(temp_map_folder, ignore_errors=True)
    os.mkdir(temp_map_folder)

    overview_segment = ""
    place_analysis_segment = ""
    od_analysis_segment = ""
    user_analysis_segment = ""

    is_all_analyses = const.ALL in mreport.analysis_selection

    config_segment = config_templates.render_config(mreport)

    if is_all_analyses | (const.OVERVIEW in mreport.analysis_selection):
        overview_segment = overview_templates.render_overview(mreport.report)

    if is_all_analyses | (const.PLACE_ANALYSIS in mreport.analysis_selection):
        place_analysis_segment = place_analysis_templates.render_place_analysis(
            mreport.report, mreport.tessellation, temp_map_folder, output_filename
        )
    if is_all_analyses | (const.OD_ANALYSIS in mreport.analysis_selection):
        od_analysis_segment = od_analysis_templates.render_od_analysis(
            mreport, top_n_flows, temp_map_folder, output_filename
        )
    if is_all_analyses | (const.USER_ANALYSIS in mreport.analysis_selection):
        user_analysis_segment = user_analysis_templates.render_user_analysis(mreport)

    return (
        template_structure.render(
            output_filename=output_filename,
            config_segment=config_segment,
            overview_segment=overview_segment,
            place_analysis_segment=place_analysis_segment,
            od_analysis_segment=od_analysis_segment,
            user_analysis_segment=user_analysis_segment,
        ),
        temp_map_folder,
    )


def create_html_assets(output_file: Path) -> None:
    path = Path(os.path.join(output_file, "assets"))
    if path.is_dir():
        shutil.rmtree(path)
    os.mkdir(path)

    asset_folder = pkg_resources.resource_filename(
        "dp_mobility_report", "report/html/html_templates/assets/"
    )

    for file_name in os.listdir(asset_folder):
        # construct full file path
        source = os.path.join(asset_folder, file_name)
        destination = os.path.join(path, file_name)
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)


def create_maps_folder(temp_map_folder: Path, output_dir: Path) -> None:
    path = Path(os.path.join(output_dir, "maps"))
    if path.is_dir():
        shutil.rmtree(path)
    os.makedirs(path)

    for file_name in os.listdir(temp_map_folder):
        # construct full file path
        source = os.path.join(temp_map_folder, file_name)
        destination = os.path.join(path, file_name)
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)

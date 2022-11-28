import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from tqdm.auto import tqdm

import pkg_resources  # type: ignore

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport, BenchmarkReport

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
    dpmreport: "DpMobilityReport", output_filename: str, top_n_flows: int = 100, disable_progress_bar = False
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

    with tqdm(  # progress bar
        total=4, desc="Create HTML Output", disable=disable_progress_bar
    ) as pbar:

        config_segment = config_templates.render_config(dpmreport)

        if not set(const.OVERVIEW_ELEMENTS).issubset(dpmreport.analysis_exclusion):
            overview_segment = overview_templates.render_overview(dpmreport.report)
        pbar.update()
        
        if not set(const.PLACE_ELEMENTS).issubset(dpmreport.analysis_exclusion):
            place_analysis_segment = place_analysis_templates.render_place_analysis(
                dpmreport.report, dpmreport.tessellation, temp_map_folder, output_filename
            )
        pbar.update()
        
        if not set(const.OD_ELEMENTS).issubset(dpmreport.analysis_exclusion):
            od_analysis_segment = od_analysis_templates.render_od_analysis(
                dpmreport, top_n_flows, temp_map_folder, output_filename
            )
        pbar.update()

        if not set(const.USER_ELEMENTS).issubset(dpmreport.analysis_exclusion):
            user_analysis_segment = user_analysis_templates.render_user_analysis(dpmreport)
        pbar.update()
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

# def render_benchmark_html(
#     benchmarkreport: "BenchmarkReport", output_filename: str, top_n_flows: int = 100
# ) -> Tuple[str, Path]:
#     template_structure = html_utils.get_template("structure.html")
#     temp_map_folder = Path(os.path.join(tempfile.gettempdir(), "maps"))

#     # remove any old temp files in case there exist any
#     shutil.rmtree(temp_map_folder, ignore_errors=True)
#     os.mkdir(temp_map_folder)

#     overview_segment = ""
#     place_analysis_segment = ""
#     od_analysis_segment = ""
#     user_analysis_segment = ""

#     config_segment = config_templates.render_benchmark_config(benchmarkreport)

#     if not set(const.OVERVIEW_ELEMENTS).issubset(benchmarkreport.analysis_exclusion):
#         overview_segment = overview_templates.render_overview(dpmreport.report)

#     if not set(const.PLACE_ELEMENTS).issubset(benchmarkreport.analysis_exclusion):
#         place_analysis_segment = place_analysis_templates.render_place_analysis(
#             dpmreport.report, dpmreport.tessellation, temp_map_folder, output_filename
#         )
#     if not set(const.OD_ELEMENTS).issubset(benchmarkreport.analysis_exclusion):
#         od_analysis_segment = od_analysis_templates.render_od_analysis(
#             dpmreport, top_n_flows, temp_map_folder, output_filename
#         )
#     if not set(const.USER_ELEMENTS).issubset(benchmarkreport.analysis_exclusion):
#         user_analysis_segment = user_analysis_templates.render_user_analysis(dpmreport)

#     return (
#         template_structure.render(
#             output_filename=output_filename,
#             config_segment=config_segment,
#             overview_segment=overview_segment,
#             place_analysis_segment=place_analysis_segment,
#             od_analysis_segment=od_analysis_segment,
#             user_analysis_segment=user_analysis_segment,
#         ),
#         temp_map_folder,
#     )


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

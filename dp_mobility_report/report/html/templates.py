from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from dp_mobility_report.md_report import MobilityDataReport

from dp_mobility_report import constants as const
from dp_mobility_report.report.html import (
    html_utils,
    od_analysis_templates,
    overview_templates,
    place_analysis_templates,
    user_analysis_templates,
)


def render_html(mdreport: "MobilityDataReport", top_n_flows: int = 100) -> str:
    template_structure = html_utils.get_template("structure.html")

    overview_segment = ""
    place_analysis_segment = ""
    od_analysis_segment = ""
    user_analysis_segment = ""

    is_all_analyses = const.ALL in mdreport.analysis_selection

    if is_all_analyses | (const.OVERVIEW in mdreport.analysis_selection):
        overview_segment = overview_templates.render_overview(mdreport.report)

    if is_all_analyses | (const.PLACE_ANALYSIS in mdreport.analysis_selection):
        place_analysis_segment = place_analysis_templates.render_place_analysis(
            mdreport.report, mdreport.tessellation
        )
    if is_all_analyses | (const.OD_ANALYSIS in mdreport.analysis_selection):
        od_analysis_segment = od_analysis_templates.render_od_analysis(
            mdreport, top_n_flows
        )
    if is_all_analyses | (const.USER_ANALYSIS in mdreport.analysis_selection):
        user_analysis_segment = user_analysis_templates.render_user_analysis(mdreport)

    return template_structure.render(
        overview_segment=overview_segment,
        place_analysis_segment=place_analysis_segment,
        od_analysis_segment=od_analysis_segment,
        user_analysis_segment=user_analysis_segment,
    )


def create_html_assets(output_file: Union[Path, str]) -> None:
    pass

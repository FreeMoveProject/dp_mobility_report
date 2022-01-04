from dp_mobility_report.report.html import (
    od_analysis_templates,
    overview_templates,
    place_analysis_templates,
    user_analysis_templates,
    utils,
)


def render_html(mdreport):
    template_structure = utils.get_template("structure.html")

    overview_segment = ""
    place_analysis_segment = ""
    od_analysis_segment = ""
    user_analysis_segment = ""

    if ("all" in mdreport.analysis_selection) | (
        "overview" in mdreport.analysis_selection
    ):
        overview_segment = overview_templates.render_overview(
            mdreport.report #, mdreport.extra_var
        )

    if ("all" in mdreport.analysis_selection) | (
        "place_analysis" in mdreport.analysis_selection
    ):
        place_analysis_segment = place_analysis_templates.render_place_analysis(
            mdreport.report, mdreport.tessellation
        )
    if ("all" in mdreport.analysis_selection) | (
        "od_analysis" in mdreport.analysis_selection
    ):
        od_analysis_segment = od_analysis_templates.render_od_analysis(mdreport)
    if ("all" in mdreport.analysis_selection) | (
        "user_analysis" in mdreport.analysis_selection
    ):
        user_analysis_segment = user_analysis_templates.render_user_analysis(mdreport)

    return template_structure.render(
        overview_segment=overview_segment,
        place_analysis_segment=place_analysis_segment,
        od_analysis_segment=od_analysis_segment,
        user_analysis_segment=user_analysis_segment,
    )


def create_html_assets(output_file) -> None:
    pass

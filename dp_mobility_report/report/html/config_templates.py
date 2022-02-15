from dp_mobility_report import constants as const
from dp_mobility_report.report.html.html_utils import fmt, get_template

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from dp_mobility_report.md_report import MobilityDataReport


def render_config(mdreport: "MobilityDataReport") -> str:

    config_table = render_config_table(mdreport)


    template_structure = get_template("config_segment.html")
    return template_structure.render(
        config_table=config_table
    )


def render_config_table(mdreport: "MobilityDataReport") -> str:

    dataset_stats_list = [
        {"name": "Max. trips per user", "value": fmt(mdreport.max_trips_per_user)},
        {"name": "Privacy budget", "value": fmt(mdreport.privacy_budget)},
        {"name": "User privacy", "value": fmt(mdreport.user_privacy)},
        {"name": "Analysis selection", "value": fmt(mdreport.analysis_selection)},
        {"name": "Evaluation dev. mode", "value": fmt(mdreport.evalu)}
    ]

    # create html from template
    template_table = get_template("table.html")
    dataset_stats_html = template_table.render(
        name="Configuration", rows=dataset_stats_list
    )
    return dataset_stats_html
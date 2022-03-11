from typing import Any, Optional, Union

import jinja2
import numpy as np
from pandas import Series

# Initializing Jinja
package_loader = jinja2.PackageLoader(
    "dp_mobility_report", "report/html/html_templates"
)
jinja2_env = jinja2.Environment(
    lstrip_blocks=True, trim_blocks=True, loader=package_loader
)


def get_template(template_name: str) -> jinja2.Template:
    return jinja2_env.get_template(template_name)


def render_summary(summary: Series, title: str = "Distribution") -> str:
    summary_list = [
        {"name": "Min.", "value": fmt(summary["min"])},
        {"name": "25%", "value": fmt(summary["25%"])},
        {"name": "Median", "value": fmt(summary["50%"])},
        {"name": "75%", "value": fmt(summary["75%"])},
        {"name": "Max.", "value": fmt(summary["max"])},
    ]
    if "mean" in summary:
        summary_list.insert(0, {"name": "Mean", "value": fmt(summary["mean"])})

    template_table = jinja2_env.get_template("table.html")
    summary_html = template_table.render(name=title, rows=summary_list)
    return summary_html


def render_outlier_info(
    outlier_count: int,
    margin_of_error=Optional[float],
    max_value: Union[int, float] = None,
) -> str:
    margin_of_error = round(margin_of_error) if margin_of_error is not None else None
    ci_interval_info = (
        f"(95% confidence interval ± {margin_of_error})"
        if margin_of_error is not None
        else ""
    )
    outlier = (
        f"outlier {ci_interval_info} has"
        if outlier_count == 1
        else f"outliers {ci_interval_info} have"
    )
    range_info = (
        f'<br>Outliers are values above " {max_value}'
        if (max_value is not None)
        else ""
    )
    return f"{outlier_count} {outlier} been excluded. {range_info}"

def render_moe_info(margin_of_error:int) -> str:
    return f"""To provide privacy, quartile values are not necessarily the true values but, e.g., instead of the true maximum value the 
        second or third highest value is displayed.
        This is achieved by the so-called exponential mechanism, where a value is drawn based on probabilites defined by the privacy budget. 
        Generally, a value closer to the true value has a higher chance of being drawn.
        The true quartile values lie with a <b>95% chance ± {margin_of_error} records</b> away from the true values."""

def fmt(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        value = round(value, 2)
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(
        value, bool
    ):
        value = f"{value:,}"
    return value

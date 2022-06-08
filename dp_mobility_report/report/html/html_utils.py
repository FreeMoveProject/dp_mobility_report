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


def render_summary(summary: Series, title: str = "") -> str:
    summary_list = [
        {"name": "Min.", "value": fmt(summary["min"])},
        {"name": "Max.", "value": fmt(summary["max"])},
    ]
    if "25%" in summary:
        summary_list.insert(1, {"name": "75%", "value": fmt(summary["75%"])})
        summary_list.insert(1, {"name": "Median", "value": fmt(summary["50%"])})
        summary_list.insert(1, {"name": "25%", "value": fmt(summary["25%"])})

    if "mean" in summary:
        summary_list.insert(0, {"name": "Mean", "value": fmt(summary["mean"])})

    template_table = jinja2_env.get_template("table.html")
    summary_html = template_table.render(name=title, rows=summary_list)
    return summary_html


def render_user_input_info(
    max_value: Optional[Union[int, float]], bin_size: Optional[Union[int, float]]
) -> str:
    return f"""User configuration for histogram chart: <br>
            maximum value: {max_value} <br>
            bin size: {bin_size}"""


def render_moe_info(margin_of_error: int) -> str:
    return """To provide privacy, quartile values are not necessarily the true values but, e.g., instead of the true maximum value the 
        second or third highest value is displayed.
        This is achieved by the so-called exponential mechanism, where a value is drawn based on probabilites defined by the privacy budget. 
        Generally, a value closer to the true value has a higher chance of being drawn."""
    # The true quartile values lie with a <b>95% chance within ± {margin_of_error} records</b> away from the true values.""" # TODO: margin_of_error reveals true record count?


def fmt(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        value = round(value, 2)
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(
        value, bool
    ):
        value = f"{value:,}"
    return value

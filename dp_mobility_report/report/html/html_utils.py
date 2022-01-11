import numbers
from typing import Any, Union

import jinja2
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


def render_summary(summary: Series) -> str:
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
    summary_html = template_table.render(name="Distribution", rows=summary_list)
    return summary_html


def render_outlier_info(outlier_count: int, max_value: Union[int, float] = None) -> str:
    outlier = "outlier has" if outlier_count == 1 else "outliers have"
    range_info = (
        "<br>Outliers are values above " + str(max_value)
        if (max_value is not None)
        else ""
    )
    return (
        "<div>"
        + str(outlier_count)
        + " "
        + outlier
        + " been excluded. "
        + range_info
        + "</div>"
    )


def fmt(value: Any) -> Any:
    # TODO: check correctly
    if isinstance(value, numbers.Number):
        value = round(value, 2)
    return value

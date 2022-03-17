from typing import Any, Optional, Union

import jinja2
import numpy as np
from pandas import DataFrame, Series
from dp_mobility_report.privacy.diff_privacy import _laplacer


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
    margin_of_error: Optional[float],
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


def render_moe_info(margin_of_error: int) -> str:
    return f"""To provide privacy, quartile values are not necessarily the true values but, e.g., instead of the true maximum value the 
        second or third highest value is displayed.
        This is achieved by the so-called exponential mechanism, where a value is drawn based on probabilites defined by the privacy budget. 
        Generally, a value closer to the true value has a higher chance of being drawn."""
    #The true quartile values lie with a <b>95% chance within ± {margin_of_error} records</b> away from the true values.""" # TODO: margin_of_error reveals true record count?


def fmt(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        value = round(value, 2)
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(
        value, bool
    ):
        value = f"{value:,}"
    return value


def _cumsum(series: Series):
    return round(
        series.sort_values(ascending=False).cumsum()
        / sum(series),
        2,
    ).reset_index(drop=True)

def cumsum_simulations(series: DataFrame, eps: float, sensitivity: int):
    df_cumsum = DataFrame()
    df_cumsum["n"] = np.arange(1, len(series) + 1)
    df_cumsum["cum_perc"] = _cumsum(series)

    for i in range(1, 50):
            sim_counts = series.apply(lambda x: _laplacer(x, eps = eps, sensitivity=sensitivity))
            sim_counts = sim_counts.apply(lambda x: int((abs(x) + x) / 2))
            df_cumsum["cum_perc_" + str(i)] = _cumsum(sim_counts)

    df_cumsum.reset_index(drop=True, inplace=True)
    return df_cumsum
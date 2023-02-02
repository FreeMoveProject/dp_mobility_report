from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from dp_mobility_report import constants as const
from dp_mobility_report.model import m_utils
from dp_mobility_report.model.section import DfSection
from dp_mobility_report.privacy import diff_privacy


def get_visits_per_tile(
    dpmreport: "DpMobilityReport",
    eps: Optional[float],
) -> DfSection:
    epsi = eps

    sensitivity = 2 * dpmreport.count_sensitivity_base
    # count number of visits for each location
    visits_per_tile = (
        dpmreport.df[
            dpmreport.df[const.TILE_ID].isin(dpmreport.tessellation.tile_id)
        ]  # only include records within tessellation
        .groupby(const.TILE_ID)
        .aggregate(visits=(const.TILE_ID, "count"))
        .sort_values("visits", ascending=False)
        .reset_index()
    )

    # number of records outside of the tessellation
    n_outliers = int(len(dpmreport.df) - visits_per_tile.visits.sum())

    visits_per_tile = visits_per_tile.merge(
        dpmreport.tessellation[[const.TILE_ID, const.TILE_NAME]],
        on=const.TILE_ID,
        how="outer",
    )
    visits_per_tile.loc[visits_per_tile.visits.isna(), "visits"] = 0

    visits_per_tile["visits"] = diff_privacy.counts_dp(
        visits_per_tile["visits"].values,
        epsi,
        sensitivity,
        allow_negative=False,
    )
    n_outliers = diff_privacy.count_dp(n_outliers, epsi, sensitivity)

    cumsum = m_utils.cumsum(
        visits_per_tile.visits.copy().to_numpy(),
        epsi,
        sensitivity,
    )

    # margin of error
    moe = diff_privacy.laplace_margin_of_error(0.95, epsi, sensitivity)

    # as counts are already dp, no further privacy mechanism needed
    dp_quartiles = visits_per_tile.visits.describe()

    return DfSection(
        data=visits_per_tile,
        privacy_budget=eps,
        sensitivity=sensitivity,
        n_outliers=n_outliers,
        quartiles=dp_quartiles,
        margin_of_error_laplace=moe,
        cumsum=cumsum,
    )


def _get_hour_bin(hour: int, timewindows: np.ndarray) -> str:
    if hour >= timewindows.min() and hour < timewindows.max():
        i = np.argwhere((timewindows) <= hour)[-1][0]
        min_v = timewindows[i]
        max_v = timewindows[i + 1]
    else:
        i = len(timewindows) - 1
        min_v = timewindows[-1]
        max_v = timewindows[0]
    return f"{i + 1}: {min_v}-{max_v}"


def get_visits_per_time_tile(
    dpmreport: "DpMobilityReport",
    eps: Optional[float],
    # trip_count: Optional[int], outlier_count: Optional[None]
) -> DfSection:
    dpmreport.df["timewindows"] = dpmreport.df[const.HOUR].apply(
        lambda x: _get_hour_bin(x, dpmreport.timewindows)
    )

    # only points within tessellation and end points
    counts_per_tile_timewindow = dpmreport.df[
        (dpmreport.df[const.POINT_TYPE] == const.END)
        & dpmreport.df[const.TILE_ID].isin(dpmreport.tessellation[const.TILE_ID])
    ][[const.TILE_ID, const.IS_WEEKEND, "timewindows"]]

    moe = diff_privacy.laplace_margin_of_error(
        0.95, eps, dpmreport.count_sensitivity_base
    )

    # create full combination of all times and tiles for application of dp
    tile_ids = dpmreport.tessellation[const.TILE_ID].unique()
    is_weekend = dpmreport.df[const.IS_WEEKEND].unique()
    timewindows = dpmreport.df.timewindows.unique()
    full_combination = pd.DataFrame(
        list(map(np.ravel, np.meshgrid(tile_ids, is_weekend, timewindows))),
        index=[const.TILE_ID, const.IS_WEEKEND, "timewindows"],
    ).T
    counts_per_tile_timewindow = pd.concat(
        [counts_per_tile_timewindow, full_combination]
    )

    counts_per_tile_timewindow = (
        counts_per_tile_timewindow.reset_index()
        .pivot_table(
            index=const.TILE_ID,
            columns=[const.IS_WEEKEND, "timewindows"],
            aggfunc="count",
        )
        .droplevel(level=0, axis=1)
    )

    # remove instance from full_combination
    counts_per_tile_timewindow = counts_per_tile_timewindow.dropna() - 1
    counts_per_tile_timewindow = counts_per_tile_timewindow.unstack()

    counts_per_tile_timewindow = pd.Series(
        index=counts_per_tile_timewindow.index,
        data=diff_privacy.counts_dp(
            counts_per_tile_timewindow.values, eps, dpmreport.count_sensitivity_base
        ),
    )

    return DfSection(
        data=counts_per_tile_timewindow.unstack(const.TILE_ID).T,
        privacy_budget=eps,
        margin_of_error_laplace=moe,
    )

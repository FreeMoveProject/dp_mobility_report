from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

from dp_mobility_report import constants as const
from dp_mobility_report.model import m_utils
from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_visits_per_tile(
    mreport: "DpMobilityReport", eps: Optional[float], record_count: Optional[int]
) -> Section:
    epsi = eps

    sensitivity = 2 * mreport.max_trips_per_user
    # count number of visits for each location
    visits_per_tile = (
        mreport.df[
            mreport.df[const.TILE_ID].isin(mreport.tessellation.tile_id)
        ]  # only include records within tessellation
        .groupby(const.TILE_ID)
        .aggregate(visits=(const.TILE_ID, "count"))
        .sort_values("visits", ascending=False)
        .reset_index()
    )

    # number of records outside of the tessellation
    n_outliers = int(len(mreport.df) - visits_per_tile.visits.sum())

    visits_per_tile = visits_per_tile.merge(
        mreport.tessellation[[const.TILE_ID, const.TILE_NAME]],
        on=const.TILE_ID,
        how="outer",
    )
    visits_per_tile.loc[visits_per_tile.visits.isna(), "visits"] = 0

    visits_per_tile["visits"] = diff_privacy.counts_dp(
        visits_per_tile["visits"].values,
        epsi,
        sensitivity,
        allow_negative=True,  # allow negative values for cum_sum simulations
    )
    n_outliers = diff_privacy.count_dp(n_outliers, epsi, sensitivity)  # type: ignore

    cumsum_simulations = m_utils.cumsum_simulations(
        visits_per_tile.visits.copy().to_numpy(),
        epsi,
        sensitivity,
    )

    # remove all negative values (needed for cumsum)
    visits_per_tile["visits"] = visits_per_tile["visits"].apply(
        diff_privacy.limit_negative_value_to_zero
    )

    # margin of error
    moe = diff_privacy.laplace_margin_of_error(0.95, epsi, sensitivity)

    # scale to record count of overview segment
    if record_count is not None:
        visists_sum = np.sum(visits_per_tile["visits"])
        if visists_sum != 0:
            visits_per_tile["visits"] = (
                visits_per_tile["visits"] / visists_sum * record_count
            ).astype(int)
            n_outliers = int(n_outliers / visists_sum * record_count)
            moe = int(moe / visists_sum * record_count)

    # as counts are already dp, no further privacy mechanism needed
    dp_quartiles = visits_per_tile.visits.describe()

    return Section(
        data=visits_per_tile,
        privacy_budget=eps,
        sensitivity=sensitivity,
        n_outliers=n_outliers,
        quartiles=dp_quartiles,
        margin_of_error_laplace=moe,
        cumsum_simulations=cumsum_simulations,
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


def get_visits_per_tile_timewindow(
    mreport: "DpMobilityReport", eps: Optional[float], record_count: Optional[int]
) -> Section:
    mreport.df["timewindows"] = mreport.df[const.HOUR].apply(
        lambda x: _get_hour_bin(x, mreport.timewindows)
    )

    # only points within tessellation and end points
    counts_per_tile_timewindow = mreport.df[
        (mreport.df[const.POINT_TYPE] == const.END)
        & mreport.df[const.TILE_ID].isin(mreport.tessellation[const.TILE_ID])
    ][[const.TILE_ID, const.IS_WEEKEND, "timewindows"]]

    # create full combination of all times and tiles for application of dp
    tile_ids = mreport.tessellation[const.TILE_ID].unique()
    is_weekend = mreport.df[const.IS_WEEKEND].unique()
    timewindows = mreport.df.timewindows.unique()
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

    counts_per_tile_timewindow = (
        counts_per_tile_timewindow.dropna() - 1
    )  # remove instance from full_combination
    counts_per_tile_timewindow = counts_per_tile_timewindow.unstack()

    counts_per_tile_timewindow = pd.Series(
        index=counts_per_tile_timewindow.index,
        data=diff_privacy.counts_dp(
            counts_per_tile_timewindow.values, eps, mreport.max_trips_per_user
        ),
    )

    moe = diff_privacy.laplace_margin_of_error(0.95, eps, mreport.max_trips_per_user)

    # scale to record count of overview segment
    if (record_count is not None) and (counts_per_tile_timewindow.sum() != 0):
        counts_sum = counts_per_tile_timewindow.sum()
        counts_per_tile_timewindow = (
            counts_per_tile_timewindow / counts_sum * record_count
        )
        moe = moe / counts_sum * record_count

    return Section(
        data=counts_per_tile_timewindow.unstack(const.TILE_ID).T,
        privacy_budget=eps,
        margin_of_error_laplace=moe,
    )

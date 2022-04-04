from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dp_mobility_report.md_report import MobilityDataReport

from dp_mobility_report import constants as const
from dp_mobility_report.model import m_utils
from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_visits_per_tile(
    mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:
    epsi = eps
    # epsi = m_utils.get_epsi(
    #     mdreport.evalu, eps, 2
    # )  
    # TODO: is this really eps / 2? bc outliers are like an own tile?

    sensitivity = 2 * mdreport.max_trips_per_user
    # count number of visits for each location
    counts_per_tile = (
        mdreport.df[
            mdreport.df[const.TILE_ID].isin(mdreport.tessellation.tile_id)
        ]  # only include records within tessellation
        .groupby(const.TILE_ID)
        .aggregate(visit_count=(const.TILE_ID, "count"))
        .sort_values("visit_count", ascending=False)
        .reset_index()
    )

    # number of records outside of the tessellation
    n_outliers = int(len(mdreport.df) - counts_per_tile.visit_count.sum())

    counts_per_tile = counts_per_tile.merge(
        mdreport.tessellation[[const.TILE_ID, const.TILE_NAME]],
        on=const.TILE_ID,
        how="outer",
    )
    counts_per_tile.loc[counts_per_tile.visit_count.isna(), "visit_count"] = 0

    counts_per_tile["visit_count"] = diff_privacy.counts_dp(
        counts_per_tile["visit_count"].values,
        epsi,
        sensitivity,
        allow_negative = True, # allow negative values for cum_sum simulations
    )
    n_outliers = diff_privacy.count_dp(n_outliers, epsi, sensitivity)  # type: ignore

    cumsum_simulations = m_utils.cumsum_simulations(
        counts_per_tile.visit_count.copy().to_numpy(),
        epsi,
        sensitivity,
    )

    # remove all negative values (needed for cumsum)
    counts_per_tile["visit_count"] = counts_per_tile["visit_count"].apply(diff_privacy.limit_negative_values_to_zero)

    # margin of error
    moe = diff_privacy.laplace_margin_of_error(0.95, epsi, sensitivity)
    
    # as percent instead of absolute values
    visists_sum = np.sum(counts_per_tile["visit_count"]) 
    if visists_sum != 0:
        counts_per_tile["visit_count"] = counts_per_tile["visit_count"] / visists_sum * 100
        n_outliers = n_outliers / visists_sum * 100
        moe = moe / visists_sum * 100

    # as counts are already dp, no further privacy mechanism needed
    dp_quartiles = counts_per_tile.visit_count.describe()



    return Section(
        data=counts_per_tile,
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
    mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:
    mdreport.df["timewindows"] = mdreport.df[const.HOUR].apply(
        lambda x: _get_hour_bin(x, mdreport.timewindows)
    )

    # only points within tessellation and end points
    counts_per_tile_timewindow = mdreport.df[
        (mdreport.df[const.POINT_TYPE] == const.END)
        & mdreport.df[const.TILE_ID].isin(mdreport.tessellation[const.TILE_ID])
    ][[const.TILE_ID, const.IS_WEEKEND, "timewindows"]]

    # create full combination of all times and tiles for application of dp
    tile_ids = mdreport.tessellation[const.TILE_ID].unique()
    is_weekend = mdreport.df[const.IS_WEEKEND].unique()
    timewindows = mdreport.df.timewindows.unique()
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
            counts_per_tile_timewindow.values, eps, mdreport.max_trips_per_user
        ),
    )

    moe = diff_privacy.laplace_margin_of_error(0.95, eps, mdreport.max_trips_per_user)

    # as percent instead of absolute values
    if counts_per_tile_timewindow.sum() != 0:
        counts_sum = counts_per_tile_timewindow.sum()
        counts_per_tile_timewindow = counts_per_tile_timewindow / counts_sum * 100
        moe = moe / counts_sum

    return Section(
        data=counts_per_tile_timewindow.unstack(const.TILE_ID).T,
        privacy_budget=eps,
        margin_of_error_laplace=moe,
    )

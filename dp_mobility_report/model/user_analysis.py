import math
from datetime import timedelta
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from dp_mobility_report.md_report import MobilityDataReport

import numpy as np
import pandas as pd
from scipy import stats

from dp_mobility_report import constants as const
from dp_mobility_report.model import m_utils
from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_trips_per_user(mdreport: "MobilityDataReport", eps: Optional[float]) -> Section:
    user_nunique = mdreport.df.groupby(const.UID).nunique()[const.TID]
    max_trips = mdreport.max_trips_per_user if mdreport.user_privacy else None

    return m_utils.hist_section(
        user_nunique,
        eps,
        sensitivity=1,
        max_value=max_trips,
        evalu=mdreport.evalu,
    )


def get_user_time_delta(
    mdreport: "MobilityDataReport", eps: Optional[float]
) -> Optional[Section]:
    epsi = m_utils.get_epsi(mdreport.evalu, eps, 6)
    epsi_quant = epsi * 5 if epsi is not None else None

    mdreport.df = mdreport.df.sort_values(
        [const.UID, const.TID, const.DATETIME]
    )  # assuming tid numbers are integers and given in a chronological order, as arranged in "preprocessing"
    same_user = mdreport.df[const.UID] == mdreport.df[const.UID].shift()
    same_tid = mdreport.df[const.TID] == mdreport.df[const.TID].shift()
    user_time_delta = mdreport.df[const.DATETIME] - mdreport.df[const.DATETIME].shift()
    user_time_delta[(same_tid) | (~same_user)] = None
    user_time_delta = user_time_delta[user_time_delta.notnull()]
    overlaps = len(user_time_delta[user_time_delta < timedelta(seconds=0)])

    if len(user_time_delta) < 1:
        return None

    n_overlaps = diff_privacy.count_dp(
        overlaps,
        epsi,
        mdreport.max_trips_per_user,
    )
    dp_quartiles = diff_privacy.quartiles_dp(
        user_time_delta, epsi_quant, mdreport.max_trips_per_user
    )

    moe = diff_privacy.laplace_margin_of_error(0.95, epsi, mdreport.max_trips_per_user)

    return Section(
        data=None,
        privacy_budget=eps,
        n_outliers=n_overlaps,
        quartiles=dp_quartiles,
        margin_of_error=moe,
    )


def get_radius_of_gyration(
    mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:
    rg = _radius_of_gyration(mdreport.df)
    return m_utils.hist_section(
        rg,
        eps,
        sensitivity=1,
        min_value=0,
        max_value=mdreport.max_radius_of_gyration,
        bin_range=mdreport.bin_range_radius_of_gyration,
        evalu=mdreport.evalu,
    )


def _radius_of_gyration(df: pd.DataFrame) -> pd.Series:
    # create a lat_lng array for each individual
    lats_lngs = (
        df.set_index(const.UID)[[const.LAT, const.LNG]].groupby(level=0).apply(np.array)
    )
    # compute the center of mass for each individual
    center_of_masses = np.array([np.mean(x, axis=0) for x in lats_lngs])
    center_of_masses_df = pd.DataFrame(
        data=center_of_masses, index=lats_lngs.index, columns=["com_lat", "com_lng"]
    )

    df_rog = df.merge(
        center_of_masses_df, how="left", left_on=const.UID, right_index=True
    )

    # compute the distance between each location and its according center of mass
    def _haversine_dist_squared(coords: List[float]) -> float:
        return m_utils.haversine_dist(coords) ** 2

    df_rog["com_dist"] = df_rog[
        [const.LAT, const.LNG, "com_lat", "com_lng"]
    ].parallel_apply(_haversine_dist_squared, axis=1)

    # compute radius of gyration
    def _mean_then_square(x: float) -> float:
        return np.sqrt(np.mean(x))

    rog = df_rog.groupby(const.UID).com_dist.apply(_mean_then_square)
    rog.name = const.RADIUS_OF_GYRATION
    return rog


def _tile_visits_by_user(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby([const.UID, const.TILE_ID], as_index=False).aggregate(
        count_by_user=(const.ID, "count")
    )


def get_location_entropy(
    mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:
    total_visits_by_tile = mdreport.df.groupby(const.TILE_ID).aggregate(
        total_visits=(const.ID, "count")
    )
    tile_visits_by_user = _tile_visits_by_user(mdreport.df).merge(
        total_visits_by_tile, left_on=const.TILE_ID, right_index=True
    )[[const.TILE_ID, "count_by_user", "total_visits"]]

    tile_visits_by_user["p"] = (
        tile_visits_by_user.count_by_user / tile_visits_by_user.total_visits
    )
    tile_visits_by_user["log2p"] = -tile_visits_by_user.p.apply(
        lambda x: math.log(x, 2)
    )
    tile_visits_by_user[const.LOCATION_ENTROPY] = (
        tile_visits_by_user.p * tile_visits_by_user.log2p
    )

    location_entropy = tile_visits_by_user.groupby(const.TILE_ID)[
        const.LOCATION_ENTROPY
    ].sum()
    location_entropy_dp = diff_privacy.entropy_dp(
        location_entropy, eps, mdreport.max_trips_per_user
    )
    data = pd.Series(
        location_entropy_dp, index=location_entropy.index, name=location_entropy.name
    )

    sensitivity = (
        2
        * mdreport.max_trips_per_user
        * (
            max(
                np.log(2),
                np.log(2 * mdreport.max_trips_per_user)
                - np.log(np.log(2 * mdreport.max_trips_per_user))
                - 1,
            )
        )
    )
    moe = diff_privacy.laplace_margin_of_error(0.95, eps, sensitivity)
    return Section(data=data, privacy_budget=eps, margin_of_error=moe)


def get_user_tile_count(
    mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:
    user_tile_count = mdreport.df.groupby(const.UID).nunique()[const.TILE_ID]

    return m_utils.hist_section(
        user_tile_count,
        eps,
        sensitivity=1,
        evalu=mdreport.evalu,
    )


def _mobility_entropy(df: pd.DataFrame) -> np.ndarray:
    total_visits_by_user = df.groupby(const.UID).aggregate(
        total_visits=(const.ID, "count")
    )
    tile_visits_by_user = _tile_visits_by_user(df).merge(
        total_visits_by_user, left_on=const.UID, right_index=True
    )
    tile_visits_by_user["probs"] = (
        tile_visits_by_user.count_by_user / tile_visits_by_user.total_visits
    )

    entropy = tile_visits_by_user.groupby(const.UID).probs.apply(
        lambda x: stats.entropy(x, base=2)
    )

    n_vals = df.groupby(const.UID)[const.TILE_ID].nunique()
    entropy = np.where(
        n_vals > 1, np.divide(entropy, np.log2(n_vals), where=n_vals > 1), 0
    )
    return entropy


def get_mobility_entropy(
    mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:
    mobility_entropy = _mobility_entropy(mdreport.df)

    return m_utils.hist_section(
        mobility_entropy,
        eps,
        sensitivity=1,
        bin_range=0.1,
        min_value=0,
        max_value=1,
        evalu=mdreport.evalu,
    )

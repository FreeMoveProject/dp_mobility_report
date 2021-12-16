import math
from datetime import timedelta

import numpy as np
import pandas as pd
from haversine import Unit, haversine
from pandarallel import pandarallel
from scipy import stats

from dp_mobility_report.model import utils
from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_traj_per_user(mdreport, eps):
    user_nunique = mdreport.df.groupby("uid").nunique().tid
    max_trips = (
        mdreport.max_trips_per_user if mdreport.user_privacy else user_nunique.max()
    )
    # TODO: define max_bins
    max_bins = max_trips if max_trips < 10 else 10

    return utils.dp_hist_section(
        user_nunique,
        eps,
        sensitivity=1,
        min_value=1,
        max_value=max_trips,
        max_bins=max_bins,
        evalu=mdreport.evalu,
    )


def get_user_time_delta(mdreport, eps):
    if mdreport.evalu is True or eps is None:
        epsi = eps
        epsi_quart = epsi
    else:
        epsi = eps / 6
        epsi_quart = 5 * epsi

    df = mdreport.df.sort_values(
        ["uid", "tid", "datetime"]
    )  # assuming tid numbers are integers and given in a chronological order
    same_user = df.uid == df.uid.shift()
    same_tid = df.tid == df.tid.shift()
    user_time_delta = df.datetime - df.datetime.shift()
    user_time_delta[(same_tid) | (~same_user)] = None
    n_deltas = len(user_time_delta[user_time_delta.notnull()])
    user_time_delta = user_time_delta[user_time_delta >= timedelta(seconds=0)]

    if len(user_time_delta) < 1:
        return None

    n_overlaps = diff_privacy.counts_dp(
        n_deltas - len(user_time_delta),
        epsi,
        mdreport.max_trips_per_user,
        parallel=True,
        nonzero=False,
    )
    dp_quartiles = diff_privacy.quartiles_dp(
        user_time_delta, epsi_quart, mdreport.max_trips_per_user
    )

    return Section(data=None, n_outliers=n_overlaps, quartiles=dp_quartiles)


def get_radius_of_gyration(mdreport, eps):
    rg = _radius_of_gyration(mdreport.df)
    return utils.dp_hist_section(
        rg,
        eps,
        sensitivity=1,
        min_value=0,
        max_value=mdreport.max_radius_of_gyration,
        bin_size=mdreport.bin_size_radius_of_gyration,
        evalu=mdreport.evalu,
    )


def _haversine_dist_squared(coords):
    return (
        haversine(
            (float(coords[0]), float(coords[1])),
            (float(coords[2]), float(coords[3])),
            unit=Unit.METERS,
        )
        ** 2
    )


def _mean_then_square(x):
    return np.sqrt(np.mean(x))


def _radius_of_gyration(df):
    # create a lat_lng array for each individual
    lats_lngs = df.set_index("uid")[["lat", "lng"]].groupby(level=0).apply(np.array)
    # compute the center of mass for each individual
    center_of_masses = np.array([np.mean(x, axis=0) for x in lats_lngs])
    center_of_masses_df = pd.DataFrame(
        data=center_of_masses, index=lats_lngs.index, columns=["com_lat", "com_lng"]
    )

    df_rg = df.merge(center_of_masses_df, how="left", left_on="uid", right_index=True)
    # compute the distance between each location and its according center of mass
    df_rg["com_dist"] = df_rg[["lat", "lng", "com_lat", "com_lng"]].parallel_apply(
        _haversine_dist_squared, axis=1
    )
    # compute radius of gyration
    rg = df_rg.groupby("uid").com_dist.apply(_mean_then_square)
    rg.name = "radius_of_gyration"
    return rg


def get_location_entropy(mdreport, eps):

    # location entropy (based on Shannon Entropy)
    tile_count_per_user = (
        mdreport.df.reset_index()
        .groupby(["tile_id", "uid"])
        .aggregate(count_by_user=("id", "count"))
        .reset_index()
    )
    total_count = (
        mdreport.df.reset_index()
        .groupby("tile_id")
        .aggregate(total_count=("id", "count"))
    )

    # entropy of single tiles
    tile_count_by_user = pd.merge(
        tile_count_per_user, total_count, left_on="tile_id", right_index=True
    )[["tile_id", "count_by_user", "total_count"]]

    tile_count_by_user["p"] = (
        tile_count_by_user.count_by_user / tile_count_by_user.total_count
    )
    tile_count_by_user["log2p"] = -tile_count_by_user.p.apply(lambda x: math.log(x, 2))
    tile_count_by_user["location_entropy"] = (
        tile_count_by_user.p * tile_count_by_user.log2p
    )

    location_entropy = tile_count_by_user.groupby("tile_id").sum().location_entropy
    location_entropy_dp = diff_privacy.entropy_dp(
        location_entropy, eps, mdreport.max_trips_per_user
    )
    return pd.Series(
        location_entropy_dp, index=location_entropy.index, name=location_entropy.name
    )


def get_user_tile_count(mdreport, eps):
    user_tile_count = mdreport.df.groupby("uid").nunique().tile_id
    # TODO: define max_bins
    max_bins = mdreport.max_trips_per_user if mdreport.max_trips_per_user < 10 else 10
    return utils.dp_hist_section(
        user_tile_count,
        eps,
        sensitivity=1,
        min_value=1,
        max_value=2 * mdreport.max_trips_per_user,
        max_bins=max_bins,
        evalu=mdreport.evalu,
    )


def _uncorrelated_entropy(df):
    total_visits_by_user = df.groupby("uid").id.count()
    user_visits_to_tiles = df.groupby(["uid", "tile_id"], as_index=False).id.count()
    user_visits_to_tiles = user_visits_to_tiles.merge(
        total_visits_by_user,
        left_on="uid",
        right_index=True,
        suffixes=["_tile_visits", "_total_visits"],
    )
    user_visits_to_tiles["probs"] = (
        1.0 * user_visits_to_tiles.id_tile_visits / user_visits_to_tiles.id_total_visits
    )
    entropy = user_visits_to_tiles.groupby("uid").probs.apply(
        lambda x: stats.entropy(x, base=2)
    )

    n_vals = df.groupby("uid").tile_id.nunique()
    entropy = np.where(
        n_vals > 1, np.divide(entropy, np.log2(n_vals), where=n_vals > 1), 0
    )
    return entropy


# TODO: define min and max values
def get_uncorrelated_entropy(mdreport, eps):
    uncorrel_entropy = _uncorrelated_entropy(mdreport.df)

    return utils.dp_hist_section(
        uncorrel_entropy,
        eps,
        sensitivity=1,
        # min_value = uncorrel_entropy.min(),
        # max_value = uncorrel_entropy.max(),
        max_bins=10,
        evalu=mdreport.evalu,
    )

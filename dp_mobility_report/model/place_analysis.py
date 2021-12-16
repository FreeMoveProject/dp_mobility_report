import numpy as np
import pandas as pd

from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_counts_per_tile(mdreport, eps):
    if mdreport.evalu is True or eps is None:
        epsi = eps
    else:
        epsi = eps / 3

    # count number of visits for each tile
    counts_per_tile = (
        mdreport.df[
            mdreport.df.tile_id.isin(mdreport.tessellation.tile_id)
        ]  # only include records within tessellation
        .groupby("tile_id")
        .aggregate(visit_count=("tile_id", "count"))
        .sort_values("visit_count", ascending=False)
        .reset_index()
    )

    # number of records outside of the tessellation
    n_outliers = len(mdreport.df) - counts_per_tile.visit_count.sum()

    counts_per_tile = counts_per_tile.merge(
        mdreport.tessellation[["tile_id", "tile_name"]], on="tile_id", how="outer"
    )
    counts_per_tile.loc[counts_per_tile.visit_count.isna(), "visit_count"] = 0

    dp_quartiles = diff_privacy.quartiles_dp(
        counts_per_tile.visit_count, epsi, mdreport.max_trips_per_user * 2
    )

    counts_per_tile["visit_count"] = diff_privacy.counts_dp(
        counts_per_tile["visit_count"],
        epsi,
        mdreport.max_trips_per_user * 2,
        parallel=True,
        nonzero=False,
    )
    # Todo saskia: for outliers more from priv budget
    n_outliers = diff_privacy.counts_dp(
        n_outliers, epsi, mdreport.max_trips_per_user * 2, parallel=True, nonzero=False
    ).item()

    # percentage
    full_count = counts_per_tile.visit_count.sum()
    counts_per_tile["visit_perc"] = (
        round(counts_per_tile.visit_count / full_count, 4) * 100
    )
    return Section(data=counts_per_tile, n_outliers=n_outliers, quartiles=dp_quartiles)


def get_counts_per_tile_timewindow(mdreport, eps):
    # hour bins
    mdreport.df.loc[
        (mdreport.df.hour >= 2) & (mdreport.df.hour < 6), "hour_bins"
    ] = "1: 2-6"
    mdreport.df.loc[
        (mdreport.df.hour >= 6) & (mdreport.df.hour < 10), "hour_bins"
    ] = "2: 6-10"
    mdreport.df.loc[
        (mdreport.df.hour >= 10) & (mdreport.df.hour < 14), "hour_bins"
    ] = "3: 10-14"
    mdreport.df.loc[
        (mdreport.df.hour >= 14) & (mdreport.df.hour < 18), "hour_bins"
    ] = "4: 14-18"
    mdreport.df.loc[
        (mdreport.df.hour >= 18) & (mdreport.df.hour < 22), "hour_bins"
    ] = "5: 18-22"
    mdreport.df.loc[
        (mdreport.df.hour >= 22) | (mdreport.df.hour < 2), "hour_bins"
    ] = "6: 22-2"

    # only use end points and records within tessellation
    counts_per_tile_timewindow = mdreport.df[
        (mdreport.df.point_type == "end")
        & mdreport.df.tile_id.isin(mdreport.tessellation.tile_id)
    ][["tile_id", "is_weekend", "hour_bins"]]

    # create full combination of all times and tiles
    tile_ids = mdreport.tessellation.tile_id.unique()
    is_weekend = mdreport.df.is_weekend.unique()
    hour_bins = mdreport.df.hour_bins.unique()
    full_combination = pd.DataFrame(
        list(map(np.ravel, np.meshgrid(tile_ids, is_weekend, hour_bins))),
        index=["tile_id", "is_weekend", "hour_bins"],
    ).T
    counts_per_tile_timewindow = pd.concat(
        [counts_per_tile_timewindow, full_combination]
    )

    counts_per_tile_timewindow = (
        counts_per_tile_timewindow.reset_index()
        .pivot_table(
            index="tile_id", columns=["is_weekend", "hour_bins"], aggfunc="count"
        )
        .droplevel(level=0, axis=1)
    )

    counts_per_tile_timewindow = (
        counts_per_tile_timewindow.dropna() - 1
    )  # remove instance from full_combination
    counts_per_tile_timewindow = counts_per_tile_timewindow.unstack()

    dp_counts_per_tile_timewindow = diff_privacy.counts_dp(
        counts_per_tile_timewindow, eps, mdreport.max_trips_per_user, nonzero=False
    )
    return dp_counts_per_tile_timewindow.unstack("tile_id").T

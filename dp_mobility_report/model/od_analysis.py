from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from dp_mobility_report import constants as const
from dp_mobility_report.model import m_utils
from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_od_shape(df: pd.DataFrame, tessellation: GeoDataFrame) -> pd.DataFrame:
    ends_od_shape = (
        df[
            (df[const.POINT_TYPE] == const.END)
            & df[const.TILE_ID].isin(tessellation[const.TILE_ID])
        ][[const.TID, const.TILE_ID, const.DATETIME, const.LAT, const.LNG]]
        .merge(tessellation[[const.TILE_ID]], on=const.TILE_ID, how="left")
        .rename(
            columns={
                const.TILE_ID: const.TILE_ID_END,
                const.LAT: const.LAT_END,
                const.LNG: const.LNG_END,
                const.DATETIME: const.DATETIME_END,
            }
        )
    )

    od_shape = (
        df[
            (df[const.POINT_TYPE] == const.START)
            & df[const.TILE_ID].isin(tessellation[const.TILE_ID])
        ][[const.TID, const.TILE_ID, const.DATETIME, const.LAT, const.LNG]]
        .merge(tessellation[[const.TILE_ID]], on=const.TILE_ID, how="left")
        .merge(ends_od_shape, on=const.TID, how="inner")
    )

    return od_shape


def get_od_flows(
    od_shape: pd.DataFrame,
    dpmreport: "DpMobilityReport",
    eps: Optional[float],
    # complete_trip_count: Optional[int],
    # outlier_count = Optional[int]
) -> Section:
    sensitivity = dpmreport.max_trips_per_user
    od_flows = (
        od_shape.groupby([const.TILE_ID, const.TILE_ID_END])
        .aggregate(flow=(const.TID, "count"))
        .reset_index()
        .rename(
            columns={
                const.TILE_ID: "origin",
                const.TILE_ID_END: "destination",
            }
        )
        .sort_values("flow", ascending=False)
    )

    # margin of error
    moe = diff_privacy.laplace_margin_of_error(0.95, eps, sensitivity)

    # fill all potential combinations with 0s for correct application of dp
    full_tile_ids = np.unique(dpmreport.tessellation[const.TILE_ID])
    full_combinations = list(map(np.ravel, np.meshgrid(full_tile_ids, full_tile_ids)))
    od_flows = pd.DataFrame(
        {"origin": full_combinations[0], "destination": full_combinations[1]}
    ).merge(od_flows, on=["origin", "destination"], how="left")
    od_flows.fillna(0, inplace=True)

    od_flows["flow"] = diff_privacy.counts_dp(
        od_flows["flow"].to_numpy(), eps, sensitivity, allow_negative=True
    )

    cumsum_simulations = m_utils.cumsum_simulations(
        od_flows.flow.copy().to_numpy(), eps, sensitivity
    )

    # remove all instances of 0 (and smaller) to reduce storage
    od_flows = od_flows[od_flows["flow"] > 0]


    # scaling not working with too many cells -> too many cells < 1
    # # plausibility check: scale total od_count if it exceeds the trip_count or goes below trip_count-outlier_count
    # if (complete_trip_count is not None) and (np.sum(od_flows["flow"]) != 0):
    #     od_sum = np.sum(od_flows["flow"])
    #     scale_data = False

    #     # scale down if od_sum is larger than trip_count
    #     if od_sum > complete_trip_count:
    #         scale_data = True

    #     # scale up if od_sum is lower than trip_count-outlier_count
    #     if outlier_count is not None and (od_sum < complete_trip_count-outlier_count):
    #         complete_trip_count = complete_trip_count-outlier_count
    #         scale_data = True

    #     if scale_data:
    #         od_flows["flow"] = (od_flows["flow"] / od_sum * complete_trip_count).astype(int)
    #         moe = int(moe / od_sum * complete_trip_count)


    # TODO: distribution with or without 0s?
    # as counts are already dp, no further privacy mechanism needed
    dp_quartiles = od_flows.flow.describe()

    return Section(
        data=od_flows,
        quartiles=dp_quartiles,
        privacy_budget=eps,
        margin_of_error_laplace=moe,
        sensitivity=sensitivity,
        cumsum_simulations=cumsum_simulations,
    )


def get_intra_tile_flows(od_flows: pd.DataFrame) -> int:
    return od_flows[(od_flows.origin == od_flows.destination)].flow.sum()


def get_travel_time(
    od_shape: pd.DataFrame, dpmreport: "DpMobilityReport", eps: Optional[float]
) -> Section:

    travel_time = od_shape[const.DATETIME_END] - od_shape[const.DATETIME]
    travel_time = travel_time.dt.seconds / 60  # as minutes

    return m_utils.hist_section(
        travel_time,
        eps,
        dpmreport.max_trips_per_user,
        hist_max=dpmreport.max_travel_time,
        bin_range=dpmreport.bin_range_travel_time,
        bin_type=int,
        evalu=dpmreport.evalu,
    )


def get_jump_length(
    od_shape: pd.DataFrame, dpmreport: "DpMobilityReport", eps: Optional[float]
) -> Section:

    # parallel computation for speed up
    jump_length = od_shape[
        [const.LAT, const.LNG, const.LAT_END, const.LNG_END]
    ].parallel_apply(m_utils.haversine_dist, axis=1)
    return m_utils.hist_section(
        jump_length,
        eps,
        dpmreport.max_trips_per_user,
        hist_max=dpmreport.max_jump_length,
        bin_range=dpmreport.bin_range_jump_length,
        evalu=dpmreport.evalu,
    )

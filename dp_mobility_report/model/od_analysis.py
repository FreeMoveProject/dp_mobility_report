from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dp_mobility_report.md_report import MobilityDataReport

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
    od_shape: pd.DataFrame, mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:
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

    # fill all potential combinations with 0s for correct application of dp
    full_tile_ids = np.unique(mdreport.tessellation[const.TILE_ID])
    full_combinations = list(map(np.ravel, np.meshgrid(full_tile_ids, full_tile_ids)))
    od_flows = pd.DataFrame(
        {"origin": full_combinations[0], "destination": full_combinations[1]}
    ).merge(od_flows, on=["origin", "destination"], how="left")
    od_flows.fillna(0, inplace=True)

    od_flows["flow"] = diff_privacy.counts_dp(
        od_flows["flow"].to_numpy(), eps, mdreport.max_trips_per_user
    )

    # remove all instances of 0 to reduce storage
    od_flows = od_flows[od_flows["flow"] > 0]

    # as counts are already dp, no further privacy mechanism needed
    dp_quartiles = od_flows.flow.describe()

    moe = diff_privacy.laplace_margin_of_error(0.95, eps, mdreport.max_trips_per_user)

    return Section(
        data=od_flows,
        quartiles=dp_quartiles,
        privacy_budget=eps,
        margin_of_error_laplace=moe,
    )


def get_intra_tile_flows(od_flows: pd.DataFrame) -> int:
    return od_flows[(od_flows.origin == od_flows.destination)].flow.sum()


def get_travel_time(
    od_shape: pd.DataFrame, mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:

    travel_time = od_shape[const.DATETIME_END] - od_shape[const.DATETIME]
    travel_time = (travel_time.dt.seconds / 60).round()  # as minutes

    return m_utils.hist_section(
        travel_time,
        eps,
        mdreport.max_trips_per_user,
        min_value=0,
        max_value=mdreport.max_travel_time,
        bin_range=mdreport.bin_range_travel_time,
        evalu=mdreport.evalu,
    )


def get_jump_length(
    od_shape: pd.DataFrame, mdreport: "MobilityDataReport", eps: Optional[float]
) -> Section:

    # parallel computation for speed up
    jump_length = od_shape[
        [const.LAT, const.LNG, const.LAT_END, const.LNG_END]
    ].parallel_apply(m_utils.haversine_dist, axis=1)
    return m_utils.hist_section(
        jump_length,
        eps,
        mdreport.max_trips_per_user,
        min_value=0,
        max_value=mdreport.max_jump_length,
        bin_range=mdreport.bin_range_jump_length,
        evalu=mdreport.evalu,
    )

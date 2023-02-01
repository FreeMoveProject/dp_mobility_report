from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dp_mobility_report import DpMobilityReport

import numpy as np
import pandas as pd

from dp_mobility_report import constants as const
from dp_mobility_report.model import m_utils
from dp_mobility_report.model.section import DfSection, TupleSection
from dp_mobility_report.privacy import diff_privacy


def get_od_shape(df: pd.DataFrame) -> pd.DataFrame:
    if const.TILE_ID in df.columns:
        columns = [const.TID, const.TILE_ID, const.DATETIME, const.LAT, const.LNG]
    else:
        columns = [const.TID, const.DATETIME, const.LAT, const.LNG]

    ends_od_shape = df[(df[const.POINT_TYPE] == const.END)][columns]

    # change column name except for TID
    ends_od_shape.columns = [
        col_name + "_" + const.END if col_name != const.TID else col_name
        for col_name in ends_od_shape.columns
    ]

    od_shape = df[(df[const.POINT_TYPE] == const.START)][columns].merge(
        ends_od_shape, on=const.TID, how="inner"
    )

    return od_shape


def get_od_flows(
    od_shape: pd.DataFrame,
    dpmreport: "DpMobilityReport",
    eps: Optional[float],
) -> DfSection:
    sensitivity = dpmreport.count_sensitivity_base
    od_flows = (
        od_shape[od_shape[const.TILE_ID].notna() & od_shape[const.TILE_ID_END].notna()]
        .groupby([const.TILE_ID, const.TILE_ID_END])
        .aggregate(flow=(const.TID, "count"))
        .reset_index()
        .rename(
            columns={
                const.TILE_ID: const.ORIGIN,
                const.TILE_ID_END: const.DESTINATION,
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
        {const.ORIGIN: full_combinations[0], const.DESTINATION: full_combinations[1]}
    ).merge(od_flows, on=[const.ORIGIN, const.DESTINATION], how="left")
    od_flows.fillna(0, inplace=True)

    od_flows[const.FLOW] = diff_privacy.counts_dp(
        od_flows[const.FLOW].to_numpy(), eps, sensitivity, allow_negative=True
    )

    # remove all instances of 0 (and smaller) to reduce storage
    od_flows = od_flows[od_flows["flow"] > 0]

    cumsum = m_utils.cumsum(od_flows.flow.copy().to_numpy(), eps, sensitivity)

    # TODO: distribution with or without 0s?
    # as counts are already dp, no further privacy mechanism needed
    dp_quartiles = od_flows.flow.describe()

    return DfSection(
        data=od_flows,
        quartiles=dp_quartiles,
        privacy_budget=eps,
        margin_of_error_laplace=moe,
        sensitivity=sensitivity,
        cumsum=cumsum,
    )


def get_intra_tile_flows(od_flows: pd.DataFrame) -> int:
    return od_flows[(od_flows[const.ORIGIN] == od_flows[const.DESTINATION])].flow.sum()


def get_travel_time(
    od_shape: pd.DataFrame, dpmreport: "DpMobilityReport", eps: Optional[float]
) -> TupleSection:

    travel_time = od_shape[const.DATETIME_END] - od_shape[const.DATETIME]
    travel_time = travel_time.dt.seconds / 60  # as minutes

    return m_utils.hist_section(
        travel_time,
        eps,
        dpmreport.count_sensitivity_base,
        hist_max=dpmreport.max_travel_time,
        bin_range=dpmreport.bin_range_travel_time,
        bin_type=int,
        evalu=dpmreport.evalu,
    )


def get_jump_length(
    od_shape: pd.DataFrame, dpmreport: "DpMobilityReport", eps: Optional[float]
) -> TupleSection:

    # parallel computation for speed up
    jump_length = od_shape[
        [const.LAT, const.LNG, const.LAT_END, const.LNG_END]
    ].parallel_apply(m_utils.haversine_dist, axis=1)
    return m_utils.hist_section(
        jump_length,
        eps,
        dpmreport.count_sensitivity_base,
        hist_max=dpmreport.max_jump_length,
        bin_range=dpmreport.bin_range_jump_length,
        evalu=dpmreport.evalu,
    )

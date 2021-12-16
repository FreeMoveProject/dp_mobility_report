import numpy as np
import pandas as pd
from haversine import Unit, haversine
from pandarallel import pandarallel

from dp_mobility_report.model import utils
from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def get_od_shape(df, tessellation):

    ends_od_shape = (
        df[(df.point_type == "end") & df.tile_id.isin(tessellation.tile_id)][
            ["tid", "tile_id", "datetime", "lat", "lng"]
        ]
        .merge(tessellation[["tile_id"]], on="tile_id", how="left")
        .rename(
            columns={
                "tile_id": "tile_id_end",
                "lat": "lat_end",
                "lng": "lng_end",
                "datetime": "datetime_end",
            }
        )
    )

    od_shape = (
        df[(df.point_type == "start") & df.tile_id.isin(tessellation.tile_id)][
            ["tid", "tile_id", "datetime", "lat", "lng"]
        ]
        .merge(tessellation[["tile_id"]], on="tile_id", how="left")
        .merge(ends_od_shape, on="tid", how="inner")
    )

    return od_shape


def get_od_flows(od_shape, eps, mdreport):
    od_flows = (
        od_shape.reset_index()[
            od_shape.tile_id.isin(mdreport.tessellation.tile_id)
            & od_shape.tile_id_end.isin(mdreport.tessellation.tile_id)
        ]
        .groupby(["tile_id", "tile_id_end"])  # "tile_name", "tile_name_end",
        .aggregate(flow=("tid", "count"))
        .reset_index()
        .rename(
            columns={
                "tile_id": "origin",
                # "tile_name": "origin_name",
                # "tile_name_end": "destination_name",
                "tile_id_end": "destination",
            }
        )
        .sort_values("flow", ascending=False)
    )

    full_tile_ids = np.unique(mdreport.tessellation.tile_id)
    full_combinations = list(map(np.ravel, np.meshgrid(full_tile_ids, full_tile_ids)))
    od_flows = pd.DataFrame(
        dict(origin=full_combinations[0], destination=full_combinations[1])
    ).merge(od_flows, on=["origin", "destination"], how="left")
    od_flows.fillna(0, inplace=True)

    od_flows["flow"] = diff_privacy.counts_dp(
        od_flows["flow"], eps, mdreport.max_trips_per_user, parallel=True, nonzero=False
    )

    # remove all instances of 0 to reduce storage
    od_flows = od_flows[od_flows["flow"] > 0]

    # flows_sum = sum(full_od_flows.flow)
    # full_od_flows["flow_perc"] = round(full_od_flows.flow / flows_sum, 4) * 100

    return od_flows


def get_intra_tile_flows(od_flows):
    return od_flows[(od_flows.origin == od_flows.destination)].flow.sum()


def get_travel_time(
    od_shape, eps, max_trips_per_user, max_value, bin_size, evalu=False
):

    travel_time = od_shape.datetime_end - od_shape.datetime
    travel_time = (travel_time.dt.seconds / 60).round()  # as minutes
    return utils.dp_hist_section(
        travel_time,
        eps,
        max_trips_per_user,
        min_value=0,
        max_value=max_value,
        bin_size=bin_size,
        evalu=evalu,
    )


def get_jump_length(
    od_shape, eps, max_trips_per_user, max_value, bin_size, evalu=False
):

    # parallel computation to speed up
    jump_length = od_shape[["lat", "lng", "lat_end", "lng_end"]].parallel_apply(
        haversine_dist, axis=1
    )
    return utils.dp_hist_section(
        jump_length,
        eps,
        max_trips_per_user,
        min_value=0,
        max_value=max_value,
        bin_size=bin_size,
        evalu=evalu,
    )


def haversine_dist(coords):
    return haversine(
        (float(coords[0]), float(coords[1])),
        (float(coords[2]), float(coords[3])),
        unit=Unit.METERS,
    )

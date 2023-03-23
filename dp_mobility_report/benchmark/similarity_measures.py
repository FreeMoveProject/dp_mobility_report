import warnings
from datetime import timedelta
from typing import TYPE_CHECKING, Any, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from haversine import Unit, haversine
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, kendalltau, wasserstein_distance
from tqdm.auto import tqdm

from dp_mobility_report import constants as const

if TYPE_CHECKING:
    from dp_mobility_report import BenchmarkReport


# catch warning from scipy
def _entropy(pk: np.array, qk: np.array) -> float:
    if (np.sum(pk) == 0) | (np.sum(qk) == 0):
        return np.nan
    else:
        return entropy(pk, qk)


# catch warning from scipy
def _jensenshannon(p: np.array, q: np.array) -> float:
    if (np.sum(p) == 0) | (np.sum(q) == 0):
        return np.nan
    else:
        return jensenshannon(p, q)


def _moving_average(arr: np.array, size: int) -> np.array:
    return np.convolve(arr, np.ones(size), "valid") / size


def _replace_inf_with_max(
    bins: np.array, max_value: Union[int, float, timedelta]
) -> np.array:
    if np.isinf(bins[-1]):
        if isinstance(max_value, timedelta):  # needed for user_time_delta
            max_value = max_value.total_seconds() / 3600
        return np.append(bins[:-1], max_value)
    return bins


def earth_movers_distance1D(
    u_hist: tuple, v_hist: tuple, u_max: Union[int, float], v_max: Union[int, float]
) -> float:
    # in emd terms, hist_bins(x-axis) are values and hist_values(y-axis) are weights

    # bins greater than defined hist_max values are summarized as a "greater" bin, named "inf"
    # to compute the emd, the actual max value is needed
    u_hist_bins = _replace_inf_with_max(u_hist[1], u_max)
    v_hist_bins = _replace_inf_with_max(v_hist[1], v_max)

    # distinguish between single int bins and ranges
    if len(u_hist[0]) == len(u_hist_bins):
        u_values = u_hist_bins
    else:  # computes moving average if hist buckets within a range
        u_values = _moving_average(u_hist_bins, 2)

    # distinguish between single int bins and ranges
    if len(v_hist[0]) == len(v_hist_bins):
        v_values = v_hist_bins
    else:  # computes moving average if hist buckets within a range
        v_values = _moving_average(v_hist_bins, 2)

    u_weights = u_hist[0]
    v_weights = v_hist[0]
    if (sum(u_weights) == 0) | (sum(v_weights) == 0):
        return np.nan
    return wasserstein_distance(u_values, v_values, u_weights, v_weights)


# keep_direction: whether to use absolute difference or keep direction of difference
def symmetric_perc_error(
    alternative: Optional[Union[int, float]],
    base: Optional[Union[int, float]],
    keep_direction: bool = False,
) -> float:
    if (alternative is None) or (base is None):
        return None
    if abs(base) + abs(alternative) == 0:
        return 0

    diff_alt_base = alternative - base
    if not keep_direction:
        diff_alt_base = np.abs(diff_alt_base)

    return diff_alt_base / ((abs(base) + abs(alternative)) / 2)


def symmetric_mape(
    alternative: Union[pd.Series, np.array],
    base: Union[pd.Series, np.array],
) -> float:
    n = len(base)
    return (
        1 / n * np.sum([symmetric_perc_error(a, b) for a, b in zip(alternative, base)])
    )


def all_perc_errors(alt_dict: dict, base_dict: dict) -> dict:
    re = {}
    for key in base_dict:
        re[key] = symmetric_perc_error(alt_dict[key], base_dict[key])
    return re


# earth movers distance
def _compute_cost_matrix(tessellation: GeoDataFrame) -> np.array:
    tile_centroids = (
        tessellation.set_index(const.TILE_ID).to_crs(3395).centroid.to_crs(4326)
    )
    tile_coords = list(zip(tile_centroids.y, tile_centroids.x))

    # get all potential combinations between all points from sig1 and sig2
    grid = np.meshgrid(range(0, len(tile_coords)), range(0, len(tile_coords)))
    tile_combinations = np.array([grid[0].flatten(), grid[1].flatten()])

    # create an empty cost matrix with the length of all possible combinations
    cost_matrix = np.empty(
        tile_combinations.shape[1], dtype=np.float32
    )  # float32 needed as input for cv2.emd!

    # compute haversine distance for all possible combinations
    for column in range(0, tile_combinations.shape[1]):
        tile_1 = tile_combinations[0, column]
        tile_2 = tile_combinations[1, column]
        cost_matrix[column] = haversine(
            tile_coords[tile_1], tile_coords[tile_2], unit=Unit.METERS
        )

    # reshape array to matrix
    return np.reshape(cost_matrix, (len(tile_coords), len(tile_coords)))


def earth_movers_distance(
    arr_alt: np.array, arr_base: np.array, cost_matrix: np.array
) -> float:  # based on haversine distance

    # set values to 1 (as before) or stay with nan, as now? especially for time windows?
    if (all(arr_alt == 0)) | (all(arr_base == 0)):
        return np.nan
    arr_base = (arr_base / arr_base.sum() * 100).round(2)
    sig_true = arr_base.astype(np.float32)

    # normalize input and assign needed type for cv2
    arr_alt = (arr_alt / arr_alt.sum() * 100).round(2)
    sig_estimate = arr_alt.astype(np.float32)

    emd_dist, _, _ = cv2.EMD(
        sig_true, sig_estimate, distType=cv2.DIST_USER, cost=cost_matrix
    )
    return emd_dist


def top_n_coverage(top_n_tiles_base: list, top_n_tiles_alt: list) -> float:
    top_n_base = len(top_n_tiles_base)
    if top_n_base == 0:
        return 0
    top_n_alternative = len([x for x in top_n_tiles_alt if x in top_n_tiles_base])
    return top_n_alternative / top_n_base


def compute_similarity_measures(
    analysis_exclusion: list,
    report_alternative: dict,
    report_base: dict,
    tessellation: Optional[GeoDataFrame],
    top_n_ranking: List[int],
    disable_progress_bar: bool,
) -> Any:
    smape_dict: dict = {}
    kld_dict: dict = {}
    jsd_dict: dict = {}
    emd_dict: dict = {}
    kendall_dict: dict = {}
    top_n_coverage_dict: dict = {}
    cost_matrix = None

    disable_emd = (
        len(tessellation) > const.DISABLE_EMD_THRESHOLD
        if (tessellation is not None)
        else True
    )

    if disable_emd and (
        (const.VISITS_PER_TILE not in analysis_exclusion)
        or (const.VISITS_PER_TIME_TILE not in analysis_exclusion)
    ):
        warnings.warn(
            "EMD computation disables for spatial analysis due to too many tiles in tessellation (results in long computation time)."
        )

    with tqdm(  # progress bar
        total=15, desc="Compute similarity measures:", disable=disable_progress_bar
    ) as pbar:

        # Statistics
        if const.DS_STATISTICS not in analysis_exclusion:
            smape_dict = dict(
                **smape_dict,
                **all_perc_errors(
                    report_alternative[const.DS_STATISTICS].data,
                    report_base[const.DS_STATISTICS].data,
                ),
            )
        pbar.update()

        # Missing values
        if const.MISSING_VALUES not in analysis_exclusion:
            smape_dict = dict(
                **smape_dict,
                **all_perc_errors(
                    report_alternative[const.MISSING_VALUES].data,
                    report_base[const.MISSING_VALUES].data,
                ),
            )
        pbar.update()
        # Temporal distributions
        if const.TRIPS_OVER_TIME not in analysis_exclusion:
            trips_over_time = report_alternative[const.TRIPS_OVER_TIME].data.merge(
                report_base[const.TRIPS_OVER_TIME].data,
                how="outer",
                on="datetime",
                suffixes=("_alternative", "_base"),
            )

            trips_over_time.fillna(0, inplace=True)
            kld_dict[const.TRIPS_OVER_TIME] = _entropy(
                pk=trips_over_time.trips_alternative, qk=trips_over_time.trips_base
            )
            jsd_dict[const.TRIPS_OVER_TIME] = _jensenshannon(
                p=trips_over_time.trips_alternative, q=trips_over_time.trips_base
            )
            smape_dict[const.TRIPS_OVER_TIME] = symmetric_mape(
                alternative=trips_over_time.trips_alternative,
                base=trips_over_time.trip_count_base,
            )
        pbar.update()

        if const.TRIPS_PER_WEEKDAY not in analysis_exclusion:
            trips_per_weekday = pd.concat(
                [
                    report_alternative[const.TRIPS_PER_WEEKDAY].data,
                    report_base[const.TRIPS_PER_WEEKDAY].data,
                ],
                join="outer",
                axis=1,
            )
            trips_per_weekday.fillna(0, inplace=True)
            kld_dict[const.TRIPS_PER_WEEKDAY] = _entropy(
                pk=trips_per_weekday.iloc[:, 0], qk=trips_per_weekday.iloc[:, 1]
            )
            jsd_dict[const.TRIPS_PER_WEEKDAY] = _jensenshannon(
                p=trips_per_weekday.iloc[:, 0], q=trips_per_weekday.iloc[:, 1]
            )
            smape_dict[const.TRIPS_PER_WEEKDAY] = symmetric_mape(
                alternative=trips_per_weekday.iloc[:, 0],
                base=trips_per_weekday.iloc[:, 1],
            )
        pbar.update()

        if const.TRIPS_PER_HOUR not in analysis_exclusion:
            trips_per_hour = report_alternative[const.TRIPS_PER_HOUR].data.merge(
                report_base[const.TRIPS_PER_HOUR].data,
                how="outer",
                on=["hour", const.TIME_CATEGORY],
                suffixes=("_alternative", "_base"),
            )
            trips_per_hour.fillna(0, inplace=True)
            kld_dict[const.TRIPS_PER_HOUR] = _entropy(
                pk=trips_per_hour.perc_alternative, qk=trips_per_hour.perc_base
            )
            jsd_dict[const.TRIPS_PER_HOUR] = _jensenshannon(
                p=trips_per_hour.perc_alternative, q=trips_per_hour.perc_base
            )
            smape_dict[const.TRIPS_PER_HOUR] = symmetric_mape(
                alternative=trips_per_hour.perc_alternative,
                base=trips_per_hour.perc_base,
            )
        pbar.update()

        # Spatial distribution
        if const.VISITS_PER_TILE not in analysis_exclusion:
            tessellation = tessellation.sort_values(by=const.TILE_ID)
            visits_per_tile = report_alternative[const.VISITS_PER_TILE].data.merge(
                report_base[const.VISITS_PER_TILE].data,
                how="outer",
                on="tile_id",
                suffixes=("_alternative", "_base"),
                sort=True,  # sort according to tessellation for cost_matrix
            )
            visits_per_tile.fillna(0, inplace=True)

            rel_counts_alternative = (
                visits_per_tile.visits_alternative
                / visits_per_tile.visits_alternative.sum()
            )
            rel_counts_base = (
                visits_per_tile.visits_base / visits_per_tile.visits_base.sum()
            )
            kld_dict[const.VISITS_PER_TILE] = _entropy(
                pk=rel_counts_alternative, qk=rel_counts_base
            )
            jsd_dict[const.VISITS_PER_TILE] = _jensenshannon(
                p=rel_counts_alternative, q=rel_counts_base
            )
            smape_dict[const.VISITS_PER_TILE] = symmetric_mape(
                alternative=rel_counts_alternative, base=rel_counts_base
            )

            most_freq_base = report_base[const.VISITS_PER_TILE].data.sort_values(
                by=["visits"], ascending=False, ignore_index=True
            )
            most_freq_alternative = report_alternative[
                const.VISITS_PER_TILE
            ].data.sort_values(by=["visits"], ascending=False, ignore_index=True)

            # mute kendall_tau warning by replacing str tile_ids with ints
            tile_id_to_index = {
                tile_id: i
                for tile_id, i in zip(
                    tessellation[const.TILE_ID], range(0, len(tessellation))
                )
            }
            kendall_dict[const.VISITS_PER_TILE_RANKING] = []
            top_n_coverage_dict[const.VISITS_PER_TILE_RANKING] = []
            for top_n in top_n_ranking:
                kt_temp, _ = kendalltau(
                    [
                        tile_id_to_index[tile_id]
                        for tile_id in most_freq_base.truncate(after=top_n)[
                            const.TILE_ID
                        ]
                    ],
                    [
                        tile_id_to_index[tile_id]
                        for tile_id in most_freq_alternative.truncate(after=top_n)[
                            const.TILE_ID
                        ]
                    ],
                )
                kendall_dict[const.VISITS_PER_TILE_RANKING] += [kt_temp]
                top_n_coverage_dict[const.VISITS_PER_TILE_RANKING] += [
                    top_n_coverage(
                        list(most_freq_base.truncate(after=top_n)[const.TILE_ID]),
                        list(
                            most_freq_alternative.truncate(after=top_n)[const.TILE_ID]
                        ),
                    )
                ]

            if not disable_emd:
                # create custom cost matrix with distances between all tiles
                cost_matrix = _compute_cost_matrix(tessellation)

                emd_dict[const.VISITS_PER_TILE] = earth_movers_distance(
                    arr_alt=visits_per_tile.visits_alternative.to_numpy(),
                    arr_base=visits_per_tile.visits_base.to_numpy(),
                    cost_matrix=cost_matrix,
                )

            # Outliers
            smape_dict[const.VISITS_PER_TILE_OUTLIERS] = symmetric_perc_error(
                report_alternative[const.VISITS_PER_TILE].n_outliers,
                report_base[const.VISITS_PER_TILE].n_outliers,
            )

            # Quartiles
            smape_dict[const.VISITS_PER_TILE_QUARTILES] = symmetric_mape(
                alternative=report_alternative[const.VISITS_PER_TILE].quartiles,
                base=report_base[const.VISITS_PER_TILE].quartiles,
            )
        pbar.update()

        # Spatio-temporal distributions
        if const.VISITS_PER_TIME_TILE not in analysis_exclusion:

            counts_timew_alternative = (
                report_alternative[const.VISITS_PER_TIME_TILE]
                .data[
                    report_alternative[const.VISITS_PER_TIME_TILE].data.index != "None"
                ]
                .unstack()
            )
            counts_timew_base = (
                report_base[const.VISITS_PER_TIME_TILE]
                .data[report_base[const.VISITS_PER_TIME_TILE].data.index != "None"]
                .unstack()
            )

            indices = np.unique(
                np.append(
                    counts_timew_alternative.index.values,
                    counts_timew_base.index.values,
                )
            )
            counts_timew_alternative = counts_timew_alternative.reindex(index=indices)
            counts_timew_alternative.fillna(0, inplace=True)

            counts_timew_base = counts_timew_base.reindex(index=indices)
            counts_timew_base.fillna(0, inplace=True)

            rel_counts_timew_alternative = (
                counts_timew_alternative / counts_timew_alternative.sum()
            )
            rel_counts_timew_base = counts_timew_base / counts_timew_base.sum()

            kld_dict[const.VISITS_PER_TIME_TILE] = _entropy(
                pk=rel_counts_timew_alternative.to_numpy().flatten(),
                qk=rel_counts_timew_base.to_numpy().flatten(),
            )
            jsd_dict[const.VISITS_PER_TIME_TILE] = _jensenshannon(
                p=rel_counts_timew_alternative.to_numpy().flatten(),
                q=rel_counts_timew_base.to_numpy().flatten(),
            )
            smape_dict[const.VISITS_PER_TIME_TILE] = symmetric_mape(
                alternative=rel_counts_timew_alternative.to_numpy().flatten(),
                base=rel_counts_timew_base.to_numpy().flatten(),
            )
            if not disable_emd:
                if cost_matrix is None:
                    tessellation = tessellation.sort_values(by=const.TILE_ID)
                    cost_matrix = _compute_cost_matrix(tessellation)

                visits_per_time_tile_emd = []
                for time_window in report_base[const.VISITS_PER_TIME_TILE].data.columns:
                    tw_base = report_base[const.VISITS_PER_TIME_TILE].data[time_window]
                    tw_base = tw_base / tw_base.sum()
                    # if time window not in proposal report, add time windows with count zero
                    if (
                        time_window
                        not in report_alternative[
                            const.VISITS_PER_TIME_TILE
                        ].data.columns
                    ):
                        tw_alternative = tw_base.copy()
                        tw_alternative[:] = 0
                    else:
                        tw_alternative = report_alternative[
                            const.VISITS_PER_TIME_TILE
                        ].data[time_window]
                        tw_alternative = tw_alternative / tw_alternative.sum()
                    tw = pd.merge(
                        tw_alternative,
                        tw_base,
                        how="outer",
                        right_index=True,
                        left_index=True,
                        suffixes=("_alternative", "_base"),
                        sort=True,  # sort according to tile_id for cost_matrix
                    )

                    tw = tw[
                        tw.notna().sum(axis=1) > 0
                    ]  # remove instances where both are NaN
                    tw.fillna(0, inplace=True)
                    visits_per_time_tile_emd.append(
                        earth_movers_distance(
                            arr_alt=tw.iloc[:, 0].to_numpy(),
                            arr_base=tw.iloc[:, 1].to_numpy(),
                            cost_matrix=cost_matrix,
                        )
                    )
                emd_dict[const.VISITS_PER_TIME_TILE] = np.mean(visits_per_time_tile_emd)

        pbar.update()

        # Origin-Destination
        if const.OD_FLOWS not in analysis_exclusion:
            all_od_combinations = pd.concat(
                [
                    report_alternative[const.OD_FLOWS].data[["origin", "destination"]],
                    report_base[const.OD_FLOWS].data[["origin", "destination"]],
                ]
            ).drop_duplicates()
            all_od_combinations["flow"] = 0

            od_flows_alternative = (
                pd.concat(
                    [report_alternative[const.OD_FLOWS].data, all_od_combinations]
                )
                .drop_duplicates(["origin", "destination"], keep="first")
                .sort_values(["origin", "destination"])
                .flow
            )
            od_flows_base = (
                pd.concat([report_base[const.OD_FLOWS].data, all_od_combinations])
                .drop_duplicates(["origin", "destination"], keep="first")
                .sort_values(["origin", "destination"])
                .flow
            )

            rel_alternative = od_flows_alternative / (od_flows_alternative.sum())
            rel_base = od_flows_base / od_flows_base.sum()

            kld_dict[const.OD_FLOWS] = _entropy(
                pk=rel_alternative.to_numpy(), qk=rel_base.to_numpy()
            )
            jsd_dict[const.OD_FLOWS] = _jensenshannon(
                p=rel_alternative.to_numpy(), q=rel_base.to_numpy()
            )
            smape_dict[const.OD_FLOWS] = symmetric_mape(
                alternative=rel_alternative.to_numpy(), base=rel_base.to_numpy()
            )

            kendall_dict[const.OD_FLOWS_RANKING] = []
            top_n_coverage_dict[const.OD_FLOWS_RANKING] = []
            for top_n in top_n_ranking:
                kt_temp, _ = kendalltau(
                    list(rel_alternative.sort_values(ascending=False).index.values)[
                        :top_n
                    ],
                    list(rel_base.sort_values(ascending=False).index.values)[:top_n],
                )
                kendall_dict[const.OD_FLOWS_RANKING] += [kt_temp]
                top_n_coverage_dict[const.OD_FLOWS_RANKING] += [
                    top_n_coverage(
                        list(
                            od_flows_base.sort_values(ascending=False)
                            .iloc[0:top_n]
                            .index
                        ),
                        list(
                            od_flows_alternative.sort_values(ascending=False)
                            .iloc[0:top_n]
                            .index
                        ),
                    )
                ]

            # Quartiles
            smape_dict[const.OD_FLOWS_QUARTILES] = symmetric_mape(
                alternative=report_alternative[const.OD_FLOWS].quartiles,
                base=report_base[const.OD_FLOWS].quartiles,
            )
        pbar.update()

        if const.TRAVEL_TIME not in analysis_exclusion:
            kld_dict[const.TRAVEL_TIME] = _entropy(
                pk=report_alternative[const.TRAVEL_TIME].data[0],
                qk=report_base[const.TRAVEL_TIME].data[0],
            )
            jsd_dict[const.TRAVEL_TIME] = _jensenshannon(
                p=report_alternative[const.TRAVEL_TIME].data[0],
                q=report_base[const.TRAVEL_TIME].data[0],
            )
            smape_dict[const.TRAVEL_TIME] = symmetric_mape(
                alternative=report_alternative[const.TRAVEL_TIME].data[0],
                base=report_base[const.TRAVEL_TIME].data[0],
            )
            emd_dict[const.TRAVEL_TIME] = earth_movers_distance1D(
                report_alternative[const.TRAVEL_TIME].data,
                report_base[const.TRAVEL_TIME].data,
                report_alternative[const.TRAVEL_TIME].quartiles["max"],
                report_base[const.TRAVEL_TIME].quartiles["max"],
            )
            # Quartiles
            smape_dict[const.TRAVEL_TIME_QUARTILES] = symmetric_mape(
                alternative=report_alternative[const.TRAVEL_TIME].quartiles,
                base=report_base[const.TRAVEL_TIME].quartiles,
            )
        pbar.update()

        if const.JUMP_LENGTH not in analysis_exclusion:
            kld_dict[const.JUMP_LENGTH] = _entropy(
                pk=report_alternative[const.JUMP_LENGTH].data[0],
                qk=report_base[const.JUMP_LENGTH].data[0],
            )
            jsd_dict[const.JUMP_LENGTH] = _jensenshannon(
                p=report_alternative[const.JUMP_LENGTH].data[0],
                q=report_base[const.JUMP_LENGTH].data[0],
            )
            smape_dict[const.JUMP_LENGTH] = symmetric_mape(
                alternative=report_alternative[const.JUMP_LENGTH].data[0],
                base=report_base[const.JUMP_LENGTH].data[0],
            )
            emd_dict[const.JUMP_LENGTH] = earth_movers_distance1D(
                report_alternative[const.JUMP_LENGTH].data,
                report_base[const.JUMP_LENGTH].data,
                report_alternative[const.JUMP_LENGTH].quartiles["max"],
                report_base[const.JUMP_LENGTH].quartiles["max"],
            )
            # Quartiles
            smape_dict[const.JUMP_LENGTH_QUARTILES] = symmetric_mape(
                alternative=report_alternative[const.JUMP_LENGTH].quartiles,
                base=report_base[const.JUMP_LENGTH].quartiles,
            )
        pbar.update()

        # User
        # TODO bin sizes do not align for kld, jsd, smape
        if const.TRIPS_PER_USER not in analysis_exclusion:

            emd_dict[const.TRIPS_PER_USER] = earth_movers_distance1D(
                report_alternative[const.TRIPS_PER_USER].data,
                report_base[const.TRIPS_PER_USER].data,
                report_alternative[const.TRIPS_PER_USER].quartiles["max"],
                report_base[const.TRIPS_PER_USER].quartiles["max"],
            )

            # Quartiles
            smape_dict[const.TRIPS_PER_USER_QUARTILES] = symmetric_mape(
                alternative=report_alternative[const.TRIPS_PER_USER].quartiles,
                base=report_base[const.TRIPS_PER_USER].quartiles,
            )
        pbar.update()

        if const.USER_TIME_DELTA not in analysis_exclusion:
            if (
                report_alternative[const.USER_TIME_DELTA] is None
            ):  # if each user only has one trip then `USER_TIME_DELTA` is None
                smape_dict[const.USER_TIME_DELTA_QUARTILES] = None
            else:
                kld_dict[const.USER_TIME_DELTA] = _entropy(
                    pk=report_alternative[const.USER_TIME_DELTA].data[0],
                    qk=report_base[const.USER_TIME_DELTA].data[0],
                )
                jsd_dict[const.USER_TIME_DELTA] = _jensenshannon(
                    p=report_alternative[const.USER_TIME_DELTA].data[0],
                    q=report_base[const.USER_TIME_DELTA].data[0],
                )
                smape_dict[const.USER_TIME_DELTA] = symmetric_mape(
                    alternative=report_alternative[const.USER_TIME_DELTA].data[0],
                    base=report_base[const.USER_TIME_DELTA].data[0],
                )
                emd_dict[const.USER_TIME_DELTA] = earth_movers_distance1D(
                    report_alternative[const.USER_TIME_DELTA].data,
                    report_base[const.USER_TIME_DELTA].data,
                    report_alternative[const.USER_TIME_DELTA].quartiles["max"],
                    report_base[const.USER_TIME_DELTA].quartiles["max"],
                )
                smape_dict[const.USER_TIME_DELTA_QUARTILES] = symmetric_mape(
                    alternative=report_alternative[
                        const.USER_TIME_DELTA
                    ].quartiles.apply(lambda x: x.total_seconds() / 3600),
                    base=report_base[const.USER_TIME_DELTA].quartiles.apply(
                        lambda x: x.total_seconds() / 3600
                    ),
                )
        pbar.update()

        if const.RADIUS_OF_GYRATION not in analysis_exclusion:

            kld_dict[const.RADIUS_OF_GYRATION] = _entropy(
                pk=report_alternative[const.RADIUS_OF_GYRATION].data[0],
                qk=report_base[const.RADIUS_OF_GYRATION].data[0],
            )
            jsd_dict[const.RADIUS_OF_GYRATION] = _jensenshannon(
                p=report_alternative[const.RADIUS_OF_GYRATION].data[0],
                q=report_base[const.RADIUS_OF_GYRATION].data[0],
            )
            emd_dict[const.RADIUS_OF_GYRATION] = earth_movers_distance1D(
                report_alternative[const.RADIUS_OF_GYRATION].data,
                report_base[const.RADIUS_OF_GYRATION].data,
                report_alternative[const.RADIUS_OF_GYRATION].quartiles["max"],
                report_base[const.RADIUS_OF_GYRATION].quartiles["max"],
            )
            smape_dict[const.RADIUS_OF_GYRATION] = symmetric_mape(
                alternative=report_alternative[const.RADIUS_OF_GYRATION].data[0],
                base=report_base[const.RADIUS_OF_GYRATION].data[0],
            )
            # Quartiles
            smape_dict[const.RADIUS_OF_GYRATION_QUARTILES] = symmetric_mape(
                alternative=report_alternative[const.RADIUS_OF_GYRATION].quartiles,
                base=report_base[const.RADIUS_OF_GYRATION].quartiles,
            )
        pbar.update()

        if const.USER_TILE_COUNT not in analysis_exclusion:
            kld_dict[const.USER_TILE_COUNT] = _entropy(
                pk=report_alternative[const.USER_TILE_COUNT].data[0],
                qk=report_base[const.USER_TILE_COUNT].data[0],
            )
            jsd_dict[const.USER_TILE_COUNT] = _jensenshannon(
                p=report_alternative[const.USER_TILE_COUNT].data[0],
                q=report_base[const.USER_TILE_COUNT].data[0],
            )
            emd_dict[const.USER_TILE_COUNT] = earth_movers_distance1D(
                report_alternative[const.USER_TILE_COUNT].data,
                report_base[const.USER_TILE_COUNT].data,
                report_alternative[const.USER_TILE_COUNT].quartiles["max"],
                report_base[const.USER_TILE_COUNT].quartiles["max"],
            )
            smape_dict[const.USER_TILE_COUNT] = symmetric_mape(
                alternative=report_alternative[const.USER_TILE_COUNT].data[0],
                base=report_base[const.USER_TILE_COUNT].data[0],
            )

            smape_dict[const.USER_TILE_COUNT_QUARTILES] = symmetric_mape(
                alternative=report_alternative[const.USER_TILE_COUNT].quartiles,
                base=report_base[const.USER_TILE_COUNT].quartiles,
            )
        pbar.update()

        if const.MOBILITY_ENTROPY not in analysis_exclusion:
            kld_dict[const.MOBILITY_ENTROPY] = _entropy(
                pk=report_alternative[const.MOBILITY_ENTROPY].data[0],
                qk=report_base[const.MOBILITY_ENTROPY].data[0],
            )
            jsd_dict[const.MOBILITY_ENTROPY] = _jensenshannon(
                p=report_alternative[const.MOBILITY_ENTROPY].data[0],
                q=report_base[const.MOBILITY_ENTROPY].data[0],
            )
            emd_dict[const.MOBILITY_ENTROPY] = earth_movers_distance1D(
                report_alternative[const.MOBILITY_ENTROPY].data,
                report_base[const.MOBILITY_ENTROPY].data,
                report_alternative[const.MOBILITY_ENTROPY].quartiles["max"],
                report_base[const.MOBILITY_ENTROPY].quartiles["max"],
            )
            smape_dict[const.MOBILITY_ENTROPY] = symmetric_mape(
                alternative=report_alternative[const.MOBILITY_ENTROPY].data[0],
                base=report_base[const.MOBILITY_ENTROPY].data[0],
            )
            # Quartiles
            smape_dict[const.MOBILITY_ENTROPY_QUARTILES] = symmetric_mape(
                alternative=report_alternative[const.MOBILITY_ENTROPY].quartiles,
                base=report_base[const.MOBILITY_ENTROPY].quartiles,
            )
        pbar.update()

    return smape_dict, kld_dict, jsd_dict, emd_dict, kendall_dict, top_n_coverage_dict


def get_selected_measures(benchmarkreport: "BenchmarkReport") -> dict:
    similarity_measures = {}

    for analysis in benchmarkreport.measure_selection.keys():
        selected_measure = benchmarkreport.measure_selection[analysis]
        try:
            if selected_measure == const.SMAPE:
                if analysis == const.DS_STATISTICS:
                    for element in const.DS_STATISTICS_ELEMENTS:
                        similarity_measures[element] = benchmarkreport.smape[element]
                elif analysis == const.MISSING_VALUES:
                    for element in const.MISSING_VALUES_ELEMENTS:
                        similarity_measures[element] = benchmarkreport.smape[element]
                else:
                    similarity_measures[analysis] = benchmarkreport.smape[analysis]
            elif selected_measure == const.KLD:
                similarity_measures[analysis] = benchmarkreport.kld[analysis]
            elif selected_measure == const.JSD:
                similarity_measures[analysis] = benchmarkreport.jsd[analysis]
            elif selected_measure == const.EMD:
                similarity_measures[analysis] = benchmarkreport.emd[analysis]
            elif selected_measure == const.KT:
                similarity_measures[analysis] = benchmarkreport.kt[analysis]
        except KeyError:
            warnings.warn(
                f"The selected measure {selected_measure} for {analysis} cannot be computed. Value for {analysis} in `self.similarity_measures` will be set to `None`."
            )
            similarity_measures[analysis] = None

    return similarity_measures

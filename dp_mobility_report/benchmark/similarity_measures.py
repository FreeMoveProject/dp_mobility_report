import warnings
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np
import pandas as pd
from haversine import Unit, haversine
from scipy.spatial import distance
from scipy.stats import entropy, wasserstein_distance

from dp_mobility_report import constants as const

if TYPE_CHECKING:
    from dp_mobility_report import BenchmarkReport


def _moving_average(arr: np.array, size: int) -> np.array:
    return np.convolve(arr, np.ones(size), "valid") / size


def earth_movers_distance1D(hist1: tuple, hist2: tuple) -> float:
    if len(hist1[0]) == len(
        hist1[1]
    ):  # checks for histogram buckets with exact sizes or ranges
        u_values = hist1[1]
        v_values = hist2[1]
    else:  # computes moving average if histogram buckets are in a range
        u_values = _moving_average(hist1[1], 2)
        v_values = _moving_average(hist2[1], 2)
    u_weights = hist1[0]
    v_weights = hist2[0]
    if (sum(u_weights) == 0) | (sum(v_weights) == 0):
        return None
    return wasserstein_distance(u_values, v_values, u_weights, v_weights)


# TODO: `n_true_positive_zeros` not needed anymore
def symmetric_mape(
    true: Union[pd.Series, np.array],
    estimate: Union[pd.Series, np.array],
    n_true_positive_zeros: int = None,
) -> float:
    n = (
        len(true)
        if n_true_positive_zeros is None
        else (len(true) + n_true_positive_zeros)
    )
    return (
        1
        / n
        * np.sum(
            np.where(
                abs(true + estimate) == 0,
                0,  # return 0 if true and estimate are both 0
                np.divide(
                    abs(estimate - true),
                    ((abs(true) + abs(estimate)) / 2),
                    where=(abs(true + estimate) != 0),
                ),
            )
        )
    )


def relative_error(true: int, estimate: int) -> int:
    if estimate is None:
        estimate = 0
    if true == 0:
        return np.abs(true - estimate)
    return np.abs(true - estimate) / true


def all_relative_errors(true_dict: dict, estimate_dict: dict) -> dict:
    re = {}
    for key in true_dict:
        re[key] = relative_error(true_dict[key], estimate_dict[key])
    return re


# earth movers distance
def _compute_cost_matrix(tile_coords: list) -> np.array:
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
    arr_true: np.array, arr_estimate: np.array, cost_matrix: np.array
) -> float:  # based on haversine distance
    # normalize input and assign needed type for cv2
    if all(
        arr_true == 0
    ):  # if all values are 0, change all to 1 (otherwise emd cannot be computed)
        arr_true = np.repeat(1, len(arr_true))
    arr_true = (arr_true / arr_true.sum() * 100).round(2)
    sig_true = arr_true.astype(np.float32)

    if all(
        arr_estimate == 0
    ):  # if all values are 0, change all to 1 (otherwise emd cannot be computed)
        arr_estimate = np.repeat(1, len(arr_estimate))
    arr_estimate = (arr_estimate / arr_estimate.sum() * 100).round(2)
    sig_estimate = arr_estimate.astype(np.float32)

    emd_dist, _, _ = cv2.EMD(
        sig_true, sig_estimate, distType=cv2.DIST_USER, cost=cost_matrix
    )
    return emd_dist


def compute_similarity_measures(
    analysis_exclusion: list,
    report_alternative: dict,
    report_base: dict,
    tessellation: pd.DataFrame,
):

    relative_error_dict: dict = {}
    kld_dict: dict = {}
    jsd_dict: dict = {}
    emd_dict: dict = {}
    smape_dict: dict = {}

    # Statistics
    if const.DS_STATISTICS not in analysis_exclusion:
        relative_error_dict = dict(
            **relative_error_dict,
            **all_relative_errors(
                report_base[const.DS_STATISTICS].data,
                report_alternative[const.DS_STATISTICS].data,
            ),
        )
    # Missing values
    if const.MISSING_VALUES not in analysis_exclusion:
        relative_error_dict = dict(
            **relative_error_dict,
            **all_relative_errors(
                report_base[const.MISSING_VALUES].data,
                report_alternative[const.MISSING_VALUES].data,
            ),
        )

    # Temporal distributions
    if const.TRIPS_OVER_TIME not in analysis_exclusion:
        trips_over_time = report_base[const.TRIPS_OVER_TIME].data.merge(
            report_alternative[const.TRIPS_OVER_TIME].data,
            how="outer",
            on="datetime",
            suffixes=("_base", "_alternative"),
        )

        trips_over_time.fillna(0, inplace=True)
        kld_dict[const.TRIPS_OVER_TIME] = entropy(
            trips_over_time.trips_base, trips_over_time.trips_alternative
        )
        jsd_dict[const.TRIPS_OVER_TIME] = distance.jensenshannon(
            trips_over_time.trips_base, trips_over_time.trips_alternative
        )
        smape_dict[const.TRIPS_OVER_TIME] = symmetric_mape(
            trips_over_time.trip_count_base, trips_over_time.trips_alternative
        )

    if const.TRIPS_PER_WEEKDAY not in analysis_exclusion:
        trips_per_weekday = pd.concat(
            [
                report_base[const.TRIPS_PER_WEEKDAY].data,
                report_alternative[const.TRIPS_PER_WEEKDAY].data,
            ],
            join="outer",
            axis=1,
        )
        trips_per_weekday.fillna(0, inplace=True)
        kld_dict[const.TRIPS_PER_WEEKDAY] = entropy(
            trips_per_weekday.iloc[:, 0], trips_per_weekday.iloc[:, 1]
        )
        jsd_dict[const.TRIPS_PER_WEEKDAY] = distance.jensenshannon(
            trips_per_weekday.iloc[:, 0], trips_per_weekday.iloc[:, 1]
        )
        smape_dict[const.TRIPS_PER_WEEKDAY] = symmetric_mape(
            trips_per_weekday.iloc[:, 0], trips_per_weekday.iloc[:, 1]
        )

    if const.TRIPS_PER_HOUR not in analysis_exclusion:
        trips_per_hour = report_base[const.TRIPS_PER_HOUR].data.merge(
            report_alternative[const.TRIPS_PER_HOUR].data,
            how="outer",
            on=["hour", "time_category"],
            suffixes=("_base", "_alternative"),
        )
        trips_per_hour.fillna(0, inplace=True)
        kld_dict[const.TRIPS_PER_HOUR] = entropy(
            trips_per_hour.perc_base, trips_per_hour.perc_alternative
        )
        jsd_dict[const.TRIPS_PER_HOUR] = distance.jensenshannon(
            trips_per_hour.perc_base, trips_per_hour.perc_alternative
        )
        smape_dict[const.TRIPS_PER_HOUR] = symmetric_mape(
            trips_per_hour.perc_base,
            trips_per_hour.perc_alternative,  # TODO im eval package change to perc
        )

    # Spatial distribution
    if const.VISITS_PER_TILE not in analysis_exclusion:
        visits_per_tile = report_base[const.VISITS_PER_TILE].data.merge(
            report_alternative[const.VISITS_PER_TILE].data,
            how="outer",
            on="tile_id",
            suffixes=("_base", "_alternative"),
        )
        visits_per_tile.fillna(0, inplace=True)

        rel_counts_base = (
            visits_per_tile.visits_base / visits_per_tile.visits_base.sum()
        )
        rel_counts_alternative = (
            visits_per_tile.visits_alternative
            / visits_per_tile.visits_alternative.sum()
        )
        kld_dict[const.VISITS_PER_TILE] = entropy(
            rel_counts_base, rel_counts_alternative
        )
        jsd_dict[const.VISITS_PER_TILE] = distance.jensenshannon(
            rel_counts_base, rel_counts_alternative
        )
        smape_dict[const.VISITS_PER_TILE] = symmetric_mape(
            rel_counts_base, rel_counts_alternative
        )

        tile_centroids = (
            tessellation.set_index("tile_id").to_crs(3395).centroid.to_crs(4326)
        )

        sorted_tile_centroids = tile_centroids.loc[visits_per_tile.tile_id]
        tile_coords = list(zip(sorted_tile_centroids.y, sorted_tile_centroids.x))
        # create custom cost matrix with distances between all tiles
        cost_matrix = _compute_cost_matrix(tile_coords)

        emd_dict[const.VISITS_PER_TILE] = earth_movers_distance(
            visits_per_tile.visits_base.to_numpy(),
            visits_per_tile.visits_alternative.to_numpy(),
            cost_matrix,
        )
        # Outliers
        relative_error_dict[const.VISITS_PER_TILE_OUTLIERS] = relative_error(
            report_base[const.VISITS_PER_TILE].n_outliers,
            report_alternative[const.VISITS_PER_TILE].n_outliers,
        )

    # Spatio-temporal distributions
    if const.VISITS_PER_TILE_TIMEWINDOW not in analysis_exclusion:
        counts_timew_base = (
            report_base[const.VISITS_PER_TILE_TIMEWINDOW]
            .data[report_base[const.VISITS_PER_TILE_TIMEWINDOW].data.index != "None"]
            .unstack()
        )
        counts_timew_alternative = (
            report_alternative[const.VISITS_PER_TILE_TIMEWINDOW]
            .data[
                report_alternative[const.VISITS_PER_TILE_TIMEWINDOW].data.index
                != "None"
            ]
            .unstack()
        )

        indices = np.unique(
            np.append(
                counts_timew_base.index.values, counts_timew_alternative.index.values
            )
        )

        counts_timew_base = counts_timew_base.reindex(index=indices)
        counts_timew_base.fillna(0, inplace=True)

        counts_timew_alternative = counts_timew_alternative.reindex(index=indices)
        counts_timew_alternative.fillna(0, inplace=True)

        rel_counts_timew_base = counts_timew_base / counts_timew_base.sum()
        rel_counts_timew_alternative = (
            counts_timew_alternative / counts_timew_alternative.sum()
        )

        kld_dict[const.VISITS_PER_TILE_TIMEWINDOW] = entropy(
            rel_counts_timew_base.to_numpy().flatten(),
            rel_counts_timew_alternative.to_numpy().flatten(),
        )
        jsd_dict[const.VISITS_PER_TILE_TIMEWINDOW] = distance.jensenshannon(
            rel_counts_timew_base.to_numpy().flatten(),
            rel_counts_timew_alternative.to_numpy().flatten(),
        )
        smape_dict[const.VISITS_PER_TILE_TIMEWINDOW] = symmetric_mape(
            rel_counts_timew_base.to_numpy().flatten(),
            rel_counts_timew_alternative.to_numpy().flatten(),
        )

        visits_per_tile_timewindow_emd = []
        # TODO visits_per_tile_timewindow should not be based on visits_per_tile and the cost_matrix from above if visits_per_tile is excluded
        for time_window in report_base[const.VISITS_PER_TILE_TIMEWINDOW].data.columns:
            tw_base = (
                report_base[const.VISITS_PER_TILE_TIMEWINDOW]
                .data[time_window]
                .loc[report_base[const.VISITS_PER_TILE].data.tile_id]
            )  # sort with `report_base[const.VISITS_PER_TILE]` to match order of cost_matrix
            tw_base = tw_base / tw_base.sum()
            # if time window not in proposal report, add time windows with count zero
            if (
                time_window
                not in report_alternative[const.VISITS_PER_TILE_TIMEWINDOW].data.columns
            ):
                tw_alternative = tw_base.copy()
                tw_alternative[:] = 0
            else:
                tw_alternative = (
                    report_alternative[const.VISITS_PER_TILE_TIMEWINDOW]
                    .data[time_window]
                    .loc[
                        report_base[
                            const.VISITS_PER_TILE
                        ].data.tile_id  # sort with `report_base[const.VISITS_PER_TILE]` to match order of cost_matrix
                    ]
                )
                tw_alternative = tw_alternative / tw_alternative.sum()
            tw = pd.merge(
                tw_base,
                tw_alternative,
                how="outer",
                right_index=True,
                left_index=True,
                suffixes=("_base", "_alternative"),
            )
            tw = tw[tw.notna().sum(axis=1) > 0]  # remove instances where both are NaN
            tw.fillna(0, inplace=True)
            visits_per_tile_timewindow_emd.append(
                earth_movers_distance(
                    tw.iloc[:, 0].to_numpy(), tw.iloc[:, 1].to_numpy(), cost_matrix
                )
            )
        emd_dict[const.VISITS_PER_TILE_TIMEWINDOW] = np.mean(
            visits_per_tile_timewindow_emd
        )

    # Origin-Destination
    if const.OD_FLOWS not in analysis_exclusion:
        all_od_combinations = pd.concat(
            [
                report_base[const.OD_FLOWS].data[["origin", "destination"]],
                report_alternative[const.OD_FLOWS].data[["origin", "destination"]],
            ]
        ).drop_duplicates()
        all_od_combinations["flow"] = 0

        true = (
            pd.concat([report_base[const.OD_FLOWS].data, all_od_combinations])
            .drop_duplicates(["origin", "destination"], keep="first")
            .sort_values(["origin", "destination"])
            .flow
        )
        estimate = (
            pd.concat([report_alternative[const.OD_FLOWS].data, all_od_combinations])
            .drop_duplicates(["origin", "destination"], keep="first")
            .sort_values(["origin", "destination"])
            .flow
        )

        rel_base = true / true.sum()
        rel_alternative = estimate / (estimate.sum())

        kld_dict[const.OD_FLOWS] = entropy(
            rel_base.to_numpy(), rel_alternative.to_numpy()
        )
        jsd_dict[const.OD_FLOWS] = distance.jensenshannon(
            rel_base.to_numpy(), rel_alternative.to_numpy()
        )
        smape_dict[const.OD_FLOWS] = symmetric_mape(
            rel_base.to_numpy(), rel_alternative.to_numpy()
        )

    if const.TRAVEL_TIME not in analysis_exclusion:
        kld_dict[const.TRAVEL_TIME] = entropy(
            report_base[const.TRAVEL_TIME].data[0],
            report_alternative[const.TRAVEL_TIME].data[0],
        )
        jsd_dict[const.TRAVEL_TIME] = distance.jensenshannon(
            report_base[const.TRAVEL_TIME].data[0],
            report_alternative[const.TRAVEL_TIME].data[0],
        )
        smape_dict[const.TRAVEL_TIME] = symmetric_mape(
            report_base[const.TRAVEL_TIME].data[0],
            report_alternative[const.TRAVEL_TIME].data[0],
        )
        emd_dict[const.TRAVEL_TIME] = earth_movers_distance1D(
            report_base[const.TRAVEL_TIME].data,
            report_alternative[const.TRAVEL_TIME].data,
        )
        # Quartiles
        smape_dict[const.TRAVEL_TIME_QUARTILES] = symmetric_mape(
            report_base[const.TRAVEL_TIME].quartiles,
            report_alternative[const.TRAVEL_TIME].quartiles,
        )

    if const.JUMP_LENGTH not in analysis_exclusion:
        kld_dict[const.JUMP_LENGTH] = entropy(
            report_base[const.JUMP_LENGTH].data[0],
            report_alternative[const.JUMP_LENGTH].data[0],
        )
        jsd_dict[const.JUMP_LENGTH] = distance.jensenshannon(
            report_base[const.JUMP_LENGTH].data[0],
            report_alternative[const.JUMP_LENGTH].data[0],
        )
        smape_dict[const.JUMP_LENGTH] = symmetric_mape(
            report_base[const.JUMP_LENGTH].data[0],
            report_alternative[const.JUMP_LENGTH].data[0],
        )
        emd_dict[const.JUMP_LENGTH] = earth_movers_distance1D(
            report_base[const.JUMP_LENGTH].data,
            report_alternative[const.JUMP_LENGTH].data,
        )
        # Quartiles
        smape_dict[const.JUMP_LENGTH_QUARTILES] = symmetric_mape(
            report_base[const.JUMP_LENGTH].quartiles,
            report_alternative[const.JUMP_LENGTH].quartiles,
        )

    # User
    # TODO bin sizes do not align
    if const.TRIPS_PER_USER not in analysis_exclusion:

        # kld_dict[const.TRIPS_PER_USER] = entropy(
        #     report_base[const.TRIPS_PER_USER].data[0],
        #     report_alternative[const.TRIPS_PER_USER].data[0],
        # )
        # jsd_dict[const.TRIPS_PER_USER] = distance.jensenshannon(
        #     report_base[const.TRIPS_PER_USER].data[0],
        #     report_alternative[const.TRIPS_PER_USER].data[0],
        # )
        # TODO discussion: potentially different results depending on histogram bin sizes
        emd_dict[const.TRIPS_PER_USER] = earth_movers_distance1D(
            report_base[const.TRIPS_PER_USER].data,
            report_alternative[const.TRIPS_PER_USER].data,
        )
        # smape_dict[const.TRIPS_PER_USER] = symmetric_mape(
        #     report_base[const.TRIPS_PER_USER].data[0],
        #     report_alternative[const.TRIPS_PER_USER].data[0],
        # )
        # Quartiles
        smape_dict[const.TRIPS_PER_USER_QUARTILES] = symmetric_mape(
            report_base[const.TRIPS_PER_USER].quartiles,
            report_alternative[const.TRIPS_PER_USER].quartiles,
        )
    if const.USER_TIME_DELTA not in analysis_exclusion:
        if (
            report_alternative[const.USER_TIME_DELTA] is None
        ):  # if each user only has one trip then `USER_TIME_DELTA` is None
            smape_dict[const.USER_TIME_DELTA_QUARTILES] = None
        else:
            # TODO comparison of histogram buckets of user time delta, currently not possible, missing jsd, kld
            smape_dict[const.USER_TIME_DELTA_QUARTILES] = symmetric_mape(
                (
                    report_base[const.USER_TIME_DELTA].quartiles.apply(
                        lambda x: x.total_seconds() / 3600
                    )
                ),
                report_alternative[const.USER_TIME_DELTA].quartiles.apply(
                    lambda x: x.total_seconds() / 3600
                ),
            )

    if const.RADIUS_OF_GYRATION not in analysis_exclusion:

        kld_dict[const.RADIUS_OF_GYRATION] = entropy(
            report_base[const.RADIUS_OF_GYRATION].data[0],
            report_alternative[const.RADIUS_OF_GYRATION].data[0],
        )
        jsd_dict[const.RADIUS_OF_GYRATION] = distance.jensenshannon(
            report_base[const.RADIUS_OF_GYRATION].data[0],
            report_alternative[const.RADIUS_OF_GYRATION].data[0],
        )
        emd_dict[const.RADIUS_OF_GYRATION] = earth_movers_distance1D(
            report_base[const.RADIUS_OF_GYRATION].data,
            report_alternative[const.RADIUS_OF_GYRATION].data,
        )
        smape_dict[const.RADIUS_OF_GYRATION] = symmetric_mape(
            report_base[const.RADIUS_OF_GYRATION].data[0],
            report_alternative[const.RADIUS_OF_GYRATION].data[0],
        )
        # Quartiles
        smape_dict[const.RADIUS_OF_GYRATION_QUARTILES] = symmetric_mape(
            report_base[const.RADIUS_OF_GYRATION].quartiles,
            report_alternative[const.RADIUS_OF_GYRATION].quartiles,
        )

    if const.USER_TILE_COUNT_QUARTILES not in analysis_exclusion:
        # TODO comparison of histogram buckets of user time delta, currently not possible, missing jsd, kld
        smape_dict[const.USER_TILE_COUNT_QUARTILES] = symmetric_mape(
            report_base[const.USER_TILE_COUNT].quartiles,
            report_alternative[const.USER_TILE_COUNT].quartiles,
        )

    # TODO create similar bins in mobility report from 0 to 1 with 0.1 bins
    if const.MOBILITY_ENTROPY not in analysis_exclusion:
        # kld_dict[const.MOBILITY_ENTROPY] = entropy(
        #     report_base[const.MOBILITY_ENTROPY].data[0],
        #     report_alternative[const.MOBILITY_ENTROPY].data[0],
        # )
        # jsd_dict[const.MOBILITY_ENTROPY] = distance.jensenshannon(
        #     report_base[const.MOBILITY_ENTROPY].data[0],
        #     report_alternative[const.MOBILITY_ENTROPY].data[0],
        # )
        emd_dict[const.MOBILITY_ENTROPY] = earth_movers_distance1D(
            report_base[const.MOBILITY_ENTROPY].data,
            report_alternative[const.MOBILITY_ENTROPY].data,
        )
        # smape_dict[const.MOBILITY_ENTROPY] = symmetric_mape(
        #     report_base[const.MOBILITY_ENTROPY].data[0],
        #     report_alternative[const.MOBILITY_ENTROPY].data[0],
        # )
        # Quartiles
        smape_dict["mobility_entropy_quartiles"] = symmetric_mape(
            report_base[const.MOBILITY_ENTROPY].quartiles,
            report_alternative[const.MOBILITY_ENTROPY].quartiles,
        )

    return relative_error_dict, kld_dict, jsd_dict, emd_dict, smape_dict


def get_selected_measures(benchmarkreport: "BenchmarkReport") -> dict:
    similarity_measures = {}

    for analysis in benchmarkreport.measure_selection.keys():
        selected_measure = benchmarkreport.measure_selection[analysis]
        try:
            if selected_measure == const.RE:
                if analysis == const.DS_STATISTICS:
                    for element in const.DS_STATISTICS_ELEMENTS:
                        similarity_measures[element] = benchmarkreport.re[element]
                elif analysis == const.MISSING_VALUES:
                    for element in const.MISSING_VALUES_ELEMENTS:
                        similarity_measures[element] = benchmarkreport.re[element]
                else:
                    similarity_measures[analysis] = benchmarkreport.re[analysis]
            elif selected_measure == const.KLD:
                similarity_measures[analysis] = benchmarkreport.kld[analysis]
            elif selected_measure == const.JSD:
                similarity_measures[analysis] = benchmarkreport.jsd[analysis]
            elif selected_measure == const.EMD:
                similarity_measures[analysis] = benchmarkreport.emd[analysis]
            elif selected_measure == const.SMAPE:
                similarity_measures[analysis] = benchmarkreport.smape[analysis]
        except KeyError:
            warnings.warn(
                f"The selected measure {selected_measure} for {analysis} cannot be computed. Value for {analysis} in `self.similarity_measures` is set to `None`."
            )
            similarity_measures[analysis] = None

    return similarity_measures

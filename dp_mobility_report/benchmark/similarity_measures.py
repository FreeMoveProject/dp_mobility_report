import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance

import cv2
from haversine import haversine, Unit
from dp_mobility_report import constants as const


def _moving_average(arr, size):
    return np.convolve(arr, np.ones(size), "valid") / size


def wasserstein_distance1D(hist1, hist2):
    u_values = _moving_average(hist1[1], 2)
    v_values = _moving_average(hist2[1], 2)
    u_weights = hist1[0]
    v_weights = hist2[0]
    if (sum(u_weights) == 0) | (sum(v_weights) == 0):
        return None
    return wasserstein_distance(u_values, v_values, u_weights, v_weights)


def symmetric_mape(true, estimate, n_true_positive_zeros=None):
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


def relative_error(true, estimate):
    if estimate == None:
        estimate = 0
    if true == 0:
        # we can return the absolute error
        return np.abs(true - estimate)
        # or Relative Percent Difference
        # return(2*(estimate-true)/(np.abs(estimate)+np.abs(true)))
    return np.abs(true - estimate) / true


def all_relative_errors(true_dict, estimate_dict, round_res=False):
    re = dict()
    for key in true_dict.data:
        re[key] = relative_error(true_dict.data[key], estimate_dict.data[key])
        if round_res:
            re[key] = round(re[key], 2)
    return re


def probability(a):
    return a / sum(a)


def kullback_leibler_divergence(a, b):
    return np.sum(
        np.where(a != 0, a * np.log(a / b, where=a != 0), 0)
    )  # double where to suppress warning


def kld_of_counts(p, q):
    p = probability(np.asarray(p, dtype=np.float))
    q = probability(np.asarray(q, dtype=np.float))
    return kullback_leibler_divergence(p, q)


def jsd_of_counts(p, q):
    p = probability(np.asarray(p, dtype=np.float))
    q = probability(np.asarray(q, dtype=np.float))
    m = (1.0 / 2.0) * (p + q)
    return (1.0 / 2.0) * kullback_leibler_divergence(p, m) + (1.0 / 2.0) * kullback_leibler_divergence(q, m)


### earth movers distance
def _compute_cost_matrix(tile_coords):
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
    arr_true, arr_estimate, cost_matrix
):  # based on haversine distance
    # normalize input and assign needed type for cv2
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


def compute_similarity_measures(report_proposal, report_benchmark, tessellation, cost_matrix=None): #analysis_selection, 

    relative_error_dict = {}
    kld_dict = {}
    jsd_dict = {}
    emd_dict = {}
    smape_dict = {}
    
    #TODO 5 dicts
    #TODO: check for each analysis if is in analysis_selection
    #TODO: KLD possible? -> None
    
    ### overview ###
    relative_error_dict = dict(
        **relative_error_dict,
        **all_relative_errors(
            report_benchmark[const.DS_STATISTICS],
            report_proposal[const.DS_STATISTICS],
            round_res=True,
        )
    )

    relative_error_dict = dict(
        **relative_error_dict,
        **all_relative_errors(
            report_benchmark[const.MISSING_VALUES],
            report_proposal[const.MISSING_VALUES],
            round_res=True,
        )
    )

    trips_over_time = report_benchmark[const.TRIPS_OVER_TIME].data.merge(
            report_proposal[const.TRIPS_OVER_TIME].data,
            how="outer",
            on="datetime",
            suffixes=("_benchmark", "_proposal"),
        )

    trips_over_time.fillna(0, inplace=True)
    jsd_dict[const.TRIPS_OVER_TIME] = jsd_of_counts(trips_over_time.trip_count_benchmark, trips_over_time.trip_count_proposal)
    smape_dict[const.TRIPS_OVER_TIME] = symmetric_mape(
        trips_over_time.trip_count_benchmark, trips_over_time.trip_count_proposal
    )
    

    trips_per_weekday = pd.concat(
        [report_benchmark[const.TRIPS_PER_WEEKDAY].data, report_proposal[const.TRIPS_PER_WEEKDAY].data],
        join="outer",
        axis=1,
    )
    trips_per_weekday.fillna(0, inplace=True)
    smape_dict[const.TRIPS_PER_WEEKDAY] = symmetric_mape(
        trips_per_weekday.iloc[:, 0], trips_per_weekday.iloc[:, 1]
    )

    trips_per_hour = report_benchmark[const.TRIPS_PER_HOUR].data.merge(
        report_proposal[const.TRIPS_PER_HOUR].data,
        how="outer",
        on=["hour", "time_category"],
        suffixes=("_benchmark", "_proposal"),
    )
    trips_per_hour.fillna(0, inplace=True)
    smape_dict[const.TRIPS_PER_HOUR] = symmetric_mape(
        trips_per_hour.perc_benchmark, trips_per_hour.perc_proposal #TODO im eval package change to perc
    )

    ### place
    visits_per_tile = report_benchmark[const.VISITS_PER_TILE].data.merge(
        report_proposal[const.VISITS_PER_TILE].data,
        how="outer",
        on="tile_id",
        suffixes=("_benchmark", "_proposal"),
    )
    visits_per_tile.fillna(0, inplace=True)

    rel_counts_benchmark = (
        visits_per_tile.visits_benchmark / visits_per_tile.visits_benchmark.sum()
    )
    rel_counts_proposal = (
        visits_per_tile.visits_proposal
        / visits_per_tile.visits_proposal.sum()
    )
    smape_dict[const.VISITS_PER_TILE] = symmetric_mape(
        visits_per_tile.visits_benchmark, visits_per_tile.visits_proposal
    )
    smape_dict[const.REL_COUNTS_PER_TILE] = symmetric_mape(
        rel_counts_benchmark, rel_counts_proposal
    )

    # speed up evaluation: cost_matrix as input so it does not have to be recomputed every time
    
    if cost_matrix is None:
        tile_centroids = (
            tessellation.set_index("tile_id").to_crs(3395).centroid.to_crs(4326)
        )

        sorted_tile_centroids = tile_centroids.loc[visits_per_tile.tile_id]
        tile_coords = list(zip(sorted_tile_centroids.y, sorted_tile_centroids.x))
        # create custom cost matrix with distances between all tiles
        cost_matrix = _compute_cost_matrix(tile_coords)


    emd_dict[const.VISITS_PER_TILE] = earth_movers_distance(
        visits_per_tile.visits_benchmark.to_numpy(),
        visits_per_tile.visits_proposal.to_numpy(),
        cost_matrix,
    )


    relative_error_dict[const.VISITS_PER_TILE_OUTLIERS] = relative_error(
        report_benchmark[const.VISITS_PER_TILE].n_outliers,
        report_proposal[const.VISITS_PER_TILE].n_outliers,
    )
    smape_dict[const.VISITS_PER_TILE_QUARTILES] = symmetric_mape(
        report_benchmark[const.VISITS_PER_TILE].quartiles,
        report_proposal[const.VISITS_PER_TILE].quartiles,
    )

    ## tile counts per timewindow
    visits_per_tile_timewindow_emd = []

    for c in report_benchmark[const.VISITS_PER_TILE_TIMEWINDOW].data.columns:
        tw_benchmark = report_benchmark[const.VISITS_PER_TILE_TIMEWINDOW].data[c].loc[
            report_benchmark[const.VISITS_PER_TILE].data.tile_id
        ]  # sort accordingly for cost_matrix
        tw_benchmark = tw_benchmark / tw_benchmark.sum()
        if c not in report_proposal[const.VISITS_PER_TILE_TIMEWINDOW].data.columns:
            tw_proposal = tw_benchmark.copy()
            tw_proposal[:] = 0
        else:
            tw_proposal = report_proposal[const.VISITS_PER_TILE_TIMEWINDOW].data[c].loc[
                report_benchmark[const.VISITS_PER_TILE].data.tile_id
            ]
            tw_proposal = tw_proposal / tw_proposal.sum()
        tw = pd.merge(
            tw_benchmark,
            tw_proposal,
            how="outer",
            right_index=True,
            left_index=True,
            suffixes=("_benchmark", "_proposal"),
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

    counts_timew_benchmark = report_benchmark[const.VISITS_PER_TILE_TIMEWINDOW].data[
        report_benchmark[const.VISITS_PER_TILE_TIMEWINDOW].data.index != "None"
    ].unstack()
    counts_timew_proposal = report_proposal[const.VISITS_PER_TILE_TIMEWINDOW].data[
        report_proposal[const.VISITS_PER_TILE_TIMEWINDOW].data.index != "None"
    ].unstack()

    indices = np.unique(
        np.append(counts_timew_benchmark.index.values, counts_timew_proposal.index.values)
    )

    counts_timew_benchmark = counts_timew_benchmark.reindex(index=indices)
    counts_timew_benchmark.fillna(0, inplace=True)

    counts_timew_proposal = counts_timew_proposal.reindex(index=indices)
    counts_timew_proposal.fillna(0, inplace=True)

    rel_counts_timew_benchmark = counts_timew_benchmark / counts_timew_benchmark.sum()
    rel_counts_timew_proposal = counts_timew_proposal / counts_timew_proposal.sum()

    smape_dict[const.VISITS_PER_TILE_TIMEWINDOW] = symmetric_mape(
        counts_timew_benchmark.to_numpy().flatten(),
        counts_timew_proposal.to_numpy().flatten(),
    )

    smape_dict[const.REL_COUNTS_PER_TILE_TIMEWINDOW] = symmetric_mape(
        rel_counts_timew_benchmark.to_numpy().flatten(),
        rel_counts_timew_proposal.to_numpy().flatten(),
    )

    ### od
    all_od_combinations = pd.concat(
        [
            report_benchmark[const.OD_FLOWS].data[["origin", "destination"]],
            report_proposal[const.OD_FLOWS].data[["origin", "destination"]],
        ]
    ).drop_duplicates()
    all_od_combinations["flow"] = 0
    n_benchmark_positive_zeros = len(tessellation) ** 2 - len(all_od_combinations)

    true = (
        pd.concat([report_benchmark[const.OD_FLOWS].data, all_od_combinations])
        .drop_duplicates(["origin", "destination"], keep="first")
        .sort_values(["origin", "destination"])
        .flow
    )
    estimate = (
        pd.concat([report_proposal[const.OD_FLOWS].data, all_od_combinations])
        .drop_duplicates(["origin", "destination"], keep="first")
        .sort_values(["origin", "destination"])
        .flow
    )

    rel_benchmark = true / true.sum()
    rel_proposal = estimate / (estimate.sum())

    smape_dict[const.OD_FLOWS] = symmetric_mape(true.to_numpy(), estimate.to_numpy())
    smape_dict[const.REL_OD_FLOWS] = symmetric_mape(
        rel_benchmark.to_numpy(), rel_proposal.to_numpy()
    )
    smape_dict[const.OD_FLOWS_ALL_FLOWS] = symmetric_mape(
        true.to_numpy(), estimate.to_numpy(), n_benchmark_positive_zeros
    )
    smape_dict[const.REL_OD_FLOWS_ALL_FLOWS] = symmetric_mape(
        rel_benchmark.to_numpy(), rel_proposal.to_numpy(), n_benchmark_positive_zeros
    )
    emd_dict[const.TRAVEL_TIME] = wasserstein_distance1D(
        report_benchmark[const.TRAVEL_TIME].data,
        report_proposal[const.TRAVEL_TIME].data,
    )
    
    smape_dict[const.TRAVEL_TIME_QUARTILES] = symmetric_mape(
        report_benchmark[const.TRAVEL_TIME].quartiles,
        report_proposal[const.TRAVEL_TIME].quartiles,
    )
    emd_dict[const.JUMP_LENGTH] = wasserstein_distance1D(
        report_benchmark[const.JUMP_LENGTH].data,
        report_proposal[const.JUMP_LENGTH].data,
    )
    
    smape_dict[const.JUMP_LENGTH_QUARTILES] = symmetric_mape(
        report_benchmark[const.JUMP_LENGTH].quartiles,
        report_proposal[const.JUMP_LENGTH].quartiles,
    )

    ## user
    if report_proposal[const.TRIPS_PER_USER] is None:
        smape_dict[const.TRAJ_PER_USER_QUARTILES] = None
        smape_dict[const.TRAJ_PER_USER_OUTLIERS] = None
    else:
        smape_dict[const.TRAJ_PER_USER_QUARTILES] = symmetric_mape(
            report_benchmark[const.TRIPS_PER_USER].quartiles,
            report_proposal[const.TRIPS_PER_USER].quartiles,
        )
    if report_proposal[const.USER_TIME_DELTA] is None:
        smape_dict[const.USER_TIME_DELTA_QUARTILES] = None
        smape_dict[const.USER_TIME_DELTA_OUTLIERS] = None
    else:
        smape_dict[const.USER_TIME_DELTA_QUARTILES] = symmetric_mape(
            (
                report_benchmark[const.USER_TIME_DELTA].quartiles.apply(
                    lambda x: x.total_seconds() / 3600
                )
            ),
            report_proposal[const.USER_TIME_DELTA].quartiles.apply(
                lambda x: x.total_seconds() / 3600
            ),
        )
    emd_dict[const.RADIUS_OF_GYRATION] = wasserstein_distance1D(
        report_benchmark[const.RADIUS_OF_GYRATION].data,
        report_proposal[const.RADIUS_OF_GYRATION].data,
    )
    smape_dict[const.RADIUS_OF_GYRATION_QUARTILES] = symmetric_mape(
        report_benchmark[const.RADIUS_OF_GYRATION].quartiles,
        report_proposal[const.RADIUS_OF_GYRATION].quartiles,
    )
    smape_dict[const.USER_TILE_COUNT_QUARTILES] = symmetric_mape(
        report_benchmark[const.USER_TILE_COUNT].quartiles,
        report_proposal[const.USER_TILE_COUNT].quartiles,
    )


    return relative_error_dict, kld_dict, jsd_dict, emd_dict, smape_dict


def get_selected_measures():
    pass
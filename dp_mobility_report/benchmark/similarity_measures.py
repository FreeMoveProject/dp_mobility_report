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


def rel_error(true, estimate):
    if estimate == None:
        estimate = 0
    if true == 0:
        # we can return the absolute error
        return np.abs(true - estimate)
        # or Relative Percent Difference
        # return(2*(estimate-true)/(np.abs(estimate)+np.abs(true)))
    return np.abs(true - estimate) / true


def rel_error_dict(true_dict, estimate_dict, round_res=False):
    re = dict()
    for key in true_dict.data:
        re[key] = rel_error(true_dict.data[key], estimate_dict.data[key])
        if round_res:
            re[key] = round(re[key], 2)
    return re


def get_prob(a):
    return a / sum(a)


def KL(a, b):
    return np.sum(
        np.where(a != 0, a * np.log(a / b, where=a != 0), 0)
    )  # double where to suppress warning


def compute_KL(p, q):
    p = get_prob(np.asarray(p, dtype=np.float))
    q = get_prob(np.asarray(q, dtype=np.float))
    return KL(p, q)


def compute_JS(p, q):
    p = get_prob(np.asarray(p, dtype=np.float))
    q = get_prob(np.asarray(q, dtype=np.float))
    m = (1.0 / 2.0) * (p + q)
    return (1.0 / 2.0) * KL(p, m) + (1.0 / 2.0) * KL(q, m)


### earth movers distance
def _get_cost_matrix(tile_coords):
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






def compute_similarity_measures(report_proposal, report_benchmark, tessellation, cost_matrix=None):

    similarity_measures = dict()

    ### overview ###
    similarity_measures = dict(
        **similarity_measures,
        **rel_error_dict(
            report_benchmark["ds_statistics"],
            report_proposal["ds_statistics"],
            round_res=True,
        )
    )

    similarity_measures = dict(
        **similarity_measures,
        **rel_error_dict(
            report_benchmark["missing_values"],
            report_proposal["missing_values"],
            round_res=True,
        )
    )

    trips_over_time = report_benchmark["trips_over_time"].data.merge(
            report_proposal["trips_over_time"].data,
            how="outer",
            on="datetime",
            suffixes=("_benchmark", "_proposal"),
        )

    trips_over_time.fillna(0, inplace=True)
    similarity_measures["trips_over_time_js"] = compute_JS(trips_over_time.trip_count_benchmark, trips_over_time.trip_count_proposal)
    similarity_measures["trips_over_time_mre"] = symmetric_mape(
        trips_over_time.trip_count_benchmark, trips_over_time.trip_count_proposal
    )
    similarity_measures["trips_over_time_quartiles"] = symmetric_mape(
        report_benchmark["trips_over_time"].quartiles.apply(lambda x: x.toordinal()),
        report_proposal["trips_over_time"].quartiles.apply(
            lambda x: x.toordinal()
        ),
    )

    trips_per_weekday = pd.concat(
        [report_benchmark["trips_per_weekday"].data, report_proposal["trips_per_weekday"].data],
        join="outer",
        axis=1,
    )
    trips_per_weekday.fillna(0, inplace=True)
    similarity_measures["trips_per_weekday"] = symmetric_mape(
        trips_per_weekday.iloc[:, 0], trips_per_weekday.iloc[:, 1]
    )

    trips_per_hour = report_benchmark["trips_per_hour"].data.merge(
        report_proposal["trips_per_hour"].data,
        how="outer",
        on=["hour", "time_category"],
        suffixes=("_benchmark", "_proposal"),
    )
    trips_per_hour.fillna(0, inplace=True)
    similarity_measures["trips_per_hour"] = symmetric_mape(
        trips_per_hour.perc_benchmark, trips_per_hour.perc_proposal #TODO im eval package change to perc
    )

    ### place
    counts_per_tile = report_benchmark["visits_per_tile"].data.merge(
        report_proposal["visits_per_tile"].data,
        how="outer",
        on="tile_id",
        suffixes=("_benchmark", "_proposal"),
    )
    counts_per_tile.fillna(0, inplace=True)

    rel_counts_benchmark = (
        counts_per_tile.visits_benchmark / counts_per_tile.visits_benchmark.sum()
    )
    rel_counts_proposal = (
        counts_per_tile.visits_proposal
        / counts_per_tile.visits_proposal.sum()
    )
    similarity_measures["counts_per_tile_smape"] = symmetric_mape(
        counts_per_tile.visits_benchmark, counts_per_tile.visits_proposal
    )
    similarity_measures["rel_counts_per_tile_smape"] = symmetric_mape(
        rel_counts_benchmark, rel_counts_proposal
    )

    # speed up evaluation: cost_matrix as input so it does not have to be recomputed every time
    
    if cost_matrix is None:
        tile_centroids = (
            tessellation.set_index("tile_id").to_crs(3395).centroid.to_crs(4326)
        )

        sorted_tile_centroids = tile_centroids.loc[counts_per_tile.tile_id]
        tile_coords = list(zip(sorted_tile_centroids.y, sorted_tile_centroids.x))
        # create custom cost matrix with distances between all tiles
        cost_matrix = _get_cost_matrix(tile_coords)


    similarity_measures["counts_per_tile_emd"] = earth_movers_distance(
        counts_per_tile.visits_benchmark.to_numpy(),
        counts_per_tile.visits_proposal.to_numpy(),
        cost_matrix,
    )


    similarity_measures["counts_per_tile_outliers"] = rel_error(
        report_benchmark["visits_per_tile"].n_outliers,
        report_proposal["visits_per_tile"].n_outliers,
    )
    similarity_measures["counts_per_tile_quartiles"] = symmetric_mape(
        report_benchmark["visits_per_tile"].quartiles,
        report_proposal["visits_per_tile"].quartiles,
    )

    ## tile counts per timewindow
    counts_per_tile_timewindow_emd = []

    for c in report_benchmark["visits_per_tile_timewindow"].data.columns:
        tw_benchmark = report_benchmark["visits_per_tile_timewindow"].data[c].loc[
            report_benchmark["visits_per_tile"].data.tile_id
        ]  # sort accordingly for cost_matrix
        tw_benchmark = tw_benchmark / tw_benchmark.sum()
        if c not in report_proposal["visits_per_tile_timewindow"].data.columns:
            tw_proposal = tw_benchmark.copy()
            tw_proposal[:] = 0
        else:
            tw_proposal = report_proposal["visits_per_tile_timewindow"].data[c].loc[
                report_benchmark["visits_per_tile"].data.tile_id
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
        counts_per_tile_timewindow_emd.append(
            earth_movers_distance(
                tw.iloc[:, 0].to_numpy(), tw.iloc[:, 1].to_numpy(), cost_matrix
            )
        )

    similarity_measures["counts_per_tile_timewindow_emd"] = np.mean(
        counts_per_tile_timewindow_emd
    )

    counts_timew_benchmark = report_benchmark["visits_per_tile_timewindow"].data[
        report_benchmark["visits_per_tile_timewindow"].data.index != "None"
    ].unstack()
    counts_timew_proposal = report_proposal["visits_per_tile_timewindow"].data[
        report_proposal["visits_per_tile_timewindow"].data.index != "None"
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

    similarity_measures["visits_per_tile_timewindow"] = symmetric_mape(
        counts_timew_benchmark.to_numpy().flatten(),
        counts_timew_proposal.to_numpy().flatten(),
    )

    similarity_measures["rel_counts_per_tile_timewindow"] = symmetric_mape(
        rel_counts_timew_benchmark.to_numpy().flatten(),
        rel_counts_timew_proposal.to_numpy().flatten(),
    )

    ### od
    all_od_combinations = pd.concat(
        [
            report_benchmark["od_flows"].data[["origin", "destination"]],
            report_proposal["od_flows"].data[["origin", "destination"]],
        ]
    ).drop_duplicates()
    all_od_combinations["flow"] = 0
    n_benchmark_positive_zeros = len(tessellation) ** 2 - len(all_od_combinations)

    true = (
        pd.concat([report_benchmark["od_flows"].data, all_od_combinations])
        .drop_duplicates(["origin", "destination"], keep="first")
        .sort_values(["origin", "destination"])
        .flow
    )
    estimate = (
        pd.concat([report_proposal["od_flows"].data, all_od_combinations])
        .drop_duplicates(["origin", "destination"], keep="first")
        .sort_values(["origin", "destination"])
        .flow
    )

    rel_benchmark = true / true.sum()
    rel_proposal = estimate / (estimate.sum())

    similarity_measures["od_flows"] = symmetric_mape(true.to_numpy(), estimate.to_numpy())
    similarity_measures["rel_od_flows"] = symmetric_mape(
        rel_benchmark.to_numpy(), rel_proposal.to_numpy()
    )
    similarity_measures["od_flows_all_flows"] = symmetric_mape(
        true.to_numpy(), estimate.to_numpy(), n_benchmark_positive_zeros
    )
    similarity_measures["rel_od_flows_all_flows"] = symmetric_mape(
        rel_benchmark.to_numpy(), rel_proposal.to_numpy(), n_benchmark_positive_zeros
    )
    similarity_measures["travel_time_emd"] = wasserstein_distance1D(
        report_benchmark["travel_time"].data,
        report_proposal["travel_time"].data,
    )
    
    similarity_measures["travel_time_quartiles"] = symmetric_mape(
        report_benchmark["travel_time"].quartiles,
        report_proposal["travel_time"].quartiles,
    )
    similarity_measures["jump_length_emd"] = wasserstein_distance1D(
        report_benchmark["jump_length"].data,
        report_proposal["jump_length"].data,
    )
    
    similarity_measures["jump_length_quartiles"] = symmetric_mape(
        report_benchmark["jump_length"].quartiles,
        report_proposal["jump_length"].quartiles,
    )

    ## user
    if report_proposal["trips_per_user"] is None:
        similarity_measures["traj_per_user_quartiles"] = None
        similarity_measures["traj_per_user_outliers"] = None
    else:
        similarity_measures["traj_per_user_quartiles"] = symmetric_mape(
            report_benchmark["trips_per_user"].quartiles,
            report_proposal["trips_per_user"].quartiles,
        )
    if report_proposal["user_time_delta"] is None:
        similarity_measures["user_time_delta_quartiles"] = None
        similarity_measures["user_time_delta_outliers"] = None
    else:
        similarity_measures["user_time_delta_quartiles"] = symmetric_mape(
            (
                report_benchmark["user_time_delta"].quartiles.apply(
                    lambda x: x.total_seconds() / 3600
                )
            ),
            report_proposal["user_time_delta"].quartiles.apply(
                lambda x: x.total_seconds() / 3600
            ),
        )
    similarity_measures["radius_gyration_emd"] = wasserstein_distance1D(
        report_benchmark["radius_of_gyration"].data,
        report_proposal["radius_of_gyration"].data,
    )
    similarity_measures["radius_gyration_quartiles"] = symmetric_mape(
        report_benchmark["radius_of_gyration"].quartiles,
        report_proposal["radius_of_gyration"].quartiles,
    )
    # shape not same
    #similarity_measures["location_entropy_mre"] = symmetric_mape(
    #    #loc_entropy_per_tile.location_entropy_benchmark,
    #    #loc_entropy_per_tile.location_entropy_proposal,
    #    report_benchmark["mobility_entropy"].data[0],
    #    report_proposal["mobility_entropy"].data[0],
    #)
    # weight and value array not the same size...?
    #similarity_measures["user_tile_count_emd"] = wasserstein_distance1D(
    #    report_benchmark["user_tile_count"].data,
    #    report_proposal["user_tile_count"].data,
    #)
    similarity_measures["user_tile_count_quartiles"] = symmetric_mape(
        report_benchmark["user_tile_count"].quartiles,
        report_proposal["user_tile_count"].quartiles,
    )


    return similarity_measures

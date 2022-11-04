import cv2
import numpy as np
import pandas as pd
from haversine import Unit, haversine
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from scipy.special import kl_div
from scipy.stats import entropy
from dp_mobility_report import constants as const


def _moving_average(arr, size):
    return np.convolve(arr, np.ones(size), "valid") / size


def earth_movers_distance1D(hist1, hist2):
    if len(hist1[0]) == len(hist1[1]): #checks for histogram buckets with exact sizes or ranges
        u_values = hist1[1]
        v_values = hist2[1]
    else: #computes moving average if histogram buckets are in a range
        u_values = _moving_average(hist1[1], 2)
        v_values = _moving_average(hist2[1], 2)
    u_weights = hist1[0]
    v_weights = hist2[0]
    if (sum(u_weights) == 0) | (sum(v_weights) == 0):
        return None
    return wasserstein_distance(u_values, v_values, u_weights, v_weights)


# TODO: `n_true_positive_zeros` not needed anymore
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


# earth movers distance
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


def compute_similarity_measures(analysis_exclusion, report_proposal, report_benchmark, tessellation): 

    relative_error_dict = {}
    kld_dict = {}
    jsd_dict = {}
    emd_dict = {}
    smape_dict = {}
        
    ### Statistics
    if not const.DS_STATISTICS in analysis_exclusion:
        relative_error_dict = dict(
            **relative_error_dict,
            **all_relative_errors(
                report_benchmark[const.DS_STATISTICS],
                report_proposal[const.DS_STATISTICS],
                round_res=True,
            )
        )
    ### Missing values
    if not const.MISSING_VALUES in analysis_exclusion:
        relative_error_dict = dict(
            **relative_error_dict,
            **all_relative_errors(
                report_benchmark[const.MISSING_VALUES],
                report_proposal[const.MISSING_VALUES],
                round_res=True,
            )
        )

    ### Temporal distributions
    if not const.TRIPS_OVER_TIME in analysis_exclusion:
        trips_over_time = report_benchmark[const.TRIPS_OVER_TIME].data.merge(
                report_proposal[const.TRIPS_OVER_TIME].data,
                how="outer",
                on="datetime",
                suffixes=("_benchmark", "_proposal"),
            )

        trips_over_time.fillna(0, inplace=True)
        kld_dict[const.TRIPS_OVER_TIME] = entropy(trips_over_time.trips_benchmark, trips_over_time.trips_proposal)
        jsd_dict[const.TRIPS_OVER_TIME] = distance.jensenshannon(trips_over_time.trips_benchmark, trips_over_time.trips_proposal)
        smape_dict[const.TRIPS_OVER_TIME] = symmetric_mape(
            trips_over_time.trip_count_benchmark, trips_over_time.trips_proposal
        )
    
    if not const.TRIPS_PER_WEEKDAY in analysis_exclusion:
        trips_per_weekday = pd.concat(
            [report_benchmark[const.TRIPS_PER_WEEKDAY].data, report_proposal[const.TRIPS_PER_WEEKDAY].data],
            join="outer",
            axis=1,
        )
        trips_per_weekday.fillna(0, inplace=True)
        kld_dict[const.TRIPS_PER_WEEKDAY] = entropy(trips_per_weekday.iloc[:, 0], trips_per_weekday.iloc[:, 1])
        jsd_dict[const.TRIPS_PER_WEEKDAY] = distance.jensenshannon(trips_per_weekday.iloc[:, 0], trips_per_weekday.iloc[:, 1])
        smape_dict[const.TRIPS_PER_WEEKDAY] = symmetric_mape(
            trips_per_weekday.iloc[:, 0], trips_per_weekday.iloc[:, 1]
        )        

    if not const.TRIPS_PER_HOUR in analysis_exclusion:
        trips_per_hour = report_benchmark[const.TRIPS_PER_HOUR].data.merge(
            report_proposal[const.TRIPS_PER_HOUR].data,
            how="outer",
            on=["hour", "time_category"],
            suffixes=("_benchmark", "_proposal"),
        )
        trips_per_hour.fillna(0, inplace=True)
        kld_dict[const.TRIPS_PER_HOUR] = entropy(trips_per_hour.perc_benchmark, trips_per_hour.perc_proposal)
        jsd_dict[const.TRIPS_PER_HOUR] = distance.jensenshannon(trips_per_hour.perc_benchmark, trips_per_hour.perc_proposal)
        smape_dict[const.TRIPS_PER_HOUR] = symmetric_mape(
            trips_per_hour.perc_benchmark, trips_per_hour.perc_proposal #TODO im eval package change to perc
        )

    if not const.TRAVEL_TIME in analysis_exclusion:
        kld_dict[const.TRAVEL_TIME] = entropy(
            report_benchmark[const.TRAVEL_TIME].data[0],
            report_proposal[const.TRAVEL_TIME].data[0]
        )
        jsd_dict[const.TRAVEL_TIME] = distance.jensenshannon(
            report_benchmark[const.TRAVEL_TIME].data[0],
            report_proposal[const.TRAVEL_TIME].data[0]
        )
        smape_dict[const.TRAVEL_TIME] = symmetric_mape(
            report_benchmark[const.TRAVEL_TIME].data[0],
            report_proposal[const.TRAVEL_TIME].data[0]
        )
        emd_dict[const.TRAVEL_TIME] = earth_movers_distance1D(
            report_benchmark[const.TRAVEL_TIME].data,
            report_proposal[const.TRAVEL_TIME].data,
        )
        #Quartiles
        smape_dict[const.TRAVEL_TIME_QUARTILES] = symmetric_mape(
            report_benchmark[const.TRAVEL_TIME].quartiles,
            report_proposal[const.TRAVEL_TIME].quartiles,
        )

    ### Spatial distribution
    if not const.VISITS_PER_TILE in analysis_exclusion:
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
            visits_per_tile.visits_proposal / visits_per_tile.visits_proposal.sum()
        )
        kld_dict[const.VISITS_PER_TILE] = entropy(rel_counts_benchmark, rel_counts_proposal)
        jsd_dict[const.VISITS_PER_TILE] = distance.jensenshannon(rel_counts_benchmark, rel_counts_proposal)
        smape_dict[const.VISITS_PER_TILE] = symmetric_mape(
            rel_counts_benchmark, rel_counts_proposal
        )

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
        #Outliers
        relative_error_dict[const.VISITS_PER_TILE_OUTLIERS] = relative_error(
            report_benchmark[const.VISITS_PER_TILE].n_outliers,
            report_proposal[const.VISITS_PER_TILE].n_outliers,
        )
        
    
    if not const.JUMP_LENGTH in analysis_exclusion:
        kld_dict[const.JUMP_LENGTH] = entropy(
            report_benchmark[const.JUMP_LENGTH].data[0],
            report_proposal[const.JUMP_LENGTH].data[0],
        )
        jsd_dict[const.JUMP_LENGTH] = distance.jensenshannon(
            report_benchmark[const.JUMP_LENGTH].data[0],
            report_proposal[const.JUMP_LENGTH].data[0],
        )
        smape_dict[const.JUMP_LENGTH] = symmetric_mape(
            report_benchmark[const.JUMP_LENGTH].data[0],
            report_proposal[const.JUMP_LENGTH].data[0],
        )
        emd_dict[const.JUMP_LENGTH] = earth_movers_distance1D(
            report_benchmark[const.JUMP_LENGTH].data,
            report_proposal[const.JUMP_LENGTH].data,
        )
        #Quartiles
        smape_dict[const.JUMP_LENGTH_QUARTILES] = symmetric_mape(
            report_benchmark[const.JUMP_LENGTH].quartiles,
            report_proposal[const.JUMP_LENGTH].quartiles,
        )

    ### Spatio-temporal distributions
    if not const.VISITS_PER_TILE_TIMEWINDOW in analysis_exclusion:
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

        kld_dict[const.VISITS_PER_TILE_TIMEWINDOW] = entropy(
            rel_counts_timew_benchmark.to_numpy().flatten(),
            rel_counts_timew_proposal.to_numpy().flatten(),
        )
        jsd_dict[const.VISITS_PER_TILE_TIMEWINDOW] = distance.jensenshannon(
            rel_counts_timew_benchmark.to_numpy().flatten(),
            rel_counts_timew_proposal.to_numpy().flatten(),
        )
        smape_dict[const.VISITS_PER_TILE_TIMEWINDOW] = symmetric_mape(
            rel_counts_timew_benchmark.to_numpy().flatten(),
            rel_counts_timew_proposal.to_numpy().flatten(),
        )
        
        
        visits_per_tile_timewindow_emd = []
        #TODO visits_per_tile_timewindow should not be based on visits_per_tile and the cost_matrix from above if visits_per_tile is excluded
        for time_window in report_benchmark[const.VISITS_PER_TILE_TIMEWINDOW].data.columns:
            tw_benchmark = report_benchmark[const.VISITS_PER_TILE_TIMEWINDOW].data[time_window].loc[
                report_benchmark[const.VISITS_PER_TILE].data.tile_id]  # sort with `report_benchmark[const.VISITS_PER_TILE]` to match order of cost_matrix
            tw_benchmark = tw_benchmark / tw_benchmark.sum()
            # if time window not in proposal report, add time windows with count zero
            if time_window not in report_proposal[const.VISITS_PER_TILE_TIMEWINDOW].data.columns:
                tw_proposal = tw_benchmark.copy()
                tw_proposal[:] = 0
            else:
                tw_proposal = report_proposal[const.VISITS_PER_TILE_TIMEWINDOW].data[time_window].loc[
                    report_benchmark[const.VISITS_PER_TILE].data.tile_id  # sort with `report_benchmark[const.VISITS_PER_TILE]` to match order of cost_matrix
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

        

    ### Origin-Destination
    if not const.OD_FLOWS in analysis_exclusion:
        all_od_combinations = pd.concat(
            [
                report_benchmark[const.OD_FLOWS].data[["origin", "destination"]],
                report_proposal[const.OD_FLOWS].data[["origin", "destination"]],
            ]
        ).drop_duplicates()
        all_od_combinations["flow"] = 0

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

        kld_dict[const.OD_FLOWS] = entropy(rel_benchmark.to_numpy(), rel_proposal.to_numpy())
        jsd_dict[const.OD_FLOWS] = distance.jensenshannon(rel_benchmark.to_numpy(), rel_proposal.to_numpy())
        smape_dict[const.OD_FLOWS] = symmetric_mape(
            rel_benchmark.to_numpy(), rel_proposal.to_numpy()
        )


    ## User
    if not const.TRIPS_PER_USER in analysis_exclusion:

        kld_dict[const.TRIPS_PER_USER] = entropy(
            report_benchmark[const.TRIPS_PER_USER].data[0],
            report_proposal[const.TRIPS_PER_USER].data[0],
        )
        jsd_dict[const.TRIPS_PER_USER] = distance.jensenshannon(
            report_benchmark[const.TRIPS_PER_USER].data[0],
            report_proposal[const.TRIPS_PER_USER].data[0],
        )
        emd_dict[const.TRIPS_PER_USER] = earth_movers_distance1D(
            report_benchmark[const.TRIPS_PER_USER].data,
            report_proposal[const.TRIPS_PER_USER].data,
        )
        smape_dict[const.TRIPS_PER_USER] = symmetric_mape(
            report_benchmark[const.TRIPS_PER_USER].data[0],
            report_proposal[const.TRIPS_PER_USER].data[0],
        )
        #Quartiles
        smape_dict[const.TRIPS_PER_USER_QUARTILES] = symmetric_mape(
            report_benchmark[const.TRIPS_PER_USER].quartiles,
            report_proposal[const.TRIPS_PER_USER].quartiles,
        )
    if not const.USER_TIME_DELTA in analysis_exclusion:
        if report_proposal[const.USER_TIME_DELTA] is None: #if each user only has one trip then `USER_TIME_DELTA` is None
            smape_dict[const.USER_TIME_DELTA_QUARTILES] = None
        else:
            #TODO comparison of histogram buckets of user time delta, currently not possible, missing jsd, kld
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

    if not const.RADIUS_OF_GYRATION in analysis_exclusion:

        kld_dict[const.RADIUS_OF_GYRATION] = entropy(
            report_benchmark[const.RADIUS_OF_GYRATION].data[0],
            report_proposal[const.RADIUS_OF_GYRATION].data[0],
        )
        jsd_dict[const.RADIUS_OF_GYRATION] = distance.jensenshannon(
            report_benchmark[const.RADIUS_OF_GYRATION].data[0],
            report_proposal[const.RADIUS_OF_GYRATION].data[0],
        )
        emd_dict[const.RADIUS_OF_GYRATION] = earth_movers_distance1D(
            report_benchmark[const.RADIUS_OF_GYRATION].data,
            report_proposal[const.RADIUS_OF_GYRATION].data,
        )
        smape_dict[const.RADIUS_OF_GYRATION] = symmetric_mape(
            report_benchmark[const.RADIUS_OF_GYRATION].data[0],
            report_proposal[const.RADIUS_OF_GYRATION].data[0],
        )
        #Quartiles
        smape_dict[const.RADIUS_OF_GYRATION_QUARTILES] = symmetric_mape(
            report_benchmark[const.RADIUS_OF_GYRATION].quartiles,
            report_proposal[const.RADIUS_OF_GYRATION].quartiles,
        )

    if not const.USER_TILE_COUNT_QUARTILES in analysis_exclusion:
        #TODO comparison of histogram buckets of user time delta, currently not possible, missing jsd, kld
        smape_dict[const.USER_TILE_COUNT_QUARTILES] = symmetric_mape(
            report_benchmark[const.USER_TILE_COUNT].quartiles,
            report_proposal[const.USER_TILE_COUNT].quartiles,
        )

    if not const.MOBILITY_ENTROPY in analysis_exclusion:
        kld_dict[const.MOBILITY_ENTROPY] = entropy(
            report_benchmark[const.MOBILITY_ENTROPY].data[0],
            report_proposal[const.MOBILITY_ENTROPY].data[0],
        )
        jsd_dict[const.MOBILITY_ENTROPY] = distance.jensenshannon(
            report_benchmark[const.MOBILITY_ENTROPY].data[0],
            report_proposal[const.MOBILITY_ENTROPY].data[0],
        )
        emd_dict[const.MOBILITY_ENTROPY] = earth_movers_distance1D(
            report_benchmark[const.MOBILITY_ENTROPY].data,
            report_proposal[const.MOBILITY_ENTROPY].data,
        )
        smape_dict[const.MOBILITY_ENTROPY] = symmetric_mape(
            report_benchmark[const.MOBILITY_ENTROPY].data[0],
            report_proposal[const.MOBILITY_ENTROPY].data[0],
        )
        #Quartiles
        smape_dict["mobility_entropy_quartiles"] = symmetric_mape(
            report_benchmark[const.MOBILITY_ENTROPY].quartiles,
            report_proposal[const.MOBILITY_ENTROPY].quartiles,
        )


    return relative_error_dict, kld_dict, jsd_dict, emd_dict, smape_dict


def get_selected_measures(benchmarkreport):
    pass

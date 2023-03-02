import os
import evaluation_config
import random
import geopandas as gpd
import pandas as pd
import sys
sys.path.append(evaluation_config.PATH_ABSOLUT)
from dp_mobility_report.benchmark.benchmarkreport import BenchmarkReport


# function to edit the dataframe and insert columns for the settings of the runs, so they can be filtered later
def edit_df(dataframe, algorithm, raw_or_average):
    dataframe.insert(0, column="Algorithm", value=algorithm)
    dataframe.insert(1, column="Raw/Average", value=raw_or_average)
    dataframe.insert(2, column="Epsilon", value=evaluation_config.EPSILON)
    dataframe.insert(3, column="User Privacy", value=evaluation_config.USER_PRIVACY)
    dataframe.insert(4, column="Max Trips", value=evaluation_config.MAX_TRIPS)
    dataframe.insert(5, column="Number of Runs", value=evaluation_config.NUMBER_OF_RUNS)


def generate_similarity_measures_synthetic(whole_csv_file_df):
    # read in synthetic datasets
    synthetic_datasets = random.sample([f for f in os.listdir(evaluation_config.PATH_DPSTAR_RESULT_DIR) if f.endswith(".csv")],
                                       evaluation_config.NUMBER_OF_RUNS)

    # generate DP-Star Reports and store similarity measures (in comparison to the raw dataset) in a dataframe
    synthetic_measures_list = []
    for dataset in synthetic_datasets:
        print(dataset)
        df_synthetic = pd.read_csv(os.path.join(evaluation_config.PATH_DPSTAR_RESULT_DIR, dataset))
        benchmarkreport = BenchmarkReport(
            df_base=base_df,
            df_alternative=df_synthetic,
            tessellation=tessellation,
            privacy_budget_base=None,
            privacy_budget_alternative=None,
            max_trips_per_user_base=evaluation_config.MAX_TRIPS,
            max_trips_per_user_alternative=evaluation_config.MAX_TRIPS,
            user_privacy_base=evaluation_config.USER_PRIVACY,
            user_privacy_alternative=evaluation_config.USER_PRIVACY,
        )
        measures = benchmarkreport.similarity_measures
        print(measures)
        synthetic_measures_list.append(measures)
    synthetic_raw_df = pd.DataFrame(synthetic_measures_list, index=None)
    edit_df(synthetic_raw_df, evaluation_config.SYNTHETIC_ALGORITHM_NAME, "RAW")
    # generate average and print it
    synthetic_average_df = pd.DataFrame(pd.DataFrame(synthetic_measures_list).mean(), index=None).T
    edit_df(synthetic_average_df, evaluation_config.SYNTHETIC_ALGORITHM_NAME, "Average")
    print(synthetic_average_df)
    whole_csv_file_df = pd.concat([whole_csv_file_df, synthetic_raw_df], ignore_index=True)
    # if you want to save the averages in the csv file use this instead:
    # whole_csv_file_df = pd.concat([whole_csv_file_df, synthetic_raw_df, synthetic_average_df], ignore_index=True)
    return whole_csv_file_df


def generate_similarity_measures_dp(whole_csv_file_df):
    # generate DPM Reports and store similarity measures (in comparison to the raw dataset) in a dataframe
    dp_measures_list = []
    for i in range(evaluation_config.NUMBER_OF_RUNS):
        benchmarkreport = BenchmarkReport(
            df_base=base_df,
            df_alternative=None,
            tessellation=tessellation,
            privacy_budget_base=None,
            privacy_budget_alternative=evaluation_config.EPSILON,
            max_trips_per_user_base=evaluation_config.MAX_TRIPS,
            max_trips_per_user_alternative=evaluation_config.MAX_TRIPS,
            user_privacy_base=evaluation_config.USER_PRIVACY,
            user_privacy_alternative=evaluation_config.USER_PRIVACY,
        )
        measures = benchmarkreport.similarity_measures
        print(measures)
        dp_measures_list.append(measures)
    dp_raw_df = pd.DataFrame(dp_measures_list, index=None)
    edit_df(dp_raw_df, "DP", "RAW")

    # generate average and store in a dataframe
    dp_average_df = pd.DataFrame(pd.DataFrame(dp_measures_list).mean(), index=None).T
    edit_df(dp_average_df, "DP", "Average")
    print(dp_average_df)
    whole_csv_file_df = pd.concat([whole_csv_file_df, dp_raw_df], ignore_index=True)
    # if you want to save the averages in the csv file use this instead:
    # whole_csv_file_df = pd.concat([whole_csv_file_df, dp_raw_df, dp_average_df], ignore_index=True)
    return whole_csv_file_df


# read in csv file
if os.path.isfile(evaluation_config.PATH_CSV_FILE):
    whole_csv_file_df = pd.read_csv(evaluation_config.PATH_CSV_FILE, index_col=False)
else:  # if it doesn't exist generate a new empty dataframe
    whole_csv_file_df = pd.DataFrame(index=None)

# read in tesselation
tessellation = gpd.read_file(evaluation_config.PATH_TESSELATION)
tessellation["tile_name"] = tessellation.tile_id

# read in raw dataframe
base_df = pd.read_csv(os.path.join(evaluation_config.PATH_RAW_DATA_DIR, evaluation_config.RAW_DATA_FILE))


if evaluation_config.GENERATE_SYNTHETIC_SIMILARITY_MEASURES:
    whole_csv_file_df = generate_similarity_measures_synthetic(whole_csv_file_df)
if evaluation_config.GENERATE_DP_SIMILARITY_MEASURES:
    whole_csv_file_df = generate_similarity_measures_dp(whole_csv_file_df)
whole_csv_file_df.to_csv(evaluation_config.PATH_CSV_FILE, na_rep="None", index=False)

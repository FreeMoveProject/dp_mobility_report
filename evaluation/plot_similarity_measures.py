import os
import sys
import evaluation_config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# State which runs you want to plot
ALGORITHMS = ["DPSTAR", "DP"]
NUMBER_OF_RUNS = 10
EPSILON = 4
MAX_TRIPS = "None"
USER_PRIVACY = False
MEASURES_TO_PLOT = ["n_records", "n_users", "n_locations", "visits_per_tile", "user_tile_count"]

# Reads in the CSV File or prints an Error Message
if os.path.isfile(evaluation_config.PATH_CSV_FILE):
    whole_csv_file_df = pd.read_csv(evaluation_config.PATH_CSV_FILE, index_col=False)
else:  # if it doesn't exist exit with error message
    sys.exit("ERROR: No CSV to read")

# filter csv for selected data (except Measures to Plot)
algorithm_datas = []
for algorithm in ALGORITHMS:
    algorithm_datas.append(whole_csv_file_df[(whole_csv_file_df["Algorithm"] == algorithm) & (whole_csv_file_df["Raw/Average"] == "RAW") & (
                whole_csv_file_df["Epsilon"] == EPSILON) & (whole_csv_file_df["User Privacy"] == USER_PRIVACY) & (
                                                         whole_csv_file_df["Max Trips"] == MAX_TRIPS)])

# filter dataframes for selected similarity measures and plot them
for measure in MEASURES_TO_PLOT:
    algorithms_selected_columns = []  # Array for datasets for each algorithm
    for algorithm_data in algorithm_datas:
        algorithms_selected_columns.append(pd.DataFrame(algorithm_data, columns=[measure]).reset_index(drop=True))

    # plot them
    fig = plt.figure()
    ax = plt.gca()
    for idx, algorithm_selected_columns in enumerate(algorithms_selected_columns):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(algorithm_selected_columns, color=color, marker="X", markersize="10", linestyle="", label=ALGORITHMS[idx] + " Raw")
        plt.axhline(np.nanmean(algorithm_selected_columns), color=color, linestyle="--", dashes=(5, 7), label=ALGORITHMS[idx] + " Average")

    plt.title(measure)
    plt.xlabel("Run")
    plt.ylabel(measure + " similarity measure")
    plt.legend()
    # generate filename
    algorithm_names = ""
    for algorithm in ALGORITHMS:
        algorithm_names += (algorithm + "_")
    filename = algorithm_names + "nr_runs_" + str(NUMBER_OF_RUNS) + "_epsilon_" + str(EPSILON) + "_maxtrips_" + str(
        MAX_TRIPS) + "_userprivacy_" + str(USER_PRIVACY) + "_measure_" + measure
    plt.savefig(evaluation_config.PATH_PLOTS + "/" + filename + ".png")

plt.show()

import os
import sys
import evaluation_config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# State which runs you want to plot
ALGORITHMS = ["DPSTAR", "DP"]
NUMBER_OF_RUNS = 10
EPSILON = [4, 10, 20, 50]
MAX_TRIPS = "None"
USER_PRIVACY = False
MEASURES_TO_PLOT = ["n_records", "n_users", "n_locations", "visits_per_tile", "user_tile_count"]

# Reads in the CSV File or prints an Error Message
if os.path.isfile(evaluation_config.PATH_CSV_FILE):
    whole_csv_file_df = pd.read_csv(evaluation_config.PATH_CSV_FILE, index_col=False)
else:  # if it doesn't exist exit with error message
    sys.exit("ERROR: No CSV to read")

averages_df = pd.DataFrame()  # Dataframe for the average values (for given settings)
#
for epsilon in EPSILON:
    algorithm_datas = []
    for algorithm in ALGORITHMS:
        algorithm_data = whole_csv_file_df[(whole_csv_file_df["Algorithm"] == algorithm) & (whole_csv_file_df["Raw/Average"] == "RAW") & (
                    whole_csv_file_df["Epsilon"] == epsilon) & (whole_csv_file_df["User Privacy"] == USER_PRIVACY) & (
                                                             whole_csv_file_df["Max Trips"] == MAX_TRIPS)]
        algorithm_data_mean = pd.DataFrame(algorithm_data.mean(numeric_only=True)).T
        algorithm_data_mean.insert(0, column="Algorithm", value=algorithm)
        averages_df = pd.concat([averages_df, pd.DataFrame(algorithm_data_mean)], ignore_index=True)

# filtering the average dataframe for the columns selected to plot
meta_columns = ["Algorithm", "Epsilon"]
selected_measures = pd.DataFrame(averages_df, columns=meta_columns + MEASURES_TO_PLOT).reset_index(drop=True)

# Filters for every selected measure and plots the averages in comparison
for measure in MEASURES_TO_PLOT:
    selected_measure = pd.DataFrame(selected_measures, columns=meta_columns + [measure]).reset_index(drop=True)
    fig = plt.figure()
    ax = plt.gca()
    for algorithm in ALGORITHMS:
        algorithm_averages = selected_measure[selected_measure["Algorithm"] == algorithm].reset_index(drop=True)
        algorithm_averages.plot(x=0, y=measure, marker="X", markersize="10", linestyle="", label=algorithm, ax=ax)
        ax.set_xticks(range(len(EPSILON)))
        ax.set_xticklabels(EPSILON)
        #algorithm_averages.plot.bar(x="Epsilon", y=measure, label=algorithm, ax=ax)
    plt.title(measure)
    plt.xlabel("Epsilon")
    plt.ylabel(measure + " similarity measure")
    plt.legend()
    # Save the plots as png
    algorithm_names = ""
    for algorithm in ALGORITHMS:
        algorithm_names += (algorithm + "_")
    filename = algorithm_names + "_epsilon_" + str(EPSILON) + "_maxtrips_" + str(
        MAX_TRIPS) + "_userprivacy_" + str(USER_PRIVACY) + "_measure_" + measure
    plt.savefig(evaluation_config.PATH_PLOTS + "/" + filename + ".png")

plt.show()

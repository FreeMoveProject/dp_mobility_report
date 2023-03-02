
# General Settings
NUMBER_OF_RUNS = 10
EPSILON = 4
MAX_TRIPS = None
USER_PRIVACY = False
SYNTHETIC_ALGORITHM_NAME = "DPSTAR"
# If you only want to generate similarity measures for one algorithm, you can set the other to False
GENERATE_SYNTHETIC_SIMILARITY_MEASURES = True
GENERATE_DP_SIMILARITY_MEASURES = True

# NEED TO INSERT ABSOLUT PATHS ON YOUR SYSTEM TO DP_STAR PROJECT AND THIS PROJECT
PATH_ABSOLUT = ""
PATH_DPSTAR = ""



# RELATIVE PATHS, NO NEED TO INSERT ANYTHING
PATH_EVALUATION = PATH_ABSOLUT + "/evaluation"
PATH_RAW_DATA_DIR = PATH_EVALUATION + "/data/raw_data"
RAW_DATA_FILE = "geolife.csv"
RAW_DATA_FILE_PEKING = "geolife_peking.csv"
PATH_TESSELATION = PATH_EVALUATION + "/data/raw_data/geolife_tessellation.gpkg"

# DPSTAR SETTINGS
# Choose Folders in this Project, so everything is in one place
PATH_DPSTAR_RESULT_DIR = PATH_EVALUATION + "/data/dpstar_synthetic_datasets"
PATH_DPSTAR_PREPROCESSED_DIR = PATH_EVALUATION + "/data/dpstar_preprocessed"
PATH_DPSTAR_PREPROCESSED_FILE = "geolife_peking.dat"


PATH_CSV_FILE = PATH_EVALUATION + "/data/evaluation_results/evaluation_database.csv"

PATH_PLOTS = PATH_EVALUATION + "/data/evaluation_results/plots"
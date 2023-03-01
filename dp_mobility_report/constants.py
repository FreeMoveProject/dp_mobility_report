from pyproj import CRS

# ANALYSIS (SEGMENTS AND ELEMENTS) NAMES
# segments
ALL = "all"  # deprecated
OVERVIEW = "overview"
PLACE_ANALYSIS = "place_analysis"
OD_ANALYSIS = "od_analysis"
USER_ANALYSIS = "user_analysis"
SEGMENTS = [OVERVIEW, PLACE_ANALYSIS, OD_ANALYSIS, USER_ANALYSIS]

# elements
DS_STATISTICS = "ds_statistics"
MISSING_VALUES = "missing_values"
TRIPS_OVER_TIME = "trips_over_time"
TRIPS_PER_WEEKDAY = "trips_per_weekday"
TRIPS_PER_HOUR = "trips_per_hour"
VISITS_PER_TILE = "visits_per_tile"
VISITS_PER_TIME_TILE = "visits_per_time_tile"
VISITS_PER_TILE_OUTLIERS = "visits_per_tile_outliers"
VISITS_PER_TILE_RANKING = "visits_per_tile_ranking"
VISITS_PER_TILE_QUARTILES = "visits_per_tile_quartiles"
OD_FLOWS = "od_flows"
OD_FLOWS_RANKING = "od_flows_ranking"
OD_FLOWS_QUARTILES = "od_flows_quartiles"
TRAVEL_TIME = "travel_time"
TRAVEL_TIME_QUARTILES = "travel_time_quartiles"
JUMP_LENGTH = "jump_length"
JUMP_LENGTH_QUARTILES = "jump_length_quartiles"
TRIPS_PER_USER = "trips_per_user"
TRIPS_PER_USER_QUARTILES = "trips_per_user_quartiles"
USER_TIME_DELTA = "user_time_delta"
USER_TIME_DELTA_QUARTILES = "user_time_delta_quartiles"
RADIUS_OF_GYRATION = "radius_of_gyration"
RADIUS_OF_GYRATION_QUARTILES = "radius_of_gyration_quartiles"
USER_TILE_COUNT = "user_tile_count"
USER_TILE_COUNT_QUARTILES = "user_tile_count_quartiles"
MOBILITY_ENTROPY = "mobility_entropy"
MOBILITY_ENTROPY_QUARTILES = "mobility_entropy_quartiles"
OVERVIEW_ELEMENTS = [
    DS_STATISTICS,
    MISSING_VALUES,
    TRIPS_OVER_TIME,
    TRIPS_PER_WEEKDAY,
    TRIPS_PER_HOUR,
]
PLACE_ELEMENTS = [VISITS_PER_TILE, VISITS_PER_TIME_TILE]
OD_ELEMENTS = [OD_FLOWS, TRAVEL_TIME, JUMP_LENGTH]
USER_ELEMENTS = [
    TRIPS_PER_USER,
    USER_TIME_DELTA,
    RADIUS_OF_GYRATION,
    USER_TILE_COUNT,
    MOBILITY_ENTROPY,
]

ELEMENTS = OVERVIEW_ELEMENTS + PLACE_ELEMENTS + OD_ELEMENTS + USER_ELEMENTS
TESSELLATION_ELEMENTS = PLACE_ELEMENTS + [OD_FLOWS, USER_TILE_COUNT, MOBILITY_ENTROPY]
TIMESTAMP_ANALYSES = [
    TRIPS_OVER_TIME,
    TRIPS_PER_WEEKDAY,
    TRIPS_PER_HOUR,
    VISITS_PER_TIME_TILE,
    TRAVEL_TIME,
    USER_TIME_DELTA,
]

SEGMENTS_AND_ELEMENTS = SEGMENTS + ELEMENTS

# DATASET COLUMNS
ID = "id"
UID = "uid"
TID = "tid"
LAT = "lat"
LNG = "lng"
DATETIME = "datetime"

POINT_TYPE = "point_type"
START = "start"
END = "end"
TILE_ID = "tile_id"
TILE_NAME = "tile_name"
GEOMETRY = "geometry"

TILE_ID_END = TILE_ID + "_" + END
LNG_END = LNG + "_" + END
LAT_END = LAT + "_" + END
DATETIME_END = DATETIME + "_" + END

# TIME
DATE = "date"
DAY_NAME = "day_name"
HOUR = "hour"
IS_WEEKEND = "is_weekend"
WEEKDAY = "weekday"
WEEKEND = "weekend"
TIME_CATEGORY = "time category"


# OTHER
DEFAULT_CRS = CRS.from_epsg(4326)
QUARTILES = "quartiles"
DATETIME_PRECISION = "datetime_precision"
PREC_DATE = "date"
PREC_WEEK = "week"
PREC_MONTH = "month"

ORIGIN = "origin"
DESTINATION = "destination"
FLOW = "flow"

# ds_statistics elements
N_RECORDS = "n_records"
N_TRIPS = "n_trips"
N_COMPLETE_TRIPS = "n_complete_trips"
N_INCOMPLETE_TRIPS = "n_incomplete_trips"
N_USERS = "n_users"
N_LOCATIONS = "n_locations"
DS_STATISTICS_ELEMENTS = [
    N_RECORDS,
    N_TRIPS,
    N_COMPLETE_TRIPS,
    N_INCOMPLETE_TRIPS,
    N_USERS,
    N_LOCATIONS,
]

# missing_values elements
MISSING_VALUES_ELEMENTS = [UID, TID, DATETIME, LAT, LNG]

# SIMILARITY MEASURES
SMAPE = "symmetric_mean_absolute_percentage_error"
JSD = "jensen_shannon_divergence"
KLD = "kullback_leibler_divergence"
EMD = "earth_movers_distance"
KT = "kendalls_tau"
TOP_N_COV = "top_n_coverage"
SIMILARITY_MEASURES = [SMAPE, JSD, KLD, EMD, KT, TOP_N_COV]

DISABLE_EMD_THRESHOLD = 1000

DARK_BLUE = "#283377"
LIGHT_BLUE = "#5D6FFF"
ORANGE = "#D9642C"
LIGHT_ORANGE = "#FFAD6F"
GREY = "#8A8A8A"
LIGHT_GREY = "#f2f2f2"
DARK_RED = "#8B0000"
LIGHT_RED = "#FFCCCB"

STANDARD_CMAP = "YlOrRd"
BASE_CMAP = "Blues"
ALT_CMAP = "Oranges"
DIVERGING_CMAP = "PiYG"

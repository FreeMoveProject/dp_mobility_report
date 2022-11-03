from pyproj import CRS

ALL = "all"
OVERVIEW = "overview"
PLACE_ANALYSIS = "place_analysis"
OD_ANALYSIS = "od_analysis"
USER_ANALYSIS = "user_analysis"
SEGMENTS = [OVERVIEW, PLACE_ANALYSIS, OD_ANALYSIS, USER_ANALYSIS]

DS_STATISTICS = "ds_statistics"
MISSING_VALUES = "missing_values"
TRIPS_OVER_TIME = "trips_over_time"
TRIPS_PER_WEEKDAY = "trips_per_weekday"
TRIPS_PER_HOUR = "trips_per_hour"
VISITS_PER_TILE = "visits_per_tile"
VISITS_PER_TILE_TIMEWINDOW = "visits_per_tile_timewindow"
VISITS_PER_TILE_OUTLIERS = "visits_per_tile_outliers"
VISITS_PER_TILE_QUARTILES = "counts_per_tile_quartiles"
REL_COUNTS_PER_TILE = "rel_counts_per_tile"
REL_COUNTS_PER_TILE_TIMEWINDOW = "rel_counts_per_tile_timewindow"
OD_FLOWS = "od_flows"
REL_OD_FLOWS = "rel_od_flows"
OD_FLOWS_ALL_FLOWS = "od_flows_all_flows"
REL_OD_FLOWS_ALL_FLOWS = "rel_od_flows_all_flows"
TRAVEL_TIME = "travel_time"
TRAVEL_TIME_QUARTILES = "travel_time_quartiles"
JUMP_LENGTH = "jump_length"
JUMP_LENGTH_QUARTILES = "jump_length_quartiles"
TRIPS_PER_USER = "trips_per_user"
TRAJ_PER_USER_QUARTILES = "traj_per_user_quartiles"
TRAJ_PER_USER_OUTLIERS = "traj_per_user_outliers"
USER_TIME_DELTA = "user_time_delta"
USER_TIME_DELTA_QUARTILES = "user_time_delta_quartiles"
USER_TIME_DELTA_OUTLIERS= "user_time_delta_outliers"
RADIUS_OF_GYRATION = "radius_of_gyration"
RADIUS_OF_GYRATION_QUARTILES = "radius_of_gyration_quartiles"
USER_TILE_COUNT = "user_tile_count"
USER_TILE_COUNT_QUARTILES = "user_tile_count_quartiles"
MOBILITY_ENTROPY = "mobility_entropy"
OVERVIEW_ELEMENTS = [
    DS_STATISTICS,
    MISSING_VALUES,
    TRIPS_OVER_TIME,
    TRIPS_PER_WEEKDAY,
    TRIPS_PER_HOUR,
]
PLACE_ELEMENTS = [VISITS_PER_TILE, VISITS_PER_TILE_TIMEWINDOW]
OD_ELEMENTS = [OD_FLOWS, TRAVEL_TIME, JUMP_LENGTH]
USER_ELEMENTS = [
    TRIPS_PER_USER,
    USER_TIME_DELTA,
    RADIUS_OF_GYRATION,
    USER_TILE_COUNT,
    MOBILITY_ENTROPY,
]

ELEMENTS = OVERVIEW_ELEMENTS + PLACE_ELEMENTS + OD_ELEMENTS + USER_ELEMENTS
TESSELLATION_ELEMENTS = (
    PLACE_ELEMENTS + OD_ELEMENTS + [USER_TILE_COUNT, MOBILITY_ENTROPY]
)  # TODO: include or exclude travel_time and jump_length?
TIMESTAMP_ANALYSES = [
    TRIPS_OVER_TIME,
    TRIPS_PER_WEEKDAY,
    TRIPS_PER_HOUR,
    VISITS_PER_TILE_TIMEWINDOW,
    TRAVEL_TIME,
    USER_TIME_DELTA,
]

SEGMENTS_AND_ELEMENTS = SEGMENTS + ELEMENTS

ID = "id"
UID = "uid"
TID = "tid"
LAT = "lat"
LNG = "lng"
DATETIME = "datetime"
TILE_ID_END = "tile_id_end"
LNG_END = "lng_end"
LAT_END = "lat_end"
DATETIME_END = "datetime_end"
DATE = "date"
DAY_NAME = "day_name"
HOUR = "hour"
IS_WEEKEND = "is_weekend"
WEEKDAY = "weekday"
WEEKEND = "weekend"
TIME_CATEGORY = "time_category"
POINT_TYPE = "point_type"
START = "start"
END = "end"

TILE_ID = "tile_id"
TILE_NAME = "tile_name"
GEOMETRY = "geometry"

DEFAULT_CRS = CRS.from_epsg(4326)
QUARTILES = "quartiles"
DATETIME_PRECISION = "datetime_precision"
PREC_DATE = "date"
PREC_WEEK = "week"
PREC_MONTH = "month"

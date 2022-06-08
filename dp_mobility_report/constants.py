from pyproj import CRS

ALL = "all"
OVERVIEW = "overview"
PLACE_ANALYSIS = "place_analysis"
OD_ANALYSIS = "od_analysis"
USER_ANALYSIS = "user_analysis"
ANALYSIS_SELECTION = [ALL, OVERVIEW, PLACE_ANALYSIS, OD_ANALYSIS, USER_ANALYSIS]

DS_STATISTICS = "ds_statistics"
MISSING_VALUES = "missing_values"
TRIPS_OVER_TIME = "trips_over_time"
TRIPS_PER_WEEKDAY = "trips_per_weekday"
TRIPS_PER_HOUR = "trips_per_hour"
VISITS_PER_TILE = "visits_per_tile"
VISITS_PER_TILE_TIMEWINDOW = "visits_per_tile_timewindow"
OD_FLOWS = "od_flows"
TRAVEL_TIME = "travel_time"
JUMP_LENGTH = "jump_length"
TRIPS_PER_USER = "trips_per_user"
USER_TIME_DELTA = "user_time_delta"
RADIUS_OF_GYRATION = "radius_of_gyration"
USER_TILE_COUNT = "user_tile_count"
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
WEEKDAY = "weekday"
HOUR = "hour"
IS_WEEKEND = "is_weekend"
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

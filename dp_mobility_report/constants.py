from pyproj import CRS

# data set constants
ID = "id"
UID = "uid"
TID = "tid"
LAT = "lat"
LNG = "lng"
DATETIME = "datetime"
DATE = "date"
DAY_NAME = "day_name"
HOUR = "hour"
IS_WEEKEND = "is_weekend"
TIME_CATEGORY = "time_category"
POINT_TYPE = "point_type"
START = "start"
END = "end"

# tessellation constants
TILE_ID = "tile_id"
TILE_NAME = "tile_name"
GEOMETRY = "geometry"

DEFAULT_CRS = CRS.from_epsg(4326)
QUARTILES = "quartiles"
DATETIME_PRECISION = "datetime_precision"
PREC_DATE = "date"
PREC_WEEK = "week"
PREC_MONTH = "month"
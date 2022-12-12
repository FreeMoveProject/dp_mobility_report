# code to create a tessellation within a certain boundary (using the scikit-mobility package)

from skmob import tessellation
from skmob.tessellation import tilers
from shapely.geometry import Polygon
import geopandas as gpd

###################### INPUT PARAMETERS ############################
# set boundaries of tessellation in latitude and longitude
# (this example is Beijing center)
MIN_LNG = 116.08
MAX_LNG = 116.69
MIN_LAT = 39.66
MAX_LAT = 40.27
TILE_DIAMETER_IN_METERS = 1000 # approximately. For h3 the most appropriate resolution is found 
TILE_TYPE = "h3_tessellation"  # other option: "squared"
OUTPUT_PATH = "tessellation.geojson"
####################################################################

base_shape = gpd.GeoDataFrame(
    index=[0],
    crs=4326,
    geometry=[
        Polygon(
            zip(
                [MIN_LNG, MAX_LNG, MAX_LNG, MIN_LNG],
                [MIN_LAT, MIN_LAT, MAX_LAT, MAX_LAT],
            )
        )
    ],
)

tessellation = tilers.tiler.get(
    TILE_TYPE,
    base_shape=base_shape,
    meters=TILE_DIAMETER_IN_METERS,
)
tessellation.rename(columns=dict(tile_ID="tile_id"), inplace=True)
tessellation.to_file(OUTPUT_PATH)

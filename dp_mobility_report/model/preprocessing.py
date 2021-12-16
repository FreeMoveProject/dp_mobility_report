import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


def preprocess_tessellation(tessellation):
    if "tile_id" not in tessellation.columns:
        raise Exception("Column tile_id must be present in tesesllation.")
    tessellation.loc[:, "tile_id"] = tessellation.tile_id.astype(str)
    if "tile_name" not in tessellation.columns:
        tessellation.loc[:, "tile_name"] = tessellation.tile_id
    if "geometry" not in tessellation.columns:
        raise Exception("Column 'geometry' must be present in tessellation.")
    if type(tessellation) is not gpd.GeoDataFrame:
        try:
            tessellation = gpd.GeoDataFrame(
                tessellation, geometry="geometry", crs="EPSG:4326"
            )
        except:
            raise Exception(
                "Tessellation cannot be cast to a geopandas.GeoDataFrame."
            ) from ex

    return tessellation[["tile_id", "tile_name", "geometry"]]


def preprocess_data(df, tessellation, extra_var, max_trips_per_user, user_privacy):
    if "uid" not in df.columns:
        raise Exception("Column 'uid' must be present in data.")
    if "tid" not in df.columns:
        raise Exception("Column 'tid' must be present in data.")
    if "lat" not in df.columns:
        raise Exception("Column 'lat' must be present in data.")
    if "lng" not in df.columns:
        raise Exception("Column 'lng' must be present in data.")
    if "datetime" not in df.columns:
        raise Exception("Column 'datetime' must be present in data.")
    df.loc[:, "id"] = range(0, len(df))

    # make sure trip ids are unique
    df["tid"] = df.groupby(["uid", "tid"]).ngroup()

    # remove unnessessary columns
    columns = ["id", "uid", "tid", "datetime", "lat", "lng"]
    if extra_var is not None:
        columns.append(extra_var)
    if "tile_id" in df.columns:
        columns.append("tile_id")

    df = df.loc[:, columns]

    # TODO: check format before converting (catch error)
    df.loc[:, "datetime"] = pd.to_datetime(df.datetime)
    df.loc[:, "hour"] = df.datetime.dt.hour
    df.loc[:, "is_weekend"] = np.select(
        [df.datetime.dt.weekday > 4], ["weekend"], default="weekday"
    )

    # remove waypoints
    df = df.sort_values("datetime").groupby("tid").nth([0, -1])
    df.reset_index(inplace=True)

    # assign start and end as point_type
    df["point_type"] = "start"
    df.sort_values("datetime", inplace=True)
    df.loc[df.groupby("tid")["point_type"].tail(1).index, "point_type"] = "end"

    # if tile assignment isn't already provided, recompute assignment (TODO: should always be computed (?))
    if "tile_id" not in df.columns:
        df = assign_points_to_tessellation(df, tessellation)

    df = sample_trips(df, max_trips_per_user, user_privacy)
    df.tile_id = df.tile_id.astype(str)
    return df


def assign_points_to_tessellation(df, tessellation):
    # TODO: add progress bar
    gdf = gpd.GeoDataFrame(
        df, geometry=[Point(xy) for xy in zip(df.lng, df.lat)], crs="EPSG:4326"
    )

    # this take some time
    df = gpd.sjoin(
        tessellation[["tile_id", "tile_name", "geometry"]], gdf, how="right"
    )  # Spatial join Points to polygons
    df.drop(["index_left", "geometry"], axis=1, inplace=True)
    return pd.DataFrame(df)


# TODO: speed up (parrellelize?)
def sample_trips(df, max_trips_per_user, user_privacy):
    if user_privacy == True:
        tid_sample = (
            df[["uid", "tid"]]
            .drop_duplicates("tid")
            .groupby("uid")
            .tid.apply(
                lambda x: np.random.choice(
                    x,
                    size=(
                        max_trips_per_user if max_trips_per_user < len(x) else len(x)
                    ),
                    replace=False,
                )
            )
        )
        return df.loc[
            df.tid.isin(np.concatenate(tid_sample.values)),
        ]
    else:
        return df

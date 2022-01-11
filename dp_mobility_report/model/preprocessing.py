import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from dp_mobility_report import constants as const


def preprocess_tessellation(tessellation: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if const.TILE_ID not in tessellation.columns:
        raise ValueError("Column tile_id must be present in tesesllation.")
    tessellation.loc[:, const.TILE_ID] = tessellation.tile_id.astype(str)
    if const.TILE_NAME not in tessellation.columns:
        tessellation.loc[:, const.TILE_NAME] = tessellation.tile_id
    if const.GEOMETRY not in tessellation.columns:
        raise ValueError("Column 'geometry' must be present in tessellation.")
    if type(tessellation) is not gpd.GeoDataFrame:
        try:
            tessellation = gpd.GeoDataFrame(
                tessellation, geometry="geometry", crs=const.DEFAULT_CRS
            )
        except TypeError as ex:
            raise TypeError(
                "Tessellation cannot be cast to a geopandas.GeoDataFrame."
            ) from ex
    if tessellation.crs != const.DEFAULT_CRS:
        tessellation.to_crs(const.DEFAULT_CRS, inplace=True)
    return tessellation[[const.TILE_ID, const.TILE_NAME, const.GEOMETRY]]


def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if const.UID not in df.columns:
        raise ValueError("Column 'uid' must be present in data.")
    df[const.UID] = df[const.UID].astype(str)
    if const.TID not in df.columns:
        raise ValueError("Column 'tid' must be present in data.")
    if const.LAT not in df.columns:
        raise ValueError("Column 'lat' must be present in data.")
    if not pd.core.dtypes.common.is_float_dtype(df[const.LAT]):
        raise TypeError("Column 'lat' is not of type float.")
    if const.LNG not in df.columns:
        raise ValueError("Column 'lng' must be present in data.")
    if not pd.core.dtypes.common.is_float_dtype(df[const.LNG]):
        raise TypeError("Column 'lng' is not of type float.")
    if const.DATETIME not in df.columns:
        raise ValueError("Column 'datetime' must be present in data.")
    try:
        df.loc[:, const.DATETIME] = pd.to_datetime(df[const.DATETIME])
    except Exception as ex:
        raise TypeError("Column 'datetime' cannot be cast to datetime.") from ex
    return df


def preprocess_data(
    df: pd.DataFrame,
    tessellation: gpd.GeoDataFrame,
    max_trips_per_user: int,
    user_privacy: bool,
) -> pd.DataFrame:
    df = _validate_columns(df)

    df.loc[:, const.ID] = range(0, len(df))

    # make sure trip ids are unique and ordered correctly
    df[const.TID] = (
        df.sort_values([const.UID, const.DATETIME])
        .groupby([const.UID, const.TID], sort=False)
        .ngroup()
    )

    # remove unnessessary columns
    columns = [const.ID, const.UID, const.TID, const.DATETIME, const.LAT, const.LNG]
    if const.TILE_ID in df.columns:
        columns.append(const.TILE_ID)
    df = df.loc[:, columns]

    # create time related variables
    df.loc[:, const.HOUR] = df[const.DATETIME].dt.hour
    df.loc[:, const.IS_WEEKEND] = np.select(
        [df[const.DATETIME].dt.weekday > 4], ["weekend"], default="weekday"
    )

    # remove waypoints
    df = df.sort_values(const.DATETIME).groupby(const.TID).nth([0, -1])
    df.reset_index(inplace=True)

    # assign start and end as point_type
    df[const.POINT_TYPE] = "start"
    df.sort_values(const.DATETIME, inplace=True)
    df.loc[
        df.groupby(const.TID)[const.POINT_TYPE].tail(1).index, const.POINT_TYPE
    ] = "end"

    # if tile assignment isn't already provided, recompute assignment
    if const.TILE_ID not in df.columns:
        df = assign_points_to_tessellation(df, tessellation)
    else:
        logging.info(
            "'tile_id' present in data. No new assignment of points to tessellation."
        )
        df.tile_id = df.tile_id.astype(str)

    df = sample_trips(df, max_trips_per_user, user_privacy)
    return df


def assign_points_to_tessellation(
    df: pd.DataFrame, tessellation: gpd.GeoDataFrame
) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df[const.LNG], df[const.LAT])],
        crs="EPSG:4326",
    )

    # Spatial join points to polygons
    df = gpd.sjoin(
        tessellation[[const.TILE_ID, const.TILE_NAME, const.GEOMETRY]],
        gdf,
        how="right",
    )
    df.drop(["index_left", const.GEOMETRY], axis=1, inplace=True)
    return pd.DataFrame(df)


def sample_trips(
    df: pd.DataFrame, max_trips_per_user: int, user_privacy: bool
) -> pd.DataFrame:
    if user_privacy:
        tid_sample = (
            df[[const.UID, const.TID]]
            .drop_duplicates(const.TID)
            .groupby(const.UID)[const.TID]
            .apply(
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
            df[const.TID].isin(np.concatenate(tid_sample.values)),
        ]
    else:
        return df

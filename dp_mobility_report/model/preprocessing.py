import logging
import warnings
from typing import Any, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely.geometry import Point

from dp_mobility_report import constants as const


def has_points_inside_tessellation(
    df: pd.DataFrame, tessellation: Optional[GeoDataFrame]
) -> bool:
    if tessellation is None:
        return True

    return not all(df[const.TILE_ID].isna())


def validate_inclusion_exclusion(
    analysis_selection: Optional[List[str]], analysis_exclusion: Optional[List[str]]
) -> Tuple[Optional[List[str]], Optional[List[str]]]:

    if analysis_selection is not None and analysis_exclusion is not None:
        warnings.warn(
            "The parameter `analysis_exclusion' will be ignored, as `analysis_selection' is set as well."
        )
        analysis_exclusion = None

    if analysis_selection == [const.ALL]:
        warnings.warn(
            "['all'] as input is deprecated. Use 'None' (default) instead to include all analyses."
        )
        analysis_selection = None

    return analysis_selection, analysis_exclusion


def validate_input(
    df: DataFrame,
    tessellation: Optional[GeoDataFrame],
    privacy_budget: Optional[Union[int, float]],
    max_trips_per_user: Optional[int],
    analysis_selection: Optional[List[str]],
    analysis_exclusion: Optional[List[str]],
    budget_split: dict,
    disable_progress_bar: bool,
    evalu: bool,
    user_privacy: bool,
    timewindows: Union[List[int], np.ndarray],
    max_travel_time: Optional[int],
    bin_range_travel_time: Optional[int],
    max_jump_length: Optional[Union[int, float]],
    bin_range_jump_length: Optional[Union[int, float]],
    max_radius_of_gyration: Optional[Union[int, float]],
    bin_range_radius_of_gyration: Optional[Union[int, float]],
    max_user_tile_count: Optional[int],
    bin_range_user_tile_count: Optional[int],
    max_user_time_delta: Optional[Union[int, float]],
    bin_range_user_time_delta: Optional[Union[int, float]],
    seed_sampling: Optional[int],
) -> None:
    if not isinstance(df, DataFrame):
        raise TypeError("'df' is not a Pandas DataFrame.")

    if tessellation is None:
        warnings.warn(
            "No tessellation has been specified. All analyses based on the tessallation will be omitted."
        )

    if not ((tessellation is None) or isinstance(tessellation, GeoDataFrame)):
        raise TypeError("'tessellation' is not a Geopandas GeoDataFrame.")

    if not ((max_trips_per_user is None) or isinstance(max_trips_per_user, int)):
        raise TypeError("'max_trips_per_user' is not numeric.")
    if (max_trips_per_user is not None) and (max_trips_per_user < 1):
        raise ValueError("'max_trips_per_user' has to be greater 0.")

    if analysis_selection is not None:
        if not isinstance(analysis_selection, list):
            raise TypeError("'analysis_selection' is not a list.")

        if not set(analysis_selection).issubset(
            const.SEGMENTS_AND_ELEMENTS + [const.ALL]
        ):
            raise ValueError(
                f"Unknown analyses in {analysis_selection}. Only elements from {const.SEGMENTS_AND_ELEMENTS} are valid inputs."
            )

    if analysis_exclusion is not None:
        if not isinstance(analysis_exclusion, list):
            raise TypeError("'analysis_exclusion' is not a list.")
        if not set(analysis_exclusion).issubset(const.SEGMENTS_AND_ELEMENTS):
            raise ValueError(
                f"Unknown analyses in {analysis_exclusion}. Only elements from {const.SEGMENTS_AND_ELEMENTS} are valid inputs."
            )

    if not isinstance(budget_split, dict):
        raise TypeError("'budget_split' is not a dict.")
    if not set(budget_split.keys()).issubset(const.ELEMENTS):
        raise ValueError(
            f"Unknown analyses in {budget_split}. Only elements from {const.ELEMENTS} are valid inputs as dictionary keys."
        )
    if not all(isinstance(x, int) for x in list(budget_split.values())):
        raise ValueError(
            f"Not all elements in 'budget_split' are integers: {list(budget_split.values())}."
        )

    if not isinstance(timewindows, (list, np.ndarray)):
        raise TypeError("'timewindows' is not a list or a numpy array.")

    timewindows = (
        np.array(timewindows) if isinstance(timewindows, list) else timewindows
    )
    if not all([np.issubdtype(item, int) for item in timewindows]):
        raise TypeError("not all items of 'timewindows' are integers.")

    _validate_bool(user_privacy, f"{user_privacy=}".split("=")[0])
    _validate_bool(evalu, f"{user_privacy=}".split("=")[0])
    _validate_bool(disable_progress_bar, f"{user_privacy=}".split("=")[0])

    if privacy_budget is not None:
        _validate_numeric_greater_zero(
            privacy_budget, f"{privacy_budget=}".split("=")[0]
        )
        if (max_trips_per_user is None) & user_privacy:
            warnings.warn(
                "Input parameter `max_trips_per_user` is `None` even though a privacy budget is given. The actual maximum number of trips per user will be used according to the data, though this violates user-level Differential Privacy."
            )

    _validate_int_greater_zero(max_travel_time, f"{max_travel_time=}".split("=")[0])
    _validate_int_greater_zero(
        bin_range_travel_time, f"{bin_range_travel_time=}".split("=")[0]
    )
    _validate_numeric_greater_zero(max_jump_length, f"{max_jump_length=}".split("=")[0])
    _validate_numeric_greater_zero(
        bin_range_jump_length, f"{bin_range_jump_length=}".split("=")[0]
    )
    _validate_numeric_greater_zero(
        max_radius_of_gyration, f"{max_radius_of_gyration=}".split("=")[0]
    )
    _validate_numeric_greater_zero(
        bin_range_radius_of_gyration, f"{bin_range_radius_of_gyration=}".split("=")[0]
    )
    _validate_int_greater_zero(
        max_user_tile_count, f"{max_user_tile_count=}".split("=")[0]
    )
    _validate_int_greater_zero(
        bin_range_user_tile_count, f"{bin_range_user_tile_count=}".split("=")[0]
    )
    _validate_numeric_greater_zero(
        max_user_time_delta, f"{max_user_time_delta=}".split("=")[0]
    )
    _validate_numeric_greater_zero(
        bin_range_user_time_delta, f"{bin_range_user_time_delta=}".split("=")[0]
    )

    if not ((seed_sampling is None) or isinstance(seed_sampling, int)):
        raise TypeError("'seed_sampling' is not an integer.")
    if (seed_sampling is not None) and (seed_sampling <= 0):
        raise ValueError("'seed_sampling' has to be greater 0.")


def _validate_int_greater_zero(var: Any, name: str) -> None:
    if not ((var is None) or isinstance(var, int)):
        raise TypeError(f"{name} is not an int.")
    if (var is not None) and (var <= 0):
        raise ValueError(f"'{name}' has to be greater 0.")


def _validate_numeric_greater_zero(var: Any, name: str) -> None:
    if not ((var is None) or isinstance(var, (int, float))):
        raise TypeError(f"{name} is not numeric.")
    if (var is not None) and (var <= 0):
        raise ValueError(f"'{name}' has to be greater 0.")


def _validate_bool(var: Any, name: str) -> None:
    if not isinstance(var, bool):
        raise TypeError(f"'{name}' is not type boolean.")


# unify different input options of analyses (segments and elements) to be included / excluded as excluded elements, i.e., 'analysis_exclusion'
def clean_analysis_exclusion(
    analysis_selection: Optional[List[str]],
    analysis_exclusion: Optional[List[str]],
    has_tessellation: bool,
    has_points_inside_tessellation: bool,
    has_timestamps: bool,
    has_od_flows: bool,
    has_consecutive_user_trips: bool,
) -> List[str]:
    def _remove_elements(elements: list, remove_list: list) -> list:
        return [e for e in elements if e not in remove_list]

    if analysis_selection is not None:
        analysis_exclusion = const.ELEMENTS

        # remove all elements of respective segments from analysis_exclusion that are given in analysis_selection
        if const.OVERVIEW in analysis_selection:
            analysis_exclusion = _remove_elements(
                analysis_exclusion, const.OVERVIEW_ELEMENTS
            )
        if const.PLACE_ANALYSIS in analysis_selection:
            analysis_exclusion = _remove_elements(
                analysis_exclusion, const.PLACE_ELEMENTS
            )
        if const.OD_ANALYSIS in analysis_selection:
            analysis_exclusion = _remove_elements(analysis_exclusion, const.OD_ELEMENTS)
        if const.USER_ANALYSIS in analysis_selection:
            analysis_exclusion = _remove_elements(
                analysis_exclusion, const.USER_ELEMENTS
            )

        # remove single elements
        analysis_exclusion = _remove_elements(analysis_exclusion, analysis_selection)

    elif analysis_exclusion is not None:
        # deduplicate list in case there are duplicates as input (otherwise `remove` might fail)
        analysis_exclusion = list(set(analysis_exclusion))

        if const.OVERVIEW in analysis_exclusion:
            analysis_exclusion += const.OVERVIEW_ELEMENTS
            analysis_exclusion.remove(const.OVERVIEW)

        if const.PLACE_ANALYSIS in analysis_exclusion:
            analysis_exclusion += const.PLACE_ELEMENTS
            analysis_exclusion.remove(const.PLACE_ANALYSIS)

        if const.OD_ANALYSIS in analysis_exclusion:
            analysis_exclusion += const.OD_ELEMENTS
            analysis_exclusion.remove(const.OD_ANALYSIS)

        if const.USER_ANALYSIS in analysis_exclusion:
            analysis_exclusion += const.USER_ELEMENTS
            analysis_exclusion.remove(const.USER_ANALYSIS)

    else:
        analysis_exclusion = []

    if not has_tessellation:
        # warning in validation
        analysis_exclusion += const.TESSELLATION_ELEMENTS

    if (has_tessellation) & (not has_points_inside_tessellation):
        analysis_exclusion += const.TESSELLATION_ELEMENTS
        warnings.warn(
            "No records are within the given tessellation. All analyses based on the tessellation will be excluded."
        )

    if not has_timestamps:
        # warning in validation
        analysis_exclusion += const.TIMESTAMP_ANALYSES

    if not has_od_flows:
        warnings.warn(
            "There are only incomplete trips, i.e., no trips with more than a single record. OD analyses cannot be conducted, thus they are excluded from the report."
        )
        analysis_exclusion += const.OD_ELEMENTS

    if (const.USER_TIME_DELTA not in analysis_exclusion) & (
        not has_consecutive_user_trips
    ):
        warnings.warn(
            "No user has more than one trip (this is also the case, if max_trips_per_user=1). No analysis of consecutive trips of a user can be conducted and will thus be excluded."
        )
        analysis_exclusion += [const.USER_TIME_DELTA]

    # deduplicate in case analyses and segments were included
    analysis_exclusion = list(set(analysis_exclusion))

    # sort according to order
    analysis_exclusion.sort(key=lambda i: const.ELEMENTS.index(i))

    return analysis_exclusion


def clean_budget_split(budget_split: dict, analysis_exclusion: List[str]) -> dict:
    intersec = set(budget_split.keys()).intersection(analysis_exclusion)
    if len(intersec) != 0:
        warnings.warn(
            f"A `budget_split`is specified for the analyses {intersec} even though they are excluded."
            "As they will be excluded, the `budget_split` specification will be ignored for these analyses."
        )

    # remove all analyses that are excluded according to `analysis_exclusion``
    remaining_analyses = set(budget_split.keys()) - set(analysis_exclusion)
    budget_split = {analysis: budget_split[analysis] for analysis in remaining_analyses}
    return budget_split


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
    if pd.core.dtypes.common.is_integer_dtype(df[const.DATETIME]):
        warnings.warn(
            "Column 'datetime' consists of type int. It is only interpreted as sequence information and thus all time related analyses are omitted."
        )
    else:
        try:
            df[const.DATETIME] = pd.to_datetime(df[const.DATETIME])
        except Exception as ex:
            raise TypeError("Column 'datetime' cannot be cast to datetime.") from ex
    return df


def preprocess_data(
    df: pd.DataFrame,
    tessellation: Optional[gpd.GeoDataFrame],
    max_trips_per_user: int,
    user_privacy: bool,
    seed: Optional[int],
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

    # only create time realted variables if df has timestamps
    if pd.core.dtypes.common.is_datetime64_dtype(df[const.DATETIME]):
        # create time related variables
        df.loc[:, const.HOUR] = df[const.DATETIME].dt.hour
        df.loc[:, const.IS_WEEKEND] = np.select(
            [df[const.DATETIME].dt.weekday > 4], ["weekend"], default="weekday"
        )

    # remove waypoints
    df = df.sort_values(const.DATETIME).groupby(const.TID, as_index=False).nth([0, -1])

    # assign start and end as point_type
    df[const.POINT_TYPE] = "start"
    df.sort_values(const.DATETIME, inplace=True)
    df.loc[
        df.groupby(const.TID)[const.POINT_TYPE].tail(1).index, const.POINT_TYPE
    ] = "end"

    # if tile assignment isn't already provided, recompute assignment
    if tessellation is not None:
        if const.TILE_ID not in df.columns:
            df = assign_points_to_tessellation(df, tessellation)
        else:
            logging.info(
                "'tile_id' present in data. No new assignment of points to tessellation."
            )
            df[const.TILE_ID] = df[const.TILE_ID].astype(str)

    df = sample_trips(df, max_trips_per_user, user_privacy, seed)
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
    df: pd.DataFrame,
    max_trips_per_user: int,
    user_privacy: bool,
    seed: Optional[int],
) -> pd.DataFrame:
    if user_privacy:
        if seed is not None:
            np.random.seed(seed)
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

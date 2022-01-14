from typing import List, Optional, Tuple, Union

import numpy as np
from haversine import haversine
from pandas import Series

from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def haversine_dist(coords: List[float]) -> float:
    # coords: provide coordinates as lat_start, lng_start, lat_end, lng_end
    return haversine((coords[0], coords[1]), (coords[2], coords[3]))


def cut_outliers(
    data: Series,
    min_value: Optional[Union[float, int]] = None,
    max_value: Optional[Union[float, int]] = None,
) -> Tuple:
    n = len(data)
    if (min_value is not None) and (max_value is not None) and (min_value > max_value):
        raise ValueError("min_value cannot be larger than max_value")
    if min_value is not None:
        data = data[data >= min_value]
    if max_value is not None:
        data = data[data <= max_value]
    outlier_count = n - len(data)
    return (data, outlier_count)


def hist_section(
    series: Union[np.ndarray, Series],
    eps: Optional[float],
    sensitivity: int,
    min_value: Optional[Union[float, int]] = None,
    max_value: Optional[Union[float, int]] = None,
    bin_range: Optional[Union[float, int]] = None,
    max_bins: Optional[int] = None,
    evalu: bool = False,
) -> Section:
    if evalu or eps is None:
        epsi = eps
        epsi_quart = eps
    else:
        epsi = eps / 7
        epsi_quart = epsi * 5

    series = Series(series) if isinstance(series, np.ndarray) else series

    if (min_value is not None) or (max_value is not None):
        series, n_outliers = cut_outliers(
            series, min_value=min_value, max_value=max_value
        )
        dp_n_outliers = diff_privacy.count_dp(n_outliers, epsi, sensitivity)
    else:
        dp_n_outliers = None

    quartiles = diff_privacy.quartiles_dp(series, epsi_quart, sensitivity)
    # TODO: or always diff private min and max values? (outliers are already cut)
    min_value = quartiles["min"] if min_value is None else min_value
    max_value = quartiles["max"] if max_value is None else max_value

    if np.issubdtype(series.dtype, np.integer) and (max_value - min_value < 10):
        min_value = int(min_value)
        max_value = int(max_value)
        dp_values = np.array(range(min_value, max_value + 1))
        counts = np.bincount(series)[min_value : max_value + 1]
    else:
        if bin_range is not None:
            max_bins = int((max_value - min_value) / bin_range)
            max_bins = max_bins if max_bins > 2 else 2
        elif max_bins is None:
            max_bins = 10  # set default of 10
        hist = np.histogram(series, bins=max_bins, range=(min_value, max_value))
        counts = hist[0]
        dp_values = hist[1]
    dp_counts = diff_privacy.counts_dp(counts, epsi, sensitivity)

    return Section(
        data=(dp_counts, dp_values),
        privacy_budget=eps,
        n_outliers=dp_n_outliers,
        quartiles=quartiles,
    )

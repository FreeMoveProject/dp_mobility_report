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
    evalu: bool = False,
) -> Section:
    epsi = get_epsi(evalu, eps, 7) # TODO: does eps need to be split between counts and outliers? (As outliers is like an extra bin)
    epsi_quant = epsi * 5 if epsi is not None else None

    series = Series(series) if isinstance(series, np.ndarray) else series

    if max_value is not None:
        series, n_outliers = cut_outliers(series, max_value=max_value)
        dp_n_outliers = diff_privacy.count_dp(n_outliers, epsi, sensitivity)
    else:
        dp_n_outliers = None

    quartiles, moe_expmech = diff_privacy.quartiles_dp(series, epsi_quant, sensitivity)
    # cut series again according to diff. priv. quartiles to that min and max values match the histogram
    series, _ = cut_outliers(
        series, min_value=quartiles["min"], max_value=quartiles["max"]
    )

    if np.issubdtype(series.dtype, np.integer) and (
        quartiles["max"] - quartiles["min"] < 10
    ):
        min_value = int(quartiles["min"])
        max_value = int(quartiles["max"])
        dp_values = np.array(range(min_value, max_value + 1))
        counts = np.bincount(series, minlength=len(dp_values))[
            min_value : max_value + 1
        ]
    else:
        # make sure min and max_value are not None
        min_value = quartiles["min"] if min_value is None else min_value
        max_value = quartiles["max"] if max_value is None else max_value

        # if range is small set bin_range to 0.1 to prevent too fine-granular bins
        if bin_range is None and (max_value - min_value) < 1:
            bin_range = 0.1

        if bin_range is not None:
        # if bin range and bounds are provided by user, use those for clean bin sizes (but remove bins according to dp_min and dp_max values)
            min_value = (
                int((quartiles["min"] - min_value) / bin_range) * bin_range + min_value
            )
            max_value = (
                max_value - int((max_value - quartiles["max"]) / bin_range) * bin_range
            )
            max_bins = int((max_value - min_value) / bin_range)
            max_bins = max_bins if max_bins > 2 else 2
        else:
            # else if no defined bin range, set bounds of hist according to dp_min and dp_max
            min_value = quartiles["min"]
            max_value = quartiles["max"]
            max_bins = 10  # set default of 10
        hist = np.histogram(series, bins=max_bins, range=(min_value, max_value))
        counts = hist[0]
        dp_values = hist[1]
    dp_counts = diff_privacy.counts_dp(counts, epsi, sensitivity)
    moe_laplace = diff_privacy.laplace_margin_of_error(0.95, epsi, sensitivity)

    return Section(
        data=(dp_counts, dp_values),
        privacy_budget=eps,
        n_outliers=dp_n_outliers,
        quartiles=quartiles,
        margin_of_error_laplace=moe_laplace,
        margin_of_error_expmech=moe_expmech
    )


def get_epsi(evalu: bool, eps: Optional[float], elements: int) -> Optional[float]:
    if evalu or eps is None:
        return eps
    else:
        return eps / elements

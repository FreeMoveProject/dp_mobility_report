import math
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from haversine import haversine
from pandas import DataFrame, Series

from dp_mobility_report.model.section import TupleSection
from dp_mobility_report.privacy import diff_privacy


def haversine_dist(coords: List[float]) -> float:
    # coords: provide coordinates as lat_start, lng_start, lat_end, lng_end
    return haversine((coords[0], coords[1]), (coords[2], coords[3]))


def _round_up(n: Union[float, int], decimals: int = 0) -> float:
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier


def _round_down(n: Union[float, int], decimals: int = 0) -> float:
    multiplier = 10**decimals
    return math.floor(n * multiplier) / multiplier


def cut_outliers(
    data: Series,
    min_value: Optional[Union[float, int]] = None,
    max_value: Optional[Union[float, int]] = None,
) -> Tuple:
    if (min_value is not None) and (max_value is not None) and (min_value > max_value):
        raise ValueError("min_value cannot be larger than max_value")
    if min_value is not None:
        data = data[data >= min_value]
    if max_value is not None:
        data = data[data <= max_value]
    return data


def hist_section(
    series: Union[np.ndarray, Series],
    eps: Optional[float],
    sensitivity: int,
    hist_max: Optional[Union[float, int]] = None,
    hist_min: Optional[Union[float, int]] = None,
    bin_range: Optional[Union[float, int]] = None,
    bin_type: Type = float,
    evalu: bool = False,
) -> TupleSection:
    epsi = get_epsi(evalu, eps, 6)
    epsi_quant = epsi * 5 if epsi is not None else None

    series = Series(series) if isinstance(series, np.ndarray) else series

    quartiles, moe_expmech = diff_privacy.quartiles_dp(series, epsi_quant, sensitivity)
    series = cut_outliers(
        series, min_value=quartiles["min"], max_value=quartiles["max"]
    )

    # determine bins for histogram
    # hist_min and hist_max determine how the output histogram looks like

    # max value of histogram: either given as input or determined by dp max
    hist_max = hist_max if (hist_max is not None) else _round_up(quartiles["max"], 3)

    # min value of histogram: either given as input or determined by dp min
    hist_min = hist_min if (hist_min is not None) else _round_down(quartiles["min"], 3)

    # if all values above defined max, create one histogram bin greater hist_max
    if hist_min > hist_max:
        bins = np.array([hist_max, np.inf])
        counts = np.array([len(series)])

    # if there are max. 10 integers, use value counts of single integers instead of bin ranges for histogram
    elif (bin_type is int) and (
        (bin_range == 1) or ((bin_range is None) and (hist_max - hist_min < 10))
    ):
        bins = np.array(range(int(quartiles["min"]), int(quartiles["max"]) + 1))
        counts = np.bincount(series, minlength=int(quartiles["max"]) + 1)[
            int(quartiles["min"]) :
        ]

        # sum all counts above hist max to single bin >max
        if bins[-1] > hist_max:
            bins = bins[bins <= hist_max]
            bins = np.append(bins, np.Inf)
            counts[len(bins) - 1] = counts[
                len(bins) - 1 :
            ].sum()  # sum up to single > max count
            counts = counts[: len(bins)]  # remove long tail

    # else use ranges for bins to create histogram
    else:
        # if hist range is small (<1), set bin_range to 0.1 to prevent too fine-granular bins
        if bin_range is None and (hist_max - hist_min) < 1:
            bin_range = 0.1

        # if bin_range is provided, snap min and max value accordingly(to create "clean" bins).
        if bin_range is not None:
            # "snap" min_value: e.g., if bin range is 5 and dp min value is 12, min_value snaps to 10
            hist_min = hist_min - (hist_min % bin_range)

        # set default of 10 bins if no bin_range is given.
        # compute bin_range from hist_max and snap max_value accordingly. This is necessary, bc hist_max and dp max might differ:
        # to create 10 bins up to hist_max but maintain bins up to dp max (to aggregate them in the following step)
        else:
            bin_range = (hist_max - hist_min) / 10

        # "snap" hist_max_input to bin_range (for pretty hist bins): E.g., if bin range is 5, and the dp max value is 23, max_value snaps to 25
        hist_max_input = (
            hist_max if hist_max > quartiles["max"] else _round_up(quartiles["max"], 3)
        )
        hist_max_input = (
            hist_max_input
            if round((hist_max_input - hist_min) % bin_range, 2) == 0
            else hist_max_input
            + (bin_range - ((hist_max_input - hist_min) % bin_range))
        )
        max_bins = int((hist_max_input - hist_min) / bin_range)
        max_bins = (
            max_bins if max_bins > 2 else 2
        )  # make sure there are at least 2 bins for histogram function to work

        hist = np.histogram(series, bins=max_bins, range=(hist_min, hist_max_input))
        counts = hist[0]
        bins = hist[1].astype(bin_type)

        # sum all counts above hist max to single bin >max
        if bins[-1] > hist_max:
            bins = bins[bins <= hist_max]
            counts[len(bins) - 1] = counts[
                len(bins) - 1 :
            ].sum()  # sum up to single > max count
            counts = counts[: len(bins)]  # remove long tail
            bins = np.append(bins, np.Inf)

        # as hist_min is currently only used for mobility_entropy with min_value = 0, we dont need to sum all counts below min accordingly
        # to be generically applicable, this would be needed!

    dp_counts = diff_privacy.counts_dp(counts, epsi, sensitivity)

    # set counts above dp_max(i.e, quartiles["max"]) to 0 (only so that bins are shown according to user input, even if they are empty)
    # if bin is "inf" it should be maintained as it is the aggregation of everything > hist_max and <= quartiles["max"]
    temp_bins_max = bins if len(bins) == len(dp_counts) else bins[:-1]
    dp_counts[
        np.logical_and(
            (temp_bins_max > quartiles["max"]), (np.not_equal(temp_bins_max, np.inf))
        )
    ] = 0
    # set counts below dp_min(i.e, quartiles["min"]) to 0
    # only needed for mobility_entropy (to show all possible bins even if below DP min)
    temp_bins_min = bins if len(bins) == len(dp_counts) else bins[1:]
    dp_counts[temp_bins_min < quartiles["min"]] = 0

    moe_laplace = diff_privacy.laplace_margin_of_error(0.95, epsi, sensitivity)

    # as percent instead of counts
    trip_counts = sum(dp_counts)
    if trip_counts > 0:
        dp_counts = dp_counts / trip_counts * 100
        moe_laplace = moe_laplace / trip_counts * 100

    return TupleSection(
        data=(dp_counts, bins),
        privacy_budget=eps,
        quartiles=quartiles,
        margin_of_error_laplace=moe_laplace,
        margin_of_error_expmech=moe_expmech,
    )


def get_epsi(evalu: bool, eps: Optional[float], elements: int) -> Optional[float]:
    if evalu or eps is None:
        return eps
    else:
        return eps / elements


def _cumsum(array: np.array) -> np.array:
    array[::-1].sort()
    if array.sum() == 0:
        return array
    return (array.cumsum() / array.sum()).round(2)


def cumsum(
    counts: np.array, eps: float, sensitivity: int, nsim: int = 10, nrow: int = 100
) -> DataFrame:
    df_cumsum = DataFrame()
    df_cumsum["n"] = np.arange(1, len(counts) + 1)
    df_cumsum["cum_perc"] = _cumsum(counts)

    # reuduce df size by only keeping max 100 values
    if len(df_cumsum) > nrow:
        nth = len(df_cumsum) // nrow
        last_row = df_cumsum.iloc[len(df_cumsum) - 1, :]
        df_cumsum = df_cumsum.iloc[::nth, :]
        # append last row
        if int(last_row.n) not in list(df_cumsum.n):
            pd.concat([df_cumsum, pd.DataFrame(last_row).T])

    df_cumsum.reset_index(drop=True, inplace=True)
    return df_cumsum

from typing import List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from haversine import haversine
from pandas import DataFrame, Series

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
    bin_range: Optional[Union[float, int]] = None,
    bin_type: Type = float,
    evalu: bool = False,
) -> Section:
    epsi = get_epsi(evalu, eps, 6)
    epsi_quant = epsi * 5 if epsi is not None else None

    series = Series(series) if isinstance(series, np.ndarray) else series
    quartiles, moe_expmech = diff_privacy.quartiles_dp(series, epsi_quant, sensitivity)
    series = cut_outliers(
        series, min_value=quartiles["min"], max_value=quartiles["max"]
    )

    # determine bins for histogram

    # max value of histogram: either given as input or determined by dp max
    hist_max = (
        hist_max
        if (hist_max is not None and hist_max < quartiles["max"])
        else quartiles["max"]
    )

    # if there are less than 10 integers, use value counts of single integers instead of bin ranges for histogram
    if (
        (bin_range is None or bin_range == 1)
        and (bin_type is int)
        and (hist_max - quartiles["min"] < 10)
    ):
        min_value = int(quartiles["min"])
        max_value = int(quartiles["max"])
        bins = np.array(range(min_value, max_value + 1))
        counts = np.bincount(series, minlength=len(bins))[min_value : max_value + 1]

        # sum all counts above hist max to single bin >max
        if bins[-1] > hist_max:
            bins = bins[bins <= hist_max]
            counts[len(bins)] = counts[len(bins) :].sum()  # sum up to single >max count
            counts = counts[: len(bins) + 1]  # remove long tail
            bins = np.append(bins, np.Inf)

    # else use ranges for bins to create histogram
    else:
        # if hist range is small (<1), set bin_range to 0.1 to prevent too fine-granular bins
        if bin_range is None and (hist_max - quartiles["min"]) < 1:
            bin_range = 0.1

        # if bin_range is provided, snap min and max value accordingly(to create "clean" bins).
        if bin_range is not None:
            # "snap" min_value: e.g., if bin range is 5 and dp min value is 12, min_value snaps to 10
            min_value = quartiles["min"] - (quartiles["min"] % bin_range)

        # set default of 10 bins if no bin_range is given.
        # compute bin_range from hist_max and snap max_value accordingly. This is necessary, bc hist_max and dp max might differ:
        # to create 10 bins up to hist_max but maintain bins up to dp max (to aggregate them in the following step)
        else:
            bin_range = (hist_max - quartiles["min"]) / 10
            min_value = quartiles["min"]

        # "snap" max_value to bin_range: E.g., if bin range is 5, and the dp max value is 23, max_value snaps to 25
        max_value = (
            quartiles["max"]
            if round((quartiles["max"] - min_value) % bin_range, 2) == 0
            else quartiles["max"]
            + (bin_range - ((quartiles["max"] - min_value) % bin_range))
        )

        max_bins = int((max_value - min_value) / bin_range)
        max_bins = (
            max_bins if max_bins > 2 else 2
        )  # make sure there are at least 2 bins for histogram function to work

        hist = np.histogram(series, bins=max_bins, range=(min_value, max_value))
        counts = hist[0]
        bins = hist[1].astype(bin_type)

        # sum all counts above hist max to single bin >max
        if bins[-1] > hist_max:
            bins = bins[bins <= hist_max]
            counts[len(bins) - 1] = counts[
                len(bins) - 1 :
            ].sum()  # sum up to single >max count
            counts = counts[: len(bins)]  # remove long tail
            bins = np.append(bins, np.Inf)

    dp_counts = diff_privacy.counts_dp(counts, epsi, sensitivity)
    moe_laplace = diff_privacy.laplace_margin_of_error(0.95, epsi, sensitivity)

    # as percent instead of counts
    trip_counts = sum(dp_counts)
    if trip_counts > 0:
        dp_counts = dp_counts / trip_counts * 100
        moe_laplace = moe_laplace / trip_counts * 100

    return Section(
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


def _cumsum(array: np.array):
    array[::-1].sort()
    return (array.cumsum() / array.sum()).round(2)


def cumsum_simulations(
    counts: np.array, eps: float, sensitivity: int, nsim: int = 10, nrow: int = 100
) -> DataFrame:
    df_cumsum = DataFrame()
    df_cumsum["n"] = np.arange(1, len(counts) + 1)

    for i in range(1, nsim):
        sim_counts = diff_privacy.counts_dp(counts, eps, sensitivity)
        df_cumsum["cum_perc_" + str(i)] = _cumsum(sim_counts)

    # once negative values have been used for simulations create cumsum of series without negative values
    df_cumsum["cum_perc"] = _cumsum(diff_privacy.limit_negative_values_to_zero(counts))

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

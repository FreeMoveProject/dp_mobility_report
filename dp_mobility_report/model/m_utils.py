import numpy as np
from haversine import Unit, haversine

from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy

def haversine_dist(coords):
    # coords: provide coordinates as lat_start, lng_start, lat_end, lng_end
    return haversine(
        (float(coords[0]), float(coords[1])),
        (float(coords[2]), float(coords[3]))#,
        #unit=Unit.METERS,
    )

def cut_outliers(data, min_value=None, max_value=None):
    n = len(data)
    if (min_value is not None) and (max_value is not None) and (min_value > max_value):
        raise ValueError('min_value cannot be larger than max_value')
    if min_value is not None:
        data = data[data >= min_value]
    if max_value is not None:
        data = data[data <= max_value]
    outlier_count = n - len(data)
    return (data, outlier_count)


def hist_section(
    series,
    eps,
    sensitivity,
    min_value=None,
    max_value=None,
    bin_range=None,
    max_bins=None,
    evalu=False,
):

    if evalu == True or eps is None:
        epsi = eps
        epsi_quart = eps
    else:
        epsi = eps / 7
        epsi_quart = epsi * 5

    if (min_value is not None) or (max_value is not None):
        series, n_outliers = cut_outliers(
            series, min_value=min_value, max_value=max_value
        )
        dp_n_outliers = diff_privacy.counts_dp(
            n_outliers, epsi, sensitivity, parallel=True, nonzero=False
        )
    else:
        dp_n_outliers = None

    quartiles = diff_privacy.quartiles_dp(series, epsi_quart, sensitivity)
    # TODO: or always diff private min and max values? (outliers are already cut)
    min_value = quartiles["min"] if min_value is None else min_value
    max_value = quartiles["max"] if max_value is None else max_value
    
    if np.issubdtype(series.dtype, np.integer) and (max_value-min_value < 10):
        min_value = int(min_value)
        max_value = int(max_value)
        dp_values = np.array(range(min_value, max_value+1))
        counts = np.bincount(series)[min_value:max_value+1]
    else:
        if bin_range is not None:
            max_bins = int((max_value - min_value) / bin_range)
            max_bins = max_bins if max_bins > 2 else 2
        elif max_bins is None:
            max_bins = 10 #set default of 10
        hist = np.histogram(series, bins=max_bins, range=(min_value, max_value))
        counts = hist[0]
        dp_values = hist[1]
    dp_counts = diff_privacy.counts_dp(
        counts, epsi, sensitivity, parallel=True, nonzero=False
    )

    return Section(
        data=(dp_counts, dp_values),
        privacy_budget=eps,
        n_outliers=dp_n_outliers,
        quartiles=quartiles,
    )

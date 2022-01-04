import numpy as np

from dp_mobility_report.model.section import Section
from dp_mobility_report.privacy import diff_privacy


def cut_outliers(data, z_score_cutoff=None, min_value=None, max_value=None):
    n = len(data)
    if z_score_cutoff is not None:
        z_score = (data - data.mean()) / data.std()
        data = data[abs(z_score) < z_score_cutoff]
    elif (min_value is not None) & (max_value is not None):
        data = data[(data >= min_value) & (data <= max_value)]
    else:
        if min_value is not None:
            data = data[data >= min_value]
        elif max_value is not None:
            data = data[data <= max_value]
        # raise NotImplementedError(
        #    "either min and max value or a zscore_cutoff need to be defined."
        # )
    outlier_count = n - len(data)
    return (data, outlier_count)


def dp_hist_section(
    series,
    eps,
    sensitivity,
    min_value=None,
    max_value=None,
    bin_size=None,
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
    min_value = quartiles["min"]
    max_value = quartiles["max"]

    if bin_size is not None:
        max_bins = int((max_value - min_value) / bin_size)
        max_bins = max_bins if max_bins > 2 else 2
    elif max_bins is None:
        raise Exception("Either max_bins or bin_size needs to be defined")
    hist = np.histogram(series, bins=max_bins, range=(min_value, max_value))
    dp_hist_counts = diff_privacy.counts_dp(
        hist[0], epsi, sensitivity, parallel=True, nonzero=False
    )

    return Section(
        data=(dp_hist_counts, hist[1]),
        n_outliers=dp_n_outliers,
        quartiles=quartiles,
    )


def hist_section(series, min_value=None, max_value=None, bin_size=None, max_bins=None):
    min_value = series.min() if min_value is None else min_value
    max_value = series.min() if min_value is None else min_value

    series, n_outliers = cut_outliers(series, min_value=min_value, max_value=max_value)

    if bin_size is not None:
        max_bins = int((max_value - min_value) / bin_size)
        max_bins = max_bins if max_bins > 2 else 2
    elif max_bins is None:
        raise Exception("Either max_bins or bin_size needs to be defiend")
    hist = np.histogram(series, bins=max_bins, range=(min_value, max_value))
    quartiles = series.quantile([0, 0.25, 0.5, 0.75, 1])
    quartiles.index = ["min", "25%", "50%", "75%", "max"]

    return Section(
        data=(hist[0], hist[1]),
        n_outliers=n_outliers,
        quartiles=quartiles,
    )

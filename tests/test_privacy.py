import numpy as np
import pandas as pd

from dp_mobility_report.privacy import diff_privacy


def test_bounds_dp():
    array = np.array([1, 10, 10, 10, 15, 15, 15, 20, 20, 1000])
    bounds = diff_privacy.bounds_dp(array, eps=None, sensitivity=None)
    assert bounds == (1, 1000)

    bounds = diff_privacy.bounds_dp(array, eps=0.1, sensitivity=1)
    assert bounds[0] <= bounds[1]
    assert bounds[0] in array
    assert bounds[1] in array


def test_quartiles_dp():
    array = np.array([1, 1, 2, 2, 4, 4, 5, 5])
    quartiles = diff_privacy.quartiles_dp(array, eps=None, sensitivity=None)
    assert quartiles.index.tolist() == ["min", "25%", "50%", "75%", "max"]
    assert quartiles.tolist() == [1, 1.75, 3, 4.25, 5]

    quartiles = diff_privacy.quartiles_dp(array, eps=0.1, sensitivity=1)
    assert quartiles.min() >= array.min()
    assert quartiles.max() <= array.max()

    array = pd.Series(
        [
            np.datetime64("2021-07-01"),
            np.datetime64("2021-07-03"),
            np.datetime64("2021-07-03"),
        ]
    )
    quartiles = diff_privacy.quartiles_dp(array, eps=0.1, sensitivity=1)
    assert quartiles.min() >= array.min()
    assert quartiles.max() <= array.max()
    assert np.issubdtype(quartiles.dtype, np.datetime64)

    array = pd.Series(
        [np.timedelta64(1, "D"), np.timedelta64(1, "D"), np.timedelta64(10, "h")]
    )
    quartiles = diff_privacy.quartiles_dp(array, eps=0.1, sensitivity=1)
    assert quartiles.min() >= array.min()
    assert quartiles.max() <= array.max()
    assert np.issubdtype(quartiles.dtype, np.timedelta64)


def test_counts_dp():
    count = 10
    dp_count = diff_privacy.count_dp(count, eps=None, sensitivity=None)
    assert dp_count == count

    count = 10
    dp_count = diff_privacy.count_dp(count, eps=0.1, sensitivity=1, nonzero=False)
    assert isinstance(dp_count, int)
    assert dp_count >= 0

    counts = np.array([100, 10, 4, 20])
    dp_counts = diff_privacy.counts_dp(counts, eps=None, sensitivity=None)
    assert all(counts == dp_counts)

    counts = np.array([100, 10, 4, 20])
    dp_counts = diff_privacy.counts_dp(counts, eps=0.1, sensitivity=1)
    assert all(dp_counts >= 0)


def test_entropy_dp():
    entropy = pd.Series([0.2, 1.4, 3.9, 2.1])
    dp_entropy = diff_privacy.entropy_dp(entropy, epsi=None, maxcontribution=10)
    assert dp_entropy.equals(entropy)

    dp_entropy = diff_privacy.entropy_dp(entropy, epsi=0.1, maxcontribution=10)
    assert all(dp_entropy >= 0)

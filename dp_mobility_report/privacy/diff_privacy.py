from typing import Optional, Tuple, Type, Union

import diffprivlib
import numpy as np
import pandas as pd
from diffprivlib.validation import clip_to_bounds
from scipy.stats import laplace


def bounds_dp(
    array: Union[np.ndarray, pd.Series], eps: Optional[float], sensitivity: int
) -> Tuple:
    if eps is None:
        return (array.min(), array.max())
    epsi = eps / 2
    result = [1, 0]
    if isinstance(array, np.ndarray):
        array = np.sort(array)

    else:
        array = array.sort_values().reset_index(drop=True)
    k = array.size

    # min can potentially be larger than max
    # if so, retry until a result is found where min < max
    while result[1] < result[0]:
        for quant in [0, 1]:
            mech = diffprivlib.mechanisms.exponential.Exponential(
                epsilon=epsi,
                sensitivity=sensitivity,
                utility=list(-np.abs(np.arange(0, k) - quant * k)),
            )
            idx = mech.randomise()
            output = array[idx]
            result[quant] = output
    return (result[0], result[1])


def quartiles_dp(
    array: pd.Series,
    eps: Optional[float],
    sensitivity: int,
    bounds: Tuple = None,
    conf_interval_perc: float = 0.95,
) -> Tuple:

    # remove nans from array
    array = array.dropna()

    if np.issubdtype(array.dtype, np.timedelta64):
        array = array.values.astype(np.int64)
        dtyp = "timedelta"
    elif np.issubdtype(array.dtype, np.datetime64):
        array = array.values.astype(np.int64)
        dtyp = "datetime"
    else:
        dtyp = "0"

    if bounds is None:
        epsi = eps / 5 if eps is not None else None
        bound_epsi = 2 * epsi if epsi is not None else None
        bounds = bounds_dp(array, bound_epsi, sensitivity)
    else:
        epsi = eps / 3 if eps is not None else None

    result = []
    result.append(bounds[0])

    array_type = array.dtype
    array = pd.Series(clip_to_bounds(np.ravel(array), bounds))
    array = array.sort_values().reset_index(drop=True)
    array = array.astype(
        array_type
    )  # clip_to_bounds converts int arrays to float arrays > convert those back to int

    k = array.size

    def utility(quant: float, k: int) -> list:
        return list(-np.abs(np.arange(0, k) - quant * k))

    if eps is None:
        result.append(array.quantile(0.25))
        result.append(array.quantile(0.5))
        result.append(array.quantile(0.75))
    else:
        for quant in [0.25, 0.5, 0.75]:
            mech = diffprivlib.mechanisms.exponential.Exponential(
                epsilon=epsi,
                sensitivity=sensitivity,
                utility=utility(quant, k),
            )
            idx = mech.randomise()
            output = array[idx]
            result.append(output)

    result.append(bounds[1])

    if dtyp == "timedelta":
        result = pd.to_timedelta(result)
    elif dtyp == "datetime":
        result = pd.to_datetime(result)

    # get margin of error

    # Calculate the probability for each element, based on its score
    counter = 0
    if eps is not None:
        probabilities = [np.exp(eps * u / (2 * sensitivity)) for u in utility(0.5, k)]
        # Normalize the probabilties so they sum to 1
        probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

        max_index = np.argmax(probabilities)
        conf = probabilities[max_index]
        while conf < conf_interval_perc:
            counter += 1
            if max_index + counter < len(probabilities):
                conf += probabilities[max_index + counter]
            if max_index - counter >= 0:
                conf += probabilities[max_index - counter]

    return (pd.Series(result, index=["min", "25%", "50%", "75%", "max"]), counter)


def _laplacer(x: int, eps: float, sensitivity: int) -> int:
    return int(
        round(
            diffprivlib.mechanisms.laplace.Laplace(
                epsilon=eps, delta=0.0, sensitivity=sensitivity
            ).randomise(x),
            0,
        )
    )


def laplace_margin_of_error(
    conf_interval_perc: float, eps: Optional[float], sensitivity: int
) -> float:
    if eps is None:
        return 0
    delta = 0
    q = conf_interval_perc + 0.5 * (1 - conf_interval_perc)
    scale = sensitivity / (eps - np.log(1 - delta))
    return laplace.ppf(q, loc=0, scale=scale)


def conf_interval(
    value: Optional[Union[float, int]], margin_of_error: float, type: Type = int
) -> Tuple:
    if value is None:
        return (None, None)
    lower_limit = (value - margin_of_error) if (margin_of_error < value) else 0
    if Type is not None:
        lower_limit = type(lower_limit)
        margin_of_error = type(margin_of_error)
    return (lower_limit, value + margin_of_error)


def count_dp(
    count: int,
    eps: Optional[float],
    sensitivity: int,
    nonzero: bool = False,
) -> Optional[int]:
    if eps is None:
        return count
    dpcount = _laplacer(count, eps, sensitivity)
    dpcount = int((abs(dpcount) + dpcount) / 2)
    if nonzero:
        return dpcount if dpcount > 0 else None
    else:
        return dpcount


def counts_dp(
    counts: Union[int, np.ndarray],  # TODO: only array?
    eps: Optional[float],
    sensitivity: int,
    allow_negative: bool = False,
) -> np.ndarray:
    if eps is None:
        return counts
    eps_local = eps  # workaround for linting error

    def _local_laplacer(x: int) -> int:
        return _laplacer(x, eps_local, sensitivity)

    vfunc = np.vectorize(_local_laplacer)
    dpcounts = vfunc(counts)
    dpcounts = (
        limit_negative_values_to_zero(dpcounts) if not allow_negative else dpcounts
    )
    return dpcounts


def limit_negative_values_to_zero(array: np.array) -> np.array:
    x = np.vectorize(limit_negative_value_to_zero)
    return x(array)


def limit_negative_value_to_zero(value: int) -> int:
    return int((abs(value) + value) / 2)

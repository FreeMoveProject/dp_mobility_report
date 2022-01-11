from typing import Optional, Tuple, Union

import diffprivlib
import numpy as np
import pandas as pd
from diffprivlib.validation import clip_to_bounds


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
    array: Union[np.ndarray, pd.Series],
    eps: Optional[float],
    sensitivity: int,
    bounds: Tuple = None,
) -> pd.Series:

    if np.issubdtype(array.dtype, np.timedelta64):
        array = array.values.astype(np.int64)
        dtyp = "timedelta"
    elif np.issubdtype(array.dtype, np.datetime64):
        array = array.values.astype(np.int64)
        dtyp = "datetime"
    else:
        dtyp = "0"

    if bounds is None:
        if eps is not None:
            epsi = eps / 5
            bound_epsi = 2 * epsi
        else:
            epsi = eps
            bound_epsi = epsi
        bounds = bounds_dp(array, bound_epsi, sensitivity)
    elif eps is not None:
        epsi = eps / 3

    result = []
    result.append(bounds[0])

    array = pd.Series(clip_to_bounds(np.ravel(array), bounds))
    array = array.sort_values().reset_index(drop=True)
    k = array.size

    if eps is None:
        result.append(array.quantile(0.25))
        result.append(array.quantile(0.5))
        result.append(array.quantile(0.75))
    else:
        for quant in [0.25, 0.5, 0.75]:
            mech = diffprivlib.mechanisms.exponential.Exponential(
                epsilon=epsi,
                sensitivity=sensitivity,
                utility=list(-np.abs(np.arange(0, k) - quant * k)),
            )
            idx = mech.randomise()
            output = array[idx]
            result.append(output)

    result.append(bounds[1])

    if dtyp == "timedelta":
        result = pd.to_timedelta(result)
    elif dtyp == "datetime":
        result = pd.to_datetime(result)

    return pd.Series(result, index=["min", "25%", "50%", "75%", "max"])


def _laplacer(x: int, eps: float, sensitivity: int) -> int:
    return int(
        round(
            diffprivlib.mechanisms.laplace.Laplace(
                epsilon=eps, delta=0.0, sensitivity=sensitivity
            ).randomise(x),
            0,
        )
    )


def count_dp(
    count: int,
    eps: Optional[float],
    sensitivity: int,
    nonzero: bool = False,
) -> int:
    if eps is None:
        return count
    dpcount = _laplacer(count, eps, sensitivity)
    dpcount = int((abs(dpcount) + dpcount) / 2)
    if nonzero:
        return dpcount if dpcount > 0 else None
    else:
        return dpcount


def counts_dp(
    counts: Union[int, np.ndarray],
    eps: Optional[float],
    sensitivity: int,
    parallel: bool = True,
    nonzero: bool = False,
) -> Union[int, np.ndarray]:
    if eps is None:
        return counts

    if parallel is False:
        epsi = eps / len([counts])
    else:
        epsi = eps

    def _local_laplacer(x: int):
        return _laplacer(x, epsi, sensitivity)

    vfunc = np.vectorize(_local_laplacer)
    dpcount = vfunc(counts)
    dpcount = ((abs(dpcount) + dpcount) / 2).astype(int)

    if nonzero:
        return dpcount[dpcount > 0]
    else:
        return dpcount


def entropy_dp(
    array: pd.Series, epsi: Optional[float], maxcontribution: int
) -> np.ndarray:
    if epsi is None:
        return array
    if maxcontribution > 1:
        sensitivity = (
            2
            * maxcontribution
            * (
                max(
                    np.log(2),
                    np.log(2 * maxcontribution)
                    - np.log(np.log(2 * maxcontribution))
                    - 1,
                )
            )
        )
    else:
        sensitivity = np.log(2)

    def _laplacer(x: int) -> int:
        return int(
            round(
                diffprivlib.mechanisms.laplace.Laplace(
                    epsilon=epsi, delta=0.0, sensitivity=sensitivity
                ).randomise(x),
                0,
            )
        )

    vfunc = np.vectorize(_laplacer)
    entropy = vfunc(array)
    entropy = (abs(entropy) + entropy) / 2
    # entropy >=0 and <= log k with k categories.
    return entropy

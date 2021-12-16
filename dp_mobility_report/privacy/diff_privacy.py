import pandas as pd
import math
import diffprivlib
import numpy as np
from diffprivlib.validation import clip_to_bounds, check_bounds


def bounds_dp(array, eps, sensitivity):
    if eps is None:
        return (array.min(), array.max())
    epsi = eps / 2
    # acc.check(epsi, 0)
    result = [1, 0]
    if isinstance(array, np.ndarray):
        array = np.sort(array)

    else:
        array = array.sort_values().reset_index(drop=True)
    k = array.size

    # ToDo: saskia Better way?
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
    # acc.spend(eps, 0)
    return (result[0], result[1])


def quartiles_dp(array, eps, sensitivity, bounds=None):

    if array.dtype == "timedelta64[ns]":
        array = array.values.astype(np.int64)
        dtyp = "timedelta"
    elif array.dtype == "datetime64[ns]":
        array = array.values.astype(
            np.int64
        )  # pd.to_datetime(array).astype(np.int64)#array.values.astype(np.int64)
        dtyp = "datetime"
    else:
        dtyp = 0

    if bounds is None:
        if eps is not None:
            epsi = eps / 5
            bound_epsi = 2 * epsi
        else:
            epsi = eps
            bound_epsi = epsi
        # acc.check(epsi, 0)
        bounds = bounds_dp(array, bound_epsi, sensitivity)
    elif eps is not None:
        epsi = eps / 3
        # acc.check(epsi, 0)

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
            # acc.spend(epsi, 0)

    result.append(bounds[1])

    if dtyp == "timedelta":
        result = pd.to_timedelta(result)
    elif dtyp == "datetime":
        result = pd.to_datetime(result)

    return pd.Series(result, index=["min", "25%", "50%", "75%", "max"])


def counts_dp(counts, eps, sensitivity, parallel=True, nonzero=True):
    if eps is None:
        return counts

    if parallel is False:
        epsi = eps / len([counts])
        # acc.spend(len([series])*epsi, 0)
    else:
        epsi = eps
        # acc.spend(epsi, 0)

    if isinstance(counts, pd.Series):
        dpcount = counts.apply(
            lambda x: int(
                round(
                    diffprivlib.mechanisms.laplace.Laplace(
                        epsilon=epsi, delta=0.0, sensitivity=sensitivity
                    ).randomise(x),
                    0,
                )
            )
        )
        dpcount = dpcount.apply(lambda x: int(round((abs(x) + x) / 2, 0)))
    else:  # isinstance(series,int #or np.ndarray):
        laplacer = lambda x: int(
            round(
                diffprivlib.mechanisms.laplace.Laplace(
                    epsilon=epsi, delta=0.0, sensitivity=sensitivity
                ).randomise(x),
                0,
            )
        )
        vfunc = np.vectorize(laplacer)
        dpcount = vfunc(counts)
        dpcount = (abs(dpcount) + dpcount) / 2

    if nonzero is True:
        if np.isscalar(dpcount):
            return dpcount if dpcount > 0 else None  # TODO: is this correct?
        return dpcount[dpcount > 0]
    else:
        return dpcount


def entropy_dp(array, epsi, maxcontribution):
    if epsi is None:
        return array
    if maxcontribution > 1:
        sensitivity = maxcontribution * (
            max(
                np.log(2), np.log(maxcontribution) - np.log(np.log(maxcontribution)) - 1
            )
        )
    else:
        sensitivity = np.log(2)

    # acc.check(epsi,0)
    laplacer = lambda x: int(
        round(
            diffprivlib.mechanisms.laplace.Laplace(
                epsilon=epsi, delta=0.0, sensitivity=sensitivity
            ).randomise(x),
            0,
        )
    )
    vfunc = np.vectorize(laplacer)
    entropy = vfunc(array)
    entropy = (abs(entropy) + entropy) / 2
    # entropy >=0 and <= log k with k categories.
    return entropy

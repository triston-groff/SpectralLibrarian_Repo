from typing import *
from multiprocessing import Pool
from bisect import bisect_left
from scipy.interpolate import interp1d

import numpy as np


def parallelize(workfunc, num_cores, argslist):
    with Pool(num_cores) as pool:
        if isinstance(argslist[0], tuple):
            results = pool.starmap(workfunc, argslist)
        else:
            results = pool.map(workfunc, argslist)
    return results


def normalizeVec(x: np.ndarray, how: str = "Max"):
    if how == "Sum":
        scale = np.sum(x)
    elif how == "Max":
        scale = np.max(x)
    else:
        raise NotImplementedError("Only 'Max' (Base peak intensity as 1) and 'Sum' (Probabilistic Interpretation) are supported for 'how' argument")

    if scale < 1e-6:
        return np.ones(x.shape) / np.sum(np.ones(x.shape))
    else:
        return x / scale


def ppm_error(mzq: Union[float, np.ndarray], mzr: Union[float, np.ndarray]):
    return (np.abs(mzq - mzr)/mzr)*1e6


def within_ppm(mzq: Union[float, np.ndarray], mzr: Union[float, np.ndarray], ppm: float):
    return abs(mzq - mzr) <= ppm_tol(mzr, ppm)


def ppm_tol(mz: Union[float, np.ndarray], ppm_val: float) -> float:
    return mz * ppm_val * 1e-6


def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return pos, myList[0]
    if pos == len(myList):
        return len(myList)-1, myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos, after
    else:
        return pos-1, before


def eic_interpolator_linear_pure(rts_itp, its_itp):
    s = interp1d(rts_itp, its_itp, kind='linear', fill_value=0, bounds_error=False)
    return s


from collections import defaultdict

def merge_list_of_dict(data: List[Dict[str, Union[float, int, str]]]):
    out = defaultdict(list)
    append_cache = {}
    extend_cache = {}

    for d in data:
        for k, v in d.items():
            if k not in append_cache:
                append_cache[k] = out[k].append
                extend_cache[k] = out[k].extend

            if type(v) is list:
                extend_cache[k](v)
            else:
                append_cache[k](v)

    return dict(out)


__all__ = ["parallelize", "normalizeVec", "ppm_error", "within_ppm", "ppm_tol", "takeClosest", "eic_interpolator_linear_pure", "merge_list_of_dict"]
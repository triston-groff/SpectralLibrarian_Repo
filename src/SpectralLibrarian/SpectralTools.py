# src/MSAnalyzer/SpectralTools.py
"""
SpectralTools – Core preprocessing, cleaning, neutral loss, entropy, mass ↔ m/z

These are the fundamental building blocks used by SimilarityTools, IsotopeTools, etc.
"""
from .utilities import within_ppm, normalizeVec

from typing import *

import matplotlib.pyplot as plt
import os
import pandas as pd, numpy as np


try:
    from ms_entropy import calculate_entropy_similarity
    # from spectral_entropy.spectral_similarity import similarity as _entropy_similarity
    HAS_ENTROPY = True
except ImportError:
    HAS_ENTROPY = False


def str2array(specstr: str) -> Union[tuple[None, None], tuple[np.ndarray, np.ndarray]]:
    peaks = specstr.split(" ")

    if (len(peaks) == 1) and (not (":" in peaks[0])):
        return None, None

    mzs, ints = [], []
    for p in peaks:
        mzs.append(float(p.split(":")[0]))
        ints.append(float(p.split(":")[1]))

    if len(ints) == 0:
        return None, None

    ints = np.array(ints)[np.argsort(mzs)]
    mzs = np.sort(mzs)
    return mzs, ints


def array2spec(mzs: np.ndarray, ints: np.ndarray) -> np.ndarray:
    return np.column_stack([mzs, ints])


def spec2array(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return spectrum[:, 0], spectrum[:, 1]


def array2str(mzs: np.ndarray, ints: np.ndarray) -> str:
    return " ".join([str(np.round(mz, 4)) + ":" + str(np.round(it, 4)) for mz, it in list(zip(mzs, ints))])


def spec2str(spectrum: np.ndarray) -> str:
    if spectrum is None:
        return ""
    else:
        return array2str(mzs=spectrum[:, 0], ints=spectrum[:, 1])


def str2spec(specstr: str) -> Union[None, np.ndarray]:
    mzs, ints = str2array(specstr=specstr)
    if not (mzs is None):
        return array2spec(mzs=mzs, ints=ints)
    else:
        return None


def normalizeSpectrum(spectrum: np.ndarray, how="Max") -> np.ndarray:
    if not (how in ["Max", "Sum"]):
        raise NotImplementedError("Only 'Max' (Base peak intensity as 1) and 'Sum' (Probabilistic Interpretation) are supported for 'how' argument")

    spectrum_new = spectrum.copy()
    spectrum_new[:, 1] = normalizeVec(x=spectrum_new[:, 1], how=how)
    return spectrum_new


def sortSpectrum(spectrum: np.ndarray, by="mz", order="ascending") -> np.ndarray:
    if by == "mz":
        if order == "ascending":
            return spectrum[np.argsort(spectrum[:,0]),:]
        else:
            return spectrum[np.argsort(spectrum[:, 0], order="descending"), :]
    elif by == "intensity":
        if order == "ascending":
            return spectrum[np.argsort(spectrum[:, 1]), :]
        else:
            return spectrum[np.argsort(spectrum[:, 1], order="descending"), :]
    else:
        raise ValueError("Invalid 'by' argument. Only 'mz' and 'intensity' are supported")


def clean_spectrum_by_abundance(spectrum: np.ndarray, abdco: float=0.0) -> np.ndarray:
    return spectrum[np.where(spectrum[:,1] > abdco)[0], :]


def sum_abundances_from_spectrum(spectrum: np.ndarray, mz_centers: Union[np.ndarray, List[float]], ppm: float, abdco: float=0.0) -> np.ndarray:
    spectrum_abdthre = clean_spectrum_by_abundance(spectrum=spectrum, abdco=abdco)
    mzs, abds = spectrum_abdthre[:,0], spectrum_abdthre[:,1]
    summed_intensities = np.array([np.sum(abds[within_ppm(mzs, c, ppm)]) for c in mz_centers])
    summed_spectrum = np.transpose([mz_centers, summed_intensities])
    return summed_spectrum # summed_spectrum[np.where(summed_spectrum[:,1] > 0)[0], :]


def mergeSpectra(spectra: List[np.ndarray], how:Literal["ppm", "bin"], bin_width:float=None, ppm: float=None):
    spectrum = sortSpectrum(spectrum=np.concatenate(spectra, axis=0), by="mz", order="ascending")

    if how == "bin":
        if bin_width is None:
            raise "bin_width (float) should be given for binning method"
        merged_spectrum, clusters = partition_mz_binning(spectrum=spectrum, bin_width=bin_width)
    elif how == "ppm":
        if ppm is None:
            raise "ppm (float) should be given for binning method"
        if len(spectrum) > 0:
            merged_spectrum, clusters = partition_mz_ppm(spectrum=spectrum, ppm=ppm)
        else:
            merged_spectrum, clusters =  [], []
    else:
        raise NotImplementedError
    return merged_spectrum, clusters


def partition_mz_binning(spectrum: np.ndarray, bin_width:float):
    mzs, ints = spectrum[:,0], spectrum[:,1]
    bins = np.arange(np.min(mzs), np.max(mzs) + bin_width, bin_width)

    clusters = []
    indices = np.digitize(mzs, bins)
    for ind in list(np.unique(indices)):
        clusters.append(list(mzs[np.where(indices == ind)]))

    new_mzs = (bins[:-1] + bins[1:]) / 2
    new_ints = np.histogram(mzs, bins=bins, weights=ints)[0]
    return np.transpose([new_mzs, new_ints]), clusters


def partition_mz_ppm(spectrum: np.ndarray, ppm: float, how: Literal["weighted", "simple"]="weighted") -> tuple[np.ndarray, List[List[tuple[float]]]]:
    """cluster peaks (mz, it) so that all the mz in a cluster are within ppm error from the (simple or weighted) centroid mz of the cluster.
    Sort the given mzs and scan from the lowest to the highest while calculating (simple or weighted) centroid of the cluster in a 'cumulative' manner
    If the mean deviates from the cluster boundary or the next mz more than the ppm, cluster is closed and the new cluster starts with the next mz.
    The complexity of this cumulative way is O(n), while that of regenerating a cluster for each iteration is O(n^2)"""
    spectrum_sorted = spectrum[np.argsort(spectrum[:,0]), :]
    sorted_mzs, sorted_ints = spectrum_sorted[:,0], spectrum_sorted[:,1]

    clusters, current = [], [(sorted_mzs[0], sorted_ints[0])]
    centroided_mzs, centroided_ints = [], []
    start, sum_it, count = 0, sorted_ints[0], 1
    sum_mz = sorted_mzs[0] * sorted_ints[0] if how == "weighted" else sorted_mzs[0]
    for i in range(1, len(sorted_mzs)):
        lo, mz, it = sorted_mzs[start], sorted_mzs[i], sorted_ints[i]
        new_sum_mz = sum_mz + (mz * it if how == "weighted" else mz)
        new_sum_it, new_count = sum_it + it, count + 1
        mean_mz = new_sum_mz / (new_sum_it if how == "weighted" else new_count)

        if within_ppm(lo, mean_mz, ppm) and within_ppm(mz, mean_mz, ppm):
            sum_mz, sum_it, count = new_sum_mz, new_sum_it, new_count
            current.append((mz, it))
        else:
            clusters.append(current)
            centroided_mzs.append(sum_mz / (sum_it if how == "weighted" else count))
            centroided_ints.append(sum_it)
            start, sum_mz, sum_it, count = i, mz * it if how == "weighted" else mz, it, 1
            current = [(mz, it)]
    clusters.append(current)
    centroided_mzs.append(sum_mz / (sum_it if how == "weighted" else count))
    centroided_ints.append(sum_it)
    return array2spec(mzs=np.array(centroided_mzs), ints=np.array(centroided_ints)), clusters


def drop_msms_precursor(spectrum: np.ndarray, precursor_mz: float, tol_da: float = None, tol_ppm:float=None) -> np.ndarray:
    """
    Removes any peaks within mz_tolerance of the precursor_mz (e.g., undissociated precursor) in an MS/MS spectrum.
    
    Args:
        spectrum: np.ndarray of [None, 2] dimension, where the first column is m/z vector, and the second is intensity vector
        precursor_mz: The precursor m/z value.
        tol_da: mz Tolerance (Da) window around precursor_mz to remove. (ex 0.01 for 0.01Da)
        tol_ppm: ppm window around precursor_mz to remove. (ex 5 for 5ppm)
    
    Returns:
        Filtered spectrum.
    """
    if (tol_da is None) and (tol_ppm is None):
        raise ValueError("At least one of mz_tolerance and ppm must be provided.")
    else:
        if np.shape(spectrum)[0] == 0:
            return spectrum
        else:
            mz_array, intensity_array = spectrum[:, 0], spectrum[:, 1]
            if tol_da is None: # use ppm
                mask = np.where(~within_ppm(mzq=mz_array, mzr=precursor_mz, ppm=tol_ppm))[0]
            else : # use mz_tolerance
                mask = np.where(np.abs(mz_array - precursor_mz) > tol_da)[0]
            return spectrum[mask,:]


def clean_spectrum(spectrum: np.ndarray, precursor_mz: float = None, noise_threshold: float = 0.01, max_mz: float = None, centroid_tol: float = 0.01) -> np.ndarray:
    """Full cleaning: noise removal, centroiding, precursor filter."""
    spectrum_norm = normalizeSpectrum(spectrum=spectrum, how="Max")
    spectrum_norm = clean_spectrum_by_abundance(spectrum=spectrum_norm, abdco=noise_threshold)
    mz, intensity = spec2array(spectrum=spectrum_norm)

    # Is this centroiding correct ? : This might be biased to low-mz values, as it scans from low-mz to high-mz by assigned the summed intensity to one of mzs with higher intensity
    if centroid_tol > 0 and len(mz) > 1:
        order = np.argsort(mz)
        mz, intensity = mz[order], intensity[order]
        keep = np.ones(len(mz), dtype=bool)
        i = 0
        while i < len(mz) - 1:
            if keep[i] and mz[i + 1] - mz[i] <= centroid_tol:
                if intensity[i] >= intensity[i + 1]:
                    intensity[i] += intensity[i + 1]
                    keep[i + 1] = False
                else:
                    intensity[i + 1] += intensity[i]
                    keep[i] = False
            i += 1
        mz, intensity = mz[keep], intensity[keep]

    if precursor_mz is not None and max_mz is None:
        max_mz = precursor_mz + 10
    if max_mz is not None:
        mask = mz <= max_mz
        mz, intensity = mz[mask], intensity[mask]

    return array2spec(mzs=mz, ints=intensity)


def spectral_entropy(spectrum: np.ndarray, normalize=True) -> float:
    """Shannon entropy of a single spectrum (information content), using natural log.
    If normalize=True, this returns normalized Shannon entropy (divided by ln(N) for [0,1] range across peak counts)."""
    if not HAS_ENTROPY:
        raise ImportError("Install spectral_entropy to compute spectral entropy")
    spectrum_norm = normalizeSpectrum(spectrum=spectrum, how="Sum")
    p = spectrum_norm[:,1] # if intensities.sum()==0: return 0.0
    entropy = -np.sum(p * np.log(p))

    if not normalize:
        return entropy
    else:
        n = np.shape(spectrum)[0]
        if n <= 1:
            return 0.0  # Handle edge cases to avoid div by zero or ln(1)=0
        return entropy / np.log(n)  # Natural log


def neutral_loss_spectrum(spectrum: np.ndarray, precursor_mz: float, max_loss: float = None, return_fragment_index: bool = False) -> np.ndarray:
    """Convert to neutral loss spectrum of size [None, 3]. The last column array contains index of neutral loss peak in the original spectrum"""
    if max_loss is None:
        max_loss = precursor_mz  # Use precursor_mz as max to capture all possible losses

    mz, intensity = spec2array(spectrum=spectrum)
    nl_mz = precursor_mz - mz
    mask = (nl_mz >= 0) & (nl_mz <= max_loss)

    nl_spec = array2spec(mzs=nl_mz[mask], ints=intensity[mask])
    if return_fragment_index:
        # Return indices into original spec['mz'] (as int array matching nl_mz order)
        nl_spec = np.concatenate([nl_spec, np.where(mask)[0].astype(int).reshape([-1, 1])], axis=1)

    # Sort and normalize
    nl_spec = sortSpectrum(spectrum=nl_spec, order="ascending")
    nl_spec = normalizeSpectrum(spectrum=nl_spec, how="Sum")
    return nl_spec


# Batch versions
def batch_drop_msms_precursor(df: pd.DataFrame, precursor_col: str = 'precursor_mz', ms2spec_col: str = 'ms2spectrum', tol_da: float = None, tol_ppm: float = None) -> pd.DataFrame:
    """
    Batch version to drop precursor peaks from MS/MS spectra (row) in a DataFrame.

    Args:
        df: Input DataFrame.
        precursor_col: Column with precursor m/z values.
        ms2spec_col: Column with ms2spectrum
        tol_da: mz Tolerance (Da) window around precursor_mz to remove. (ex 0.01 for 0.01Da)
        tol_ppm: ppm window around precursor_mz to remove. (ex 5 for 5ppm)

    Returns:
        Updated DataFrame with filtered spectrum in place
    """
    df = df.copy()

    # Vectorized implementation: Stack all arrays and precursors
    if len(df) == 0:
        return df

    # Get original lengths
    mz_arrays, int_arrays = zip(*df[ms2spec_col].apply(str2array))
    lengths = [len(mza) for mza in mz_arrays]

    # Stack mz, intensity, and repeat precursors and row indices
    all_mz = np.concatenate(mz_arrays)
    all_intensity = np.concatenate(int_arrays)
    all_precursors = np.repeat(df[precursor_col].values, lengths)
    row_indices = np.repeat(np.arange(len(df)), lengths)

    # Vectorized mask
    if tol_da is None:
        mask = np.where(~within_ppm(mzq=all_mz, mzr=all_precursors, ppm=tol_ppm))[0]
    else:
        mask = np.where(np.abs(all_mz - all_precursors) > tol_da)[0]

    # Filter stacked arrays and row indices
    filtered_mz = all_mz[mask]
    filtered_intensity = all_intensity[mask]
    filtered_row_indices = row_indices[mask]

    # Compute new lengths using bincount (handles variable removals correctly)
    new_lengths = np.bincount(filtered_row_indices, minlength=len(df))
    new_cum_lengths = np.cumsum(new_lengths)

    # Split back into per-row arrays
    filtered_mz_list = np.split(filtered_mz, new_cum_lengths[:-1])
    filtered_intensity_list = np.split(filtered_intensity, new_cum_lengths[:-1])

    ms2spec_new = []
    numpeaks_new = []
    for mz_array, int_array in zip(filtered_mz_list, filtered_intensity_list):
        ms2spec_new.append(array2spec(mzs=mz_array, ints=int_array))
        numpeaks_new.append(len(mz_array))
    df[ms2spec_col] = ms2spec_new
    df['Num Peaks'] = numpeaks_new  # Add/update 'Num Peaks'

    return df


def batch_clean(df: pd.DataFrame, precursor_col: str = "precursor_mz", **kwargs) -> pd.DataFrame:
    df = df.copy()

    def clean_row(row):
        cleaned_spectrum = clean_spectrum(spectrum=str2spec(specstr=row["ms2spectrum"]), precursor_mz=row.get(precursor_col), **kwargs)
        return spec2str(spectrum=cleaned_spectrum)

    df["ms2spectrum"] = df.apply(clean_row, axis=1)
    df['Num Peaks'] = df["ms2spectrum"].map(lambda x: len(x.split(" ")), axis=1)
    return df


def batch_normalize(df: pd.DataFrame, how:str="Max") -> pd.DataFrame:
    if not (how in ["Max", "Sum"]):
        raise ValueError("Invalid value for 'how' parameter. Only 'Max' and 'Sum' are valid.")

    df = df.copy()
    df["ms2spectrum"] = df["ms2spectrum"].map(lambda x: spec2str(spectrum=normalizeSpectrum(spectrum=str2spec(specstr=x))), axis=1)
    df['Num Peaks'] = df["ms2spectrum"].map(lambda x: len(x.split(" ")), axis=1)
    return df


def batch_spectral_entropy(df: pd.DataFrame, normalize=True) -> pd.DataFrame:
    df = df.copy()
    newcol = "normalized_spectral_entropy" if normalize else "spectral_entropy"
    df[newcol] = df.apply(lambda r: spectral_entropy(spectrum=str2spec(specstr=r["ms2spectrum"]), normalize=normalize), axis=1)
    return df


def batch_neutral_loss(df: pd.DataFrame, precursor_col: str = "precursor_mz", max_loss_col: str = None, return_fragment_index: bool = False) -> pd.DataFrame:
    df = df.copy()

    def nl_row(row):
        spectrum = str2spec(specstr=row["ms2spectrum"])
        max_loss_val = row[max_loss_col] if max_loss_col and max_loss_col in row else None
        nl = neutral_loss_spectrum(spectrum=spectrum, precursor_mz=row[precursor_col], max_loss=max_loss_val, return_fragment_index=return_fragment_index)
        res = {"nl_mz": nl["mz"], "nl_intensity": nl["intensity"]}
        if return_fragment_index:
            res["nl_fragment_index"] = nl["nl_fragment_index"]
        return pd.Series(res)

    nl = df.apply(nl_row, axis=1)
    df["nl_mz"] = nl["nl_mz"]
    df["nl_intensity"] = nl["nl_intensity"]
    if return_fragment_index:
        df["nl_fragment_index"] = nl["nl_fragment_index"]

    return df


__all__ = ["str2array", "array2spec", "spec2array", "array2str", "spec2str", "str2spec", "normalizeSpectrum", "sortSpectrum",
           "clean_spectrum_by_abundance", "sum_abundances_from_spectrum", "mergeSpectra", "partition_mz_binning", "partition_mz_ppm",
           "drop_msms_precursor", "clean_spectrum", "spectral_entropy", "neutral_loss_spectrum",
           "batch_drop_msms_precursor", "batch_clean", "batch_normalize", "batch_spectral_entropy", "batch_neutral_loss"]
# src/SpectralLibrarian/SpectralTools.py
"""
SpectralTools – Core preprocessing, cleaning, neutral loss, entropy, mass ↔ m/z

Now fully standardized: all batch functions accept mz_array_col, intensity_array_col,
num_peaks_col, and precursor_col (where relevant) for maximum flexibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

# Try to import spectral_entropy for entropy calculation
try:
    from spectral_entropy.spectral_similarity import similarity as _entropy_similarity
    HAS_ENTROPY = True
except ImportError:
    HAS_ENTROPY = False


def drop_msms_precursor(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    precursor_mz: float,
    mz_tolerance: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Drop peaks corresponding to the precursor ion in an MS/MS spectrum."""
    if len(mz_array) == 0:
        return mz_array, intensity_array
    
    mask = np.abs(mz_array - precursor_mz) > mz_tolerance
    return mz_array[mask], intensity_array[mask]


def batch_drop_msms_precursor(
    df: pd.DataFrame,
    mz_array_col: str = 'mz_array',
    intensity_array_col: str = 'intensity_array',
    precursor_col: str = 'PRECURSORMZ',
    num_peaks_col: str = 'NUM PEAKS',
    mz_tolerance: float = 0.01
) -> pd.DataFrame:
    """Batch version to drop precursor peaks."""
    df = df.copy()
    if len(df) == 0:
        return df

    lengths = df[mz_array_col].apply(len)

    # Vectorized (robust to any index)
    all_mz = np.concatenate(df[mz_array_col].to_numpy())
    all_intensity = np.concatenate(df[intensity_array_col].to_numpy())
    all_precursors = np.repeat(df[precursor_col].values, lengths)
    row_indices = np.repeat(np.arange(len(df)), lengths)

    mask = np.abs(all_mz - all_precursors) > mz_tolerance

    filtered_mz = all_mz[mask]
    filtered_intensity = all_intensity[mask]
    filtered_row_indices = row_indices[mask]

    new_lengths = np.bincount(filtered_row_indices, minlength=len(df))
    new_cum_lengths = np.cumsum(new_lengths)

    filtered_mz_list = np.split(filtered_mz, new_cum_lengths[:-1])
    filtered_intensity_list = np.split(filtered_intensity, new_cum_lengths[:-1])

    df[mz_array_col] = filtered_mz_list
    df[intensity_array_col] = filtered_intensity_list

    # Update peak count using the configurable column name
    df[num_peaks_col] = pd.to_numeric(df[mz_array_col].apply(len), downcast='integer')
    return df


def clean_spectrum(
    spec: Dict[str, np.ndarray],
    precursor_mz: float = None,
    noise_threshold: float = 0.01,
    max_mz: float = None,
    centroid_tol: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Full cleaning: noise removal, centroiding, precursor filter."""
    mz = spec["mz"].copy()
    intensity = spec["intensity"].copy()

    if intensity.max() > 0:
        intensity /= intensity.max()
    mask = intensity >= noise_threshold
    mz, intensity = mz[mask], intensity[mask]

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

    return {"mz": mz, "intensity": intensity}


def standardize_spectrum(spec: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Sort by m/z and normalize intensities to sum=1."""
    spec = spec.copy()
    order = np.argsort(spec["mz"])
    spec["mz"] = spec["mz"][order]
    spec["intensity"] = spec["intensity"][order]
    total = spec["intensity"].sum()
    if total > 0:
        spec["intensity"] /= total
    return spec


def neutral_loss_spectrum(
    spec: Dict[str, np.ndarray],
    precursor_mz: float,
    max_loss: float = None,
    return_fragment_index: bool = False
) -> Dict[str, np.ndarray]:
    """Convert to neutral loss spectrum."""
    if max_loss is None:
        max_loss = precursor_mz
    
    nl_mz = precursor_mz - spec["mz"]
    mask = (nl_mz >= 0) & (nl_mz <= max_loss)
    
    nl_spec = {"mz": nl_mz[mask], "intensity": spec["intensity"][mask]}
    if return_fragment_index:
        nl_spec["nl_fragment_index"] = np.where(mask)[0].astype(int)
    
    if return_fragment_index:
        order = np.argsort(nl_spec["mz"])
        nl_spec["mz"] = nl_spec["mz"][order]
        nl_spec["intensity"] = nl_spec["intensity"][order]
        nl_spec["nl_fragment_index"] = nl_spec["nl_fragment_index"][order]
    else:
        nl_spec = standardize_spectrum(nl_spec)
    
    return nl_spec


def spectral_entropy(spec: Dict[str, np.ndarray]) -> float:
    """Shannon entropy of a single spectrum (information content), using natural log."""
    if not HAS_ENTROPY:
        raise ImportError("Install spectral_entropy to compute spectral entropy")
    intensities = spec["intensity"]
    if intensities.sum() == 0:
        return 0.0
    p = intensities / intensities.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def normalized_spectral_entropy(spec: Dict[str, np.ndarray]) -> float:
    """Normalized Shannon entropy (divided by ln(N) for [0,1] range)."""
    entropy = spectral_entropy(spec)
    n = len(spec["intensity"])
    if n <= 1:
        return 0.0
    return entropy / np.log(n)


# ====================== BATCH FUNCTIONS (now fully standardized) ======================

def batch_clean(
    df: pd.DataFrame,
    mz_array_col: str = 'mz_array',
    intensity_array_col: str = 'intensity_array',
    num_peaks_col: str = 'NUM PEAKS',
    precursor_col: str = 'PRECURSORMZ',
    **kwargs
) -> pd.DataFrame:
    """Batch cleaning with configurable column names."""
    df = df.copy()
    def clean_row(row):
        spec = {"mz": row[mz_array_col], "intensity": row[intensity_array_col]}
        cleaned = clean_spectrum(spec, precursor_mz=row.get(precursor_col), **kwargs)
        return pd.Series({mz_array_col: cleaned["mz"], intensity_array_col: cleaned["intensity"]})
    cleaned = df.apply(clean_row, axis=1)
    df[mz_array_col] = cleaned[mz_array_col]
    df[intensity_array_col] = cleaned[intensity_array_col]
    df[num_peaks_col] = pd.to_numeric(df[mz_array_col].apply(len), downcast='integer')
    return df


def batch_standardize(
    df: pd.DataFrame,
    mz_array_col: str = 'mz_array',
    intensity_array_col: str = 'intensity_array',
    num_peaks_col: str = 'NUM PEAKS'
) -> pd.DataFrame:
    """Batch standardize (sort + normalize) with configurable column names."""
    df = df.copy()
    def std_row(row):
        spec = {"mz": row[mz_array_col], "intensity": row[intensity_array_col]}
        std = standardize_spectrum(spec)
        return pd.Series({mz_array_col: std["mz"], intensity_array_col: std["intensity"]})
    std = df.apply(std_row, axis=1)
    df[mz_array_col] = std[mz_array_col]
    df[intensity_array_col] = std[intensity_array_col]
    df[num_peaks_col] = pd.to_numeric(df[mz_array_col].apply(len), downcast='integer')
    return df


def batch_neutral_loss(
    df: pd.DataFrame,
    mz_array_col: str = 'mz_array',
    intensity_array_col: str = 'intensity_array',
    precursor_col: str = 'PRECURSORMZ',
    max_loss_col: str = None,
    return_fragment_index: bool = False
) -> pd.DataFrame:
    """Batch neutral loss with configurable input columns."""
    df = df.copy()
    def nl_row(row):
        spec = {"mz": row[mz_array_col], "intensity": row[intensity_array_col]}
        max_loss_val = row[max_loss_col] if max_loss_col and max_loss_col in row else None
        nl = neutral_loss_spectrum(
            spec,
            row[precursor_col],
            max_loss=max_loss_val,
            return_fragment_index=return_fragment_index
        )
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


def batch_spectral_entropy(
    df: pd.DataFrame,
    mz_array_col: str = 'mz_array',
    intensity_array_col: str = 'intensity_array'
) -> pd.DataFrame:
    """Batch spectral entropy with configurable column names."""
    df = df.copy()
    df["spectral_entropy"] = df.apply(
        lambda r: spectral_entropy({"mz": r[mz_array_col], "intensity": r[intensity_array_col]}),
        axis=1
    )
    return df


def batch_normalized_spectral_entropy(
    df: pd.DataFrame,
    mz_array_col: str = 'mz_array',
    intensity_array_col: str = 'intensity_array'
) -> pd.DataFrame:
    """Batch normalized spectral entropy with configurable column names."""
    df = df.copy()
    df["normalized_spectral_entropy"] = df.apply(
        lambda r: normalized_spectral_entropy({"mz": r[mz_array_col], "intensity": r[intensity_array_col]}),
        axis=1
    )
    return df

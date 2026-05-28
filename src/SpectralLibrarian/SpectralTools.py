# src/SpectralLibrarian/SpectralTools.py
"""
SpectralTools – Core preprocessing, cleaning, neutral loss, entropy, mass ↔ m/z

These are the fundamental building blocks used by SimilarityTools, IsotopeTools, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path

# Optional: re-export from adduct if you keep them there
# from .AdductTools import mass_to_mz, mz_to_mass

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
    """
    Drop peaks corresponding to the precursor ion in an MS/MS spectrum.
    
    Removes any peaks within mz_tolerance of the precursor_mz (e.g., undissociated precursor).
    
    Args:
        mz_array: Array of m/z values.
        intensity_array: Array of corresponding intensities (same length as mz_array).
        precursor_mz: The precursor m/z value.
        mz_tolerance: Tolerance window around precursor_mz to remove (default 0.01 Da).
    
    Returns:
        Filtered (mz_array, intensity_array).
    """
    if len(mz_array) == 0:
        return mz_array, intensity_array
    
    mask = np.abs(mz_array - precursor_mz) > mz_tolerance
    return mz_array[mask], intensity_array[mask]


def batch_drop_msms_precursor(
    df: pd.DataFrame,
    mz_array_col: str = 'mz_array',
    intensity_array_col: str = 'intensity_array',
    precursor_col: str = 'precursor_mz',
    mz_tolerance: float = 0.01
) -> pd.DataFrame:
    """
    Batch version to drop precursor peaks from MS/MS spectra in a DataFrame.
    
    Applies drop_msms_precursor to each row, using the specified columns.
    
    Args:
        df: Input DataFrame.
        mz_array_col: Column with m/z arrays.
        intensity_array_col: Column with intensity arrays.
        precursor_col: Column with precursor m/z values.
        mz_tolerance: Tolerance for removal (default 0.01 Da).
    
    Returns:
        Updated DataFrame with filtered arrays in place.
    """
    df = df.copy()
    
    # Vectorized implementation: Stack all arrays and precursors
    if len(df) == 0:
        return df
    
    # Get original lengths
    lengths = df[mz_array_col].apply(len)
    
    # Stack mz, intensity, and repeat precursors and row indices
    all_mz = np.concatenate(df[mz_array_col])
    all_intensity = np.concatenate(df[intensity_array_col])
    all_precursors = np.repeat(df[precursor_col].values, lengths)
    row_indices = np.repeat(np.arange(len(df)), lengths)
    
    # Vectorized mask
    mask = np.abs(all_mz - all_precursors) > mz_tolerance
    
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
    
    df[mz_array_col] = filtered_mz_list
    df[intensity_array_col] = filtered_intensity_list
    
    # Add/update 'Num Peaks'
    if 'Num Peaks' not in df.columns:
        df['Num Peaks'] = 0
    df['Num Peaks'] = pd.to_numeric(df[mz_array_col].apply(len), downcast='integer')
    
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
    max_loss: float = None,  # None means use precursor_mz for all losses
    return_fragment_index: bool = False
) -> Dict[str, np.ndarray]:
    """Convert to neutral loss spectrum."""
    if max_loss is None:
        max_loss = precursor_mz  # Use precursor_mz as max to capture all possible losses
    
    nl_mz = precursor_mz - spec["mz"]
    mask = (nl_mz >= 0) & (nl_mz <= max_loss)
    
    nl_spec = {"mz": nl_mz[mask], "intensity": spec["intensity"][mask]}
    if return_fragment_index:
        # Return indices into original spec['mz'] (as int array matching nl_mz order)
        nl_spec["nl_fragment_index"] = np.where(mask)[0].astype(int)
    
    # Standardize after calculation (sort and normalize)
    # Note: If return_fragment_index, reorder the indices to match the sorted nl_mz
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
    return -np.sum(p * np.log(p))  # Natural log


def normalized_spectral_entropy(spec: Dict[str, np.ndarray]) -> float:
    """Normalized Shannon entropy (divided by ln(N) for [0,1] range across peak counts)."""
    entropy = spectral_entropy(spec)
    n = len(spec["intensity"])
    if n <= 1:
        return 0.0  # Handle edge cases to avoid div by zero or ln(1)=0
    return entropy / np.log(n)  # Natural log


# Batch versions
def batch_clean(df: pd.DataFrame, precursor_col: str = "precursor_mz", **kwargs) -> pd.DataFrame:
    df = df.copy()
    def clean_row(row):
        spec = {"mz": row["mz_array"], "intensity": row["intensity_array"]}
        cleaned = clean_spectrum(spec, precursor_mz=row.get(precursor_col), **kwargs)
        return pd.Series({"mz_array": cleaned["mz"], "intensity_array": cleaned["intensity"]})
    cleaned = df.apply(clean_row, axis=1)
    df["mz_array"] = cleaned["mz_array"]
    df["intensity_array"] = cleaned["intensity_array"]
    
    # Add/update 'Num Peaks'
    if 'Num Peaks' not in df.columns:
        df['Num Peaks'] = 0
    df['Num Peaks'] = pd.to_numeric(df['mz_array'].apply(len), downcast='integer')
    
    return df


def batch_standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def std_row(row):
        spec = {"mz": row["mz_array"], "intensity": row["intensity_array"]}
        std = standardize_spectrum(spec)
        return pd.Series({"mz_array": std["mz"], "intensity_array": std["intensity"]})
    std = df.apply(std_row, axis=1)
    df["mz_array"] = std["mz_array"]
    df["intensity_array"] = std["intensity_array"]
    
    # Add/update 'Num Peaks'
    if 'Num Peaks' not in df.columns:
        df['Num Peaks'] = 0
    df['Num Peaks'] = pd.to_numeric(df['mz_array'].apply(len), downcast='integer')
    
    return df


def batch_neutral_loss(
    df: pd.DataFrame, 
    precursor_col: str = "precursor_mz",
    max_loss_col: str = None,  # Optional column for per-row max_loss; else use function default
    return_fragment_index: bool = False
) -> pd.DataFrame:
    df = df.copy()
    def nl_row(row):
        spec = {"mz": row["mz_array"], "intensity": row["intensity_array"]}
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


def batch_spectral_entropy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spectral_entropy"] = df.apply(
        lambda r: spectral_entropy({"mz": r["mz_array"], "intensity": r["intensity_array"]}),
        axis=1
    )
    return df


def batch_normalized_spectral_entropy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["normalized_spectral_entropy"] = df.apply(
        lambda r: normalized_spectral_entropy({"mz": r["mz_array"], "intensity": r["intensity_array"]}),
        axis=1
    )
    return df
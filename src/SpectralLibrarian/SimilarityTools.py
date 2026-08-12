# src/SpectralLibrarian/SimilarityTools.py
from __future__ import annotations

"""
SimilarityTools – Ultra-fast, publication-grade spectral similarity scoring

Tolerance API
-------------
Pass exactly one of:
    mz_tol_da   – absolute window in Daltons
    mz_tol_ppm  – window in ppm

Each backend receives the unit it expects; conversion is automatic via a
reference m/z (precursor if available, else median fragment m/z).
    modified cosine / matchms  → Da
    spectral_entropy           → ppm
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
from scipy.optimize import linear_sum_assignment
import warnings
import gc
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

# SpectralEntropy
try:
    from spectral_entropy.spectral_similarity import (
        similarity,
        multiple_similarity,
        all_similarity,
    )
    HAS_ENTROPY = True
except ImportError:
    HAS_ENTROPY = False
    warnings.warn("spectral_entropy not installed – entropy metrics disabled")

# ====================== Robust matchms import ======================
HAS_MATCHMS = False
MatchmsSpectrum = None
CosineGreedy = None
CosineHungarian = None
ModifiedCosineGreedy = None
ModifiedCosineHungarian = None

try:
    from matchms import Spectrum as MatchmsSpectrum
    from matchms.similarity import CosineGreedy, CosineHungarian

    try:
        from matchms.similarity import ModifiedCosineGreedy, ModifiedCosineHungarian
    except ImportError:
        try:
            from matchms.similarity import ModifiedCosine as _ModCos
            ModifiedCosineGreedy = _ModCos
        except ImportError:
            ModifiedCosineGreedy = None
        ModifiedCosineHungarian = None

    HAS_MATCHMS = True
except ImportError:
    warnings.warn("matchms not installed – matchms similarity methods disabled")

import dask.bag as db


# ===================================================================
# TOLERANCE RESOLUTION (Da ↔ ppm)
# ===================================================================
def _resolve_tolerances(
    mz_tol_da: Optional[float],
    mz_tol_ppm: Optional[float],
    ref_mz: float,
) -> tuple[float, float]:
    """
    Return (tol_da, tol_ppm) from whichever argument the caller supplied.

    Conversion uses ref_mz:
        ppm = (da / ref_mz) * 1e6
        da  = (ppm * ref_mz) / 1e6
    """
    if mz_tol_da is None and mz_tol_ppm is None:
        raise ValueError("Provide exactly one of: mz_tol_da or mz_tol_ppm")
    if mz_tol_da is not None and mz_tol_ppm is not None:
        raise ValueError("Provide only one of: mz_tol_da or mz_tol_ppm (not both)")

    ref = max(float(ref_mz), 1.0)

    if mz_tol_da is not None:
        tol_da = float(mz_tol_da)
        tol_ppm = (tol_da / ref) * 1e6
    else:
        tol_ppm = float(mz_tol_ppm)
        tol_da = (tol_ppm * ref) / 1e6

    return tol_da, tol_ppm


# ===================================================================
# SINGLE PAIR SCORING
# ===================================================================
def score_similarity(
    spec1: Dict[str, np.ndarray],
    spec2: Dict[str, np.ndarray],
    method: Union[str, List[str]] = "modified_cosine_greedy",
    mz_tol_da: Optional[float] = None,
    mz_tol_ppm: Optional[float] = None,
    precursor_diff: float = 0.0,
    precursor_mz1: float = None,
    precursor_mz2: float = None,
    mz_power: float = 0.0,
    intensity_power: float = 1.0,
) -> Dict[str, float]:
    """
    Compute one or more spectral similarity scores between two spectra.

    Tolerance (provide exactly one):
        mz_tol_da   – absolute window in Daltons
        mz_tol_ppm  – window in ppm

    Backends automatically receive the correct unit:
        modified cosine / matchms  → Da
        spectral_entropy           → ppm
    """
    from .SpectralTools import standardize_spectrum

    spec1 = standardize_spectrum(spec1)
    spec2 = standardize_spectrum(spec2)

    q_mz = np.asarray(spec1["mz"]).ravel()
    q_int = np.asarray(spec1["intensity"]).ravel()
    l_mz = np.asarray(spec2["mz"]).ravel()
    l_int = np.asarray(spec2["intensity"]).ravel()

    # Reference m/z for Da ↔ ppm conversion
    ref_mz = precursor_mz1 or precursor_mz2
    if ref_mz is None or ref_mz <= 0:
        ref_mz = float(np.median(q_mz)) if len(q_mz) else 200.0

    tol_da, tol_ppm = _resolve_tolerances(mz_tol_da, mz_tol_ppm, ref_mz)

    scores: Dict[str, float] = {}
    methods = [method] if isinstance(method, str) else list(method)

    CUSTOM_METHODS = {
        "modified_cosine",
        "modified_cosine_greedy",
        "modified_cosine_hungarian",
    }
    MATCHMS_METHODS = {m for m in methods if m.startswith("matchms_")}

    # ------------------------------------------------------------------
    # 1. Custom modified cosine (absolute Da)
    # ------------------------------------------------------------------
    def _modified_cosine_greedy(q_mz, q_int, l_mz, l_int, delta, tol, mz_power, intensity_power):
        if len(q_mz) == 0 or len(l_mz) == 0:
            return 0.0
        q_w = (q_mz ** mz_power) * (q_int ** intensity_power)
        l_w = (l_mz ** mz_power) * (l_int ** intensity_power)
        norm_q = np.sqrt(np.sum(q_w ** 2)) or 1.0
        norm_l = np.sqrt(np.sum(l_w ** 2)) or 1.0

        cost_d = np.abs(q_mz[:, None] - l_mz[None, :])
        i_d, j_d = np.where(cost_d <= tol)
        prods_d = q_w[i_d] * l_w[j_d]

        l_shift = l_mz + delta
        cost_s = np.abs(q_mz[:, None] - l_shift[None, :])
        i_s, j_s = np.where(cost_s <= tol)
        prods_s = q_w[i_s] * l_w[j_s]

        candidates = [
            (float(pr), int(ii), int(jj)) for ii, jj, pr in zip(i_d, j_d, prods_d)
        ]
        candidates += [
            (float(pr), int(ii), int(jj)) for ii, jj, pr in zip(i_s, j_s, prods_s)
        ]
        if not candidates:
            return 0.0
        candidates.sort(key=lambda x: x[0], reverse=True)

        used_q, used_l = set(), set()
        numer = 0.0
        for pr, i, j in candidates:
            if i not in used_q and j not in used_l:
                used_q.add(i)
                used_l.add(j)
                numer += pr
        return numer / (norm_q * norm_l)

    def _modified_cosine_hungarian(q_mz, q_int, l_mz, l_int, delta, tol, mz_power, intensity_power):
        if len(q_mz) == 0 or len(l_mz) == 0:
            return 0.0
        q_w = (q_mz ** mz_power) * (q_int ** intensity_power)
        l_w = (l_mz ** mz_power) * (l_int ** intensity_power)
        norm_q = np.sqrt(np.sum(q_w ** 2)) or 1.0
        norm_l = np.sqrt(np.sum(l_w ** 2)) or 1.0

        l_shift = l_mz + delta
        cost = np.minimum(
            np.abs(q_mz[:, None] - l_mz[None, :]),
            np.abs(q_mz[:, None] - l_shift[None, :]),
        )
        prod = q_w[:, None] * l_w[None, :]
        assign_cost = np.where(cost <= tol, -prod, 1e10)

        row_ind, col_ind = linear_sum_assignment(assign_cost)
        valid = cost[row_ind, col_ind] <= tol
        numer = np.sum(prod[row_ind[valid], col_ind[valid]]) if valid.any() else 0.0
        return numer / (norm_q * norm_l)

    for m in methods:
        if m in ("modified_cosine", "modified_cosine_greedy"):
            key = "modified_cosine" if m == "modified_cosine" else m
            scores[key] = _modified_cosine_greedy(
                q_mz, q_int, l_mz, l_int, precursor_diff, tol_da, mz_power, intensity_power
            )
        elif m == "modified_cosine_hungarian":
            scores[m] = _modified_cosine_hungarian(
                q_mz, q_int, l_mz, l_int, precursor_diff, tol_da, mz_power, intensity_power
            )

    # ------------------------------------------------------------------
    # 2. matchms methods (absolute Da)
    # ------------------------------------------------------------------
    if HAS_MATCHMS and MATCHMS_METHODS:
        q_meta = {"precursor_mz": precursor_mz1} if precursor_mz1 is not None else {}
        l_meta = {"precursor_mz": precursor_mz2} if precursor_mz2 is not None else {}
        q_sp = MatchmsSpectrum(mz=q_mz, intensities=q_int, metadata=q_meta)
        l_sp = MatchmsSpectrum(mz=l_mz, intensities=l_int, metadata=l_meta)

        if "matchms_cosine_greedy" in methods and CosineGreedy:
            try:
                scores["matchms_cosine_greedy"] = CosineGreedy(
                    tolerance=tol_da, mz_power=mz_power, intensity_power=intensity_power
                ).pair(q_sp, l_sp)["score"]
            except Exception:
                pass

        if "matchms_modified_cosine_greedy" in methods and ModifiedCosineGreedy:
            try:
                scores["matchms_modified_cosine_greedy"] = ModifiedCosineGreedy(
                    tolerance=tol_da, mz_power=mz_power, intensity_power=intensity_power
                ).pair(q_sp, l_sp)["score"]
            except Exception:
                pass

        if "matchms_modified_cosine_hungarian" in methods and ModifiedCosineHungarian:
            try:
                scores["matchms_modified_cosine_hungarian"] = ModifiedCosineHungarian(
                    tolerance=tol_da, mz_power=mz_power, intensity_power=intensity_power
                ).pair(q_sp, l_sp)["score"]
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 3. spectral_entropy methods (ppm)
    # ------------------------------------------------------------------
    if HAS_ENTROPY:
        entropy_methods = [
            m
            for m in methods
            if m not in CUSTOM_METHODS and not m.startswith("matchms_")
        ]
        if entropy_methods:
            q_arr = np.column_stack([q_mz, q_int])
            l_arr = np.column_stack([l_mz, l_int])
            try:
                entropy_scores = multiple_similarity(
                    q_arr, l_arr, methods=entropy_methods, ms2_ppm=tol_ppm
                )
                scores.update(entropy_scores)
            except Exception:
                for m in entropy_methods:
                    try:
                        scores[m] = similarity(
                            q_arr, l_arr, method=m, ms2_ppm=tol_ppm
                        )
                    except Exception:
                        scores[m] = np.nan

    return scores


# ===================================================================
# DASK BATCH SCORING
# ===================================================================
def batch_score_similarity(
    pairs_df: Optional[pd.DataFrame] = None,
    query_mz_array_col: str = "mz_array_1",
    query_intensity_array_col: str = "intensity_array_1",
    query_precursor_mz_col: str = "PRECURSORMZ_1",
    library_mz_array_col: str = "mz_array_2",
    library_intensity_array_col: str = "intensity_array_2",
    library_precursor_mz_col: str = "PRECURSORMZ_2",
    method: Union[str, List[str]] = "modified_cosine_greedy",
    mz_tol_da: Optional[float] = None,
    mz_tol_ppm: Optional[float] = None,
    mz_power: float = 0.0,
    intensity_power: float = 1.0,
    npartitions: int = 64,
    scheduler: str = "processes",
) -> pd.DataFrame:
    """
    Batch wrapper with the same unit-agnostic tolerance rules as
    ``score_similarity``.

    Provide exactly one of ``mz_tol_da`` or ``mz_tol_ppm``.
    """
    if pairs_df is None:
        raise ValueError("pairs_df is required")
    if (mz_tol_da is None) == (mz_tol_ppm is None):
        raise ValueError("Provide exactly one of: mz_tol_da or mz_tol_ppm")

    df = pairs_df.copy()
    df = df.assign(
        query_mz=df[query_mz_array_col],
        query_intensity=df[query_intensity_array_col],
        query_precursor_mz=df[query_precursor_mz_col],
        library_mz=df[library_mz_array_col],
        library_intensity=df[library_intensity_array_col],
        library_precursor_mz=df[library_precursor_mz_col],
    )
    df["precursor_diff"] = df["query_precursor_mz"] - df["library_precursor_mz"]

    def score_row(row_dict):
        q = {
            "mz": np.asarray(row_dict["query_mz"]),
            "intensity": np.asarray(row_dict["query_intensity"]),
        }
        l = {
            "mz": np.asarray(row_dict["library_mz"]),
            "intensity": np.asarray(row_dict["library_intensity"]),
        }
        return score_similarity(
            q,
            l,
            method=method,
            mz_tol_da=mz_tol_da,
            mz_tol_ppm=mz_tol_ppm,
            precursor_diff=row_dict["precursor_diff"],
            precursor_mz1=row_dict["query_precursor_mz"],
            precursor_mz2=row_dict["library_precursor_mz"],
            mz_power=mz_power,
            intensity_power=intensity_power,
        )

    bag = db.from_sequence(df.to_dict("records"), npartitions=npartitions)
    scored = bag.map(score_row)

    print(f"Computing with Dask ({npartitions} partitions, scheduler={scheduler})...")
    results = scored.compute(scheduler=scheduler)

    result_df = df.copy()
    for i, sc in enumerate(results):
        for k, v in sc.items():
            result_df.at[df.index[i], k] = v

    print("Batch scoring finished")
    return result_df


# ===================================================================
# PAIRWISE FUNCTIONS
# ===================================================================
def process_group_chunk(chunk, match_cols, dont_match_cols, tol_dict, id_col):
    n = len(chunk)
    if n < 2:
        return []
    chunk = chunk.reset_index(drop=True)
    i, j = np.triu_indices(n, k=1)
    mask = np.ones(len(i), dtype=bool)
    for col in match_cols:
        vals = chunk[col].values
        mask &= vals[i] == vals[j]
    for col in dont_match_cols:
        vals = chunk[col].values
        mask &= vals[i] != vals[j]
    for col, tol in tol_dict.items():
        vals = chunk[col].values
        mask &= np.abs(vals[i] - vals[j]) <= tol

    valid_i = i[mask]
    valid_j = j[mask]
    combined_rows = []
    cols = chunk.columns.tolist()
    for vi, vj in zip(valid_i, valid_j):
        row1 = chunk.iloc[vi]
        row2 = chunk.iloc[vj]
        row1_dict = {k + "_1": row1[k] for k in cols}
        row2_dict = {k + "_2": row2[k] for k in cols}
        row1_dict[f"{id_col}_1"] = row1[id_col]
        row2_dict[f"{id_col}_2"] = row2[id_col]
        combined_rows.append({**row1_dict, **row2_dict})
    return combined_rows


def process_group(group, match_cols, dont_match_cols, tol_dict, id_col, chunk_size=1000):
    n = len(group)
    if n < 2:
        return []
    chunks = [group.iloc[i : i + chunk_size] for i in range(0, n, chunk_size)]
    partial_process_chunk = partial(
        process_group_chunk,
        match_cols=match_cols,
        dont_match_cols=dont_match_cols,
        tol_dict=tol_dict,
        id_col=id_col,
    )
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(partial_process_chunk, chunk) for chunk in chunks]
        results = [future.result() for future in as_completed(futures)]
    return [item for sublist in results for item in sublist]


def pairwise_combinations_df(
    filtered_df,
    match_cols=None,
    dont_match_cols=None,
    tol_dict=None,
    id_col="original_row_index",
    group_chunk_size=1000,
):
    match_cols = match_cols or []
    dont_match_cols = dont_match_cols or []
    tol_dict = tol_dict or {}
    all_keys = set(match_cols + dont_match_cols + list(tol_dict.keys()) + [id_col])
    if not all_keys.issubset(filtered_df.columns):
        raise ValueError(f"Missing columns: {all_keys - set(filtered_df.columns)}")

    groups = (
        [g for _, g in filtered_df.groupby(match_cols, dropna=False)]
        if match_cols
        else [filtered_df]
    )

    partial_process = partial(
        process_group,
        match_cols=match_cols,
        dont_match_cols=dont_match_cols,
        tol_dict=tol_dict,
        id_col=id_col,
        chunk_size=group_chunk_size,
    )

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(partial_process, g) for g in groups]
        results = [future.result() for future in as_completed(futures)]

    flat_results = [item for sublist in results for item in sublist]
    if not flat_results:
        return pd.DataFrame()

    chunk_size = 100000
    df_chunks = [
        pd.DataFrame(flat_results[i : i + chunk_size])
        for i in range(0, len(flat_results), chunk_size)
    ]
    result_df = pd.concat(df_chunks, ignore_index=True)
    del flat_results, df_chunks
    gc.collect()
    return result_df

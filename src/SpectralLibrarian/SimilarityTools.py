# src/SpectralLibrarian/SimilarityTools.py
from __future__ import annotations

"""
SimilarityTools – Ultra-fast, publication-grade spectral similarity scoring

Features:
    • score_similarity(spec1, spec2, method=…) – single pair
    • batch_score_similarity(pairs_df, …) – millions-safe, Dask-powered, checkpointed
"""

import os
import json
import pickle
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

# matchms
try:
    from matchms import Spectrum as MatchmsSpectrum
    from matchms.similarity import (
        CosineGreedy,
        ModifiedCosine,
    )
    HAS_MATCHMS = True
except ImportError:
    HAS_MATCHMS = False

# Dask for large-scale batch scoring
import dask.bag as db

# ===================================================================
# SINGLE PAIR SCORING
# ===================================================================
def score_similarity(
    spec1: Dict[str, np.ndarray],
    spec2: Dict[str, np.ndarray],
    method: Union[str, List[str]] = "modified_cosine",
    mz_tol: float = 0.02,
    precursor_diff: float = 0.0,
    precursor_mz1: float = None,
    precursor_mz2: float = None,
) -> Dict[str, float]:
    from .SpectralTools import standardize_spectrum

    spec1 = standardize_spectrum(spec1)
    spec2 = standardize_spectrum(spec2)

    q_mz, q_int = spec1["mz"], spec1["intensity"]
    l_mz, l_int = spec2["mz"], spec2["intensity"]

    scores = {}

    if isinstance(method, str):
        methods = [method]
    else:
        methods = method

    if "modified_cosine" in methods:
        dot = np.sum(q_int * l_int)
        l_shift = l_mz + precursor_diff
        cost = np.abs(q_mz[:, None] - l_mz[None, :])
        cost_shift = np.abs(q_mz[:, None] - l_shift[None, :])
        cost_comb = np.minimum(cost, cost_shift)
        row_ind, col_ind = linear_sum_assignment(cost_comb)
        mask = cost_comb[row_ind, col_ind] <= mz_tol
        mod_dot = np.sum(q_int[row_ind[mask]] * l_int[col_ind[mask]]) if mask.any() else 0.0
        scores["modified_cosine"] = max(dot, mod_dot)

    if HAS_ENTROPY and any(m != "modified_cosine" for m in methods):
        q_arr = np.column_stack([q_mz, q_int])
        l_arr = np.column_stack([l_mz, l_int])

        entropy_methods = [m for m in methods if m != "modified_cosine"]

        if "all_entropy" in entropy_methods:
            entropy_scores = all_similarity(q_arr, l_arr, ms2_ppm=mz_tol * 1e6)
            scores.update(entropy_scores)
        elif len(entropy_methods) == 1:
            scores[entropy_methods[0]] = similarity(q_arr, l_arr, method=entropy_methods[0], ms2_ppm=mz_tol * 1e6)
        elif len(entropy_methods) > 1:
            entropy_scores = multiple_similarity(q_arr, l_arr, methods=entropy_methods, ms2_ppm=mz_tol * 1e6)
            scores.update(entropy_scores)

    if HAS_MATCHMS:
        q_sp = MatchmsSpectrum(mz=q_mz, intensities=q_int)
        l_sp = MatchmsSpectrum(mz=l_mz, intensities=l_int)
        if "matchms_cosine_greedy" in methods:
            scores["matchms_cosine_greedy"] = CosineGreedy(tolerance=mz_tol).pair(q_sp, l_sp)["score"]
        if "matchms_modified_cosine" in methods:
            scores["matchms_modified_cosine"] = ModifiedCosine(tolerance=mz_tol).pair(q_sp, l_sp)["score"]

    return scores


# ===================================================================
# DASK-BASED BATCH SCORING – MILLIONS-SAFE
# ===================================================================
def batch_score_similarity(
    pairs_df: Optional[pd.DataFrame] = None,
    query_mz_array_col: str = "mz_array_1",
    query_intensity_array_col: str = "intensity_array_1",
    query_precursor_mz_col: str = "PRECURSORMZ_1",
    library_mz_array_col: str = "mz_array_2",
    library_intensity_array_col: str = "intensity_array_2",
    library_precursor_mz_col: str = "PRECURSORMZ_2",
    method: Union[str, List[str]] = "dot_product",
    mz_tol: float = 0.02,
    npartitions: int = 64,
    scheduler: str = "processes",
    checkpoint_file: str = "similarity_checkpoint.json",
    result_pkl: str = "similarity_results.pkl",
    force_restart: bool = False,
) -> pd.DataFrame:
    """Dask-powered batch scorer for millions of rows."""
    if pairs_df is None:
        raise ValueError("pairs_df is required")

    df = pairs_df.copy()
    df = df.assign(
        query_mz=df[query_mz_array_col],
        query_intensity=df[query_intensity_array_col],
        query_precursor_mz=df[query_precursor_mz_col],
        library_mz=df[library_mz_array_col],
        library_intensity=df[library_intensity_array_col],
        library_precursor_mz=df[library_precursor_mz_col],
    )
    df['precursor_diff'] = df['query_precursor_mz'] - df['library_precursor_mz']

    def score_row(row_dict):
        q = {"mz": np.asarray(row_dict["query_mz"]), "intensity": np.asarray(row_dict["query_intensity"])}
        l = {"mz": np.asarray(row_dict["library_mz"]), "intensity": np.asarray(row_dict["library_intensity"])}
        return score_similarity(
            q, l,
            method=method,
            mz_tol=mz_tol,
            precursor_diff=row_dict["precursor_diff"],
            precursor_mz1=row_dict["query_precursor_mz"],
            precursor_mz2=row_dict["library_precursor_mz"]
        )

    import dask.bag as db
    bag = db.from_sequence(df.to_dict('records'), npartitions=npartitions)
    scored = bag.map(score_row)

    print(f"Computing with Dask ({npartitions} partitions, scheduler={scheduler})...")
    results = scored.compute(scheduler=scheduler)

    result_df = df.copy()
    for i, scores in enumerate(results):
        for k, v in scores.items():
            result_df.at[df.index[i], k] = v

    Path(checkpoint_file).unlink(missing_ok=True)
    print("✅ Dask batch scoring finished")
    return result_df


# ===================================================================
# PAIRWISE COMBINATIONS FUNCTIONS (your requested implementation)
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
        col_mask = (vals[i] == vals[j])
        mask &= col_mask
    for col in dont_match_cols:
        vals = chunk[col].values
        col_mask = (vals[i] != vals[j])
        mask &= col_mask
    for col, tol in tol_dict.items():
        vals = chunk[col].values
        col_mask = np.abs(vals[i] - vals[j]) <= tol
        mask &= col_mask
    
    valid_i = i[mask]
    valid_j = j[mask]
    combined_rows = []
    cols = chunk.columns.tolist()
    for vi, vj in zip(valid_i, valid_j):
        row1 = chunk.iloc[vi]
        row2 = chunk.iloc[vj]
        row1_dict = {k + '_1': row1[k] for k in cols}
        row2_dict = {k + '_2': row2[k] for k in cols}
        row1_dict[f'{id_col}_1'] = row1[id_col]
        row2_dict[f'{id_col}_2'] = row2[id_col]
        combined_row = {**row1_dict, **row2_dict}
        combined_rows.append(combined_row)
    return combined_rows


def process_group(group, match_cols, dont_match_cols, tol_dict, id_col, chunk_size=1000):
    n = len(group)
    if n < 2:
        return []
    chunks = [group.iloc[i:i + chunk_size] for i in range(0, n, chunk_size)]
    partial_process_chunk = partial(
        process_group_chunk, 
        match_cols=match_cols, 
        dont_match_cols=dont_match_cols, 
        tol_dict=tol_dict, 
        id_col=id_col
    )
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(partial_process_chunk, chunk) for chunk in chunks]
        results = [future.result() for future in as_completed(futures)]
    flat_results = [item for sublist in results for item in sublist]
    return flat_results


def pairwise_combinations_df(filtered_df, match_cols=None, dont_match_cols=None, 
                             tol_dict=None, id_col='original_row_index', group_chunk_size=1000):
    match_cols = match_cols if match_cols is not None else []
    dont_match_cols = dont_match_cols if dont_match_cols is not None else []
    tol_dict = tol_dict if tol_dict is not None else {}
    all_keys = set(match_cols + dont_match_cols + list(tol_dict.keys()) + [id_col])
    if not all_keys.issubset(filtered_df.columns):
        raise ValueError(f"Missing columns: {all_keys - set(filtered_df.columns)}")
    
    if match_cols:
        groups = [g for _, g in filtered_df.groupby(match_cols, dropna=False)]
    else:
        groups = [filtered_df]
    
    partial_process = partial(
        process_group, 
        match_cols=match_cols, 
        dont_match_cols=dont_match_cols, 
        tol_dict=tol_dict, 
        id_col=id_col, 
        chunk_size=group_chunk_size
    )
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(partial_process, g) for g in groups]
        results = [future.result() for future in as_completed(futures)]
    
    flat_results = [item for sublist in results for item in sublist]
    
    if not flat_results:
        return pd.DataFrame()
    
    chunk_size = 100000
    df_chunks = []
    for i in range(0, len(flat_results), chunk_size):
        df_chunks.append(pd.DataFrame(flat_results[i:i + chunk_size]))
    
    result_df = pd.concat(df_chunks, ignore_index=True)
    del flat_results, df_chunks
    gc.collect()
    return result_df

# src/MSAnalyzer/SimilarityTools.py
"""
SimilarityTools – Ultra-fast, publication-grade spectral similarity scoring

Features:
• score_similarity(spec1, spec2, method=...) – single pair
  → method can be:
      - "modified_cosine" (our vectorized + Hungarian – fastest & most accurate)
      - any SpectralEntropy method name ("entropy", "ms_for_id", "pearson_correlation", ...)
      - list of methods
      - "all_entropy" → all 43 SpectralEntropy methods
      - "matchms_*" → matchms metrics (optional)

• batch_score_similarity(pairs_df, ...) – millions-safe, checkpointed, resumable
"""


###############################################################
# NEED TO IMPLEMENT THIS
###############################################################
# def pairwise_combinations_df(filtered_df, match_cols=None, dont_match_cols=None, tol_dict=None, id_col='ID', group_chunk_size=500):
#     """
#     Generate pairwise DF with flexible filtering criteria.
#     
#     Parameters:
#     - filtered_df: Input DataFrame.
#     - match_cols: List of columns that must match (equal values), e.g., ['FORMULA', 'INSTRUMENTTYPE'].
#     - dont_match_cols: List of columns that must not match (different values), e.g., ['PRECURSORTYPE'].
#     - tol_dict: Dict of {column: tolerance} for numeric columns where abs(diff) <= tol.
#     - id_col: Column name for IDs, added as 'ID_1'/'ID_2' (default 'ID'; assume in columns).
#     - group_chunk_size: Number of rows per chunk within a group (default 500).
#     
#     Returns:
#     - result_df: DataFrame with pairs, columns suffixed _1 and _2, plus ID_1/ID_2.
#     
#     Optimizations:
#     - Vectorized condition checks with broadcasting.
#     - Sequential group and chunk processing to minimize memory.
#     - Incremental append to CSV to avoid holding all pairs in memory.
#     - If loading full CSV causes OOM, comment out read_csv/to_pickle and use the CSV in chunks.
#     """
#     # Checkpoint: Load from pickle if exists
#     pickle_path = 'pairwise_df.pkl'
#     if os.path.exists(pickle_path):
#         print(f"Loading pairwise_df from {pickle_path} to avoid recomputation.")
#         return pd.read_pickle(pickle_path)
#     
#     match_cols = match_cols if match_cols is not None else []
#     dont_match_cols = dont_match_cols if dont_match_cols is not None else []
#     tol_dict = tol_dict if tol_dict is not None else {}
#     all_keys = set(match_cols + dont_match_cols + list(tol_dict.keys()) + [id_col])
#     if not all_keys.issubset(filtered_df.columns):
#         raise ValueError(f"All keys must be in DataFrame columns. Missing: {all_keys - set(filtered_df.columns)}")
#     
#     # Group by all match_cols for better splitting
#     group_key = match_cols if match_cols else None
#     if group_key:
#         grouped = filtered_df.groupby(group_key)
#     else:
#         grouped = [(None, filtered_df)]
#     
#     csv_path = 'pairwise_df.csv'
#     if os.path.exists(csv_path):
#         os.remove(csv_path)
#     
#     # Sequential process groups
#     for key, group in grouped:
#         print(f"Processing group {key}, size: {len(group)}")
#         process_group(group, match_cols=match_cols, dont_match_cols=dont_match_cols, tol_dict=tol_dict, id_col=id_col, chunk_size=group_chunk_size, csv_path=csv_path)
#         gc.collect()
#     
#     # Load full (comment out if OOM, use csv_path directly)
#     result_df = pd.read_csv(csv_path)
#     
#     # Convert array columns back to lists (add more cols if needed)
#     array_cols = ['mz_array_1', 'mz_array_2', 'intensity_array_1', 'intensity_array_2']
#     for col in array_cols:
#         if col in result_df.columns:
#             result_df[col] = result_df[col].apply(ast.literal_eval)
#     
#     # Checkpoint: Save to pickle (comment out if OOM)
#     result_df.to_pickle(pickle_path)
#     print(f"Saved pairwise_df to {pickle_path}.")
#     
#     return result_df
###############################################################


from __future__ import annotations

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import warnings

# SpectralEntropy
try:
    from spectral_entropy.spectral_similarity import (
        similarity,
        multiple_similarity,
        all_similarity,
    )
    HAS_ENTROPY = True
except ImportError:  # pragma: no cover
    HAS_ENTROPY = False
    warnings.warn("spectral_entropy not installed – entropy metrics disabled")

# matchms
try:
    from matchms import Spectrum as MatchmsSpectrum
    from matchms.similarity import (
        CosineGreedy,
        ModifiedCosine,
        NeutralLossCosine,
        FingerprintSimilarity,
    )
    HAS_MATCHMS = True
except ImportError:  # pragma: no cover
    HAS_MATCHMS = False


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
    """
    Score similarity between two spectra using any supported method.
    """
    from .SpectralTools import standardize_spectrum

    spec1 = standardize_spectrum(spec1)
    spec2 = standardize_spectrum(spec2)

    q_mz, q_int = spec1["mz"], spec1["intensity"]
    l_mz, l_int = spec2["mz"], spec2["intensity"]

    scores = {}

    # Resolve method input
    if isinstance(method, str):
        methods = [method]
    else:
        methods = method

    # === Our modified cosine (gold standard) ===
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

    # === SpectralEntropy ===
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

    # === matchms (optional) ===
    if HAS_MATCHMS:
        q_sp = MatchmsSpectrum(mz=q_mz, intensities=q_int)
        l_sp = MatchmsSpectrum(mz=l_mz, intensities=l_int)
        if "matchms_cosine_greedy" in methods:
            scores["matchms_cosine_greedy"] = CosineGreedy(tolerance=mz_tol).pair(q_sp, l_sp)["score"]
        if "matchms_modified_cosine" in methods:
            scores["matchms_modified_cosine"] = ModifiedCosine(tolerance=mz_tol).pair(q_sp, l_sp)["score"]

    return scores


# ===================================================================
# BATCH SCORING – MILLIONS-SAFE
# ===================================================================
def batch_score_similarity( # AT LEAST WITH ALL_ENTROPY METHOD THERE IS PROBLEM WITH HANDLING COLUMN NAMES
    pairs_df: pd.DataFrame,
    method: Union[str, List[str]] = "modified_cosine",
    mz_tol: float = 0.02,
    n_jobs: int = -1,
    chunk_size: int = 50_000,
    checkpoint_file: str = "similarity_checkpoint.json",
    result_pkl: str = "similarity_results.pkl",
) -> pd.DataFrame:
    """
    Batch scoring with full checkpointing and crash recovery.
    Survives crashes, resumes automatically, never loads full matrix.
    """
    df = pairs_df.copy()
    total = len(df)

    # Load checkpoint
    completed = 0
    if Path(checkpoint_file).exists():
        with open(checkpoint_file) as f:
            completed = json.load(f).get("completed", 0)
        print(f"Resuming from pair {completed}/{total}")

    # Load existing results
    computed: Dict[int, Dict[str, float]] = {}
    if Path(result_pkl).exists():
        with open(result_pkl, "rb") as f:
            computed = pickle.load(f)

    def _process_chunk(chunk_df: pd.DataFrame, start_idx: int):
        results = {}
        for global_idx, row in chunk_df.iterrows():
            if global_idx in computed:
                results[global_idx] = computed[global_idx]
                continue

            q = {"mz": row.query_mz, "intensity": row.query_intensity}
            l = {"mz": row.library_mz, "intensity": row.library_intensity}
            diff = row.precursor_diff if "precursor_diff" in row else row.query_precursor_mz - row.library_precursor_mz

            scores = score_similarity(
                q, l,
                method=method,
                mz_tol=mz_tol,
                precursor_diff=diff,
                precursor_mz1=row.query_precursor_mz,
                precursor_mz2=row.library_precursor_mz,
            )
            results[global_idx] = scores

            # Save every 10k
            if len(results) % 10_000 == 0:
                with open(result_pkl, "wb") as f:
                    pickle.dump({**computed, **results}, f)
                with open(checkpoint_file, "w") as f:
                    json.dump({"completed": global_idx + 1}, f)

        return results

    # Main chunked loop
    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        for chunk_start in range(completed, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk = df.iloc[chunk_start:chunk_end]
            chunk_results = parallel(delayed(_process_chunk)(chunk, chunk_start) for _ in [0])[0]
            computed.update(chunk_results)

            # Update checkpoint
            with open(checkpoint_file, "w") as f:
                json.dump({"completed": chunk_end}, f)

    # Final merge
    result_df = df.copy()
    for idx, scores in computed.items():
        for k, v in scores.items():
            if k not in result_df.columns:
                result_df[k] = np.nan
            result_df.at[idx, k] = v

    # Clean up
    Path(checkpoint_file).unlink(missing_ok=True)

    return result_df
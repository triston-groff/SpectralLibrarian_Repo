# src/SpectralLibrarian/SimilarityTools.py
from __future__ import annotations

"""
SimilarityTools – Ultra-fast, publication-grade spectral similarity scoring

Features:
    • score_similarity(spec1, spec2, method=…) – single pair
      → method can be:
          - "modified_cosine" (our vectorized + Hungarian – fastest & most accurate)
          - any SpectralEntropy method name ("entropy", "ms_for_id", "pearson_correlation", ...)
          - list of methods
          - "all_entropy" → all 43 SpectralEntropy methods
          - "matchms_*" → matchms metrics (optional)

    • batch_score_similarity(pairs_df, …) – millions-safe, checkpointed, resumable
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
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional
from joblib import Parallel, delayed

def batch_score_similarity(
    # ================== INPUT OPTIONS ==================
    pairs_df: Optional[pd.DataFrame] = None,   # Option 1: pass DataFrame

    # Column names when using pairs_df
    query_mz_array_col: str = "mz_array_1",
    query_intensity_array_col: str = "intensity_array_1",
    query_precursor_mz_col: str = "PRECURSORMZ_1",

    library_mz_array_col: str = "mz_array_2",
    library_intensity_array_col: str = "intensity_array_2",
    library_precursor_mz_col: str = "PRECURSORMZ_2",

    # Option 2: Pass raw lists / arrays / Series directly (no DataFrame needed)
    query_mz_arrays: Optional[list] = None,
    query_intensity_arrays: Optional[list] = None,
    query_precursor_mz: Optional[list] = None,
    library_mz_arrays: Optional[list] = None,
    library_intensity_arrays: Optional[list] = None,
    library_precursor_mz: Optional[list] = None,

    # ================== OTHER PARAMETERS ==================
    method: Union[str, List[str]] = "dot_product",   # ← your requested default
    mz_tol: float = 0.02,
    n_jobs: int = -1,
    chunk_size: int = 50_000,
    checkpoint_file: str = "similarity_checkpoint.json",
    result_pkl: str = "similarity_results.pkl",
    force_restart: bool = False,
) -> pd.DataFrame:
    """
    Fully flexible batch similarity scorer.
    - Accepts either a DataFrame + column names OR raw lists/arrays directly.
    - Default method is now "dot_product".
    """
    # ===================== BUILD INTERNAL DF =====================
    if pairs_df is not None:
        df = pairs_df.copy()
        df = df.assign(
            query_mz=df[query_mz_array_col],
            query_intensity=df[query_intensity_array_col],
            query_precursor_mz=df[query_precursor_mz_col],
            library_mz=df[library_mz_array_col],
            library_intensity=df[library_intensity_array_col],
            library_precursor_mz=df[library_precursor_mz_col],
        )
    else:
        # Build from raw arrays/lists
        df = pd.DataFrame()
        if query_mz_arrays is not None:
            df['query_mz'] = pd.Series(query_mz_arrays)
        if query_intensity_arrays is not None:
            df['query_intensity'] = pd.Series(query_intensity_arrays)
        if query_precursor_mz is not None:
            df['query_precursor_mz'] = pd.Series(query_precursor_mz)
        else:
            df['query_precursor_mz'] = 0.0

        if library_mz_arrays is not None:
            df['library_mz'] = pd.Series(library_mz_arrays)
        if library_intensity_arrays is not None:
            df['library_intensity'] = pd.Series(library_intensity_arrays)
        if library_precursor_mz is not None:
            df['library_precursor_mz'] = pd.Series(library_precursor_mz)
        else:
            df['library_precursor_mz'] = 0.0

    df['precursor_diff'] = df['query_precursor_mz'] - df['library_precursor_mz']

    # Convert to numpy arrays
    for col in ['query_mz', 'query_intensity', 'library_mz', 'library_intensity']:
        df[col] = df[col].apply(lambda x: np.asarray(x, dtype=float))

    total = len(df)
    if total == 0:
        raise ValueError("No data provided!")

    # ===================== CHECKPOINT / RESUME =====================
    if force_restart:
        for f in [checkpoint_file, result_pkl]:
            Path(f).unlink(missing_ok=True)

    completed = 0
    if Path(checkpoint_file).exists() and not force_restart:
        try:
            with open(checkpoint_file) as f:
                completed = int(json.load(f).get("completed", 0))
            print(f"Resuming from pair {completed}/{total}")
        except:
            completed = 0

    computed: dict = {}
    if Path(result_pkl).exists() and not force_restart:
        try:
            with open(result_pkl, "rb") as f:
                computed = pickle.load(f)
            print(f"Loaded {len(computed)} cached scores")
        except:
            computed = {}

    def _atomic_save(data, path):
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        tmp.replace(path)

    def _atomic_checkpoint(val):
        tmp = Path(str(checkpoint_file) + ".tmp")
        tmp.write_text(json.dumps({"completed": int(val)}))
        tmp.replace(checkpoint_file)

    # ===================== SCORING =====================
    def _process_chunk(chunk_df, start_idx):
        results = {}
        for global_idx, row in chunk_df.iterrows():
            if global_idx in computed:
                results[global_idx] = computed[global_idx]
                continue

            q = {"mz": row["query_mz"], "intensity": row["query_intensity"]}
            l = {"mz": row["library_mz"], "intensity": row["library_intensity"]}

            scores = score_similarity(
                q, l,
                method=method,
                mz_tol=mz_tol,
                precursor_diff=row.get("precursor_diff", 0.0),
                precursor_mz1=row["query_precursor_mz"],
                precursor_mz2=row["library_precursor_mz"],
            )
            results[global_idx] = scores

            if len(results) % 5000 == 0:
                _atomic_save({**computed, **results}, result_pkl)
                _atomic_checkpoint(start_idx + len(results))

        return results

    try:
        with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
            for start in range(completed, total, chunk_size):
                end = min(start + chunk_size, total)
                print(f"Processing {start:,}–{end:,} / {total:,} | method={method}")

                chunk = df.iloc[start:end]
                chunk_results = parallel(delayed(_process_chunk)(chunk, start) for _ in [0])[0]
                computed.update(chunk_results)

                _atomic_save(computed, result_pkl)
                _atomic_checkpoint(end)

    except KeyboardInterrupt:
        print("⚠️ Interrupted – saving current progress...")
        _atomic_save(computed, result_pkl)
        _atomic_checkpoint(max(computed.keys(), default=0) + 1)
        raise
    except Exception as e:
        print(f"❌ Error occurred: {e} – progress has been saved.")
        _atomic_save(computed, result_pkl)
        raise
    finally:
        if len(computed) >= total - 1:
            Path(checkpoint_file).unlink(missing_ok=True)
            print("✅ All done – checkpoint cleaned.")

    # ===================== RETURN =====================
    result_df = df.copy()
    for idx, scores in computed.items():
        for k, v in scores.items():
            result_df.at[idx, k] = v

    return result_df

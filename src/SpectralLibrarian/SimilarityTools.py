# src/MSAnalyzer/SimilarityTools.py
# from .SpectralTools import *
# from .utilities import *

from .SpectralTools import str2spec
from .utilities import parallelize

from typing import *
from itertools import combinations
from collections import defaultdict

import pandas as pd, pickle as pkl
import gc


# TODO : New
def pairwise_combinations_df(inputpath:str, resultpath:str, id_col:str, match_cols:Union[str, List[str]], dont_match_cols:Union[str, List[str]],
                             method:Union[str, List[str]] = "modified_cosine", mz_tol: float = 0.02, tol_dict:Dict[str, float]={}):
    """
    Generate pairwise DF with flexible filtering criteria.

    Parameters:
    - filtered_df: Input DataFrame.
    - resultpath: Path of output file, should be txt, tsv, csv, or xlsx file
    - match_cols: List of columns that must match (equal values), e.g., ['FORMULA', 'INSTRUMENTTYPE'].
    - dont_match_cols: List of columns that must not match (different values), e.g., ['PRECURSORTYPE'].
    - tol_dict: Dict of {column: tolerance} for numeric columns where abs(diff) <= tol.
    - id_col: Column name for IDs, added as 'ID_1'/'ID_2' (default 'ID'; assume in columns).

    Returns:
    - result_df: DataFrame with pairs, columns suffixed _1 and _2, plus ID_1/ID_2.

    Optimizations:
    - Vectorized condition checks with broadcasting.
    - Sequential group and chunk processing to minimize memory.
    - Incremental append to CSV to avoid holding all pairs in memory.
    - If loading full CSV causes OOM, comment out read_csv/to_pickle and use the CSV in chunks.
    """
    ext2sep = {"txt": "\t", "tsv": "\t", "csv": ",", "xlsx": None, "pkl": None}
    inext = inputpath.split(".")[-1]
    if not (inext in ext2sep.keys()):
        raise ValueError("The extender of 'inputpath' should be one of the followings : " + str(list(ext2sep.keys())))

    if inext == "pkl":
        with open(inputpath, "rb") as f:
            input_df = pkl.load(f)
    elif inext == "xlsx":
        input_df = pd.read_excel(inputpath)
    else:
        input_df = pd.read_csv(inputpath, sep=inext)

    outext = resultpath.split(".")[-1]
    if not (outext in ext2sep.keys()):
        raise ValueError("The extender of 'resultpath' should be one of the followings : " + str(list(ext2sep.keys())))

    # Argument QC
    match_cols = [match_cols] if type(match_cols)==str else match_cols
    dont_match_cols = [dont_match_cols] if type(dont_match_cols)==str else dont_match_cols
    all_keys = set(match_cols + dont_match_cols + list(tol_dict.keys()) + [id_col])
    if not all_keys.issubset(input_df.columns):
        raise ValueError(f"All keys must be in DataFrame columns. Missing: {all_keys - set(input_df.columns)}")

    # Checkpoint: Load previous file if exists
    if os.path.exists(resultpath):
        print(f"Loading pairwise_df from {resultpath} to avoid recomputation.")
        if outext == "xlsx":
            df_completed = pd.read_excel(resultpath)
        elif outext == "pkl":
            with open(resultpath, "rb") as f:
                df_completed = pkl.load(f)
        else:
            df_completed = pd.read_csv(resultpath, sep=ext2sep[outext])
        completed_ids = np.unique(df_completed[[id_col+"_1", id_col+"_2"]].values.reshape(-1))
        df_compute = input_df.loc[~input_df[id_col].isin(completed_ids)]
    else:
        df_completed = pd.DataFrame()
        df_compute = input_df.copy()

    # Group by all match_cols for better splitting : Sequential process across groups, parallel process within groups
    compute_groups = list(df_compute.groupby(match_cols))
    for i, kg in enumerate(compute_groups):
        key, df_group = kg
        if len(df_group) > 1:
            print(str((i + 1) * 100 / len(compute_groups)) + " % groups")
            print(f"Processing group {key}")
            df_group = df_group.reset_index(drop=True)
            df_scores = process_group(df_group, dont_match_cols=dont_match_cols, method=method, mz_tol=mz_tol)
            df_completed = pd.concat([df_completed.reset_index(drop=True), df_scores.reset_index(drop=True)], axis=0)
            gc.collect()

    if len(df_completed) == 0:
        print("No pairs to be compared. Relieve constraints 'match_cols' and 'dont_match_cols', and try again")

    if not os.path.exists(os.path.dirname(resultpath)):
        os.makedirs(os.path.dirname(resultpath), exist_ok=True)

    if outext == "xlsx":
        df_completed.to_excel(resultpath, index=False)
    elif outext == "pkl":
        with open(resultpath, "wb") as f:
            pkl.dump(df_completed, f)
    else: # .csv
        df_completed.to_csv(resultpath, sep=ext2sep[outext], index=False)


def process_group(df_group:pd.DataFrame, dont_match_cols:List[str], method: Union[str, List[str]] = "modified_cosine", mz_tol: float = 0.02):
    ids = list((df_group["ID"]))
    precmzs = list(df_group["Precursor m/z"])
    paircombidxs = combinations(list(range(len(df_group))), 2)

    idmat, arglist = [], []
    for idx1, idx2 in paircombidxs:
        skip_this = False
        for dmc in dont_match_cols:
            if df_group.iloc[idx1][dmc] == df_group.iloc[idx2][dmc]:
                skip_this = True # Should not be identical, so skip this pair
                break

        if skip_this:
            continue

        for id1, id2 in [[idx1, idx2], [idx2, idx1]]:
            spec1 = str2spec(specstr=df_group.iloc[id1]["ms2spectrum"])
            spec2 = str2spec(specstr=df_group.iloc[id2]["ms2spectrum"])
            if not ((spec1 is None) or (spec2 is None)):
                idmat.append([ids[id1], ids[id2]])
                pmz1, pmz2 = precmzs[id1], precmzs[id2]
                arglist.append((spec1, spec2, method, mz_tol, pmz1-pmz2, pmz1, pmz2))

    print("Found " + str(len(arglist)) + " not matching by " + str(dont_match_cols))
    if len(arglist) > 0:
        df_idpairs = pd.DataFrame.from_records(idmat, columns=["ID_1", "ID_2"])
        if len(arglist) > 15:
            group_results = parallelize(workfunc=score_similarity, num_cores=os.cpu_count()-2, argslist=arglist)
        else:
            group_results = []
            for arg in arglist:
                spec1, spec2, method, mz_tol, precursor_diff, pmz1, pmz2 = arg
                group_results.append(score_similarity(spec1, spec2, method, mz_tol, precursor_diff, pmz1, pmz2))
        results_dict = merge_list_of_dict(data=group_results)
        df_scores = pd.concat([df_idpairs, pd.DataFrame.from_dict(results_dict)], axis=1)
    else:
        df_scores = pd.DataFrame()
    return df_scores


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
##############################################################


# TODO: Following does not work

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
    from ms_entropy import calculate_entropy_similarity
    # from spectral_entropy.spectral_similarity import similarity, multiple_similarity, all_similarity
    # from ms_entropy.
    HAS_ENTROPY = True
except ImportError:  # pragma: no cover
    HAS_ENTROPY = False
    warnings.warn("spectral_entropy not installed – entropy metrics disabled")

# matchms
try:
    from matchms import Spectrum
    from matchms.similarity import *
    HAS_MATCHMS = True
    matchms_mapper = {
        # "BinnedEmbeddingSimilarity": BinnedEmbeddingSimilarity,
        "BlinkCosine": BlinkCosine,
        "CosineGreedy": CosineGreedy,
        "CosineHungarian": CosineHungarian,
        # "FingerprintSimilarity": FingerprintSimilarity,
        "FlashSimilarity": FlashSimilarity,
        # "IntersectMz": IntersectMz,
        # "MetadataMatch": MetadataMatch,
        "ModifiedCosineGreedy": ModifiedCosineGreedy,
        "ModifiedCosineHungarian": ModifiedCosineHungarian,
        "NeutralLossesCosine": NeutralLossesCosine,
        # "ParentMassMatch": ParentMassMatch,
        # "PrecursorMzMatch": PrecursorMzMatch,
    }
except ImportError:  # pragma: no cover
    HAS_MATCHMS = False


from .SpectralTools import normalizeSpectrum, spec2array

def score_similarity(qspec: np.ndarray, lspec: np.ndarray, method: Union[str, List[str]] = "modified_cosine",
                     mz_tol: float = 0.02, precursor_diff: float = 0.0, precursor_mz1: float = None, precursor_mz2: float = None) -> Dict[str, float]:
    """
    Score similarity between two spectra using any supported method.
    method: string ("modified_cosine" (default), 'entropy_similarity' (by ms_entropy), 'other matchms similarity'), or a List of string of those methods
    """
    qspec_norm = normalizeSpectrum(qspec)
    lspec_norm = normalizeSpectrum(lspec)

    # Resolve method input
    methods = [method] if isinstance(method, str) else method
    scores = {}

    # === SpectralEntropy ===
    if HAS_ENTROPY and ("entropy_similarity" in methods):
        scores["entropy_similarity"] = calculate_entropy_similarity(qspec_norm, lspec_norm)

    q_mz, q_int = spec2array(qspec_norm)
    l_mz, l_int = spec2array(lspec_norm)
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

    # === matchms (optional) ===
    if HAS_MATCHMS and any(m in matchms_mapper for m in methods):
        q_sp = Spectrum(mz=q_mz, intensities=q_int, metadata={"precursor_mz": precursor_mz1})
        l_sp = Spectrum(mz=l_mz, intensities=l_int, metadata={"precursor_mz": precursor_mz2})
        for m in methods:
            if m in matchms_mapper:
                if m in ["FlashSimilarity"]:
                    scores[m] = float(matchms_mapper[m](tolerance=mz_tol).pair(q_sp, l_sp))
                else:
                    scores[m] = float(matchms_mapper[m](tolerance=mz_tol).pair(q_sp, l_sp)["score"])

    return scores


# ===================================================================
# BATCH SCORING – MILLIONS-SAFE
# ===================================================================
# AT LEAST WITH ALL_ENTROPY METHOD THERE IS PROBLEM WITH HANDLING COLUMN NAMES
def batch_score_similarity(
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
            results[global_idx] = score_similarity(q, l, method=method, mz_tol=mz_tol, precursor_diff=diff, precursor_mz1=row.query_precursor_mz, precursor_mz2=row.library_precursor_mz)

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

__all__ = ["pairwise_combinations_df", "score_similarity", "batch_score_similarity"]
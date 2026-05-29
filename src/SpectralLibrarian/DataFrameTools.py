# src/MSAnalyzer/DataFrameTools.py
"""
dataframe_utils – Clean, reusable pandas utilities for spectral library handling
Modeled after Corinna Brungs' msn_tree_library/pandas_utils.py (2025 standard)
"""

from typing import *
import pandas as pd


def isnull(o: Any) -> bool:
    if o is None:
        return True
    if pd.isna(o):
        return True
    if isinstance(o, str):
        s = o.lower()
        return s in {"", "na", "n/a", "nan", "<na>"}
    return False


def notnull(o: Any) -> bool:
    return not isnull(o)


def isnull_or_empty(o: Any) -> bool:
    return isnull(o) or (hasattr(o, "__len__") and len(o) == 0)


def enforce_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Add missing columns as NaN (in-place False, returns df for chaining)."""
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def reorder_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Put specified columns first, keep rest in original order."""
    extra = [c for c in df.columns if c not in columns]
    return df[list(columns) + extra]


def enforce_dtypes(df: pd.DataFrame, dtype_map: dict[str, str]) -> pd.DataFrame:
    """Apply dtypes in one shot – respects pandas nullable types."""
    for col, dtype in dtype_map.items():
        if col in df.columns:
            if dtype == "category":
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(dtype)
    return df


def clean_empty_lists_and_strings(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: None if isnull_or_empty(x) else x)
    return df


def update_columns_in_dfs(dfs, colmap: dict[str, dict], use_regex=False):
    """
    Updates values in specified columns across multiple DataFrames based on a mapping dictionary.
    - For each key-value in mapping where value is not None, replaces matching values in the columns.
    - For each key where value is None, drops rows where the column value matches the key.
    - Modifications are done in-place on the DataFrames.

    Parameters:
    - dfs: List of pandas DataFrames to update.
    - colmap: Dictionary of column (key) and mapping (value) .
    - use_regex: Boolean, whether to treat keys as regex patterns (default: False).
    """
    dfs_new = []
    for df in dfs:
        df_new = df.copy()
        for col, mapping in colmap.items():
            if col in df.columns:
                replace_dict = {k: v for k, v in mapping.items() if v is not None}
                drop_keys = [k for k, v in mapping.items() if v is None]

                # Handle replacements (exclude None values)
                if replace_dict:
                    df_new[col] = df_new[col].replace(replace_dict, regex=use_regex)

                # Handle drops (keys where value is None)
                if drop_keys:
                    mask = df_new[col].str.contains('|'.join(drop_keys), regex=True, na=False) if use_regex else df_new[col].isin(drop_keys)
                    df_new.drop(df_new[mask].index, inplace=True) # Drop matching rows
                    df_new.reset_index(drop=True, inplace=True) # Reset index after drops (optional, but helps avoid fragmented indices)
        dfs_new.append(df_new)
    return dfs_new


def extract_unique_values(dfs, columns=None):
    """
    Extracts unique values for specified columns across multiple DataFrames.

    Parameters:
    - dfs: List of pandas DataFrames (e.g., [spx.df, msn_lib.df, mona.df, nist23.df]).
    - columns: List of column names to extract uniques from (default: the ones you specified).

    Returns:
    - A tuple of sorted lists: (unique_smiles_harmonized, unique_inchikey_harmonized, unique_precursortype,
      unique_instrument, unique_instrumenttype, unique_ionization)
    - If a column is missing in all DFs, its list will be empty.
    """
    if columns is None:
        columns = ['SMILES_harmonized', 'INCHIKEY_harmonized', 'PRECURSORTYPE', 'INSTRUMENT', 'INSTRUMENTTYPE',
                   'IONIZATION']

    uniques = {col: set() for col in columns}

    for df in dfs:
        for col in columns:
            if col in df.columns:
                uniques[col].update(df[col].dropna().unique())

    # Convert to sorted lists
    unique_lists = [sorted(uniques[col]) for col in columns]

    return tuple(unique_lists)


def rename_columns(df, rename: dict, inplace=True):
    rename_new = {k: v for k, v in rename.items() if k in df.columns}
    if inplace:
        df.rename(columns=rename_new, inplace=True)
    else:
        return df.rename(columns=rename_new)


def cast_dtype(df: pd.DataFrame, dtype_dict: dict) -> pd.DataFrame:
    dtype_apply = {k: v for k, v in dtype_dict.items() if k in df.columns}
    return df.astype(dtype=dtype_apply)


def lower_columns(df, protected_cols=None):
    df_new = df.copy()
    df_new.rename(columns=lambda x: x.lower(), inplace=True)  # But skip protected cols
    if not (protected_cols is None):
        for col in protected_cols:
            if col.lower() in df_new.columns:
                df_new.rename(columns={col.lower(): col}, inplace=True)
    return df_new


def drop_duplicated_columns(df):
    #  Handle potential duplicate columns (e.g., 'smiles_isomeric' and 'SMILES_isomeric'), Keep uppercase if both exists
    df_new = df.copy()
    cols = df_new.columns
    for col in cols:
        lower_col = col.lower()
        if lower_col != col and lower_col in cols:
            df_new.drop(columns=[lower_col], inplace=True)
    return df_new


def replace_realcol_na(df, col_dtype_dict, fillas=0):
    df_new = df.copy()
    int_cols = [col for col, dtype in col_dtype_dict.items() if dtype == int]
    float_cols = [col for col, dtype in col_dtype_dict.items() if dtype == float]

    for col in int_cols + float_cols:
        if col in df_new.columns:
            df_new[col] = df[col].fillna(fillas)
    return df_new


def remove_columns(df, exclude_patterns, keep_columns):
    query_cols_to_keep = [col for col in df.columns if (not any(pat in col.lower() for pat in exclude_patterns)) or (col in keep_columns)]
    return df[query_cols_to_keep].copy()


__all__ =["isnull", "notnull", "isnull_or_empty", "enforce_columns", "reorder_columns", "enforce_dtypes", "clean_empty_lists_and_strings", "update_columns_in_dfs", "extract_unique_values",
          "rename_columns", "cast_dtype", "lower_columns", "drop_duplicated_columns", "replace_realcol_na", "remove_columns"]

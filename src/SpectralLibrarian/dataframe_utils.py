# src/SpectralLibrarian/dataframe_utils.py
"""
dataframe_utils – Clean, reusable pandas utilities for spectral library handling
Modeled after Corinna Brungs' msn_tree_library/pandas_utils.py (2025 standard)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Iterable, Sequence, Any


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
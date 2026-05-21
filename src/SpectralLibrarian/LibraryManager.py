# src/MSAnalyzer/LibraryManager.py
"""
LibraryManager – MSP/MGF/JSON parser with annotated peak support
- mz_array, intensity_array, annotations_array all guaranteed same length
- annotations_array uses np.ndarray[object] (None for peaks without annotation)
- Now supports GNPS-style .json (JSON Lines / NDJSON) libraries
- ROBUSTNESS FIX (v0.4.1): Handles scalar None / 0-d arrays / Num Peaks=1 cases
  that survived heavy dataframe filtering/renaming
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union


class LibraryManager:
    def __init__(
        self,
        input: Union[str, Path, pd.DataFrame],
        library_type: str = "auto",
        df: pd.DataFrame | None = None,
    ):
        if df is not None:
            input = df

        if isinstance(input, (str, Path)):
            file_path = Path(input)
            if library_type == "auto":
                library_type = file_path.suffix.lower().lstrip(".")
            
            if library_type not in {"msp", "mgf", "json"}:
                raise ValueError("library_type must be 'msp', 'mgf', or 'json'")

            if library_type == "msp":
                parser = self._parse_msp
            elif library_type == "mgf":
                parser = self._parse_mgf
            elif library_type == "json":
                parser = self._parse_json
            else:
                raise ValueError("library_type must be 'msp', 'mgf', or 'json'")

            raw_df = parser(file_path)
        elif isinstance(input, pd.DataFrame):
            raw_df = input.copy()
        else:
            raise ValueError("Input must be a file path (str/Path) or a pandas DataFrame")

        # === Enforce consistent array columns ===
        for col, dtype in [("mz_array", np.float32), 
                           ("intensity_array", np.float32), 
                           ("annotations_array", object)]:
            if col not in raw_df.columns:
                raw_df[col] = [np.array([], dtype=dtype) for _ in range(len(raw_df))]
            else:
                raw_df[col] = raw_df[col].apply(
                    lambda x: np.array(x, dtype=dtype) 
                    if isinstance(x, (list, np.ndarray)) 
                    else np.array([], dtype=dtype)
                )

        # === Standardize 'Num Peaks' ===
        potential_cols = [col for col in raw_df.columns if re.sub(r'[ _]', '', col.lower()) == 'numpeaks']
        if potential_cols:
            keep_col = potential_cols[0]
            for extra in potential_cols[1:]:
                raw_df.drop(extra, axis=1, inplace=True)
            if keep_col != 'Num Peaks':
                raw_df.rename(columns={keep_col: 'Num Peaks'}, inplace=True)

        # Force 'Num Peaks' to match actual array length
        raw_df['Num Peaks'] = raw_df['mz_array'].apply(len)

        # === Ensure annotations_array always matches mz_array length (ROBUST VERSION) ===
        def ensure_annotation_length(row):
            mz_len = len(row['mz_array'])
            ann = row.get('annotations_array')
            
            # CRITICAL FIX: handle scalar None, 0-d arrays, non-arrays, wrong length
            if (ann is None or 
                not isinstance(ann, np.ndarray) or 
                ann.ndim != 1 or 
                len(ann) != mz_len):
                row['annotations_array'] = np.array([None] * mz_len, dtype=object)
            return row

        raw_df = raw_df.apply(ensure_annotation_length, axis=1)

        self.df = raw_df

    # ... (all _parse_* methods unchanged) ...

    def _parse_peak_line(self, line: str):
        """Parse mz intensity [annotation] — supports tab or space separated."""
        parts = line.strip().split('\t')
        if len(parts) < 2:
            parts = line.strip().split()

        try:
            mz = float(parts[0])
            intensity = float(parts[1])
        except (ValueError, IndexError):
            return None, None, None

        annotation = None
        if len(parts) > 2:
            ann_raw = ' '.join(parts[2:]).strip()
            if ann_raw.startswith('"') and ann_raw.endswith('"'):
                annotation = ann_raw[1:-1].strip()
            else:
                annotation = ann_raw.strip()

        return mz, intensity, annotation

    def _parse_msp(self, file_path: str) -> pd.DataFrame:
        # (unchanged)
        records = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            meta = {}
            mz_list = []
            inty_list = []
            ann_list = []

            for line in f:
                line = line.strip()
                if not line:
                    if meta or mz_list:
                        if mz_list:
                            meta['mz_array'] = np.array(mz_list, dtype=np.float32)
                            meta['intensity_array'] = np.array(inty_list, dtype=np.float32)
                            meta['annotations_array'] = np.array(ann_list, dtype=object)
                        records.append(meta)
                        meta = {}
                        mz_list.clear()
                        inty_list.clear()
                        ann_list.clear()
                    continue

                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if key.lower() in ['num peaks', 'numpeaks']:
                        num_peaks = int(value)
                        for _ in range(num_peaks):
                            peak_line = f.readline().strip()
                            mz, intensity, annotation = self._parse_peak_line(peak_line)
                            if mz is not None:
                                mz_list.append(mz)
                                inty_list.append(intensity)
                                ann_list.append(annotation)
                    else:
                        meta[key] = value
                else:
                    mz, intensity, annotation = self._parse_peak_line(line)
                    if mz is not None:
                        mz_list.append(mz)
                        inty_list.append(intensity)
                        ann_list.append(annotation)

            if meta or mz_list:
                if mz_list:
                    meta['mz_array'] = np.array(mz_list, dtype=np.float32)
                    meta['intensity_array'] = np.array(inty_list, dtype=np.float32)
                    meta['annotations_array'] = np.array(ann_list, dtype=object)
                records.append(meta)

        return pd.DataFrame(records)

    def _parse_mgf(self, file_path: str) -> pd.DataFrame:
        # (unchanged - already good)
        records = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if line != "BEGIN IONS":
                continue

            meta: Dict[str, Any] = {}
            peaks = []

            while i < len(lines):
                line = lines[i].strip()
                i += 1
                if line == "END IONS":
                    break
                if not line or line.startswith(("#", ";")):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key in meta:
                        if isinstance(meta[key], list):
                            meta[key].append(val)
                        else:
                            meta[key] = [meta[key], val]
                    else:
                        meta[key] = val
                else:
                    try:
                        m, i_val, *_ = line.split()
                        peaks.append((float(m), float(i_val)))
                    except:
                        pass

            if peaks:
                mz, inty = zip(*peaks)
                meta['mz_array'] = np.array(mz, dtype=np.float32)
                meta['intensity_array'] = np.array(inty, dtype=np.float32)
                meta['annotations_array'] = np.array([None] * len(mz), dtype=object)
            records.append(meta)

        return pd.DataFrame(records)

    def _parse_json(self, file_path: str) -> pd.DataFrame:
        # (unchanged - already good)
        df = pd.read_json(
            file_path,
            lines=True,
            encoding_errors="ignore"
        )

        def process_peaks(row):
            peaks = row.get("peaks", [])
            mz_list: list[float] = []
            int_list: list[float] = []
            if isinstance(peaks, list):
                for p in peaks:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        try:
                            mz_list.append(float(p[0]))
                            int_list.append(float(p[1]))
                        except (ValueError, TypeError):
                            pass
            mz_arr = np.array(mz_list, dtype=np.float32)
            int_arr = np.array(int_list, dtype=np.float32)
            ann_arr = np.array([None] * len(mz_arr), dtype=object)

            row["mz_array"] = mz_arr
            row["intensity_array"] = int_arr
            row["annotations_array"] = ann_arr
            return row

        df = df.apply(process_peaks, axis=1)

        exclude = {"mz_array", "intensity_array", "annotations_array", "peaks"}
        for col in [c for c in df.columns if c not in exclude]:
            def to_array(x):
                if isinstance(x, list):
                    try:
                        return np.array(x, dtype=np.float32)
                    except (ValueError, TypeError):
                        return np.array(x, dtype=object)
                return x
            df[col] = df[col].apply(to_array)

        if "peaks" in df.columns:
            df = df.drop(columns=["peaks"])

        return df

    def to_msp(self, output_path: str) -> None:
        """Export to MSP format. Works whether annotations_array exists or not.
        ROBUST VERSION: handles scalar None and 0-d arrays gracefully."""
        with open(output_path, 'w') as f:
            for _, row in self.df.iterrows():
                # Write metadata
                for col in self.df.columns:
                    if col in ['mz_array', 'intensity_array', 'annotations_array', 'Num Peaks']:
                        continue
                    
                    value = row[col]
                    if (not isinstance(value, (np.ndarray, list, tuple, pd.Series)) and
                        pd.notna(value)):
                        f.write(f"{col.upper()}: {value}\n")

                # === Peak block ===
                mz_arr = row['mz_array']
                int_arr = row['intensity_array']
                ann_arr = row.get('annotations_array')

                f.write(f"Num Peaks: {len(mz_arr)}\n")

                # ROBUST check for annotations_array
                if not (isinstance(ann_arr, np.ndarray) and ann_arr.ndim == 1 and len(ann_arr) == len(mz_arr)):
                    ann_arr = np.array([None] * len(mz_arr), dtype=object)

                for i in range(len(mz_arr)):
                    if ann_arr[i] is not None:
                        f.write(f"{mz_arr[i]} {int_arr[i]} \"{ann_arr[i]}\"\n")
                    else:
                        f.write(f"{mz_arr[i]} {int_arr[i]}\n")
                f.write("\n")


__all__ = ["LibraryManager"]
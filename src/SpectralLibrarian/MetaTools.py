# src/SpectralLibrarian/MetaTools.py
"""
MetaTools – PubChem search + RDKit-based metadata harmonization, classification, and fragment neutralization
"""

from __future__ import annotations

import time
import pandas as pd
import concurrent.futures
from typing import List, Iterable, Any
from pubchempy import Compound, get_compounds, get_cids
from datetime import datetime


#### UPDATE HARMONIZATION: THIS WORKED WELL IN DATAFRAME ####
#### Double check that InChI and InChIKey contain stereochem info ####
# from rdkit import Chem
# import pandas as pd
# 
# def harmonize_molecules(df, smiles_col='SMILES'):
#     """
#     Efficiently adds harmonized SMILES (canonical and isomeric), InChI, and InChIKey columns to the DataFrame.
#     Processes only unique SMILES to avoid redundant computations on duplicates.
#     
#     Parameters:
#     - df: pandas DataFrame containing the SMILES column.
#     - smiles_col: Name of the column containing SMILES strings (default: 'SMILES').
#     
#     Returns:
#     - The modified DataFrame with new columns: 'SMILES_canonical', 'SMILES_isomeric', 'INCHI_harmonized', 'INCHIKEY_harmonized'.
#     """
#     # Extract unique non-null SMILES and create a temp DF for processing
#     unique_df = df[[smiles_col]].drop_duplicates().dropna()
#     if unique_df.empty:
#         # If no valid SMILES, add empty columns and return
#         df['SMILES_canonical'] = None
#         df['SMILES_isomeric'] = None
#         df['INCHI_harmonized'] = None
#         df['INCHIKEY_harmonized'] = None
#         return df
#     
#     # Create RDKit Mol objects for unique SMILES
#     unique_df['ROMol'] = unique_df[smiles_col].apply(Chem.MolFromSmiles)
#     
#     # Compute harmonized values, handling invalid Mols (None)
#     unique_df['SMILES_canonical'] = unique_df['ROMol'].apply(
#         lambda m: Chem.MolToSmiles(m, canonical=True, isomericSmiles=False) if m else None
#     )
#     unique_df['SMILES_isomeric'] = unique_df['ROMol'].apply(
#         lambda m: Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m else None
#     )
#     unique_df['INCHI_harmonized'] = unique_df['ROMol'].apply(
#         lambda m: Chem.inchi.MolToInchi(m) if m else None
#     )
#     unique_df['INCHIKEY_harmonized'] = unique_df['ROMol'].apply(
#         lambda m: Chem.inchi.MolToInchiKey(m) if m else None
#     )
#     
#     # Create mapping dictionaries from original SMILES to harmonized values
#     map_can = dict(zip(unique_df[smiles_col], unique_df['SMILES_canonical']))
#     map_iso = dict(zip(unique_df[smiles_col], unique_df['SMILES_isomeric']))
#     map_inchi = dict(zip(unique_df[smiles_col], unique_df['INCHI_harmonized']))
#     map_inchikey = dict(zip(unique_df[smiles_col], unique_df['INCHIKEY_harmonized']))
#     
#     # Apply mappings to the original DF (NaN SMILES will map to NaN/None)
#     df['SMILES_canonical'] = df[smiles_col].map(map_can)
#     df['SMILES_isomeric'] = df[smiles_col].map(map_iso)
#     df['INCHI_harmonized'] = df[smiles_col].map(map_inchi)
#     df['INCHIKEY_harmonized'] = df[smiles_col].map(map_inchikey)
#     
#     # Optionally, if you need the 'ROMol' column in the full DF, uncomment the next line
#     # (but it's memory-intensive for large DFs with duplicates, so it's skipped by default)
#     # df['ROMol'] = df[smiles_col].apply(Chem.MolFromSmiles)
#     
#     return df
################################################################################################





##### CLEAN SALTS IDENTIFIERS: This worked well in dataframe ####
#from rdkit import Chem
#from rdkit.Chem.MolStandardize import rdMolStandardize
#from rdkit.Chem import inchi
#from rdkit.Chem.rdmolops import AssignStereochemistry
#
## Create uncharger once (efficient)
#uncharger = rdMolStandardize.Uncharger()
#
#def get_cleaned_neutral_data(smiles):
#    """
#    Processes a (possibly disconnected/salt) SMILES and returns cleaned neutral versions:
#    - smiles: isomeric SMILES of the neutral largest fragment
#    - inchi: Standard InChI with stereochemistry (/SNon)
#    - inchikey: Corresponding InChIKey (includes stereo if present)
#    
#    Returns dict with None values if processing fails.
#    """
#    if not isinstance(smiles, str) or smiles.strip() == '':
#        return {'smiles': None, 'inchi': None, 'inchikey': None}
#    
#    mol = Chem.MolFromSmiles(smiles)
#    if mol is None:
#        return {'smiles': None, 'inchi': None, 'inchikey': None}
#    
#    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
#    if not frags:
#        return {'smiles': None, 'inchi': None, 'inchikey': None}
#    
#    # Select the largest fragment (discards small ions like Cl-, Na+, Cu+2, OH-, etc.)
#    frag = frags[0] if len(frags) == 1 else max(frags, key=lambda m: m.GetNumAtoms())
#    
#    # Neutralize common organic charges
#    neutral_frag = uncharger.uncharge(frag)
#    
#    # Force stereochemistry perception (important for isomeric input)
#    AssignStereochemistry(neutral_frag, cleanIt=True, force=True)
#    
#    # Sanitize the neutralized molecule
#    try:
#        Chem.SanitizeMol(neutral_frag)
#    except:
#        return {'smiles': None, 'inchi': None, 'inchikey': None}
#    
#    # Generate cleaned isomeric SMILES
#    try:
#        cleaned_smiles = Chem.MolToSmiles(neutral_frag, isomericSmiles=True)
#    except:
#        cleaned_smiles = None
#    
#    # Generate InChI and InChIKey WITH stereochemistry
#    try:
#        cleaned_inchi = inchi.MolToInchi(neutral_frag, options="/SNon")
#        cleaned_inchikey = inchi.InchiToInchiKey(cleaned_inchi) if cleaned_inchi else None
#    except:
#        cleaned_inchi = None
#        cleaned_inchikey = None
#    
#    return {
#        'smiles': cleaned_smiles,
#        'inchi': cleaned_inchi,
#        'inchikey': cleaned_inchikey
#    }
##################################################################################




# === ULTIMATE VERSION-PROOF RDKit MolStandardize IMPORT (works 2018 → 2025+) ===
try:
    # RDKit 2024.03 + (current standard as of 2025)
    from rdkit.Chem.MolStandardize.rdMolStandardize import (
        LargestFragmentChooser,
        Uncharger,
        TautomerEnumerator,
    )
    _molstd_source = "rdMolStandardize.rdMolStandardize"
except ImportError:
    try:
        # RDKit 2023.xx – 2024.03
        from rdkit.Chem.MolStandardize.fragment import LargestFragmentChooser
        from rdkit.Chem.MolStandardize.charge import Uncharger
        from rdkit.Chem.MolStandardize.tautomer import TautomerEnumerator
        _molstd_source = "submodules"
    except ImportError:
        try:
            # Older RDKit (pre-2023)
            from rdkit.Chem.MolStandardize import (
                LargestFragmentChooser,
                Uncharger,
                TautomerEnumerator,
            )
            _molstd_source = "direct"
        except ImportError:
            raise ImportError(
                "Unable to import MolStandardize tools. Your RDKit version is not supported."
            )

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt


def isnull_or_empty(o: Any) -> bool:
    if o is None:
        return True
    try:
        if pd.isna(o):
            return True
    except (ValueError, TypeError):
        pass
    try:
        s = str(o).strip().lower()
        if s in {"", "<na>", "n/a", "na", "nan", "nat", "none"}:
            return True
        if hasattr(o, "__len__") and len(o) == 0:
            return True
    except Exception:
        pass
    return False


def _compound_score(comp: Compound) -> int:
    """Prefer SMILES with fewer dots (less fragmented). Version-proof for modern PubChemPy."""
    smiles = (
        getattr(comp, "smiles", None)
        or getattr(comp, "connectivity_smiles", None)
        or getattr(comp, "canonical_smiles", None)
        or getattr(comp, "isomeric_smiles", None)
        or ""
    )
    return 1000 - smiles.count(".")


def _get_parent_cids(cid: int | None) -> List[int]:
    """Fetch parent CIDs for a given CID (PubChem cids_type=parent). Returns empty list on failure."""
    if cid is None or cid < 0:
        return []
    try:
        parents = get_cids(
            identifier=cid,
            namespace="cid",
            domain="compound",
            cids_type="parent",
        )
        if isinstance(parents, int):
            return [parents]
        if isinstance(parents, (list, tuple)):
            return [int(p) for p in parents if p is not None]
        return []
    except Exception:
        return []


def _search_one(
    original_query: Any,
    field: str,
    include_synonyms: bool = True,
    include_parents: bool = False,
) -> dict | None | str:
    """
    Search a single query.
    original_query is kept with its original Python type for the output 'query' column.
    Internally the value is converted to str for the PubChem API.
    """
    if isnull_or_empty(original_query):
        return None

    query_str = str(original_query).strip()

    try:
        compounds = get_compounds(query_str, field)
        if not compounds:
            return None

        compounds.sort(key=_compound_score, reverse=True)
        best = compounds[0]
        result = best.to_dict()

        # --- name + synonyms ---
        iupac_name = result.get("iupac_name")
        if include_synonyms:
            try:
                synonyms = best.synonyms or []
                result["synonyms"] = synonyms
                friendly_name = synonyms[0] if synonyms else None
            except Exception:
                friendly_name = None
                result["synonyms"] = []
        else:
            friendly_name = None
            result["synonyms"] = None

        result["name"] = (
            friendly_name
            or iupac_name
            or result.get("molecular_formula")
            or query_str
        )

        # --- parents (optional extra request) ---
        if include_parents:
            cid = result.get("cid")
            result["parent_cids"] = _get_parent_cids(cid)
        else:
            result["parent_cids"] = None

        # Preserve original query value + type
        result["query"] = original_query
        result["query_field"] = field

        return result

    except Exception as e:
        return f"Error: {str(e)}"


def search(
    queries: Iterable[Any],
    field: str = "name",
    max_workers: int = 5,
    include_synonyms: bool = True,
    include_parents: bool = False,
) -> pd.DataFrame:
    return parallel_search(
        queries,
        field=field,
        max_workers=max_workers,
        include_synonyms=include_synonyms,
        include_parents=include_parents,
    )


def parallel_search(
    queries: Iterable[Any],
    field: str = "name",
    max_workers: int = 5,
    max_retries: int = 5,
    include_synonyms: bool = True,
    include_parents: bool = False,
) -> pd.DataFrame:
    """
    Parallel PubChem search with type-preserving query column and clean dtypes.

    Parameters
    ----------
    queries : Iterable
        Can be list / Series of str, int, etc. Internally converted to str for the API.
        The output 'query' column keeps the original values and their original types.
    field : str
        PubChem namespace ('name', 'cid', 'smiles', 'inchikey', ...)
    max_workers : int
        Thread pool size.
    max_retries : int
        Retries for 503 ServerBusy responses.
    include_synonyms : bool
        If True (default), fetches the full synonym list and builds a smart 'name'
        column (first synonym preferred – matches what PubChem website usually shows).
    include_parents : bool
        If True, also fetches parent CIDs via cids_type=parent (extra API call per hit).

    Returns
    -------
    pd.DataFrame
        Missing / failed rows have cid = -1. Column order is cleaned for convenience.
    """
    # Keep original objects so we can put them back later (type preservation)
    if isinstance(queries, pd.Series):
        original_list = queries.tolist()
    else:
        original_list = list(queries)

    if not original_list:
        return pd.DataFrame()

    pending_idx = list(range(len(original_list)))
    all_results: list[dict | None] = [None] * len(original_list)
    attempt = 0

    while pending_idx and attempt < max_retries:
        attempt += 1
        print(f"PubChem query attempt {attempt}/{max_retries} – {len(pending_idx)} remaining")

        current_queries = [original_list[i] for i in pending_idx]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                executor.map(
                    lambda q: _search_one(
                        q,
                        field,
                        include_synonyms=include_synonyms,
                        include_parents=include_parents,
                    ),
                    current_queries,
                )
            )

        retry_idx = []
        for idx, res in zip(pending_idx, results):
            if isinstance(res, dict):
                all_results[idx] = res
            elif res is None:
                # No hit
                all_results[idx] = {
                    "query": original_list[idx],
                    "query_field": field,
                    "cid": -1,
                    "name": None,
                    "synonyms": [] if include_synonyms else None,
                    "parent_cids": [] if include_parents else None,
                }
            else:
                # Error string
                if "503" in str(res) and "ServerBusy" in str(res):
                    retry_idx.append(idx)
                else:
                    all_results[idx] = {
                        "query": original_list[idx],
                        "query_field": field,
                        "cid": -1,
                        "name": None,
                        "pubchem_error": res,
                        "synonyms": [] if include_synonyms else None,
                        "parent_cids": [] if include_parents else None,
                    }

        pending_idx = retry_idx
        if pending_idx:
            time.sleep(10)

    # Any remaining after max retries
    for idx in pending_idx:
        all_results[idx] = {
            "query": original_list[idx],
            "query_field": field,
            "cid": -1,
            "name": None,
            "pubchem_error": "Max retries exceeded (503 ServerBusy)",
            "synonyms": [] if include_synonyms else None,
            "parent_cids": [] if include_parents else None,
        }

    df = pd.DataFrame(all_results)

    # ---------- dtype enforcement ----------
    if "cid" in df.columns:
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce").fillna(-1).astype("int64")

    if "query_field" in df.columns:
        df["query_field"] = df["query_field"].astype("string")

    for col in [
        "name",
        "iupac_name",
        "molecular_formula",
        "smiles",
        "connectivity_smiles",
        "inchi",
        "inchikey",
        "coordinate_type",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # query column is intentionally left alone so original int/str types are preserved

    # Nice column order
    preferred = [
        "query",
        "name",
        "cid",
        "iupac_name",
        "synonyms",
        "parent_cids",
        "query_field",
        "molecular_formula",
        "smiles",
        "inchikey",
    ]
    existing = [c for c in preferred if c in df.columns]
    others = [c for c in df.columns if c not in existing]
    df = df[existing + others]

    return df


# ===================================================================
# GOLD STANDARD RDKit HARMONIZATION (from msn_tree_library)
# ===================================================================

def harmonize_smiles_rdkit(
    smiles: str,
    tautomer_limit: float = 900.0,
    remove_stereo: bool = False,        # ← default False (keeps stereo)
    prefer_organic: bool = True,
) -> str:
    """
    Corinna Brungs' gold-standard pipeline – updated for RDKit 2023+.
    - Largest organic fragment
    - Tautomer canonicalization (<900 Da)
    - Uncharged
    - Stereochemistry preserved by default
    """
    if isnull_or_empty(smiles):
        return ""

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""

        # 1. Largest fragment (prefer organic)
        lfc = LargestFragmentChooser(prefer_organic=prefer_organic)
        mol = lfc.choose(mol)

        # 2. Tautomer canonicalization (only if small)
        if CalcExactMolWt(mol) < tautomer_limit:
            try:
                te = TautomerEnumerator()
                te.SetMaxTautomers(1000)
                canon = te.Canonicalize(mol)
                if canon:
                    mol = canon
            except Exception:
                pass  # ignore tautomer failures

        # 3. Uncharge
        uc = Uncharger()
        mol = uc.uncharge(mol)

        # 4. Final fragment cleanup (in case uncharging split anything)
        mol = lfc.choose(mol)

        # 5. Optional: remove stereochemistry
        if remove_stereo:
            Chem.RemoveStereochemistry(mol)

        # 6. Return SMILES (preserve stereo unless removed)
        return Chem.MolToSmiles(mol, isomericSmiles=not remove_stereo)

    except Exception as e:
        # Only print once per unique SMILES to avoid spam
        if not hasattr(harmonize_smiles_rdkit, "seen_errors"):
            harmonize_smiles_rdkit.seen_errors = set()
        key = str(smiles)[:50]  # truncate long ones
        if key not in harmonize_smiles_rdkit.seen_errors:
            harmonize_smiles_rdkit.seen_errors.add(key)
            print(f"RDKit harmonization failed for '{smiles}': {e}")
        return ""


def batch_harmonize_smiles(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    output_col: str = "smiles_harmonized",
    date_col: str | None = None,
    overwrite: bool = False,
    **kwargs,
) -> pd.DataFrame:
    df = df.copy()

    mask = df[smiles_col].notnull()
    if not overwrite and output_col in df.columns:
        mask &= (df[output_col].isnull()) | (df[output_col] == "")

    if not mask.any():
        print("No SMILES to harmonize.")
        return df

    print(f"Harmonizing {mask.sum():,} SMILES (stereo preserved by default)...")
    df.loc[mask, output_col] = df.loc[mask, smiles_col].apply(
        harmonize_smiles_rdkit, **kwargs
    )

    if date_col is not None:
        df.loc[mask, date_col] = datetime.now().isoformat()

    return df


# Keep old names for backward compatibility
pubchem_search = search
pubchem_parallel_search = parallel_search


__all__ = [
    "search",
    "parallel_search",
    "pubchem_search",
    "pubchem_parallel_search",
    "isnull_or_empty",
    "harmonize_smiles_rdkit",
    "batch_harmonize_smiles",
]

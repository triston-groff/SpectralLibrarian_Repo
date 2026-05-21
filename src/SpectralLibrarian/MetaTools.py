# src/SpectralLibrarian/MetaTools.py
"""
MetaTools – PubChem search + RDKit-based metadata harmonization, classification, and fragment neutralization
"""

from __future__ import annotations

import time
import pandas as pd
import concurrent.futures
from typing import List, Iterable, Any
from pubchempy import Compound, get_compounds
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
    if pd.isna(o):
        return True
    try:
        s = str(o).strip().lower()
        if s in {"", "<na>", "n/a", "na", "nan", "nat"}:
            return True
        if hasattr(o, "__len__") and len(o) == 0:
            return True
    except:
        pass
    return False


def _compound_score(comp: Compound) -> int:
    """Prefer SMILES with fewer dots (less fragmented)."""
    smiles = comp.canonical_smiles or comp.isomeric_smiles or ""
    return 1000 - smiles.count(".")


def _search_one(query: str, field: str) -> dict | None | str:
    if isnull_or_empty(query):
        return None
    try:
        compounds = get_compounds(query, field)
        if not compounds:
            return None
        compounds.sort(key=_compound_score, reverse=True)
        best = compounds[0]
        result = best.to_dict()
        result["query"] = query
        result["query_field"] = field
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def search(queries: Iterable[str], field: str = "name", max_workers: int = 5) -> pd.DataFrame:
    return parallel_search(queries, field=field, max_workers=max_workers)


def parallel_search(
    queries: Iterable[str],
    field: str = "name",
    max_workers: int = 5,
    max_retries: int = 5,
) -> pd.DataFrame:
    if isinstance(queries, pd.Series):
        queries = queries.astype(str).replace(["nan", "<NA>"], None).tolist()
    elif not isinstance(queries, list):
        queries = list(queries)

    if not queries:
        return pd.DataFrame()

    pending = list(queries)
    all_results = []
    attempt = 0

    while pending and attempt < max_retries:
        attempt += 1
        print(f"PubChem query attempt {attempt}/{max_retries} – {len(pending)} remaining")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda q: _search_one(q, field), pending))

        retry = []
        for query, res in zip(pending, results):
            row = {"query": query, "query_field": field}
            if isinstance(res, dict):
                row.update(res)
                all_results.append(row)
            elif res is None:
                all_results.append(row)
            else:
                if "503" in res and "ServerBusy" in res:
                    retry.append(query)
                else:
                    row["pubchem_error"] = res
                    all_results.append(row)

        pending = retry
        if pending:
            time.sleep(10)

    for query in pending:
        all_results.append({
            "query": query,
            "query_field": field,
            "pubchem_error": "Max retries exceeded (503 ServerBusy)"
        })

    return pd.DataFrame(all_results)


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
            except:
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
    **kwargs
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
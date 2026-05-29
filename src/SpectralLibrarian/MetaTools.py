from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdmolops import AssignStereochemistry

from pubchempy import Compound, get_compounds

from typing import *

import pandas as pd, pickle as pkl, molmass as mm
import os, time, re, ast, concurrent.futures

# src/MSAnalyzer/MetaTools.py


# === ULTIMATE VERSION-PROOF RDKit MolStandardize IMPORT (works 2018 → 2025+) ===
try:
    # RDKit 2024.03 + (current standard as of 2025)
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser, Uncharger, TautomerEnumerator
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
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
            from rdkit.Chem.MolStandardize import LargestFragmentChooser, Uncharger, TautomerEnumerator
            _molstd_source = "direct"
        except ImportError:
            raise ImportError("Unable to import MolStandardize tools. Your RDKit version is not supported.")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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


def mol_to_inchi(mol, to_key=False):
    if mol is None:
        return None
    try:
        if to_key:
            return Chem.inchi.MolToInchiKey(mol)
        else:
            return Chem.inchi.MolToInchi(mol)
    except:
        return None


def inchi_to_mol(inchi):
    if pd.isna(inchi):
        return None
    try:
        return Chem.inchi.MolFromInchi(inchi)
    except:
        return None


def mol_to_smiles(mol, isomeric=False):
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except:
        return None


def smiles_to_mol(smiles):
    if smiles is None:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None


def smiles_to_inchikey(smiles):
    # Direct is impossible. So, smiles -> mol -> inchikey
    if pd.isna(smiles):
        return None
    mol = smiles_to_mol(smiles=smiles)
    if mol is None:
        return None
    try:
        return mol_to_inchi(mol=mol, to_key=True)
    except:
        return None


def smiles_to_inchi(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return mol_to_inchi(mol=mol, to_key=False)
    except:
        return None


def inchi_to_inchikey(inchikey):
    if pd.isna(inchikey):
        return None
    try:
        return Chem.inchi.InchiToInchiKey(str(inchikey))  # works for most real cases
    except:
        return None


def inchi_to_smiles(inchi_str):
    # Direct is impossible. So, inchi -> mol -> smiles
    if pd.isna(inchi_str):
        return None
    try:
        mol = inchi_to_mol(str(inchi_str))
        if mol is None:
            return None
        return mol_to_smiles(mol=mol, isomeric=True)
    except:
        return None


def has_disconnected_components(smiles):
    if not isinstance(smiles, str):
        return True  # Treat NaN or non-string as problematic
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return True
        frags = Chem.GetMolFrags(mol)
        return len(frags) > 1
    except:
        return True


# def harmonize_molecules(df, smiles_col='SMILES', appendROMol=False):
#     #### Double check that InChI and InChIKey contain stereochem info ####
#     """
#     Efficiently adds harmonized SMILES (canonical and isomeric), InChI, and InChIKey columns to the DataFrame.
#     Processes only unique SMILES to avoid redundant computations on duplicates.
#
#     Parameters:
#     - df: pandas DataFrame containing the SMILES column.
#     - smiles_col: Name of the column containing SMILES strings (default: 'SMILES').
#     - appendROMol : Whether to append 'ROMols' column to the df (default: False (memory-intensive)).
#
#     Returns:
#     - The modified DataFrame with new columns: 'SMILES_canonical', 'SMILES_isomeric', 'INCHI_harmonized', 'INCHIKEY_harmonized', and 'ROMols' if required
#     """
#     print("Harmonizing molecules ... ")
#     unique_smiles = list(df.loc[pd.notnull(df[smiles_col])][smiles_col].unique())
#     ROMols, SMILES_canonical, SMILES_isomeric, INCHI_harmonized, INCHIKEY_harmonized = [], [], [], [], []
#     for smiles in unique_smiles:
#         mol = smiles_to_mol(smiles) # SMILES -> RDKit Mol
#         # Optionally, if you need the 'ROMol' column in the full DF, uncomment the next line (but it's memory-intensive for large DFs with duplicates, so it's skipped by default)
#         if appendROMol:
#             ROMols.append(mol)
#
#         # Mol -> Harmonized identifiers (SMILES, INCHI, INCHIKEY)
#         SMILES_canonical.append(mol_to_smiles(mol=mol, isomeric=False))
#         SMILES_isomeric.append(mol_to_smiles(mol=mol, isomeric=True))
#         INCHI_harmonized.append(mol_to_inchi(mol=mol, to_key=False))
#         INCHIKEY_harmonized.append(mol_to_inchi(mol=mol, to_key=True))
#
#     unique_df = pd.DataFrame({smiles_col: unique_smiles, "SMILES_canonical": SMILES_canonical, "SMILES_isomeric": SMILES_isomeric, "INCHI_harmonized": INCHI_harmonized, "INCHIKEY_harmonized": INCHIKEY_harmonized})
#     if appendROMol:
#         unique_df['ROMols'] = ROMols
#
#     return pd.merge(df, unique_df, on=smiles_col, how='left')

def extract_pubchem_dict(comment):
    if pd.isna(comment):
        return {}
    match = re.search(r'\{.*\}', comment, re.DOTALL)
    if match:
        dict_str = match.group(0)
        try:
            # ast.literal_eval handles the Python dict syntax with single quotes
            parsed_dict = ast.literal_eval(dict_str)
            return parsed_dict
        except (ValueError, SyntaxError):
            return {}
    return {}


def harmonize_molecules(df, smiles_col='SMILES', inchi_col='INCHI', appendROMol=False):
    """
    Harmonize molecules to return harmonized identifiers (SMILES_canonical, SMILES_isomeric, INCHI_harmonized, INCHIKEY_harmonized columns in the returned dataframe)
    At least smiles_col and inchi_col column should exist in the dataframe given. If both exist and are not null, it returned identifiers harmonized using the opposite identifier.
    That is, SMILES_canonical and SMILE_isomeric are derived from INCHI, while INCHI_harmonized and INCHIKEY_harmonized are derived from SMILES.
    If only one of smiles_col or inchi_col is available or not null, it returns identifiers harmonized using that available or not-null identifier

    Parameters
    ----------
    df : pandas Dataframe with row of molecule and column of properties including identifiers
    smiles_col : column to be used as 'SMILES'
    inchi_col : column to be used as 'InChI'
    appendROMol : whether to return mol object. Default = False, because it's memory-intensive for large DFs with duplicates

    Returns
    pandas Dataframe where SMILES_canonical, SMILES_isomeric, INCHI_harmonized, INCHIKEY_harmonized columns are appended to the given dataframe. If a row in the given dataframe has not any match, the row is deleted
    -------

    """
    has_smiles = smiles_col in df.columns
    has_inchi = inchi_col in df.columns

    if has_smiles and has_inchi:
        df_smiles_or_inchi = df.loc[pd.notnull(df[smiles_col]) | pd.notnull(df[inchi_col])]
    elif has_smiles:
        df_smiles_or_inchi = df.loc[pd.notnull(df[smiles_col])]
    elif has_inchi:
        df_smiles_or_inchi = df.loc[pd.notnull(df[inchi_col])]
    else:
        raise Exception("At least '" + smiles_col + "' or '" + inchi_col + "' column should exist in the dataframe given")

    # For each unique SMILES, get harmonized molecule and retrieve their identifiers
    if has_smiles:
        ROMols, SMILES_canonical, SMILES_isomeric, INCHI_harmonized, INCHIKEY_harmonized = [], [], [], [], []
        unique_smiles = list(set(df_smiles_or_inchi[smiles_col]))
        for smiles in unique_smiles:
            mol = smiles_to_mol(smiles) # SMILES -> RDKit Mol
            if appendROMol:
                ROMols.append(mol)

            SMILES_canonical.append(mol_to_smiles(mol=mol, isomeric=False))
            SMILES_isomeric.append(mol_to_smiles(mol=mol, isomeric=True))
            INCHI_harmonized.append(mol_to_inchi(mol=mol, to_key=False))
            INCHIKEY_harmonized.append(mol_to_inchi(mol=mol, to_key=True))
        unique_df_bysmile = pd.DataFrame({smiles_col: unique_smiles, "SMILES_canonical_smiles": SMILES_canonical, "SMILES_isomeric_smiles": SMILES_isomeric, "INCHI_harmonized_smiles": INCHI_harmonized, "INCHIKEY_harmonized_smiles": INCHIKEY_harmonized})
        if appendROMol:
            unique_df_bysmile['ROMols'] = ROMols

        # Append the result together
        df_harmonized = pd.merge(df_smiles_or_inchi, unique_df_bysmile, on=smiles_col, how="left")
    else:
        df_harmonized = df_smiles_or_inchi.copy()

    # For each unique InChI, get harmonized molecule and retrieve their identifiers
    if has_inchi:
        ROMols, SMILES_canonical, SMILES_isomeric, INCHI_harmonized, INCHIKEY_harmonized = [], [], [], [], []
        unique_inchis = list(set(df_smiles_or_inchi[inchi_col]))
        for inchi in unique_inchis:
            mol = inchi_to_mol(inchi) # InChI -> RDKit Mol
            if appendROMol:
                ROMols.append(mol)

            SMILES_canonical.append(mol_to_smiles(mol=mol, isomeric=False))
            SMILES_isomeric.append(mol_to_smiles(mol=mol, isomeric=True))
            INCHI_harmonized.append(mol_to_inchi(mol=mol, to_key=False))
            INCHIKEY_harmonized.append(mol_to_inchi(mol=mol, to_key=True))
        unique_df_byinchi = pd.DataFrame({inchi_col: unique_inchis, "SMILES_canonical_inchi": SMILES_canonical, "SMILES_isomeric_inchi": SMILES_isomeric, "INCHI_harmonized_inchi": INCHI_harmonized, "INCHIKEY_harmonized_inchi": INCHIKEY_harmonized})
        if appendROMol:
            unique_df_byinchi['ROMols'] = ROMols

        df_harmonized = pd.merge(df_harmonized, unique_df_byinchi, on=inchi_col, how="left")

    # Remove any rows without any harmonized identifiers from either smile or inchi
    final_metadict = {"SMILES_canonical": [], "SMILES_isomeric": [], "INCHI_harmonized": [], "INCHIKEY_harmonized": []}
    for k in final_metadict.keys():
        if has_smiles and has_inchi:
            df_harmonized = df_harmonized.loc[pd.notnull(df_harmonized[k+"_smiles"]) | pd.notnull(df_harmonized[k+"_inchi"])]
        elif has_smiles:
            df_harmonized = df_harmonized.loc[pd.notnull(df_harmonized[k + "_smiles"])]
        else:
            df_harmonized = df_harmonized.loc[pd.notnull(df_harmonized[k + "_inchi"])]
    df_harmonized.reset_index(drop=True, inplace=True)

    # Pruning the harmonized result by comparing those by SMILES and INCHI, for the 4 fields : SMILES_canonical, SMILES_isomeric, INCHI_harmonized, INCHIKEY_harmonized
    for i, df_row in df_harmonized.iterrows():
        for k in final_metadict.keys():
            if has_smiles and has_inchi:
                id_by_smile, id_by_inchi = df_row[k + "_smiles"], df_row[k + "_inchi"]
                if id_by_smile is None:
                    final_id = id_by_inchi
                elif id_by_inchi is None:
                    final_id = id_by_smile
                else:
                    # If both are not null, use the one retrieved by the other identifier because that's the one harmonized by rdkit.Chem.Mol
                    final_id = id_by_inchi if k.startswith("SMILES") else id_by_smile
            elif has_smiles:
                final_id = df_row[k + "_smiles"]
            else:
                final_id = df_row[k + "_inchi"]
            final_metadict[k].append(final_id)
    df_final = pd.concat([df_harmonized, pd.DataFrame(final_metadict)], axis=1)

    # delete intermediate columns
    for k in final_metadict.keys():
        if has_smiles:
            del df_final[k + "_smiles"]

        if has_inchi:
            del df_final[k + "_inchi"]

    for k in final_metadict.keys():
        df_final = df_final.loc[pd.notnull(df_final[k])]

    return df_final


def get_molecule_fp_bit_msadf(msadf, radius=3, nBits=2048):
    # Create the Morgan generator
    morgan_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    ao = Chem.rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()

    fingerprints = [] # List of fingerprints
    bit_infos = []  # List of dicts: one per molecule {bit_id: [(atom_idx, eff_radius), ...]}
    for mol in msadf['ROMol']:
        # AllChem.GetMorganFingerprintAsBitVect is deprecated
        fp = morgan_gen.GetFingerprint(mol, additionalOutput=ao)
        fingerprints.append(list(fp))  # Convert to list of 0/1 for DF
        bit_infos.append(ao.GetBitInfoMap())

    # Create the Morgan fingerprint DataFrame
    bit_columns = [f'Bit_{i}' for i in range(nBits)]
    df_morgan = pd.DataFrame(fingerprints, index=msadf['SMILES_isomeric'], columns=bit_columns)

    # Global bit meanings: unique substructs per bit across all molecules (handles collisions)
    global_bit_meanings = {bit: set() for bit in range(nBits)}
    for i, mol in enumerate(msadf['ROMol']):
        for bit_id in bit_infos[i]:
            meanings = get_bit_meaning(mol, bit_id, bit_infos[i])
            global_bit_meanings[bit_id].update(meanings)

    return df_morgan, global_bit_meanings


def get_bit_meaning(mol, bit_id, bit_info):
    """
    For a given Mol, bit_id (0-2047), and its bit_info dict, returns list of rooted SMILES
    representing the substructures that set that bit.
    """
    if bit_id not in bit_info:
        return []  # Bit not set in this molecule

    substructs = []
    for atom_idx, sub_radius in bit_info[bit_id]:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, sub_radius, atom_idx)
        atom_map = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
        if atom_idx in atom_map:
            rooted_smiles = Chem.MolToSmiles(submol, rootedAtAtom=atom_map[atom_idx], canonical=False)
            substructs.append(rooted_smiles)
        else:
            substructs.append("Invalid mapping")  # Rare error case

    return substructs


def pubchem_query_from_file(inputfile, sheet_name:str=None, column="CAS Number", field="name", sleeptime:float=0.333, save_for_desalt:bool=False):
    inputdir = os.path.dirname(inputfile)
    filename = os.path.splitext(os.path.basename(inputfile))[0]

    if inputfile.endswith(".pkl"):
        with open(inputfile, "rb") as f:
            df_input = pkl.load(f)
    elif inputfile.endswith(".xlsx"):
        if not (sheet_name is None):
            df_input = pd.read_excel(inputfile, sheet_name=sheet_name)
        else:
            df_input = pd.read_excel(inputfile)
    else:
        raise FileNotFoundError("The inputfile should be a .pkl file containing pd.DataFrame or a .xlsx file")
    queries = list(set(df_input.loc[pd.notnull(df_input[column])][column]))

    pklfiledir = os.path.join(inputdir, filename + "_pubchemquery_" + column + ".pkl") # pkl file containing pd.DataFrame()
    if os.path.exists(pklfiledir):
        with open(pklfiledir, "rb") as f:
            df_complete = pkl.load(f)
        completed_queries = list(df_complete[field])
    else:
        df_complete = pd.DataFrame()
        completed_queries = []
    queries_left = [x for x in queries if not (x in completed_queries)]

    new_results = []
    for i, query in enumerate(queries_left):
        result = pubchem_query_one(query=query, field=field)

        row = {field: query}
        if isinstance(result, dict):
            row.update(result)
            new_results.append(row)
            print(field + " : " + str(i + 1) + "-th query / " + str(len(queries_left)) + " queries : " + query + " : success")
        elif result is None:
            new_results.append(row)
            print(field + " : " + str(i + 1) + "-th query / " + str(len(queries_left)) + " queries : " + query + " : success. NOTHING RETURNED")
        else:
            print(field + " : " + str(i + 1) + "-th query / " + str(len(queries_left)) + " queries : " + query + " : ERROR!")
        time.sleep(sleeptime)

        if len(new_results) % 10 == 0:
            df_new = pd.DataFrame(new_results).reindex(columns=pubchem_query_resultcols.keys())
            df_complete = pd.concat([df_complete.reset_index(drop=True), df_new.reset_index(drop=True)], axis=0)
            os.makedirs(os.path.dirname(pklfiledir), exist_ok=True)
            with open(pklfiledir, "wb") as f:
                pkl.dump(df_complete, f)
            new_results = []

    if len(new_results) != 0:
        df_new = pd.DataFrame(new_results).reindex(columns=pubchem_query_resultcols.keys())
        df_complete = pd.concat([df_complete.reset_index(drop=True), df_new.reset_index(drop=True)], axis=0)
        os.makedirs(os.path.dirname(pklfiledir), exist_ok=True)
        with open(pklfiledir, "wb") as f:
            pkl.dump(df_complete, f)

    print("Done !!!")
    with open(pklfiledir, "rb") as f:
        df_complete = pkl.load(f)
        outexcelfile = os.path.join(inputdir, filename + "_pubchemquery_" + column + ".xlsx")
        try:
            df_complete.to_excel(outexcelfile, index=False)
        except:
            print("The output dataframe was saved in a .pkl file but failed as an xlsx file. Check the .pkl file instead")

        # save for desalting and neutralizing
        if save_for_desalt:
            salt_excelfile = os.path.join(inputdir, filename + "_pubchemquery_" + column + "_smiles_salt.xlsx")
            df_complete_salt = df_complete.rename(columns={"smiles": "smiles_salt"})[["smiles_salt"]]
            df_complete_salt.to_excel(salt_excelfile, index=False)


def pubchem_query_multiple(queries: Iterable[str], field: str = "name", parallel:bool=True, max_workers: int = 5, max_retries: int = 5) -> pd.DataFrame:
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

        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(lambda q: pubchem_query_one(q, field), pending))
        else:
            results = [pubchem_query_one(q, field) for q in pending]

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
        all_results.append({ "query": query, "query_field": field, "pubchem_error": "Max retries exceeded"})

    return pd.DataFrame(all_results)


def pubchem_query_one(query: str, field: str) -> dict | None | str:
    if isnull_or_empty(query):
        return None
    try:
        compounds = get_compounds(query, field) # -> list[Compound] | pd.DataFrame:
        if len(compounds) == 0:
            return None
        compounds.sort(key=pubchem_compound_score, reverse=True)
        result = compounds[0].to_dict()
        result[field] = query
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def pubchem_compound_score(comp: Compound) -> int:
    """Prefer SMILES with fewer dots (less fragmented)."""
    # smiles = comp.canonical_smiles or comp.isomeric_smiles or ""
    smiles = comp.connectivity_smiles or comp.isomeric_smiles or ""
    return 1000 - smiles.count(".")


pubchem_query_resultcols = {'name': str, 'cid': str, 'smiles': str, 'inchi': str, 'inchikey': str, 'iupac_name': str, 'connectivity_smiles': str,
                            'elements': str, 'atoms': str, 'bonds': str, 'coordinate_type': str, 'charge': int, 'molecular_formula': str,
                            'exact_mass': float, 'monoisotopic_mass': float, 'molecular_weight': float,
                            'xlogp': float, 'tpsa': float, 'complexity': float,
                            'h_bond_donor_count': int, 'h_bond_acceptor_count': int, 'rotatable_bond_count': int, 'fingerprint': str,
                            'cactvs_fingerprint': str, 'heavy_atom_count': int, 'isotope_atom_count': int,
                            'atom_stereo_count': int, 'defined_atom_stereo_count': int, 'undefined_atom_stereo_count': int,
                            'bond_stereo_count': int, 'defined_bond_stereo_count':int, 'undefined_bond_stereo_count': int,'covalent_unit_count': int}

                            # 'volume_3d': float, 'multipoles_3d': float, 'conformer_rmsd_3d': float, 'effective_rotor_count_3d': float, 'pharmacophore_features_3d': float,
                            # 'mmff94_partial_charges_3d': float, 'mmff94_energy_3d': float, 'conformer_id_3d': float, 'shape_selfoverlap_3d': float,
                            # 'feature_selfoverlap_3d': float, 'shape_fingerprint_3d': float}
# 'fingerprint': str, 'h_bond_acceptor_count': int, 'complexity': int,'isotope_atom_count': int, 'SYNON': None



# Salt Neutralizer
from rdkit.Chem.MolStandardize import rdMolStandardize

uc = rdMolStandardize.Uncharger()

def get_neutral_data_from_smiles(inputfile:str, outputfile:str, smiles_col:str="SMILES", only_smiles:bool=False, clean:bool=True):
    if not (inputfile.endswith(".xlsx") and outputfile.endswith(".xlsx")):
        raise FileNotFoundError("Both inputfile and outputfile directories should be .xlsx file")

    df_input = pd.read_excel(inputfile).reset_index(drop=True)
    salt_smiles_list = list(set(df_input[smiles_col]))

    neutral_data = []
    for i, salt_smiles in enumerate(salt_smiles_list):
        neutral_data.append(get_largest_neutral_data(salt_smiles, only_smiles=only_smiles, clean=clean))
        print(str(i + 1) + " / " + str(len(salt_smiles_list)) + " completed")

    if len(neutral_data) > 0:
        data_dict = {smiles_col: salt_smiles_list}
        for k in neutral_data[0].keys():
            data_dict[k] = []

        for ndict in neutral_data:
            for k, v in ndict.items():
                data_dict[k].append(v)
    else:
        if only_smiles:
            data_dict = {smiles_col: [], "smiles": [], "charged_smiles": []}
        else:
            data_dict = {smiles_col: [], 'smiles': [], "charged_smiles": [], 'inchi': [], 'inchikey': [], 'molecular formula': [], 'monoisotopic mass': []}
    df_neutral = pd.DataFrame.from_dict(data_dict)
    df_final = pd.merge(df_input, df_neutral, on=smiles_col, how='left')
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    df_final.to_excel(outputfile, index=False)


def get_largest_neutral_data(smiles, only_smiles=False, clean=True) -> Union[Any, dict[str, Any]]:
    """
    Returns a SMILES or dict with cleaned SMILES, InChI, and InChIKey for the neutralized largest fragment.
    Returns the neutralized isomeric SMILES of the largest organic fragment. Discards small inorganic ions, then neutralizes charges on the organic part.
    Returns None for all if processing fails.
    """
    nonreturn = {'smiles': None, 'charged_smiles': None} if only_smiles else {'smiles': None, 'charged_smiles': None, 'inchi': None, 'inchikey': None, 'molecular formula': None, 'monoisotopic mass': None}

    if not isinstance(smiles, str) or smiles.strip() == '':
        return nonreturn

    # SMILES -> Mol -> MolFrags
    try:
        mol = Chem.MolFromSmiles(smiles) # WARNING: not removing hydrogen atom without neighbors ??
        if mol is None:
            return nonreturn

        frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
        if len(frags) == 0:
            return nonreturn

        charged_parts = []
        for frag in frags:
            # Look at charges. If total charge is not 0, consider it a charged part/counterion
            total_charge = Chem.GetFormalCharge(frag)
            if total_charge != 0:
                charged_parts.append(frag)
        charged_smiles = [Chem.MolToSmiles(m) for m in charged_parts]
        charged_smiles_str = ','.join(charged_smiles)

        if len(frags) > 1:
            frags.sort(key=lambda m: m.GetNumAtoms()) # this is increasing order
        largest_frag = frags[-1] # if len(frags) == 1 else max(frags, key=lambda m: m.GetNumAtoms()) # Select largest fragment
        neutral_largest_frag = uc.uncharge(largest_frag) # Neutralization (e.g., [O-] → O with added H)
        AssignStereochemistry(neutral_largest_frag, cleanIt=clean, force=True)  # Force stereochemistry perception (important for isomeric input)
        Chem.SanitizeMol(neutral_largest_frag) # Sanitize
        cleaned_smiles = Chem.MolToSmiles(neutral_largest_frag, isomericSmiles=True) # Generate cleaned (neutralized) isomeric SMILES
        if only_smiles:
            return {'smiles': cleaned_smiles, 'charged_smiles': charged_smiles_str}

        cleaned_inchi = Chem.inchi.MolToInchi(neutral_largest_frag)
        cleaned_inchikey = None if cleaned_inchi is None else Chem.inchi.InchiToInchiKey(cleaned_inchi)
        mf = CalcMolFormula(mol)
        mimw = mm.Formula(mf).monoisotopic_mass
        return {'smiles': cleaned_smiles, 'charged_smiles': charged_smiles_str, 'inchi': cleaned_inchi, 'inchikey': cleaned_inchikey, 'molecular formula':mf, 'monoisotopic mass': mimw}
    except Exception as e:
        return nonreturn


__all__ = ["isnull_or_empty", "mol_to_inchi", "inchi_to_mol", "mol_to_smiles", "smiles_to_mol", "smiles_to_inchikey", "smiles_to_inchi", "inchi_to_inchikey", "inchi_to_smiles", "has_disconnected_components",
           "harmonize_molecules", "get_molecule_fp_bit_msadf", "get_bit_meaning", "pubchem_query_from_file", "pubchem_query_multiple", "pubchem_query_one", "pubchem_compound_score",
           "get_neutral_data_from_smiles", "get_largest_neutral_data", "extract_pubchem_dict"]
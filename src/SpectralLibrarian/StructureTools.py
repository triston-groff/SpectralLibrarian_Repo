# MSAnalyzer/structural_analysis.py
"""
StructuralAnalysis - Ultimate RDKit-based hierarchical structural categorizer
November 2025 - Final Gold Standard Version
"""
from typing import *
from collections import defaultdict
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import rdmolops, Fragments, rdFingerprintGenerator
from rdkit.Chem.rdFingerprintGenerator import GetMorganFeatureAtomInvGen
from scipy.sparse import csr_matrix

import pandas as pd, numpy as np
import os




ALLOWED_ATOMIC_NUMS = {1, 6, 7, 8, 15, 16, 9, 17, 35, 53}  # H,C,N,O,P,S,F,Cl,Br,I


HIERARCHICAL_GROUPS = {
    "carbonyl": ["ketone", "aldehyde", "carboxyl", "ester", "amide", "imide", "lactone", "anhydride", "carbamate"],
    "carbon_nitrogen_multiple_bond": ["imine", "oxime", "hydrazone", "semicarbazone", "nitrone", "hydroxamate", "O_acylhydroxylamine", "isourea"],
    "carbon_multiple_bond": [],
    "alcohol": ["alcohol_primary", "alcohol_secondary", "alcohol_tertiary", "phenol", "enol", "alpha_hydroxy_acid", "beta_hydroxy_acid"],
    "amine": ["amine_primary", "amine_secondary", "amine_tertiary", "amine_quaternary"],
    "amide": ["amide_primary", "amide_secondary", "amide_tertiary", "lactam", "peptide_bond"],
    "sulfur_oxygen": ["sulfoxide", "sulfone", "sulfonic_acid", "sulfonamide", "sulfamate"],
    "phosphorus_oxygen": ["phosphate", "phosphonate", "phosphinate", "phosphoramidate"],
    "halogen": ["fluoride", "chloride", "bromide", "iodide"],
    "multiple_bond": ["alkene", "alkyne", "allene", "conjugated_diene"],
    "monosaccharide": ["triose", "tetrose", "pentose", "hexose", "heptose"],
    "amino_acid": ["alpha_amino_acid", "beta_amino_acid", "gamma_amino_acid"],
    "keto_acid": ["alpha_keto_acid", "beta_keto_acid", "gamma_keto_acid"],
    "hydroxy_acid": ["alpha_hydroxy_acid", "beta_hydroxy_acid"],
}

HIERARCHICAL_GROUPS["carbon_multiple_bond"] = (
    HIERARCHICAL_GROUPS["carbonyl"] + HIERARCHICAL_GROUPS["carbon_nitrogen_multiple_bond"]
)


FUNCTIONAL_GROUPS = {
    "ketone": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "carboxylate": "[CX3](=O)[O-]",
    "ester": "[CX3](=O)[OX2][CX4]",
    "lactone": "[CX3](=O)[OX2R][CR]",
    "amide_primary": "[NX3][CX3](=O)[H]",
    "amide_secondary": "[NX3][CX3](=O)[CX4]",
    "amide_tertiary": "[NX3][CX3](=O)[CX4][CX4]",
    "lactam": "[NR][CR](=O)",
    "imide": "[CX3](=O)[NX3][CX3](=O)",
    "anhydride": "[CX3](=O)[OX2][CX3](=O)",
    "carbamate": "[NX3][CX3](=O)[OX2]",
    "peptide_bond": "[NX3][CX3](=O)[NX3]",
    "alpha_keto_acid": "[CX3](=O)[CX4][CX3](=O)[OH1]",
    "beta_keto_acid": "[CX3](=O)[CX4][CX4][CX3](=O)[OH1]",
    "gamma_keto_acid": "[CX3](=O)[CX4][CX4][CX4][CX3](=O)[OH1]",
    "alpha_hydroxy_acid": "[CH1X4][OH1][CX3](=O)[OH1]",
    "beta_hydroxy_acid": "[CH2X4][OH1][CH1X4][CX3](=O)[OH1]",
    "alpha_amino_acid": "[NX3][CH1X4][CX3](=O)[OH1]",
    "beta_amino_acid": "[NX3][CX4][CH1X4][CX3](=O)[OH1]",
    "gamma_amino_acid": "[NX3][CX4][CX4][CH1X4][CX3](=O)[OH1]",
    "imine": "[CX3]=[NX2]",
    "oxime": "[CX3]=[NX2][OH1]",
    "hydrazone": "[CX3]=[NX2][NX3]",
    "hydroxamate": "[CX3](=O)[NX2][OH1]",
    "O_acylhydroxylamine": "[CX3](=O)[OX2][NX3]",
    "hydroxyl": "[OX2H1]",
    "phenol": "c[OH1]",
    "alcohol_primary": "[CH2X4][OH1]",
    "alcohol_secondary": "[CH1X4][OH1]",
    "alcohol_tertiary": "[CX4;!H][OH1]",
    "enol": "[CX3]=[CX3][OH1]",
    "ether": "[OD2](-[#6])-[#6]",
    "epoxide": "[CR0X3R]1[OX2R][CR0X3R]1",
    "amine_primary": "[NX3;H2]",
    "amine_secondary": "[NX3;H1]",
    "amine_tertiary": "[NX3;H0]",
    "amine_quaternary": "[NX4+]",
    "nitrile": "[CX2]#[NX1]",
    "nitro": "[NX3](=O)=O",
    "thiol": "[SX2H1]",
    "thioether": "[SX2]",
    "disulfide": "[SX2][SX2]",
    "sulfoxide": "[SX3]=O",
    "sulfone": "[SX4](=O)=O",
    "sulfonic_acid": "[SX4](=O)(=O)[OH1]",
    "sulfonamide": "[SX4][OX2][NX3]",
    "phosphate": "[PX4](=O)([OX2])([OX2])[OX2]",
    "fluoride": "[F]",
    "chloride": "[Cl]",
    "bromide": "[Br]",
    "iodide": "[I]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "allene": "[CX2]=[CX2]=[CX2]",
    "acetal_hemiacetal": "[CX4]([OX2H0][CX4])[OX2H0][CX4]",  # or better pattern if you prefer
    "o_glycosidic": "[CX4][OX2][CX4]1[CX4][CX4][CX4][CX4]1",  # simplified – can be refined
    "n_glycosidic": "[NX3][CX4]1[CX4][CX4][CX4][CX4]1",
    "urea": "[NX3][CX3](=O)[NX3]",
    "thiourea": "[NX3][CX3](=S)[NX3]",
    "guanidine": "[CX3](=[NX2])[NX3][NX3]",
    "azide": "[NX1-]-[NX2+]=[NX1]",
    "peroxide": "[OX2][OX2]",
    "phosphonate": "[PX4](=O)(O)O",  # adjust as needed
    "sulfamate": "[NX3][SX4](=O)(=O)O"
}


CARBOHYDRATE_PATTERNS = {
    "aldotriose": "[CX3H1](=O)[CH1X4][OH1][CH2X4][OH1]",
    "aldotetrose": "[CX3H1](=O)[CH1X4][OH1][CH1X4][OH1][CH2X4][OH1]",
    "aldopentose": "[CX3H1](=O)[CH1X4][OH1][CH1X4][OH1][CH1X4][OH1][CH2X4][OH1]",
    "aldohexose": "[CX3H1](=O)[CH1X4][OH1][CH1X4][OH1][CH1X4][OH1][CH1X4][OH1][CH2X4][OH1]",
    "aldoheptose": "[CX3H1](=O)[CH1X4][OH1][CH1X4][OH1][CH1X4][OH1][CH1X4][OH1][CH1X4][OH1][CH2X4][OH1]",
    "ketotriose": "[CH2X4][OH1][CX3](=O)[CH2X4][OH1]",
    "ketotetrose": "[CH2X4][OH1][CX3](=O)[CH1X4][OH1][CH2X4][OH1]",
    "ketopentose": "[CH2X4][OH1][CX3](=O)[CH1X4][OH1][CH1X4][OH1][CH2X4][OH1]",
    "ketohexose": "[CH2X4][OH1][CX3](=O)[CH1X4][OH1][CH1X4][OH1][CH1X4][OH1][CH2X4][OH1]",
    "ketoheptose": "[CH2X4][OH1][CX3](=O)[CH1X4][OH1][CH1X4][OH1][CH1X4][OH1][CH1X4][OH1][CH2X4][OH1]",
    "furanose": "[CR]1[OX2R][CR][CR][CR]1",
    "pyranose": "[CR]1[OX2R][CR][CR][CR][CR]1",
    "alpha_anomer": "[OX2R]1[CH1X4@@][OH1][CR]C1",
    "beta_anomer": "[OX2R]1[CH1X4@][OH1][CR]C1",
}

from .utilities import parallelize

def classify_molecules_from_smiles(inputfile:str, outputfile:str, smiles_col="smiles", num_cores:int=os.cpu_count()-2, usertags_excelfile:str=None):
    if not (inputfile.endswith(".xlsx") and outputfile.endswith(".xlsx")):
        raise FileNotFoundError("Both inputfile and outputfile should be .xlsx file")

    df = pd.read_excel(inputfile)
    neutral_smiles = list(df[smiles_col])

    usertags = None
    if not (usertags_excelfile is None):
        if not (usertags_excelfile.endswith(".xlsx")):
            raise FileNotFoundError("Both usertag inputfile should be .xlsx file")
        df_tag =  pd.read_excel(usertags_excelfile)
        usertags = dict(zip(list(df_tag["group_name"]), list(df_tag["smarts"])))

    molecules = []
    argslist = []
    for smiles in neutral_smiles:
        try:
            molecule = Chem.MolFromSmiles(SMILES=smiles)
        except:
            molecule = None
        molecules.append(molecule)
        argslist.append((molecule, usertags))

    # categories_list = categorize_molecules(mols=molecules, usertags=usertags)
    categories_list = parallelize(workfunc=categorize_molecule, num_cores=num_cores, argslist=argslist)
    sparse_mat, feature_list = categories_to_sparse(categories_list)

    df_result = pd.DataFrame.from_records(np.array(sparse_mat.todense()), columns=feature_list, index=neutral_smiles)
    df_result = df_result.reset_index(drop=False).rename(columns={"index": smiles_col})
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    df_result.to_excel(outputfile, index=False)

def categorize_molecules(mols: Iterable[Chem.Mol], usertags:Dict[str, str]=None) -> List[Dict[str, int]]:
    results = []
    for i, m in enumerate(mols):
        print("Functional Group Processing : " + str(i + 1) + " / " + str(len(mols)) + " completed")
        results.append(categorize_molecule(m, usertags=usertags))
    return results

def categorize_molecule(mol: Chem.Mol, usertags:Dict[str, str]=None) -> Dict[str, int]:
    if mol is None or not _is_allowed(mol):
        return {}

    mol = rdmolops.AddHs(mol)
    cats: Dict[str, int] = {}

    # Element thresholds
    elem_count = defaultdict(int)
    for a in mol.GetAtoms():
        elem_count[a.GetSymbol()] += 1

    for sym, cnt in elem_count.items():
        for thr in [1, 2, 4, 6, 8]:
            if cnt >= thr:
                cats[f"{sym}>={thr}"] = 1

    # Custom functional groups
    for name, smarts in FUNCTIONAL_GROUPS.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            cats[name] = 1

    # RDKit Fragments
    for fr_name in [k for k in dir(Fragments) if k.startswith('fr_')]:
        count = getattr(Fragments, fr_name)(mol)
        if count > 0:
            cats[fr_name.replace('fr_', '')] = count

    # user-defined tags
    if not (usertags is None):
        for k, v in usertags.items():
            if mol.HasSubstructMatch(Chem.MolFromSmarts(v)):
                cats[k] = 1

    # Hierarchy propagation
    for parent, children in HIERARCHICAL_GROUPS.items():
        if any(child in cats for child in children):
            cats[parent] = 1

    # Carbohydrates
    for name, smarts in CARBOHYDRATE_PATTERNS.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            cats[name] = 1

    # Property categories
    cats.update(_get_property_categories(mol))

    # O–O distances (fixed variable name!)
    o_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(o_idxs) >= 2:
        dist_mat = Chem.GetDistanceMatrix(mol)
        for i, j in combinations(o_idxs, 2):
            d = int(dist_mat[i, j])
            if 2 <= d <= 6:
                path = Chem.GetShortestPath(mol, i, j)
                bonds = [mol.GetBondBetweenAtoms(path[k], path[k+1]) for k in range(len(path)-1)]
                rotatable = sum(1 for b in bonds if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing())
                rigid = len(bonds) - rotatable
                if 2 <= rotatable <= 6:
                    cats[f"O-O_rot_{rotatable}"] = 1
                if 2 <= rigid <= 6:
                    cats[f"O-O_rig_{rigid}"] = 1

    # Polyol stereochemistry
    oh_matches = mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H1]"))
    if len(oh_matches) >= 4:
        cats["≥4_hydroxyl"] = 1
        sig = "".join(
            "R" if mol.GetAtomWithIdx(m[0]).GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW
            else "S" if mol.GetAtomWithIdx(m[0]).GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
            else "?" for m in oh_matches
        )
        cats[f"OH_stereo_{len(oh_matches)}{sig}"] = 1

    return cats


def _is_allowed(mol: Chem.Mol) -> bool:
    return all(a.GetAtomicNum() in ALLOWED_ATOMIC_NUMS for a in mol.GetAtoms())


def _get_property_categories(mol: Chem.Mol) -> Dict[str, int]:
    cats = {}

    if (mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]")) or
        mol.HasSubstructMatch(Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1]")) or
        mol.HasSubstructMatch(Chem.MolFromSmarts("[PX4](=O)([OX2H])[OX2H]")) or
        mol.HasSubstructMatch(Chem.MolFromSmarts("c[OH1]"))):
        cats["acidic"] = 1

    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N-C=[O,S,N]);!$(N-S(=O)=O)]")):
        cats["basic"] = 1

    if mol.HasSubstructMatch(Chem.MolFromSmarts("[NX4+,NX3+;H3,H2,H1]")):
        cats["cationic"] = 1
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[O-,S-,P-,N-]")):
        cats["anionic"] = 1
    if cats.get("cationic") and cats.get("anionic"):
        cats["zwitterionic"] = 1

    if mol.HasSubstructMatch(Chem.MolFromSmarts("[N,O,S,F,Cl,Br,I;X1,X2,X3,X4;!+;!-]")):
        cats["lewis_base_lone_pair"] = 1

    if (mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=[CX3]")) or any(a.GetIsAromatic() for a in mol.GetAtoms())):
        cats["lewis_base_pi"] = 1

    if cats.get("lewis_base_lone_pair") or cats.get("lewis_base_pi"):
        cats["lewis_base"] = 1

    if mol.HasSubstructMatch(Chem.MolFromSmarts("[B;!$(B-[#6]);D2,D3]")) or mol.HasSubstructMatch(Chem.MolFromSmarts("[P,S;X5,X6]")):
        cats["lewis_acid"] = 1

    if any(a.GetIsAromatic() for a in mol.GetAtoms()):
        cats["aromatic"] = 1

    return cats

def categories_to_sparse(categories_list: List[Dict[str, int]], feature_list: Optional[List[str]] = None) -> Tuple[csr_matrix, List[str]]:
    if feature_list is None:
        feature_set = set()
        for d in categories_list:
            feature_set.update(d.keys())
        feature_list = sorted(feature_set)
    
    col_index = {feat: i for i, feat in enumerate(feature_list)}
    rows, cols, data = [], [], []

    for row_idx, cat_dict in enumerate(categories_list):
        for feat, val in cat_dict.items():
            if feat in col_index:
                rows.append(row_idx)
                cols.append(col_index[feat])
                data.append(1 if val == 1 else val)

    sparse_mat = csr_matrix((data, (rows, cols)), shape=(len(categories_list), len(feature_list)))
    return sparse_mat, feature_list




# Pharmacophoric atom invariant generator
_feat_gen = GetMorganFeatureAtomInvGen()

# Hashed fingerprint size (16 777 216 bits per Morgan generator)
HASHED_FP_SIZE = 2**24

# Fixed size of the RDKit topological fingerprint layer when used in count mode
RDKIT_COUNT_SIZE = 8192

# Default ensemble
DEFAULT_GENERATORS = {
    "morgan_r2": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=HASHED_FP_SIZE),
    "morgan_r3": rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=HASHED_FP_SIZE),
    "morgan_r2_count": rdFingerprintGenerator.GetMorganGenerator(radius=2, countSimulation=True, fpSize=HASHED_FP_SIZE),
    "morgan_r3_count": rdFingerprintGenerator.GetMorganGenerator(radius=3, countSimulation=True, fpSize=HASHED_FP_SIZE),
    "feat_morgan_r2_count_hashed": rdFingerprintGenerator.GetMorganGenerator(
        radius=2, atomInvariantsGenerator=_feat_gen, countSimulation=True, fpSize=HASHED_FP_SIZE
    ),
    "feat_morgan_r3_count_hashed": rdFingerprintGenerator.GetMorganGenerator(
        radius=3, atomInvariantsGenerator=_feat_gen, countSimulation=True, fpSize=HASHED_FP_SIZE
    ),
    "rdkit_count": rdFingerprintGenerator.GetRDKitFPGenerator(
        countSimulation=True, fpSize=RDKIT_COUNT_SIZE
    ),
}

def _process_single_molecule(args) -> Tuple[list, list, list]:
    mol, mol_idx, generators = args
    if mol is None:
        return [], [], []

    rows, cols, vals = [], [], []

    for gen_name, gen in generators.items():
        # Column offset for hashed Morgan generators
        if "morgan" in gen_name:
            offset = list(DEFAULT_GENERATORS.keys()).index(gen_name) * HASHED_FP_SIZE
        else:
            # RDKit count layer is placed after all hashed blocks
            offset = len([k for k in DEFAULT_GENERATORS.keys() if "morgan" in k]) * HASHED_FP_SIZE

        try:
            fp = gen.GetSparseCountFingerprint(mol)
            nz = fp.GetNonzeroElements()
            for bit, count in nz.items():
                rows.append(mol_idx)
                cols.append(offset + bit)
                vals.append(count)
        except Exception:
            pass

    return rows, cols, vals


def combine_with_structural_features(a: csr_matrix, b: csr_matrix) -> csr_matrix:
    from scipy.sparse import hstack
    return hstack([a, b], format="csr")


__all__ = [
    "classify_molecules_from_smiles",
    "categorize_molecule",
    "categorize_molecules",
    "categories_to_sparse",
    "ALLOWED_ATOMIC_NUMS",
    "HIERARCHICAL_GROUPS",
    "FUNCTIONAL_GROUPS",
    "CARBOHYDRATE_PATTERNS",
    "combine_with_structural_features",
    "DEFAULT_GENERATORS"
]


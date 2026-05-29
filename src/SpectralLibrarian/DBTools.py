from .LibraryManager import LibraryManager
from .MetaTools import harmonize_molecules
from .DataFrameTools import rename_columns
from .InstrumentalTools import extract_ionization

from matchms import Spectrum
from matchms.exporting import save_as_msp
from matchms.importing import load_from_msp

from typing import *
import numpy as np, pandas as pd, pickle as pkl, molmass as mm, networkx as nx
import xmltodict, sqlite3, os, re


class ms2dbfile():
    def __init__(self, specdb_path):
        self.specdb_path = specdb_path
        self.readfile = True
        if os.path.exists(self.specdb_path):
            print(self.specdb_path + " exists. Connecting to " + self.specdb_path)
            print("Connecting to SQLite database for spectrum data:", self.specdb_path)
            self.conn = sqlite3.connect(self.specdb_path)
            self.cursor = self.conn.cursor()
        else:
            os.makedirs(os.path.dirname(specdb_path), exist_ok=True)
            self.create_dbfiles()
            self.create_indices()

    def create_dbfiles(self):
        self.conn = sqlite3.connect(self.specdb_path)
        self.cursor = self.conn.cursor()
        self.conn.execute('PRAGMA journal_mode = MEMORY')
        self.conn.execute('PRAGMA synchronous = OFF')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS ms2_library (smiles TEXT, inchi TEXT, inchikey TEXT, formula, TEXT, instrument TEXT, ionization TEXT, ionmode TEXT, adduct TEXT, precursormz float, collision_gas TEXT, collision_energy REAL, ms2spectrum TEXT)''')
        self.conn.commit()
        self.drop_indices()

    def create_indices(self):
        # Generate index for speed up in searching
        self.conn.execute('PRAGMA journal_mode = MEMORY')  # save transaction log (journal) into memory for speedup instead of disk
        self.conn.execute('PRAGMA synchronous = OFF')  # turn off disk synchronization
        self.conn.execute('PRAGMA cache_size = -100000')   # Adjust cache size 100,000 page in KB -> 100MB
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_mz ON ms2_library(smiles)')  # make index for precursor_mz in the table 'data'
        self.conn.commit()

    def drop_indices(self):
        # delete index if exists to speed up inserting
        self.conn.execute('PRAGMA journal_mode = MEMORY')
        self.conn.execute('PRAGMA synchronous = OFF')
        self.cursor.execute('DROP INDEX IF EXISTS idx_mz')
        self.conn.commit()

    def close(self):
        self.conn.close()

    def insert_data(self, ms2specdata):
        self.cursor.executemany("INSERT INTO ms2_library (smiles, inchi, inchikey, formula, instrument, ionization, ionmode, adduct, precursormz, collision_gas, collision_energy, ms2spectrum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ms2specdata)
        self.conn.commit()

    def search_spectra(self, mz:float=None, ms1ppm:float=None, ionization="ESI", ionmode="Positive", collision_gas="N2", collision_energy=30.0):
        if not (mz is None):
            if not (ms1ppm is None):
                query = ("SELECT smiles, inchi, inchikey, formula, instrument, ionization, ionmode, adduct, precursor m/z, collision_gas, collision_energy, ms2spectrum FROM ms2_library "
                         "WHERE (ionization == ?) AND (ionmode == ?) AND (precursormz <= ?) AND (precursormz >= ?) AND (collision_gas == ?) AND (collision_energy == ?)")
                self.cursor.execute(query, (ionization, ionmode, mz * (1 + ms1ppm / 1e6), mz * (1 - ms1ppm / 1e6), collision_gas, collision_energy))
            else:
                raise ValueError('ms1ppm should be given if mz is given')
        else:
            if not (ms1ppm is None):
                print('ms1ppm is ignored since mz is not given')
            query = ("SELECT smiles, inchi, inchikey, formula, instrument, ionization, ionmode, adduct, precursormz, collision_gas, collision_energy, ms2spectrum FROM ms2_library "
                     "WHERE (ionization == ?) AND (ionmode == ?) AND (collision_gas == ?) AND (collision_energy == ?)")
            self.cursor.execute(query,(ionization, ionmode, collision_gas, collision_energy))

        results = pd.DataFrame.from_records(self.cursor.fetchall(), columns=[des[0] for des in self.cursor.description])
        return results


def read_publicDBs(specdbdir, dbnames:List[str]):
    dfs = []
    for dbname in dbnames:
        print("Reading " + dbname)
        if dbname == "HMDB":
            db = read_hmdb(hmdbdir = os.path.join(specdbdir, "HMDB")) # {'IONIZATION', 'INSTRUMENTTYPE'}
        elif dbname == "NIST23":
            db = read_nist(nistdir=os.path.join(specdbdir, "NIST23"))  # Intellectual Property Applied, {'IONIZATION', 'INSTRUMENTTYPE', 'INSTRUMENT'}
        elif dbname == "MoNA":
            db = read_mona(monadir=os.path.join(specdbdir, "MoNA")) # {'IONIZATION', 'INSTRUMENTTYPE', 'INSTRUMENT'}
        elif dbname == "MSnLib":
            db = read_msnlib(msnlibdir=os.path.join(specdbdir, "MSnLib Mass Spectral Libraries", "ms2_mgf")) # {'IONIZATION', 'INSTRUMENTTYPE', 'INSTRUMENT'}
        elif dbname == "Spx":
            db = read_spx(spxdir=specdbdir)  # 1267 by 32 # {'IONIZATION', 'INSTRUMENTTYPE', 'INSTRUMENT'}
        else:
            raise NotImplementedError('HMDB, NIST23, MoNA, MSnLIb, Spx are supported')
        dfs.append(db.df)

        if not ("INSTRUMENT" in db.df):
            print("INSTRUMENT field not in " + dbname)

    # if len(dfs) > 0:
    #     dfs_new = update_columns_in_dfs(dfs, colmap={'INSTRUMENTTYPE': instrument_type_mappings, 'IONIZATION': ionization_mapping}, use_regex=False)
    #     apply_instrument_corrections(dfs_new)
    return dfs


# TODO : NIST
def read_nist(nistdir):
    nist_dfs = []
    for pol in ["Pos", "Neg"]:
        filename = os.path.join(nistdir, "NIST23_" + pol + "_PubChem_Matches_HiResLC.msp")
        if os.path.exists(filename):
            nist_dfs.append(LibraryManager(filename).df)
    nist23_df = pd.concat(nist_dfs, axis=0, ignore_index=True) # 1331085 by 34
    nist23_df['PRECURSORTYPE'] = standardize_precursor_type_safe(nist23_df['PRECURSORTYPE'])
    nist23_df = nist23_df.dropna(subset=['PRECURSORTYPE'])
    nist23_df = harmonize_molecules(nist23_df, smiles_col='SMILES', inchi_col="INCHI")
    return LibraryManager(nist23_df)


NIST23_COLUMN_MAPPING = {
    # ==== Primary name ====
    # Only real "NAME" fields map to NAME – DESCRIPTION and Synon are left out on purpose
    'Name': 'NAME',   # MoNA primary name

    # ==== Identifiers ====
    'ID':           'ID',
    'FEATURE_ID':   'ID',
    'DB#':          'ID',
    'SPECTRUMID':   'SPECTRUMID',

    # ==== Molecular properties ====
    'FORMULA':                 'FORMULA',
    'Formula':                 'FORMULA', # msnlib
    'INCHI':                   'INCHI',
    'INCHIAUX':                'INCHIKEY', # msnlib
    'SHORTINCHI':              'INCHI',
    'INCHIKEY':                'INCHIKEY',
    'InChIKey':                'INCHIKEY',
    'SMILES':                  'SMILES',
    'CONNECTIVITY_SMILES':     'SMILES',
    'CAS':                     'CAS',
    'CID':                     'CID',
    'NIST':                    'NIST',
    'EXACT_MASS':              'EXACT_MASS',
    'ExactMass':               'EXACT_MASS',
    'EXACTMASS':               'EXACT_MASS',
    'PARENTMASS':              'PARENTMASS',
    'NOMINALMASS':             'NOMINALMASS',
    'XLOGP':                   'XLOGP',
    'IUPAC_NAME':              'IUPAC_NAME',

    # ==== Precursor / ion info ====
    'PRECURSORMZ':             'PRECURSORMZ',
    'PrecursorMZ':             'PRECURSORMZ',
    'PEPMASS':                 'PRECURSORMZ',

    'PRECURSORTYPE':           'PRECURSORTYPE',
    'Precursor_type':          'PRECURSORTYPE',
    'ADDUCT':                  'PRECURSORTYPE',

    'CHARGE':                  'CHARGE',
    'IONMODE':                 'IONMODE',
    'Ion_mode':                'IONMODE',

    # ==== Instrument & acquisition ====
    'INSTRUMENTTYPE':          'INSTRUMENTTYPE',
    'Instrument_type':         'INSTRUMENTTYPE',
    'INSTRUMENT_TYPE':         'INSTRUMENTTYPE',
    'INSTRUMENT':              'INSTRUMENT',
    'Instrument':              'INSTRUMENT',
    'SOURCE_INSTRUMENT':       'INSTRUMENT',

    'IONIZATION':              'IONIZATION',
    'ION_SOURCE':              'IONIZATION',

    'COLLISIONENERGY':         'COLLISIONENERGY',
    'Collision_energy':        'COLLISIONENERGY',
    'COLLISION_ENERGY':        'COLLISIONENERGY',
    'COLLISIONGAS':            'COLLISIONGAS',
    'IN_SOURCE_VOLTAGE':       'IN_SOURCE_VOLTAGE',

    'MSLEVEL':                 'MSLEVEL',

    # ==== Peaks ====
    'NUM PEAKS':               'NUM PEAKS',
    'Num peaks':               'NUM PEAKS',

    'mz_array':                'mz_array',
    'intensity_array':         'intensity_array',

    # ==== Comment ====
    'COMMENT':                 'COMMENT',
    'Comments':                'COMMENT',

    # ==== Peptide (kept because they exist in some nist23 entries) ====
    'PEPTIDE_SEQUENCE':        'PEPTIDE_SEQUENCE',
    'PEPTIDE_MODIFICATIONS':   'PEPTIDE_MODIFICATIONS',
}


# TODO : MSnlib

def read_msnlib(msnlibdir):
    msn_sublibs = ["20241003_enamdisc", "20241003_enammol", "20241003_mcebio", "20241003_mcedrug",
                   "20241003_mcescaf", "20241003_nihnp", "20241003_otavapep", "20250828_mcediv_50k_sub"]
    msnlib_dfs = []
    for pol in ["neg", "pos"]:
        for sublib in msn_sublibs:
            filedir = os.path.join(msnlibdir, sublib + "_" + pol + "_ms2.mgf")
            if os.path.exists(filedir):
                msnlib_dfs.append(LibraryManager(filedir).df)
    msn_lib_df = pd.concat(msnlib_dfs, axis=0)
    msn_lib_df = rename_columns(msn_lib_df, rename=NIST23_COLUMN_MAPPING, inplace=False)
    msn_lib_df['PRECURSORTYPE'] = standardize_precursor_type_safe(msn_lib_df['PRECURSORTYPE'])
    msn_lib_df = msn_lib_df.dropna(subset=['PRECURSORTYPE'])
    msn_lib_df = harmonize_molecules(msn_lib_df, smiles_col='SMILES', inchi_col="INCHI") # has IONIZATION, and INSTRUMENTTYPE, SMILES, SMILES_isomeric field
    return LibraryManager(msn_lib_df)


# TODO : MoNA
def read_mona(monadir):
    mona = LibraryManager(os.path.join(monadir, "MoNA-export-LC-MS-MS_Spectra.msp"))  # .df 153624 by 17
    mona.df = rename_columns(mona.df, rename=NIST23_COLUMN_MAPPING, inplace=False)
    mona.df = extract_real_identifiers_from_mona_comment(mona.df, 'COMMENT')
    mona.df['IONMODE'] = mona.df['IONMODE'].replace({'N': 'NEGATIVE', 'P': 'POSITIVE'})
    mona.df = mona.df.mask(mona.df.isin(['NA', 'n/a', 'N/A', 'None', 'none', 'NAN', 'NaN', 'nan', '', 'null', 'NULL', 'missing', 'MISSING'])) # remove these values as 'NA'
    mona.df['IONIZATION'] = mona.df['INSTRUMENTTYPE'].apply(extract_ionization)
    mona.df['PRECURSORTYPE'] = standardize_precursor_type_safe(mona.df['PRECURSORTYPE'])
    mona.df = mona.df.dropna(subset=['PRECURSORTYPE'])
    mona.df = harmonize_molecules(mona.df, smiles_col='SMILES', inchi_col="INCHI")
    return mona


def extract_real_identifiers_from_mona_comment(df: pd.DataFrame, comment_col: str = 'COMMENT') -> pd.DataFrame:
    """
    Extracts REAL structure identifiers from MoNA COMMENT field.
    Handles the annoying 'computed SMILES=...', 'computed InChI=...' format that MoNA uses.
    Ignores all computed adduct m/z garbage.
    """
    df = df.copy()

    # Compile once, reuse
    regex_patterns = {
        'SMILES': re.compile(r'(?:^|" ?)SMILES[= ]"?([^" ]+)"?', re.IGNORECASE),
        'INCHI': re.compile(r'(?:^|") ?InChI[= ]"?([^" ]+)', re.IGNORECASE),
        'CAS': re.compile(r'"?cas[= ]"?([0-9-]+)"?', re.IGNORECASE),
        'CID': re.compile(r'(?:pubchem cid|cid)[= ]"?(\d+)"?', re.IGNORECASE),
        'SID': re.compile(r'pubchem sid[= ]"?(\d+)"?', re.IGNORECASE),
        'KEGG': re.compile(r'"?kegg[= ]"?(C?\d{5})"?', re.IGNORECASE),
        'CHEBI': re.compile(r'"?chebi[= ]"?(\d+)"?', re.IGNORECASE),
        'ACCESSION': re.compile(r'"?accession[= ]"?([^" ]+)"?', re.IGNORECASE),
        'EXACT_MASS': re.compile(r'"?exact mass[= ]"?([0-9\.]+)"?', re.IGNORECASE),
        'SPLASH': re.compile(r'"?SPLASH[= ]"?([^" ]+)"?', re.IGNORECASE),
        'LICENSE': re.compile(r'"?license[= ]"?([^" ]+)"?', re.IGNORECASE),
        'MoNA_Rating': re.compile(r'"?MoNA Rating[= ]"?([0-9\.]+)"?', re.IGNORECASE),
        'Computed_Spectral_Entropy': re.compile(r'"?computed spectral entropy[= ]"?([0-9\.]+)"?', re.IGNORECASE),
        'Computed_Normalized_Entropy': re.compile(r'"?computed normalized entropy[= ]"?([0-9\.]+)"?', re.IGNORECASE),
    }

    # Special case: MoNA often puts REAL identifiers as "computed SMILES=...", "computed InChI=..."
    computed_ids = {'SMILES': re.compile(r'"computed SMILES[= ]"?([^" ]+)"?', re.IGNORECASE),
                    'INCHI': re.compile(r'"computed InChI[= ]"?([^" ]+)"?', re.IGNORECASE)}

    def parse_row(comment: str):
        if pd.isna(comment):
            return {k: None for k in regex_patterns.keys()}

        result = {}
        # Try real (non-computed) fields first, and then override SMILES/INCHI with the "computed" ones (this is the real structure in MoNA!)
        for key, pat in list(regex_patterns.items()) + list(computed_ids.items()):
            m = pat.search(comment)
            if m:
                result[key] = m.group(1).strip()

        for k in regex_patterns.keys():
            result.setdefault(k, None) # Fill missing with None

        return result

    # Apply to entire column (vectorized-ish, but safe)
    parsed = df[comment_col].apply(parse_row)
    parsed_df = pd.DataFrame(parsed.tolist(), index=df.index)

    # Attach to original dataframe
    return pd.concat([df, parsed_df], axis=1)


# TODO :  HMDB
# Experimental MS2
def parse_hmdb_to_msp(hmdbdir):
    metaboliteInfo = parse_hmdb_metabolite_infos(hmdbdir)

    expms2dir = os.path.join(hmdbdir, "hmdb_experimental_msms_spectra")
    ms2xmlfiles = [x for x in os.listdir(expms2dir) if os.path.isfile(os.path.join(expms2dir, x)) and x.endswith(".xml")]
    goodfiles, badfiles, badreason = [], [], []

    ms2_refmspspectra = []
    for cnt, xmlf in enumerate(ms2xmlfiles):
        xml_string = open(os.path.join(expms2dir, xmlf), "r", encoding='utf-8').read()
        data_dict = xmltodict.parse(xml_string)['ms-ms']

        databaseID = data_dict.get('database-id', None)
        if (databaseID in metaboliteInfo) and ('ms-ms-peaks' in data_dict.keys()) and (data_dict['predicted'] == 'false'):
            try:
                # Molecular info (cas_registry_number, chebi_id, chemical_formula, drugbank_id, inchi, inchikey,
                # kegg_id, metlin_id, monisotopic_molecular weight, name, pdb_id, pubchem_compound_id,
                # secondary_accessions, smiles, stats
                minfo = metaboliteInfo[databaseID]
                name = minfo.get('name', None)
                smiles = minfo.get('smiles', None)
                inchi = minfo.get('inchi', None)
                inchikey = minfo.get('inchikey', None)
                formula = minfo.get('chemical_formula', None)
                try:
                    monomass = mm.Formula(formula).monoisotopic_mass
                except: # weird chemical formula, so ignore this ms2 spectrum
                    raise Exception('Abnormal Formula')

                note = data_dict.get('notes', None)
                if type(note) == dict:
                    note = None
                structureID = data_dict.get('structure-id', None) # STRUCTUREID

                # Instrument (LC, MS) info
                chromtype = data_dict.get('chromatography-type', None) # CHROMATOGRAPHYTYPE
                instype = data_dict.get('instrument-type', None) # INSTRUMENTTYPE
                if type(instype) == dict:
                    instype = None
                ionizetype = data_dict.get('ionization-type', None) # IONIZATION_TYPE

                # Polarity
                polarity = data_dict.get('ionization-mode', None) # IONIZATIONMODE
                if (polarity is None) and (type(data_dict['charge-type']) == dict):
                    raise Exception("Unknown Ionization Mode")
                else:
                    if polarity is None:
                        polarity = data_dict.get('charge-type', None) # CHARGETYPE

                # Adduct info
                adduct_type = data_dict.get('adduct-type', None) # ADDUCTTYPE (+, -)
                adduct = data_dict.get('adduct', None) # PRECURSORTYPE (M+H, M-H, etc)
                if (adduct is None) or (type(adduct) == dict) or (type(adduct_type) == dict) or (type(data_dict["adduct-mass"]) == dict):
                    raise Exception("Unknown adduct form")
                else:
                    precursormz = data_dict.get('adduct-mass', 0.0)

                # Spectrum ID
                if "references" in data_dict:
                    tem = data_dict['references']
                    if "reference" in tem:
                        tem2 = tem['reference']
                        if 'spectra-id' in tem2:
                            spectra_id = tem2['spectra-id']
                        else:
                            spectra_id = None
                    else:
                        spectra_id = None
                else:
                    spectra_id = None

                # MS2 spectrum peak
                ms2peaks = data_dict['ms-ms-peaks']["ms-ms-peak"]
                if type(ms2peaks) == dict:
                    m2zs = ms2peaks["mass-charge"]
                    its = ms2peaks["intensity"]
                    if type(m2zs) == str and type(its) == str:
                        mzs = np.array([float(m2zs)])
                        ints = np.array([float(its)])
                        ints = np.array(ints) / np.max(ints)
                    elif type(m2zs) == list and type(its) == list:
                        mzs = np.array(ms2peaks["mass-charge"])  # np.array([float(ms2spectrum["mass-charge"])])
                        ints = np.array(ms2peaks["intensity"])  # np.array([float(ms2spectrum["intensity"])])
                        ints = np.array(ints) / np.max(ints)
                    else:
                        raise Exception("Unexpected MS2 spectrum data type")
                else: # ms2peaks = List[dict]
                    mzs, ints = [], []
                    for p in ms2peaks:
                        mzs.append(float(p["mass-charge"]))
                        ints.append(float(p["intensity"]))
                    mzs = np.array(mzs)
                    ints = np.array(ints) / np.max(ints)

                # Collision Energy
                ce = data_dict.get('collision-energy-voltage', None)
                if ce is None:
                    raise Exception("Unknown CE")

                msp_metadata = {"NAME": name, "SMILES": smiles, "INCHI": inchi, "INCHIKEY": inchikey, "Formula": formula, "MOLECULARMZ": monomass, "NOTE": note, "DB#": databaseID, "STRUCTUREID": structureID,
                                "SPECTRUMID": spectra_id, "CHROMATOGRAPHYTYPE": chromtype, "INSTRUMENTTYPE": instype, "IONIZATION": ionizetype, "IONMODE": polarity,
                                "ADDUCTTYPE": adduct_type, "ADDUCT": adduct, "PRECURSORMZ": precursormz, "COLLISIONENERGY": ce}
                ms2_refmspspectra.append(Spectrum(mz=mzs, intensities=ints, metadata=msp_metadata))

                goodfiles.append(xmlf)
                print(str(cnt + 1) + " / " + str(len(ms2xmlfiles)) + " : " + xmlf + " : SUCCESS!")
            except Exception as e:
                badfiles.append(xmlf)
                badreason.append("Unexpected MS2 data type")
                print(str(cnt + 1) + " / " + str(len(ms2xmlfiles)) + " : " + xmlf + " : FAIL!")

    if os.path.exists(os.path.join(hmdbdir, "HMDB_experimental.msp")):
        os.remove(os.path.join(hmdbdir, "HMDB_experimental.msp"))
    save_as_msp(ms2_refmspspectra, os.path.join(hmdbdir, "HMDB_experimental.msp"))

    sfx2obj = {"good": goodfiles, "bad": badfiles, "bad_reason": badreason}
    for k, obj in sfx2obj.items():
        with open(os.path.join(hmdbdir, "HMDB_all_metabolites_" + k + ".pkl"), "wb") as f:
            pkl.dump(obj, f)

def parse_hmdb_metabolite_infos(hmdbdir):
    """Read hmdb_metabolites.xml file, convert it to dict, save it as pkl and return it. If the pkl file exist, read and return the dict read"""
    allmeta_pkl = os.path.join(hmdbdir, "HMDB_all_metabolites.pkl")
    metaboliteInfo = {}
    if not os.path.exists(allmeta_pkl):
        allmeta_xml = os.path.join(hmdbdir, "hmdb_metabolites", "hmdb_metabolites.xml")
        xml_string = open(allmeta_xml, encoding="utf-8").read()
        metabocards = xmltodict.parse(xml_string)['hmdb']['metabolite']

        features = ["status", "secondary_accessions", "name", "chemical_formula", "monisotopic_molecular_weight",
                    "cas_registry_number", "smiles", "inchi", "inchikey", "pubchem_compound_id", "kegg_id", "chebi_id",
                    "drugbank_id", "metlin_id", "pdb_id"]
        for m in metabocards:
            print(m["accession"])
            metaboliteInfo[m["accession"]] = {k: m.get(k, None) for k in features}

        with open(allmeta_pkl, "wb") as f:
            pkl.dump(metaboliteInfo, f)
    else:
        with open(allmeta_pkl, "rb") as f:
            metaboliteInfo = pkl.load(f)
    print(str(len(metaboliteInfo.keys())) + " metabolites found")
    return metaboliteInfo

def read_hmdb(hmdbdir):
    hmdbmspfile = os.path.join(hmdbdir, "HMDB_experimental.msp")
    if not os.path.exists(hmdbmspfile):
        parse_hmdb_to_msp(hmdbdir)
    hmdb = LibraryManager(hmdbmspfile)  # .df 153624 by 17
    rename_columns(hmdb.df, rename=NIST23_COLUMN_MAPPING)
    hmdb.df = hmdb.df.mask(hmdb.df.isin(['NA', 'n/a', 'N/A', 'None', 'none', 'NAN', 'NaN', 'nan', '', 'null', 'NULL', 'missing', 'MISSING'])) # remove these values as 'NA'
    hmdb.df['PRECURSORTYPE'] = standardize_precursor_type_safe(hmdb.df['PRECURSORTYPE'])
    hmdb.df = hmdb.df.dropna(subset=['PRECURSORTYPE'])
    hmdb.df = harmonize_molecules(hmdb.df, smiles_col='SMILES', inchi_col="INCHI")
    hmdb.df.to_csv(os.path.join(hmdbdir, "HMDB_experimental.csv"), index=False)
    return hmdb

def print_isomers(hmdbdir):
    hmdbmspfile = os.path.join(hmdbdir, "HMDB_experimental.msp")
    if not os.path.exists(hmdbmspfile):
        parse_hmdb_to_msp(hmdbdir)
    hmdbspectra = load_from_msp(hmdbmspfile)

    cpdnames, formulas, smiles, inchis, mimws = [], [], [], [], []
    elenums = {"C": [], "H": [], "O": [], "N": [], "S": [], "P": [], "Na": [], "Mg": [], "K": [], "Ca": [], "Se": [], "F": [], "Cl": [], "Br": []}
    for spectrum in hmdbspectra:
        specmeta = spectrum.metadata
        f = specmeta.get("formula", None)
        if not (f is None):
            mmf = mm.Formula(f)
            for e, cnts in elenums.items():
                if e in mmf._elements:
                    cnts.append(mmf._elements[e][0])
                else:
                    cnts.append(0)

            cpdnames.append(specmeta.get("compound_name", None))
            formulas.append(f)
            smiles.append(specmeta.get("smiles", None))
            inchis.append(specmeta.get("inchi", None))
            mimws.append(mmf.monoisotopic_mass)

    df = pd.DataFrame.from_dict({"Name": cpdnames, "SMILES": smiles, "INCHI": inchis, "Formula": formulas, "MIMW": mimws})
    df_e = pd.DataFrame.from_dict(elenums)
    df = pd.concat([df, df_e], axis=1).dropna(axis=1)
    df.drop_duplicates(subset="Name", keep="first", inplace=True)  # subset=["SMILES", "INCHI", "Formula"]

    # Isomers
    enums_diff = np.bool(np.ones(shape=[len(df), len(df)]))
    for e in elenums.keys():
        enums = np.array(df[e])
        enums_diff = np.multiply(enums_diff, (enums[:, np.newaxis] == enums[np.newaxis, :]))
    np.fill_diagonal(enums_diff, False)
    isopairs = np.column_stack(np.where(np.triu(enums_diff, k=1)))

    G = nx.Graph()
    G.add_edges_from(list(zip(isopairs[:, 0], isopairs[:, 1])))
    ccs = list(nx.connected_components(G))
    for cc in ccs:
        print(" : ".join([list(df["Name"])[c] for c in cc]) + " == " + list(df["Formula"])[list(cc)[0]] + " ( " + str(list(df["MIMW"])[list(cc)[0]]) + " )")

    # # Isobars within ms1ppm
    # mzs = np.array([mm.Formula(f).monoisotopic_mass for f in formulas])
    # # mzdiff = np.multiply(np.divide(abs((mzs[:, np.newaxis] - mzs[np.newaxis, :])), mzs)*1e6 < 20, np.divide(abs((mzs[:, np.newaxis] - mzs[np.newaxis, :])), mzs)*1e6 < 20
    # df.values[:,5:]


# TODO : sugar phosphate
from .PrecursorTools import standardize_precursor_type_safe

def read_spx(spxdir):
    spx = LibraryManager(os.path.join(spxdir, "20251206_pos_neg_all_adducts_merged_v3.msp"))  # .df 1267 by 25
    spx.df = spx.df.rename(columns=NIST23_COLUMN_MAPPING)
    spx.df['PRECURSORTYPE'] = standardize_precursor_type_safe(spx.df['PRECURSORTYPE'])
    spx.df = spx.df.dropna(subset=['PRECURSORTYPE'])
    spx.df = harmonize_molecules(spx.df, smiles_col='SMILES')
    return spx




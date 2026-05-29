# src/SpectralLibrarian/PredictionTools.py
"""
PredictionTools – Transformation product prediction (exact + template modes)

Now supports your ORDerly extraction layout:
- ord_parquet_path can be a directory (e.g. "ord_parquet/extracted_ords")
- Auto-detects the reaction SMILES column
- Processes parquet files one-by-one for low memory usage
"""
from .SpectralTools import normalizeSpectrum
from collections import defaultdict
from typing import *

from huggingface_hub.utils import capture_output
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

import pandas as pd, pickle as pkl, matplotlib.pyplot as plt
import warnings, time, json, os

# === OPTIONAL DEPENDENCIES ===
try:
    from rxnutils.chem.reaction import ChemicalReaction
    HAS_RXNUTILS = True
except ImportError:
    HAS_RXNUTILS = False
    warnings.warn("rxnutils not installed → ORD template extraction disabled.", ImportWarning)

cts_base_url = "https://qed.epa.gov"
cts_socketio_path = "cts/ws"
try:
    import socketio
    HAS_CTS = True
except ImportError:
    HAS_CTS = False
    warnings.warn("socketio not installed → CTS extraction disabled.", ImportWarning)

try:
    from enviPath_python import enviPath
    HAS_ENVIPATH = True
except ImportError:
    HAS_ENVIPATH = False
    warnings.warn("enviPath-python not installed → curated rules disabled.", ImportWarning)
enviPath_package_uri: str = "http://envipath.org/package/32de3cf4-e3e6-4168-956e-32fa5ddb0ce1"


class PredictTransformation:
    """
    Predict transformation products using exact lookup OR/AND generalized templates.
    Fully compatible with your ORDerly-extracted parquet directory.
    """

    def __init__(self, ord_parquet_path: str, cache_dir: str  = "tp_cache", methods: Optional[List[str]] = None, max_ord_templates: int = 2000, ord_template_radius: int = 1):
        self.cache_dir = os.path.join(os.path.expanduser("~"), cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        if methods is None:
            print("Methods are not given. Use the default method setting (running all the 4 methods)")
            self.methods = ["exact", "template", "cts", "enviPath"]
        else:
            self.methods = methods
        self.use_exact = ("exact" in self.methods) and bool(ord_parquet_path)
        self.use_cts = ("cts" in self.methods) and HAS_CTS
        self.use_templates = ("template" in self.methods) and HAS_RXNUTILS and bool(ord_parquet_path)
        self.use_envi = ("enviPath" in self.methods) and HAS_ENVIPATH

        self.exact_lookup: Dict[str, List[str]] = {}
        self.ord_rxns: List[AllChem.ChemicalReaction] = []
        self.envi_rxns: List[tuple[AllChem.ChemicalReaction, str]] = []

        if self.use_exact or self.use_templates:
            self._parquet_source = ord_parquet_path
            if not os.path.exists(self._parquet_source):
                raise FileNotFoundError(f"ORD parquet path not found: {ord_parquet_path}")

        if self.use_exact:
            self._build_exact_lookup()

        if self.use_templates:
            self._load_or_extract_ord_templates(max_ord_templates, ord_template_radius)

        if self.use_envi:
            self._load_enviPath_rules(enviPath_package_uri)

        print(f"PredictTransformation loaded → "
              f"exact: {len(self.exact_lookup)} reactants | "
              f"ORD templates: {len(self.ord_rxns)} | "
              f"enviPath rules: {len(self.envi_rxns)}")

    # ====================== HELPER: Find reaction SMILES column ======================
    @staticmethod
    def _get_reaction_smiles_col(df: pd.DataFrame) -> str:
        """Auto-detect the column containing reaction SMILES."""
        candidates = ["reaction_smiles", "rxn_smiles", "smiles", "reaction", "standardized_rxn_smiles"]
        for col in candidates:
            if col in df.columns:
                print(f"✓ Using reaction SMILES column: '{col}'")
                return col
        # Fallback: any column containing "smiles" or "rxn"
        for col in df.columns:
            if "smiles" in col.lower() or "rxn" in col.lower():
                print(f"✓ Using reaction SMILES column (detected): '{col}'")
                return col
        raise KeyError(
            f"No reaction SMILES column found!\n"
            f"Available columns: {list(df.columns)}\n"
            f"Try printing df.columns.tolist() on one of your parquet files."
        )

    # ====================== EXACT LOOKUP ======================
    @staticmethod
    def _canonical_smiles(smiles: str) -> str:
        if not smiles:
            return ""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return smiles

    def _build_exact_lookup(self):
        cache_file = os.path.join(self.cache_dir, "exact_lookup.pkl")
        if os.path.exists(cache_file):
            print("Loading cached exact lookup...")
            with open(cache_file, "rb") as f:
                self.exact_lookup = pkl.load(f)
            return

        print("Building exact canonical lookup from ORDerly parquet files...")
        lookup: Dict[str, set] = defaultdict(set)

        parquet_files = self._get_parquet_files()
        for p in tqdm(parquet_files, desc="Processing parquet files"):
            df = pd.read_parquet(p)

            reactcols = [x for x in df.columns if x.startswith("reactant_")]
            prodtcols = [x for x in df.columns if x.startswith("product_")]
            df_rp = df[reactcols+prodtcols]
            df_rp = df_rp.replace('<missing>', None)
            df_rp.dropna(how='all', inplace=True)
            df_rp.reset_index(drop=True, inplace=True)

            for _, df_rxn in df_rp.iterrows():
                try:
                    reactants = [df_rxn[r] for r in reactcols if not (df_rxn[r] is None)]
                    products = [df_rxn[p] for p in prodtcols if not (df_rxn[p] is None)]
                    for r in reactants:
                        canon_r = self._canonical_smiles(r)
                        if canon_r:
                            for p in products:
                                canon_p = self._canonical_smiles(p)
                                if canon_p:
                                    lookup[canon_r].add(canon_p)
                except Exception:
                    continue

        self.exact_lookup = {k: sorted(v) for k, v in lookup.items()}

        with open(cache_file, "wb") as f:
            pkl.dump(self.exact_lookup, f)
        print(f"Cached exact lookup for {len(self.exact_lookup)} reactants")

    # ====================== TEMPLATE EXTRACTION ======================
    def _load_or_extract_ord_templates(self, max_templates: int, radius: int):
        cache_file = self.cache_dir / f"ord_templates_r{radius}_top{max_templates}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                template_smarts_list = pkl.load(f)
        else:
            print("Extracting reaction templates from ORDerly parquet files...")
            template_count = defaultdict(int)
            parquet_files = self._get_parquet_files()

            for p in tqdm(parquet_files, desc="Extracting templates"):
                df = pd.read_parquet(p)

                reactcols = [x for x in df.columns if x.startswith("reactant_")]
                prodtcols = [x for x in df.columns if x.startswith("product_")]
                df_rp = df[reactcols + prodtcols]
                df_rp = df_rp.replace('<missing>', None)
                df_rp.dropna(how='all', inplace=True)
                df_rp.reset_index(drop=True, inplace=True)

                for _, df_rxn in df_rp.iterrows():
                    try:
                        reactants = [df_rxn[r] for r in reactcols if not (df_rxn[r] is None)]
                        products = [df_rxn[p] for p in prodtcols if not (df_rxn[p] is None)]
                        reaction = ".".join(reactants)+">>"+".".join(products)
                        rxn = ChemicalReaction(reaction)
                        templates = rxn.generate_reaction_template(radius=radius)
                        for t in templates:
                            if t.smarts:
                                template_count[t.smarts] += 1
                    except Exception:
                        continue

            sorted_templates = sorted(template_count.items(), key=lambda x: x[1], reverse=True)
            template_smarts_list = [smarts for smarts, _ in sorted_templates[:max_templates]]
            with open(cache_file, "wb") as f:
                pkl.dump(template_smarts_list, f)

        for smarts in tqdm(template_smarts_list, desc="Compiling ORD templates"):
            try:
                rxn = AllChem.ReactionFromSmarts(smarts)
                if rxn:
                    self.ord_rxns.append(rxn)
            except Exception:
                continue

    def _get_parquet_files(self) -> List[str]:
        """Return list of parquet files (handles file or directory)."""
        if os.path.exists(self._parquet_source):
            return [os.path.join(self._parquet_source, x) for x in os.listdir(self._parquet_source) if x.endswith("parquet")]
        else:
            return []

    # ====================== ENVIPATH (unchanged) ======================
    def _load_enviPath_rules(self, package_uri: str):
        cache_file = os.path.join(self.cache_dir, "enviPath_rules.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.envi_rxns = pkl.load(f)
            return

        print("Fetching enviPath rules...")
        eP = enviPath("https://envipath.org")
        bbd = eP.get_package(package_uri)
        rules = bbd.get_rules()

        for rule in tqdm(rules, desc="Compiling enviPath rules"):
            try:
                smirks = rule.get_smirks()
                if smirks:
                    name = rule.get_name() or f"enviPath_{rule.get_id()}"
                    rxn = AllChem.ReactionFromSmarts(smirks)
                    if rxn:
                        self.envi_rxns.append((rxn, name))
            except Exception:
                continue

        with open(cache_file, "wb") as f:
            pkl.dump(self.envi_rxns, f)


    def predict(self, smiles_list: Iterable[str], max_products_per_mol: int = 15, as_dataframe: bool = True) -> pd.DataFrame | Dict[str, List[Dict[str, str]]]:
        smiles_list = list(smiles_list)
        results = defaultdict(list)

        for smiles in tqdm(smiles_list, desc="Predicting TPs"):
            if not smiles or not isinstance(smiles, str):
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            dedup = set()
            for mode in self.methods:
                if mode == "exact":
                    self.predict_by_exactORD(smiles, results, dedup)
                elif mode == "template":
                    self.predict_by_templateORD(mol, smiles, results, dedup)
                elif (mode == "cts") and self.use_cts:
                    self.predict_by_cts(smiles, results, dedup)
                elif mode == "enviPath":
                    self.predict_by_enviPath(mol, smiles, results, dedup)
                else:
                    NotImplementedError("Only 4 modes are supported currently : 'exact', 'template', 'cts', and 'enviPath'")

            results[smiles] = results[smiles][:max_products_per_mol]

        if as_dataframe:
            rows = []
            for input_smi, preds in results.items():
                for p in preds:
                    rows.append({"input_smiles": input_smi, "product_smiles": p["product_smiles"], "source": p["source"], "rule_name": p["rule_name"]})
            return pd.DataFrame(rows)

        return dict(results)

    def predict_by_exactORD(self, smiles:str, results:Dict[str, List[Dict[str, Any]]], dedup:set):
        # return products by matching reactant with that of reactions in ORD
        canon = self._canonical_smiles(smiles)
        for p in self.exact_lookup.get(canon, []):
            if p not in dedup:
                dedup.add(p)
                results[smiles].append({"product_smiles": p, "source": "ORD_exact", "rule_name": None})

    def predict_by_templateORD(self, mol: Chem.Mol, smiles:str, results:Dict[str, List[Dict[str, Any]]], dedup:set):
        # return products by matching reactant template in ORD
        for rxn in self.ord_rxns:
            try:
                ps = rxn.RunReactants([mol])
                for prod_tuple in ps:
                    for p in prod_tuple:
                        if p:
                            p_smiles = Chem.MolToSmiles(p, canonical=True, isomericSmiles=True)
                            if p_smiles and p_smiles not in dedup:
                                dedup.add(p_smiles)
                                results[smiles].append(
                                    {"product_smiles": p_smiles, "source": "ORD_template", "rule_name": None})
            except Exception:
                continue

    def predict_by_enviPath(self, mol: Chem.Mol, smiles:str, results:Dict[str, List[Dict[str, Any]]], dedup:set):
        # predict by enviPath. This requires account.
        for rxn, rule_name in self.envi_rxns:
            try:
                ps = rxn.RunReactants([mol])
                for prod_tuple in ps:
                    for p in prod_tuple:
                        if p:
                            p_smiles = Chem.MolToSmiles(p, canonical=True, isomericSmiles=True)
                            if p_smiles and p_smiles not in dedup:
                                dedup.add(p_smiles)
                                results[smiles].append(
                                    {"product_smiles": p_smiles, "source": "enviPath", "rule_name": rule_name})
            except Exception:
                continue


    def predict_by_cts(self, smiles, results:Dict[str, List[Dict[str, Any]]], dedup:set, gen_limit=1, prop="cyp450", ph="7.4", timeout=120):
        sio = socketio.Client(logger=False, engineio_logger=False)

        products = []
        done = {"flag": False}

        @sio.event
        def connect():
            payload = {"chemical": smiles, "ph": str(ph), "pchem_request": {}, "service": "getTransProducts", "workflow": "gentrans",
                       "chem_info": "", "run_type": "single", "calc": "biotrans", "gen_limit": gen_limit,
                       "metabolizer_post": {"chemical": smiles,"gen_limit": gen_limit, "prop": prop}}
            sio.emit("get_data", json.dumps(payload))

        @sio.on("message")
        def on_message(msg):
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    return

            if not isinstance(msg, dict):
                return

            if msg.get("calc") != "biotrans":
                return

            tree = msg.get("data", {})
            children = tree.get("children", [])
            for child in children:
                child_data = child.get("data", {})
                product_smiles = child_data.get("smiles", None)

                if product_smiles:
                    products.append({"input_smiles": smiles, "product_smiles": product_smiles, "route": child_data.get("routes"), "accumulation": child_data.get("accumulation"),
                                     "production": child_data.get("production"), "likelihood": child_data.get("likelihood"), "generation": child.get("id"),
                                     "source": "CTS/BioTransformer", "prop": prop, "generation_limit": gen_limit})
            done["flag"] = True

        sio.connect(cts_base_url, socketio_path=cts_socketio_path, transports=["websocket", "polling"])

        start = time.time()
        while time.time() - start < timeout and not done["flag"]:
            sio.sleep(1)
        sio.disconnect()

        for p in products:
            p_smiles = p["product_smiles"]
            if p_smiles and p_smiles not in dedup:
                dedup.add(p_smiles)
                results[smiles].append({"product_smiles": p_smiles, "source": p["source"], "rule_name": p["route"]})


from .MetaTools import harmonize_molecules
from rdkit import Chem
from rdkit.Chem import Draw


def PredictTransformation_table(inputfile: str, outdir: str, methods:List[str], ord_parquet_path:str=None, sheet_name:str=None, smiles_cols="SMILES", inchi_cols="INCHI"):
    figdir = os.path.join(outdir, "Structure")
    os.makedirs(figdir, exist_ok=True)
    finalfile = os.path.join(outdir, os.path.splitext(os.path.basename(inputfile))[0] + "_transformed.xlsx")
    if not os.path.exists(finalfile):
        if not (sheet_name is None):
            df = pd.read_excel(inputfile, sheet_name=sheet_name)
        else:
            df = pd.read_excel(inputfile)

        has_smiles = smiles_cols in df.columns
        has_inchis = inchi_cols in df.columns

        if (not has_smiles) and (not has_inchis):
            raise KeyError(f"At least " + smiles_cols + " or " + inchi_cols + " should be given in this case. Available columns: {list(lib.df.columns)}")

        if not has_smiles:
            print("SMILES column " + smiles_cols + " was not detected, while InChI column "+ inchi_cols + " was. Using the " + inchi_cols + " column to retrieve the isomeric SMILES to be used for transformation prediction")
            df = harmonize_molecules(df=df, inchi_col=inchi_cols)
            smiles_cols = "SMILES_isomeric"

        unique_smiles = (df[smiles_cols].dropna().astype(str).str.strip().unique())
        unique_smiles_list = [s for s in unique_smiles if s and s.lower() not in {"nan", "none", ""}]
        print(f"Found {len(unique_smiles_list)} unique SMILES")

        predictor = PredictTransformation(ord_parquet_path=ord_parquet_path, methods=methods)  # methods = ["exact", "template", "cts", "enviPath"]
        df_output = pd.DataFrame(predictor.predict(unique_smiles_list))
        df_final = pd.merge(df, df_output, left_on=smiles_cols, right_on="input_smiles", how="left")
        df_final.to_excel(os.path.join(outdir, os.path.splitext(os.path.basename(inputfile))[0] + "_transformed.xlsx"), index=False)
    else:
        df_final = pd.read_excel(finalfile)

    df_final_output = df_final.loc[pd.notnull(df_final["product_smiles"])]
    inputsmiles_list = list(set(df_final_output["input_smiles"]))
    for inputsmiles in inputsmiles_list:
        mol = Chem.MolFromSmiles(inputsmiles)
        img = Draw.MolToImage(mol)
        plt.imshow(img)
        plt.savefig(os.path.join(figdir, "_".join(["Reactant", inputsmiles]) + ".png"), dpi=900)
        # plt.savefig(os.path.join(figdir, "_".join(["Reactant", inputsmiles]) + ".pdf"), dpi=900)
        plt.close()

        df_asp = df_final_output.loc[(df_final_output["input_smiles"] == inputsmiles) & (df_final_output["source"] == "CTS/BioTransformer")][["input_smiles", "product_smiles", "rule_name"]]
        prd_smiles = list(df_asp["product_smiles"])
        for i, s in enumerate(prd_smiles):
            mol = Chem.MolFromSmiles(s)
            img = Draw.MolToImage(mol) # size=(500, 500))
            plt.imshow(img)
            plt.savefig(os.path.join(figdir, "_".join(["Reactant", inputsmiles, "Product", str(i+1), s]) + ".png"), dpi=900)
            plt.savefig(os.path.join(figdir, "_".join(["Reactant", inputsmiles, "Product", str(i+1), s]) + ".pdf"), dpi=900)
            plt.close()


import os, shutil, subprocess, tempfile
from typing import List, Literal



class PredictSpectra:
    def __init__(self, image="wishartlab/cfmid:latest", output_dir="./predicted_spectra", prob_thresh=0.001):
        self.image = image
        self.output_dir = output_dir
        self.prob_thresh = prob_thresh

    def _run(self, cmd, check=True):
        return subprocess.run(cmd, text=True, capture_output=True, check=check)

    def check_docker(self):
        if shutil.which("docker") is None:
            raise RuntimeError("Docker is not installed or not available in PATH.")

        self._run(["docker", "--version"])

    def image_exists(self):
        result = self._run(["docker", "image", "inspect", self.image], check=False)
        return result.returncode == 0

    def pull_image(self):
        print(f"Pulling Docker image: {self.image}")
        subprocess.run(["docker", "pull", self.image], check=True)

    def check_models(self):
        cmd = ["docker", "run", "--rm", self.image, "sh", "-c",
               ("test -f '/trained_models_cfmid4.0/[M+H]+/param_output.log' && "
                "test -f '/trained_models_cfmid4.0/[M+H]+/param_config.txt' && "
                "test -f '/trained_models_cfmid4.0/[M-H]-/param_output.log' && "
                "test -f '/trained_models_cfmid4.0/[M-H]-/param_config.txt'")]

        result = self._run(cmd, check=False)
        return result.returncode == 0

    def check_or_prepare_image(self):
        self.check_docker()

        if not self.image_exists():
            self.pull_image()

        if not self.check_models():
            raise RuntimeError("CFM-ID image exists, but trained_models_cfmid4.0 model files were not found inside the container.")

    def _format_targets(self, smiles_list):
        lines = []

        for i, smi in enumerate(smiles_list, start=1):
            smi = str(smi).strip()
            if smi:
                lines.append(f"mol{i} {smi}\n")

        return lines

    def predict(self, smiles_list, pols=("pos", "neg")):
        self.check_or_prepare_image()

        if isinstance(pols, str):
            pols = (pols,)

        os.makedirs(self.output_dir, exist_ok=True)
        targets = self._format_targets(smiles_list)

        targets_incomplete = []
        for t in targets:
            logtxtfilename = t.split(" ")[0]+".log"
            if not all([os.path.exists(os.path.join(self.output_dir, pol, logtxtfilename)) for pol in pols]):
                targets_incomplete.append(t)
        targets = targets_incomplete.copy()

        if len(targets) == 0:
            raise ValueError("No valid SMILES strings were provided.")

        # Create temporary input file in the user's temp directory
        fd, tmp_fn = tempfile.mkstemp(prefix="cfmid_", suffix=".txt")
        os.close(fd)

        try:
            with open(tmp_fn, "w", encoding="utf-8") as f:
                f.writelines(targets)

            # Docker can only access files in mounted directories.
            # Here, we mount the temp directory containing tmp_fn.
            mount_dir = os.path.dirname(tmp_fn)
            input_name = os.path.basename(tmp_fn)

            # Output path inside the mounted temp directory
            docker_output_root = "/cfmid/public/cfmid_output"

            for pol in pols:
                if pol == "neg":
                    model = "/trained_models_cfmid4.0/[M-H]-/"
                elif pol == "pos":
                    model = "/trained_models_cfmid4.0/[M+H]+/"
                else:
                    raise ValueError("pols must contain only 'pos' and/or 'neg'.")

                param_output = model + "param_output.log"
                param_config = model + "param_config.txt"

                docker_output_path = f"{docker_output_root}/{pol}/"

                docker_cmd = (
                    f"mkdir -p {docker_output_path}; "
                    f"cd /cfmid/public; "
                    f"cfm-predict "
                    f"{input_name} "
                    f"{self.prob_thresh} "
                    f"'{param_output}' "
                    f"'{param_config}' "
                    f"0 "
                    f"{docker_output_path}"
                )

                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{mount_dir}:/cfmid/public",
                    "-i", self.image,
                    "sh", "-c", docker_cmd
                ]

                print(f"Running CFM-ID {pol} prediction...")
                result = self._run(cmd, check=False)

                if result.stdout:
                    print(result.stdout)

                if result.stderr:
                    print(result.stderr)

                if result.returncode != 0:
                    raise RuntimeError(f"CFM-ID prediction failed for polarity: {pol}")

            # Copy output from temp-mounted directory to requested output_dir
            temp_output_root = os.path.join(mount_dir, "cfmid_output")

            for pol in pols:
                src_dir = os.path.join(temp_output_root, pol)
                dst_dir = os.path.join(self.output_dir, pol)

                os.makedirs(dst_dir, exist_ok=True)

                if os.path.isdir(src_dir):
                    for fn in os.listdir(src_dir):
                        src = os.path.join(src_dir, fn)
                        dst = os.path.join(dst_dir, fn)

                        if os.path.isfile(src):
                            shutil.copy2(src, dst)

        finally:
            if os.path.exists(tmp_fn):
                os.remove(tmp_fn)

            temp_output_root = os.path.join(os.path.dirname(tmp_fn), "cfmid_output")
            if os.path.isdir(temp_output_root):
                shutil.rmtree(temp_output_root)

        return targets


import re
import numpy as np


def parse_cfmid_log(log_file):
    metadata_list = []
    spectra_list = []

    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    smiles, formula, adduct, precursor_mass = None, None, None, None
    energy_map = {"energy0": "low", "energy1": "medium", "energy2": "high"}

    # Common metadata
    for line in lines:
        if line.startswith("#SMILES="):
            smiles = line.split("=", 1)[1]

        elif line.startswith("#Formula="):
            formula = line.split("=", 1)[1]

        elif line.startswith("#In-silico ESI-MS/MS"):
            match = re.search(r"\[(M[^\]]+)\][+-]", line)
            if match:
                adduct = "[" + match.group(1) + "]" + line[-1]

        elif line.startswith("#PMass="):
            precursor_mass = float(line.split("=", 1)[1])

    current_energy = None
    current_peaks = []

    def _save_current_spectrum():
        if current_energy is None:
            return

        spectrum = np.array(current_peaks, dtype=float)
        if spectrum.size == 0:
            spectrum = np.empty((0, 2), dtype=float)

        metadata = {"SMILES": smiles, "Formula": formula, "adduct": adduct, "precursor_mass": precursor_mass, "energy": current_energy, "energy_label": energy_map[current_energy],}
        metadata_list.append(metadata)
        spectra_list.append(spectrum)

    for line in lines:
        if line in energy_map:
            _save_current_spectrum()
            current_energy = line
            current_peaks = []

        elif current_energy is not None and not line.startswith("#"):
            parts = line.split()

            if len(parts) == 2:
                mz = float(parts[0])
                intensity = float(parts[1])
                current_peaks.append([mz, intensity])

    _save_current_spectrum()

    return metadata_list, spectra_list


def PredictSpectra_table(inputfile, smiles_col:str="product_smiles", output_dir:str="PredictedSpectra"):
    df = pd.read_excel(inputfile)
    if not (smiles_col in df.columns):
        raise ValueError(smiles_col + " column should exist in the dataframe given.")

    # input_smiles = list(set(df[smiles_col]))
    # predictor = PredictSpectra(output_dir=output_dir)
    # targets = predictor.predict(input_smiles, pols=["pos", "neg"])
    # filenames = [t.split(" ")[0] for t in targets]

    for pol in ["pos", "neg"]:
        output_poldir = os.path.join(output_dir, pol)
        metadata_list, spectrum_list, fignames = [], [], []
        # for f in filenames:
        logfiles = [x for x in os.listdir(output_poldir) if x.endswith(".log")]
        for f in logfiles:
            log_file = os.path.join(output_poldir, f)
            metadatas, spectra = parse_cfmid_log(log_file=log_file)
            metadata_list.extend(metadatas)
            spectrum_list.extend(spectra)
            fignames.extend([os.path.splitext(os.path.basename(log_file))[0]]*len(metadatas))

        for metadata, spectrum, f in zip(metadata_list, spectrum_list, fignames):
            fig = plt.figure(num=1, figsize=(15, 10))
            if len(np.shape(spectrum)) == 1:
                spectrum = np.reshape(spectrum, [-1, 2])
            spec_norm = normalizeSpectrum(spectrum)
            _, stemlines1, _ = plt.stem(spec_norm[:, 0], spec_norm[:, 1], linefmt="#0073CF", markerfmt='', basefmt=" ")
            plt.setp(stemlines1, linewidth=5)
            plt.axhline(0, color='black', linewidth=1)
            plt.xlabel("m/z", fontdict={"fontsize": 25, "fontweight": "bold"})
            plt.ylabel("Relative Intensity", fontdict={"fontsize": 25, "fontweight": "bold"})
            # plt.xlim([0, float(np.ceil(metadata["precursor_mass"]/10)*10)])
            plt.xlim([40, 120])
            plt.ylim([0, 1.1])
            plt.title(" : ".join([metadata["SMILES"], metadata["Formula"], metadata["adduct"], metadata["energy"]]), fontdict={"fontsize": 20, "fontweight": "bold"})
            plt.tight_layout()

            for figformat in ["png", "pdf"]:
                figfile = os.path.join(output_poldir, "_".join(["Spectrum", f, metadata["energy"]]) + "." + figformat)
                os.makedirs(os.path.dirname(figfile), exist_ok=True)
                plt.savefig(figfile, dpi=600)
                print("figure saved at : " + figfile)
            plt.close(fig)


# Convenience alias
predict_transformation_products = PredictTransformation

__all__ = ["PredictTransformation",
           "PredictTransformation_table",
           "PredictSpectra",
           "PredictSpectra_table"]
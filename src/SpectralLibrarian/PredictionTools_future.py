# src/SpectralLibrarian/PredictionTools.py
"""
PredictionTools – Transformation product prediction (exact + template modes)

Now supports your ORDerly extraction layout:
- ord_parquet_path can be a directory (e.g. "ord_parquet/extracted_ords")
- Auto-detects the reaction SMILES column
- Processes parquet files one-by-one for low memory usage
"""
from collections import defaultdict
from typing import *

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

import pandas as pd
import pickle as pkl
import warnings, time, json

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

import os

# TODO : for ORD, do the following first
# from huggingface_hub import snapshot_download
#
# userdir = os.path.expanduser("~")
# ord_data_hfdir = os.path.join(userdir, "ord-data-hf")
#
# # TODO : Download data from Hugging Face
# snapshot_download(
#     repo_id="open-reaction-database/ord-data",
#     repo_type="dataset",
#     local_dir=ord_data_hfdir,
#     allow_patterns="data/**/*.pb.gz",
# )
# # TODO : To Generate parquet using ORDerly, go to the 'userdir' defined above, open the command window, and type the following command
# # python -m orderly.extract --data_path="ord-data-hf/data" --output_path="ord_parquet"
#
# # TODO : Now, come back to the pycharm and run the following
# import pandas as pd
# from pathlib import Path
#
# # df.columns does not have "reaction_smiles"
#
# parquet_dir = Path(os.path.join(userdir, "ord_parquet", "extracted_ords"))
# parquet_files = list(parquet_dir.glob("*.parquet"))
#
# df_pqs_list = []
# total_rows = 0
# for p in parquet_files:
#     df = pd.read_parquet(p)
#     df_pqs_list.append(df)
#     total_rows += len(df)
#     print(p, df.shape, df.columns.tolist()[:10])
#
# df_pqs = pd.concat(df_pqs_list, ignore_index=True)
#
#
# for p in Path(os.path.join(userdir, "ord_parquet", "extracted_ords")).rglob("*.parquet"):
#     df = pd.read_parquet(p)
#     print(p, df.columns.tolist()[:20])
# TODO : However, the following sucks..
# import json
# from pathlib import Path
# import pandas as pd
#
#
# from google.protobuf import json_format
# from ord_schema.proto import reaction_pb2, dataset_pb2  # or dataset_pb2 if you have full datasets
# from ord_schema.message_helpers import messages_to_dataframe, load_message, write_message
# from google.protobuf.json_format import MessageToJson
#
# pbgzdir = os.path.join(ord_data_hfdir, "data/0a")
# input_fname = "ord_dataset-0a66204fc43e49c2922e6f9107e6b62f.pb.gz"
# dataset = load_message(os.path.join(pbgzdir, input_fname), dataset_pb2.Dataset)
#
# # take one reaction message from the dataset for example
# rxn = dataset.reactions[0]
# rxn_json = json.loads(
#     MessageToJson(
#         message=rxn,
#         including_default_value_fields=False,
#         preserving_proto_field_name=True,
#         indent=2,
#         sort_keys=False,
#         use_integers_for_enums=False,
#         descriptor_pool=None,
#         float_precision=None,
#         ensure_ascii=True,
#     )
# )
#
# jsondir = os.path.join(userdir, "jsons")
# os.makedirs(jsondir, exist_ok=True)
#
# with open(os.path.join(jsondir, "jsontest.json"), "w") as f:
#     json.dump(rxn_json, f)
#
#
# # Option A: List of individual JSON files (your case with thousands of entries)
# jsonpath = os.path.join(jsondir, "jsontest.json")
# # json_files = list(Path("path/to/jsons").glob("*.json"))  # or however your files are organized
# json_files = [jsonpath]
#
# reactions = []
# for fpath in json_files:
#     with open(fpath, "r", encoding="utf-8") as f:
#         json_data = json.load(f)
#
#     reaction = reaction_pb2.Reaction()
#     json_format.ParseDict(json_data, reaction)  # or Parse(json_str, reaction) if you have a string
#     reactions.append(reaction)


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

            # rxn_col = self._get_reaction_smiles_col(df)
            # for rxn_smiles in df[rxn_col].dropna():
            #     try:
            #         if ">>" not in str(rxn_smiles):
            #             continue
            #         reactants_str, products_str = str(rxn_smiles).split(">>", 1)
            #         reactants = [r.strip() for r in reactants_str.split(".") if r.strip()]
            #         products = [p.strip() for p in products_str.split(".") if p.strip()]
            #
            #         for r in reactants:
            #             canon_r = self._canonical_smiles(r)
            #             if canon_r:
            #                 for p in products:
            #                     canon_p = self._canonical_smiles(p)
            #                     if canon_p:
            #                         lookup[canon_r].add(canon_p)
            #     except Exception:
            #         continue

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

                # rxn_col = self._get_reaction_smiles_col(df)
                # for rxn_smiles in df[rxn_col].dropna():
                #     try:
                #         rxn = ChemicalReaction.from_smiles(str(rxn_smiles))
                #         templates = rxn.generate_reaction_template(radius=radius)
                #         for t in templates:
                #             if t.smarts:
                #                 template_count[t.smarts] += 1
                #     except Exception:
                #         continue

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

#
# from dataclasses import dataclass, asdict
# from rdkit.Chem import Descriptors, rdMolDescriptors
#
# @dataclass
# class TransformationProduct:
#     parent_smiles: str
#     product_smiles: str
#     source: str
#     reaction_name: Optional[str] = None
#     generation: Optional[int] = None
#     score: Optional[float] = None
#     formula: Optional[str] = None
#     exact_mass: Optional[float] = None
#     inchikey: Optional[str] = None
#
#
# def canonicalize_smiles(smiles: str) -> str:
#     if not smiles:
#         return ""
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is None:
#             return smiles
#         return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
#     except Exception:
#         return smiles
#
# def annotate_product(parent_smiles, product_smiles, source, reaction_name=None, generation=None, score=None):
#     mol = Chem.MolFromSmiles(product_smiles)
#     if mol is None:
#         return None
#
#     return TransformationProduct(
#         parent_smiles=parent_smiles,
#         product_smiles=Chem.MolToSmiles(mol, canonical=True),
#         source=source,
#         reaction_name=reaction_name,
#         generation=generation,
#         score=score,
#         formula=rdMolDescriptors.CalcMolFormula(mol),
#         exact_mass=Descriptors.ExactMolWt(mol),
#         inchikey=Chem.MolToInchiKey(mol),
#     )
#
# import requests
# from enviPath_python import enviPath
# from enviPath_python.objects import Pathway
#
# class EnviPathClient:
#     def __init__(self, username=None, password=None):
#         self.ep = enviPath("https://envipath.org")
#         if username and password:
#             self.ep.login(username, password)
#
#
#     def predict_products(self, smiles: str) -> List[TransformationProduct]:
#         smiles = canonicalize_smiles(smiles)
#
#         # 실제 prediction 호출 함수는 enviPath-python 문서/API 예제에 맞춰 조정 필요
#         # search는 database hit 검색이고, prediction은 별도 pathway prediction API일 수 있음.
#         result = self.ep.search(smiles, [])
#
#         products = []
#         # result parsing은 반환 구조 확인 후 맞춤 구현
#         return products
#
#     def predict_pathway(self, smiles: str) -> List[Pathway]:
#         smiles = canonicalize_smiles(smiles)
#         me = self.ep.who_am_i()
#         package = me.get_default_package()
#
#         pw = Pathway.create(package, smiles=smiles)
#         return pw
#
#
# import json
# import time
# import socketio
#
#
# class CTSClient_new2:
#     def __init__(self, base_url="https://qed.epa.gov", socketio_path="cts/ws"):
#         self.base_url = base_url
#         self.socketio_path = socketio_path
#
#     def predict_products(self, smiles, gen_limit=1, prop="cyp450", ph="7.4", timeout=120):
#         sio = socketio.Client(logger=False, engineio_logger=False)
#
#         products = []
#         done = {"flag": False}
#
#         @sio.event
#         def connect():
#             payload = {
#                 "chemical": smiles,
#                 "ph": str(ph),
#                 "pchem_request": {},
#                 "service": "getTransProducts",
#                 "workflow": "gentrans",
#                 "chem_info": "",
#                 "run_type": "single",
#                 "calc": "biotrans",
#                 "gen_limit": gen_limit,
#                 "metabolizer_post": {
#                     "chemical": smiles,
#                     "gen_limit": gen_limit,
#                     "prop": prop,
#                 },
#             }
#             sio.emit("get_data", json.dumps(payload))
#
#         @sio.on("message")
#         def on_message(msg):
#             if isinstance(msg, str):
#                 try:
#                     msg = json.loads(msg)
#                 except json.JSONDecodeError:
#                     return
#
#             if not isinstance(msg, dict):
#                 return
#
#             if msg.get("calc") != "biotrans":
#                 return
#
#             tree = msg.get("data", {})
#             children = tree.get("children", [])
#
#             for child in children:
#                 child_data = child.get("data", {})
#                 product_smiles = child_data.get("smiles")
#
#                 if product_smiles:
#                     products.append({
#                         "input_smiles": smiles,
#                         "product_smiles": product_smiles,
#                         "route": child_data.get("routes"),
#                         "accumulation": child_data.get("accumulation"),
#                         "production": child_data.get("production"),
#                         "likelihood": child_data.get("likelihood"),
#                         "generation": child.get("id"),
#                         "source": "CTS/BioTransformer",
#                         "prop": prop,
#                         "generation_limit": gen_limit,
#                     })
#
#             done["flag"] = True
#
#         sio.connect(
#             self.base_url,
#             socketio_path=self.socketio_path,
#             transports=["websocket", "polling"],
#         )
#
#         start = time.time()
#         while time.time() - start < timeout and not done["flag"]:
#             print(time.time() - start)
#             sio.sleep(0.2)
#
#         sio.disconnect()
#
#         unique = {}
#         for p in products:
#             unique[p["product_smiles"]] = p
#
#         return list(unique.values())
#
#
# cts = CTSClient_new2()
#
# products = cts.predict_products(
#     "CC(=O)OC1=CC=CC=C1C(O)=O",
#     gen_limit=1,
#     prop="cyp450",
# )
#
# for p in products:
#     print(p["product_smiles"], "|", p["route"])


# Convenience alias
predict_transformation_products = PredictTransformation

__all__ = ["PredictTransformation", "predict_transformation_products"]
from .RawDataTools import readmzML
from .SpectralTools import str2array, spec2str, clean_spectrum_by_abundance, normalizeSpectrum, mergeSpectra
from .SimilarityTools import score_similarity
from .PrecursorTools import get_precursor, sum_abundances_from_spectrum
from .utilities import parallelize, ppm_error

from matchms.exporting import save_as_msp
from matchms import Spectrum
from typing import *

import numpy as np, pandas as pd, pickle as pkl
import itertools, os

essential_metafields = ["Molecular Name", "Molecular Formula", "RT left (min)", "RT right (min)", "Precursor Adduct"]


def construct_library(meta_excelfile, output_excelfile, output_mspdir, specdir:str=None, relabdco:float=0.05):
    if not (meta_excelfile.endswith(".xlsx") and output_excelfile.endswith(".xlsx") and output_mspdir.endswith(".msp")):
        raise FileNotFoundError("Check the extendoer of the all three files in the argument")

    df_prec = pd.read_excel(meta_excelfile, sheet_name="Transition")
    df_mzfile = pd.read_excel(meta_excelfile, sheet_name="mzML")

    libraries = []
    df_libs = pd.DataFrame()
    for _, mzfile in df_mzfile.iterrows():
        m = mzfile["Molecular Name"]
        pol = mzfile["Polarity"]
        precursors = df_prec.loc[(df_prec["Molecular Name"] == m) & (df_prec["Polarity"] == pol)]
        if len(precursors) > 0:
            mzmldir = mzfile["FilePath"]
            full_width = mzfile["Isolation Window(Full)"]
            ce = float(mzfile["Collision Energy"])
            ms1intco = float(mzfile["Precursor Intensity Cutoff"])
            ppms = {"ms1": mzfile["ms1ppm"], "ms2": mzfile["ms2ppm"]}
            abdcos = {"ms1": mzfile["ms1abdco"], "ms2": mzfile["ms2abdco"]}
            library, df_lib, args = build_ms2library(mzmldir, ppms, abdcos=abdcos, full_width=full_width, ce=ce, precursors=precursors, ms1intco=ms1intco, relabdco=relabdco, specdir=specdir)
            libraries.extend(library)
            df_libs = pd.concat([df_libs, df_lib], axis=0)
    df_libs.reset_index(drop=True, inplace=True)
    df_libs.sort_values(by=essential_metafields)
    df_libs.to_excel(output_excelfile, index=False)

    os.makedirs(os.path.dirname(output_mspdir), exist_ok=True)
    save_as_msp(libraries, output_mspdir)


def build_ms2library(mzmldir, ppms:Dict[str, float], abdcos:Dict[str, float], full_width:float, ce:float, precursors:pd.DataFrame, ms1intco:float=10000.0, relabdco:float=0.05, mzs_to_exclude:List[float]=None, num_cores:int=os.cpu_count()-2, specdir:str=None):
    ms1ppm, ms2ppm = ppms["ms1"], ppms["ms2"]
    ms1abdco, ms2abdco = abdcos["ms1"], abdcos["ms2"]
    procob = readmzML(mzmldir=mzmldir, ppms=[ms1ppm, ms2ppm], abdcos=[ms1abdco, ms2abdco], half_width=full_width/2, ce=ce)

    if not set(essential_metafields).issubset(precursors.keys()):
        raise ValueError(str(essential_metafields) + " are all required filed in the argument 'precursors' which is a pandas DataFrame")

    df_precursors = precursors.copy()
    for ess in essential_metafields:
        df_precursors = df_precursors.loc[pd.notnull(df_precursors[ess])]
    df_precursors.reset_index(drop=True, inplace=True)

    filename = os.path.splitext(os.path.basename(mzmldir))[0]
    idxs2process, molnames, adducts, precmzs, argslist = [], [], [], [], []
    for idx, df_p in df_precursors.iterrows():
        molname = df_p["Molecular Name"]
        molfmla = df_p["Molecular Formula"]
        rtmin, rtmax = df_p["RT left (min)"], df_p["RT right (min)"]
        adduct = df_p["Precursor Adduct"]
        _, precfmla, precursor_mz = get_precursor(molfmla=molfmla, adduct=adduct)
        if precfmla is None:
            print(adduct + " adduct of " + molname + " (" + molfmla + ") is impossible... removed")
            continue
        molnames.append(molname)
        adducts.append(adduct)
        idxs2process.append(idx)
        precmzs.append(precursor_mz)
        prec_ints, _, ms2spectra, ms2rts = procob.gatherMS2spec(rtRange=[rtmin, rtmax], precursor_mz=precursor_mz, ce=ce, mzs_to_exclude=mzs_to_exclude)
        argslist.append((prec_ints, ms2spectra, ms2ppm, ms1intco, ms2abdco, relabdco))

        # save spectra used to extract consensus spectra
        if not (specdir is None):
            specpklfile = os.path.join(specdir, molname, adduct, "CE-"+str(ce), "RawSpectrum",  filename+ ".pkl")
            if not os.path.exists(specpklfile):
                os.makedirs(os.path.dirname(specpklfile), exist_ok=True)
                with open(specpklfile, "wb") as f:
                    pkl.dump(ms2spectra, f)

    # Extract consensus MS2 spectra
    results = parallelize(workfunc=getConsensusMS2, num_cores=num_cores, argslist=argslist)
    # results = [] # for debugging
    # for i, args in enumerate(argslist):
    #     try:
    #         result = getConsensusMS2(*args)
    #         results.append(result)
    #     except Exception as e:
    #         print("Failed index:", i)
    #         print("Args:", args)
    #         raise

    # save extracted spectra (Consensus MS2 spectra)
    ms2specstrs = []
    for i, r in enumerate(results):
        ms2specstrs.append(spec2str(spectrum=r[0]))
        if not (specdir is None):
            specpklfile = os.path.join(specdir, molnames[i], adducts[i], "CE-"+str(ce), "ConsensusSpectrum", filename + ".pkl")
            if not os.path.exists(specpklfile):
                os.makedirs(os.path.dirname(specpklfile), exist_ok=True)
                with open(specpklfile, "wb") as f:
                    pkl.dump([r[0]], f)

    df_ms2spec = pd.DataFrame.from_dict({"Precursor m/z": precmzs, "Collision energy": [ce]*len(ms2specstrs), "MS2 spectrum": ms2specstrs})
    df_ms2spec.index = idxs2process

    df_final = pd.concat([df_precursors[essential_metafields], df_ms2spec], axis=1)
    df_lib = df_final.loc[pd.notnull(df_final["MS2 spectrum"])]
    library = []
    for _, df_s in df_lib.iterrows():
        mzs, ints = str2array(specstr=df_s["MS2 spectrum"])
        if not (mzs is None):
            metadata = {"Name": df_s["Molecular Name"], "Formula": df_s["Molecular Formula"], "Adduct": df_s["Precursor Adduct"], "Precursor m/z": df_s["Precursor m/z"], "Collision energy": df_s["Collision energy"]}
            library.append(Spectrum(mz=mzs, intensities=ints, metadata=metadata))

    return library, df_lib, argslist


def getConsensusMS2(prec_ints: List[float], ms2spectra: List[np.ndarray], ms2ppm: float, ms1intco: float=10000.0, ms2abdco:float=100.0, relabdco:float=0.05):
    # Remove scan with the precursor intensity lower than 'ms1intco'
    pico = np.max([ms1intco, np.quantile(prec_ints, 0.8)]) # top 20 percentage of precursor
    idxuse = np.where(np.array(prec_ints) > pico)[0]
    N = len(idxuse)
    if N >= 3:
        # precursor peak / isotopic peak / obvious background peak 제거
        # Remove low intensity peaks in each MS2 spectrum and normalized the remaining peaks
        ms2spectra_norm = []
        for i in idxuse:
            ms2spectrum_clean = clean_spectrum_by_abundance(spectrum=ms2spectra[i], abdco=ms2abdco) # apply intensity threshold
            if len(ms2spectrum_clean) > 0:
                ms2spectra_norm.append(normalizeSpectrum(spectrum=ms2spectrum_clean, how="Max"))

        # Vectorize the left peaks by merging through ppm-grid (sum or weighted) or binning (sum), to get the mz grid for similarity measure
        mergedSpectrum, _ = mergeSpectra(ms2spectra_norm, ppm=ms2ppm, how="ppm")
        vectorized_mz = mergedSpectrum[:,0]

        # Gather spectra for similarity measure
        ms2spectra_sim_norm = []
        for spectrum in ms2spectra_norm:
            spectrum_abdsum = sum_abundances_from_spectrum(spectrum=spectrum, mz_centers=vectorized_mz, ppm=ms2ppm)
            ms2spectra_sim_norm.append(normalizeSpectrum(spectrum=spectrum_abdsum, how="Max"))

        # Calculate similarity matrix using cosine similarity
        combos = list(itertools.combinations(list(range(0, N)), 2))
        pairsim_matrix = np.zeros((N, N))
        for i, j in combos:
            pairsim_matrix[i,j] = score_similarity(qspec=ms2spectra_sim_norm[i], lspec=ms2spectra_sim_norm[j], method="CosineGreedy")["CosineGreedy"] # or entropy_similarity
            pairsim_matrix[j,i] = score_similarity(qspec=ms2spectra_sim_norm[j], lspec=ms2spectra_sim_norm[i], method="CosineGreedy")["CosineGreedy"] # or entropy_similarity
        pairsim_mean = np.sum(pairsim_matrix, axis=1)/(N - 1)

        # Remove the ms2spectra with mean similarity lower than the median
        pairsim_idx = list(np.where(np.array(pairsim_mean) >= np.median(pairsim_mean))[0])
        ms2spectra_to_be_aligned = [clean_spectrum_by_abundance(spectrum=ms2spectra_sim_norm[i], abdco=0) for i in pairsim_idx] # remove zero-intensity to avoid division-by-zero

        # Merge the remaining ms2spectra to generate the consensus MS2 spectrum
        spectrum, clusters = mergeSpectra(spectra=ms2spectra_to_be_aligned, ppm=ms2ppm, how="ppm")
        frag_mzs = spectrum[:,0]
        weighted_average_intensities = []

        ms2spectra_consensus_abds_matrix = []  # column (spectra) and row (fragment m/z)
        for spectrum in ms2spectra_to_be_aligned:
            spectrum_abdsum = sum_abundances_from_spectrum(spectrum=spectrum, mz_centers=frag_mzs, ppm=ms2ppm)
            spectrum_abdsum_norm = normalizeSpectrum(spectrum=spectrum_abdsum, how="Max")
            ms2spectra_consensus_abds_matrix.append(spectrum_abdsum_norm[:,1])
        ms2spectra_consensus_abds_matrix = np.array(ms2spectra_consensus_abds_matrix)

        # Build the consensus spectrum by pairwise similarity-weighted averaging intensities
        # Calculate statistics (support ratio, weighted median intensity, mz dispersion) for each fragment ion
        # 필요하면 high-intensity subset consensus도 함께 저장
        pairsim_matrix_selected = pairsim_matrix[:, pairsim_idx][pairsim_idx, :]
        weights = np.sum(pairsim_matrix_selected, axis=1)/(len(pairsim_idx) - 1)

        support_ratios = np.mean(ms2spectra_consensus_abds_matrix != 0, axis=0)
        mz_dispersions = []
        mz_maxppms = []
        for j, fmz in enumerate(frag_mzs):
            nonzero_idx = np.where(ms2spectra_consensus_abds_matrix[:,j] > 0)[0]
            nonzero_weights = weights[nonzero_idx]
            nonzero_ints = ms2spectra_consensus_abds_matrix[nonzero_idx,j]
            try:
                weighted_average_intensities.append(np.average(a=nonzero_ints, weights=nonzero_weights))
            except ZeroDivisionError:
                weighted_average_intensities.append(np.mean(nonzero_ints))

            cluster_mzs = [p[0] for p in clusters[j]]
            mz_dispersions.append(np.std(cluster_mzs))
            mz_maxppms.append(np.max(ppm_error(np.array(cluster_mzs), fmz)))

        consensusMS2spectrum = normalizeSpectrum(spectrum=np.transpose([frag_mzs, np.array(weighted_average_intensities)]), how="Max")
        consensusMS2spectrum = clean_spectrum_by_abundance(spectrum=consensusMS2spectrum, abdco=relabdco)
        return consensusMS2spectrum, pico, list(support_ratios), mz_dispersions, mz_maxppms
    else:
        return None, None, None, None, None

__all__ = ["construct_library", "build_ms2library", "getConsensusMS2"]
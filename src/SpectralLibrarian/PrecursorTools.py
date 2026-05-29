from .utilities import normalizeVec, parallelize
from .SpectralTools import sum_abundances_from_spectrum

from typing import *
import numpy as np, pandas as pd, molmass as mm
import re, os


# Adduct type defined in the Fiehn Lab : https://fiehnlab.ucdavis.edu/staff/kind/metabolomics/ms-adduct-calculator/
abbdict = {'MeOH': 'CH3OH', 'ACN': 'CH3CN', 'IsoProp': 'C3H8O', 'DMSO': '(CH3)2SO', 'FA': 'HCOOH', 'Hac': 'CH3COOH', 'TFA': 'CF3COOH'}
# mass_e = mm.Formula('+').monoisotopic_mass
mass_e = 0.000548579909 # electron mass

# test_adducts = ['[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+MeOH+H]+', '[M+K]+', '[M+ACN+H]+', '[M+2Na-H]+', '[M+IsoProp+H]+', '[M+ACN+Na]+', '[M+2K-H]+', '[M+DMSO+H]+', '[M+2ACN+H]+', '[M+IsoProp+Na+H]+', '[M+H-H2O]+',
#                 '[2M+H]+', '[2M+NH4]+', '[2M+Na]+', '[2M+K]+', '[2M+ACN+H]+', '[2M+ACN+Na]+',
#                 '[M+2H]2+', '[M+H+NH4]2+', '[M+H+Na]2+', '[M+H+K]2+', '[M+ACN+2H]2+', '[M+2Na]2+', '[M+2ACN+2H]2+', '[M+3ACN+2H]2+',
#                 '[M+3H]3+', '[M+2H+Na]3+', '[M+H+2Na]3+', '[M+3Na]3+',
#                 '[M-H-H2O]-', '[M-H]-', '[M+Na-2H]-', '[M+Cl]-', '[M+K-2H]-', '[M+FA-H]-', '[M+Hac-H]-', '[M+Br]-', '[M+TFA-H]-',
#                 '[2M-H]-', '[2M+FA-H]-', '[2M+Hac-H]-', '[3M-H]-', '[M-CO2-H]-', '[M+HCO3-]-', '[M-2H]2-']
#
# dms = [parse_adduct(adduct=a)[1] for a in test_adducts]
# df_new = pd.DataFrame.from_records({"Adduct": test_adducts, "delta_mass_new": dms})
#
# adduct_table = pd.DataFrame([
#     ('[M+3H]3+', 1.007276), ('[M+2H+Na]3+', 8.334590), ('[M+H+2Na]3+', 15.7661904),
#     ('[M+3Na]3+', 22.989218), ('[M+2H]2+', 1.007276), ('[M+H+NH4]2+', 9.520550),
#     ('[M+H+Na]2+', 11.998247), ('[M+H+K]2+', 19.985217), ('[M+2Na]2+', 22.989218),
#     ('[M+H]+', 1.007276), ('[M+NH4]+', 18.033823), ('[M+Na]+', 22.989218),
#     ('[M+MeOH+H]+', 33.033489), ('[M+K]+', 38.963158), ('[M+ACN+H]+', 42.033823),
#     ('[M+2Na-H]+', 44.971160), ('[M+IsoProp+H]+', 61.06534), ('[M+DMSO+H]+', 79.02122),
#     ('[M-H]-', -1.007276), ('[M-H-H2O]-', -19.01839), ('[M+Cl]-', 34.969402),
#     ('[M+FA-H]-', 44.998201), ('[M+Hac-H]-', 59.013851), ('[M+Br]-', 78.918885),
#     ('[M+TFA-H]-', 112.985586), ('[2M+H]+', 1.007276), ('[2M+Na]+', 22.989218),
#     ('[3M-H]-', -1.007276),
# ], columns=['Adduct', 'delta_mass_old'])
#
# df_compare = pd.merge(adduct_table, df_new, on='Adduct', how='left')


def standardize_precursor_type_safe(series: pd.Series) -> pd.Series:
    # example usage
    # mona.df['PRECURSORTYPE'] = standardize_precursor_type_safe(mona.df['PRECURSORTYPE'])
    # mona.df = mona.df.dropna(subset=['PRECURSORTYPE'])
    s = series.astype(str).str.strip()

    # Garbage → NaN (drop later)
    garbage = {
        'nan', 'carotenoid', 'carotenoids', 'MSMS', 'Protonated',
        'Deprotonated molecule', 'M+1', 'M+', 'M-', 'carotenoid',
        '[M]+*', '[M-H]+', '[M-OH]+', '[M+Ca-H]+', '[M-H]+', '[M-H]2-', '[M+2H]+', '[M+H]2+', '[M-2H]-'
    }
    s = s.replace(garbage, np.nan)

    # Only safe fixes
    safe_fixes = {
        # Missing brackets/charge
        'M+': '[M]+', 'M-': '[M]-',
        'M+H': '[M+H]+', 'M-H': '[M-H]-',
        'M+Na': '[M+Na]+', 'M+K': '[M+K]+',
        'M+2Na': '[M+2Na]+',
        'M+NH4': '[M+NH4]+', 'M+Cl': '[M+Cl]-',
        'M+H-H2O': '[M+H-H2O]+',

        # Double charge notation
        '[M-2H]--': '[M-2H]2-', '[M+2H]++': '[M+2H]2+',
        '[M]++': '[M]2+',

        # Typos
        '[M-H20-H]-': '[M-H2O-H]-',

        # Missing charge
        '[M+H-H2O]': '[M+H-H2O]+',
        '[M-H]': '[M-H]-',
        '[M-H]1-': '[M-H]-',
        '[M+H]': '[M+H]+',
        '(M+H)+': '[M+H]+',
        '[M-H2O+H]': '[M+H-H2O]+',
        'M+Na-2H': '[M+Na-2H]-',

        # Synonyms (mass identical)
        '[M+CH3COO]-': '[M+CH3COOH-H]-',
        '[M+HAc-H]-': '[M+CH3COOH-H]-',
        '[M+Ac-H]-': '[M+CH3COOH-H]-',
        '[M-H+Ac]-': '[M+CH3COOH-H]-',
        '[M+Hac-H]-': '[M+CH3COOH-H]-',
        '[M+FA-H]-': '[M+HCOOH-H]-',
        '[M+HCOO]-': '[M+HCOOH-H]-',
        '[M+ACN+H]+': '[M+CH3CN+H]+',
    }
    s = s.replace(safe_fixes)

    # Final format check
    valid = r'^\[.*\][0-9]*[+-]?$'
    s = s.where(s.str.match(valid, na=True), np.nan)
    return s


def get_precursor(molfmla:str, adduct: str) -> Tuple[Union[int, None], Union[str, None], Union[float, None]]:
    try:
        k, delta_mass, charge_str, mmexp_term = parse_adduct(adduct)
        mmexp_prec = str(k) + "*mm.Formula('" + molfmla + "')" + mmexp_term
        mm_precursor_nocharge = eval(mmexp_prec)
        pfmla = "[" + mm_precursor_nocharge.formula + "]" + charge_str
        precursor = mm.Formula(pfmla)
        return precursor.charge, precursor.formula, precursor.monoisotopic_mass / abs(precursor.charge)
    except Exception as e: # impossible adduct form for the given molecular formula
        return None, None, None


def parse_adduct(adduct: str):
    if not adduct.startswith('[') or ']' not in adduct:
        raise ValueError("Adduct must be in format like '[3M+4ACN-CO2+2H]2+'")

    bracket_end = adduct.find(']')
    bracket = adduct[1:bracket_end].strip() # [3M-CO2+2H]2+ -> 3M-CO2+2H
    charge_str = adduct[bracket_end + 1:].strip() # [3M-CO2+2H]2+ -> 2+

    # Charge
    if not charge_str: # Uncharged adduct cannot be detected, so return None
        raise ValueError("Adduct must end with a charge string")

    # Multiplier k to the M : 3M-CO2+2H -> 3
    k = 1
    if bracket and bracket[0].isdigit():
        i = 0
        while i < len(bracket) and bracket[i].isdigit():
            i += 1
        k = int(bracket[:i])
        bracket = bracket[i:].strip()

    # Now bracket becomes M+4ACN-CO2+2H
    if not bracket.startswith('M'):
        raise ValueError("Adduct must contain 'M'")
    terms = bracket[1:].strip() # terms = 4ACN-CO2+2H

    delta_mass, mmexp_term = parse_terms(terms=terms)
    return k, delta_mass, charge_str, mmexp_term


def parse_terms(terms: str):
    pattern = re.compile(r'([+-]?)(\d*)([A-Za-z][A-Za-z0-9]*)')

    result = []
    for m in pattern.finditer(terms):
        sign, coeff_str, formula = m.groups()

        coeff = int(coeff_str) if coeff_str else 1
        if sign == '-':
            coeff = -coeff
        result.append((coeff, formula))
    result.sort(key=lambda x: x[0], reverse=True) # sort by the coefficient (Descending)

    mmexp_term = ""
    delta_mass = 0.0
    for r in result:
        adfmla = abbdict.get(r[1], r[1]) # This might be unstable, if precursor type is not standardized or abbreviations are not fully annotated
        if r[0] > 0:
            mmexp_term += "+" + str(r[0]) + "*mm.Formula('" + adfmla +"')"
        else:
            mmexp_term += str(r[0]) + "*mm.Formula('" + adfmla + "')"
        delta_mass += r[0] * mm.Formula(adfmla).monoisotopic_mass

    return delta_mass, mmexp_term


def mz_to_mimw(mz: float, adduct: str) -> float:
    try:
        k, delta_mass, charge_str, _ = parse_adduct(adduct)
        z = int(charge_str[:-1]) if charge_str.endswith("+") else -int(charge_str[:-1])
        return (mz*abs(z) - (delta_mass -z*mass_e))/k
    except Exception as e:
        return 0.0


def mimw_to_mz(monoisotopic_mass: float, adduct: str):
    try:
        k, delta_mass, charge_str, _ = parse_adduct(adduct)
        z = -int(charge_str[:-1]) if charge_str.endswith("+") else int(charge_str[:-1])
        return (k*monoisotopic_mass + delta_mass + z*mass_e)/abs(z)
    except Exception as e:
        return 0.0


def get_isodist(precursor_fmla: str):
    mf = mm.Formula(precursor_fmla)
    df = mf.spectrum().dataframe()[["Relative mass", "Fraction"]]  # DataFrame containing the theoretical isotopologue distribution of this adduct
    return list(df["Relative mass"]), list(df["Fraction"]), mf.charge


def eval_isodist(spectrum: np.ndarray, precursor_fmla: str, ppm: float, abdco: float):
    """Measure the cosine similarity between observed and theoreical isotopologue intensities"""

    exactmasses, fracs, precursor_charge = get_isodist(precursor_fmla)

    mzs = [m / abs(precursor_charge) for m in exactmasses] # Isotopologue's m/z
    summed_spectrum = sum_abundances_from_spectrum(spectrum=spectrum, mz_centers=mzs, ppm=ppm, abdco=abdco)

    if np.shape(summed_spectrum)[0] < 2:  # MS1 intensity cutoff to validate isotope distribution consistency:
        isocosim = None
    else:
        observed = normalizeVec(summed_spectrum[:, 1], how="Sum")
        theoretical = normalizeVec(np.array(fracs), how="Sum")
        isocosim = np.dot(observed, theoretical) / (np.linalg.norm(observed) * np.linalg.norm(theoretical))

    return isocosim


def add_precursor_info_to_df(df: pd.DataFrame, formula_col: str, adduct_col: str, num_cores: int=os.cpu_count()-2) -> pd.DataFrame:
    if not ({formula_col, adduct_col}.issubset(df.columns)):
        raise Exception("Both " + formula_col + " and " + adduct_col + " columns should exist in the dataframe given")

    if len(df) > 1000:
        args = list(zip(list(df[formula_col]), list(df[adduct_col])))
        results = parallelize(workfunc=get_precursor, num_cores=num_cores, argslist=args)
    else:
        results = [get_precursor(molfmla=fmla, adduct=adt) for fmla, adt in zip(list(df[formula_col]), list(df[adduct_col]))]
    df_pinfo = pd.DataFrame.from_records(results, columns=["Precursor Charge", "Precursor Formula", "Precursor m/z"])
    df_final = pd.concat([df.reset_index(drop=True), df_pinfo], axis=1)
    return df_final

__all__ = ["standardize_precursor_type_safe", "get_precursor", "parse_adduct", "parse_terms", "mz_to_mimw", "mimw_to_mz", "get_isodist", "eval_isodist", "add_precursor_info_to_df"]
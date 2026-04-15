# src/MSAnalyzer/adduct.py
"""
Adduct ↔ m/z conversion tools
Usage: msa.adduct.mass_to_mz(), msa.adduct.mz_to_mass()
"""

from __future__ import annotations

import re
import pandas as pd
import numpy as np
from typing import Iterable, Union, Tuple
from pyteomics import mass


# Precompiled common adducts – will move to constants module later
_adduct_table = pd.DataFrame([
    ('[M+3H]3+', 1.007276), ('[M+2H+Na]3', 8.334590), ('[M+H+2Na]3+', 15.7661904),
    ('[M+3Na]3+', 22.989218), ('[M+2H]2+', 1.007276), ('[M+H+NH4]2+', 9.520550),
    ('[M+H+Na]2+', 11.998247), ('[M+H+K]2+', 19.985217), ('[M+2Na]2+', 22.989218),
    ('[M+H]+', 1.007276), ('[M+NH4]+', 18.033823), ('[M+Na]+', 22.989218),
    ('[M+CH3OH+H]+', 33.033489), ('[M+K]+', 38.963158), ('[M+ACN+H]+', 42.033823),
    ('[M+2Na-H]+', 44.971160), ('[M+IsoProp+H]+', 61.06534), ('[M+DMSO+H]+', 79.02122),
    ('[M-H]-', -1.007276), ('[M-H2O-H]-', -19.01839), ('[M+Cl]-', 34.969402),
    ('[M+FA-H]-', 44.998201), ('[M+Hac-H]-', 59.013851), ('[M+Br]-', 78.918885),
    ('[M+TFA-H]-', 112.985586), ('[2M+H]+', 1.007276), ('[2M+Na]+', 22.989218),
    ('[3M-H]-', -1.007276),
], columns=['adduct', 'delta_mz'])


def _parse_adduct(adduct: str) -> Tuple[int, float, int]:
    if not adduct.startswith('[') or ']' not in adduct:
        raise ValueError("Adduct must be in format like '[M+H]+'")

    bracket_end = adduct.find(']')
    bracket = adduct[1:bracket_end].strip()
    charge_str = adduct[bracket_end + 1:].strip()

    # Charge
    if not charge_str:
        z = 0
    else:
        m = re.match(r'^(\d*)([+-]?)$', charge_str)
        num = 1 if m.group(1) == '' else int(m.group(1))
        sign = 1 if m.group(2) in ('', '+') else -1
        z = sign * num

    # Multiplier k
    k = 1
    if bracket and bracket[0].isdigit():
        i = 0
        while i < len(bracket) and bracket[i].isdigit():
            i += 1
        k = int(bracket[:i])
        bracket = bracket[i:].strip()

    if not bracket.startswith('M'):
        raise ValueError("Adduct must contain 'M'")

    terms = bracket[1:].strip()
    delta = 0.0
    if terms:
        for sign_str, term in re.findall(r'([+-])([^+-]+)', terms):
            sign = 1 if sign_str == '+' else -1
            try:
                comp = mass.Composition(formula=term.replace('D', 'H[2]').replace('13C', 'C[13]'))
                delta += sign * mass.calculate_mass(composition=comp)
            except:
                m2 = re.match(r'(\d+)(.+)', term)
                if m2:
                    n, elem = m2.groups()
                    comp = mass.Composition(formula=elem.replace('D', 'H[2]').replace('13C', 'C[13]'))
                    delta += sign * mass.calculate_mass(composition=comp) * int(n)

    return k, delta, z


def mass_to_mz(monoisotopic_mass: float, adduct: str) -> float:
    if adduct in _adduct_table['adduct'].values:
        delta = _adduct_table.loc[_adduct_table['adduct'] == adduct, 'delta_mz'].item()
        k, _, z = _parse_adduct(adduct)
        if z == 0:
            return k * monoisotopic_mass + delta
        return k * monoisotopic_mass / abs(z) + delta
    else:
        k, delta_atoms, z = _parse_adduct(adduct)
        if z == 0:
            return k * monoisotopic_mass + delta_atoms
        e = 0.000548579909  # electron mass
        return k * monoisotopic_mass / abs(z) + delta_atoms / abs(z) - (z // abs(z)) * e


def mz_to_mass(mz: float, adduct: str) -> float:
    if adduct in _adduct_table['adduct'].values:
        delta = _adduct_table.loc[_adduct_table['adduct'] == adduct, 'delta_mz'].item()
        k, _, z = _parse_adduct(adduct)
        if z == 0:
            return (mz - delta) / k
        return (mz - delta) * abs(z) / k
    else:
        k, delta_atoms, z = _parse_adduct(adduct)
        if z == 0:
            return (mz - delta_atoms) / k
        e = 0.000548579909
        return (mz * abs(z) - delta_atoms + (z // abs(z) * abs(z) * e)) / k


def batch_adduct_mz(
    monoisotopic_masses: Iterable[float],
    adducts: Union[str, Iterable[str]]
) -> np.ndarray:
    masses = np.asarray(monoisotopic_masses)
    if isinstance(adducts, str):
        adducts = [adducts] * len(masses)
    return np.array([mass_to_mz(m, a) for m, a in zip(masses, adducts)])


def add_adduct_mz_to_df(
    df: pd.DataFrame,
    mass_col: str,
    adduct_col: Union[str, Iterable[str]]
) -> pd.DataFrame:
    df = df.copy()
    masses = df[mass_col]
    if isinstance(adduct_col, str) and adduct_col in df.columns:
        adducts = df[adduct_col]
    else:
        adducts = adduct_col  # single adduct string for all rows
    df['mz'] = batch_adduct_mz(masses, adducts)
    return df
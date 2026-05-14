# src/SpectralLibrarian/IsotopeTools.py
"""
Theoretical isotopic distribution tools
Usage: sl.IsotopeTools.batch_isotopic_distribution(), sl.IsotopeTools.add_isotopic_distribution_to_df(), etc
"""

from __future__ import annotations

from typing import List, Tuple, Iterable, Dict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pyteomics.mass import Composition, isotopologues, calculate_mass
from collections import defaultdict


def _make_labels(iso_comp: Composition) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    elements = defaultdict(dict)
    for key, count in iso_comp.items():
        if '[' not in key:
            continue
        elem, mass_str = key.split('[')
        mass_num = int(mass_str.rstrip(']'))
        elements[elem][mass_num] = count

    for elem in elements:
        data = pyteomics.mass.nist_mass[elem]
        base = max((k for k in data if k != 0), key=lambda k: data[k][1])
        for m_num, cnt in elements[elem].items():
            if cnt > 0:
                labels[elem if m_num == base else f"{m_num}{elem}"] = cnt
    return labels


def compute_isotopic_distribution(
    formula: str,
    overall_threshold: float = 1e-4,
    isotope_threshold: float = 0.0,
    fine_distribution: bool = False,
    mass_bin_ppm: float = None,
    mass_bin_da: float = None,
) -> List[Tuple[float, float, Dict[str, int]]]:
    comp = Composition(formula=formula)
    dist = []
    for iso_comp, abun in isotopologues(
        composition=comp,
        report_abundance=True,
        overall_threshold=overall_threshold,
        isotope_threshold=isotope_threshold,
    ):
        m = calculate_mass(composition=iso_comp)
        labels = _make_labels(iso_comp)
        dist.append((m, abun, labels))
    dist.sort(key=lambda x: x[0])

    if fine_distribution or (mass_bin_ppm is None and mass_bin_da is None):
        return dist

    # Binning
    binned = []
    current = [dist[0]]
    sum_ma = dist[0][0] * dist[0][1]
    sum_a = dist[0][1]

    for m, a, l in dist[1:]:
        center = sum_ma / sum_a
        delta = m - center
        thresh = mass_bin_da if mass_bin_da is not None else center * mass_bin_ppm / 1e6
        if delta <= thresh:
            current.append((m, a, l))
            sum_ma += m * a
            sum_a += a
        else:
            # take peak with highest abundance as representative
            best = max(current, key=lambda x: x[1])
            binned.append((best[0], sum(x[1] for x in current), best[2]))
            current = [(m, a, l)]
            sum_ma = m * a
            sum_a = a

    if current:
        best = max(current, key=lambda x: x[1])
        binned.append((best[0], sum(x[1] for x in current), best[2]))

    return binned


def batch_isotopic_distribution(
    formulas: Iterable[str],
    overall_threshold: float = 1e-4,
    isotope_threshold: float = 0.0,
    fine_distribution: bool = False,
    mass_bin_ppm: float = None,
    mass_bin_da: float = None,
    max_workers: int = None,
) -> np.ndarray:
    formulas = list(formulas)
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        results = list(exe.map(compute_isotopic_distribution, formulas,
                              [overall_threshold]*len(formulas),
                              [isotope_threshold]*len(formulas),
                              [fine_distribution]*len(formulas),
                              [mass_bin_ppm]*len(formulas),
                              [mass_bin_da]*len(formulas)))
    return np.array(results, dtype=object)


def add_isotopic_distribution_to_df(
    df: pd.DataFrame,
    formula_col: str,
    overall_threshold: float = 1e-4,
    isotope_threshold: float = 0.0,
    fine_distribution: bool = False,
    mass_bin_ppm: float = None,
    mass_bin_da: float = None,
    max_workers: int = None,
) -> pd.DataFrame:
    df = df.copy()
    dists = batch_isotopic_distribution(
        df[formula_col],
        overall_threshold=overall_threshold,
        isotope_threshold=isotope_threshold,
        fine_distribution=fine_distribution,
        mass_bin_ppm=mass_bin_ppm,
        mass_bin_da=mass_bin_da,
        max_workers=max_workers,
    )
    masses = []
    abundances = []
    labels = []
    for d in dists:
        if len(d):
            m, a, l = zip(*d)
        else:
            m, a, l = [], [], []
        masses.append(list(m))
        abundances.append(list(a))
        labels.append(list(l))

    df['isotopic_masses'] = masses
    df['isotopic_abundances'] = abundances
    df['isotopologues'] = labels
    return df
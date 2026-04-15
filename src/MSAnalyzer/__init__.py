# src/MSAnalyzer/__init__.py
"""
MSAnalyzer – Advanced Mass Spectrometry & Chemoinformatics Toolkit
===================================================================

Version: 0.3.0 (November 20, 2025)
Author: Triston Groff
License: MIT
"""

from __future__ import annotations

__version__ = "0.3.0"
__author__ = "Triston Groff"
__license__ = "MIT"

# Core
from .LibraryManager import LibraryManager

# Spectral preprocessing
from .SpectralTools import (
    clean_spectrum,
    standardize_spectrum,
    neutral_loss_spectrum,
    spectral_entropy,
    batch_clean,
    batch_standardize,
    batch_neutral_loss,
    batch_spectral_entropy,
    drop_msms_precursor,
    batch_drop_msms_precursor,
    normalized_spectral_entropy,
    batch_normalized_spectral_entropy,
)

# Similarity scoring
from .SimilarityTools import (
    score_similarity,
    batch_score_similarity,
)

# Structural analysis & fingerprints
from .StructureTools import (
    categorize_molecule,
    categorize_molecules,
    categories_to_sparse,
    get_fingerprint_ensemble,
    combine_with_structural_features,
)

# Adduct tools
from .AdductTools import (
    mass_to_mz,
    mz_to_mass,
    batch_adduct_mz,
    add_adduct_mz_to_df,
)

# Isotope tools
from .IsotopeTools import (
    compute_isotopic_distribution,
    batch_isotopic_distribution,
    add_isotopic_distribution_to_df,
)

# Metadata tools
from .MetaTools import (
    search as pubchem_search,
    parallel_search as pubchem_parallel_search,
)

# Utilities
from .dataframe_utils import (
    isnull,
    notnull,
    isnull_or_empty,
    enforce_columns,
    reorder_columns,
    enforce_dtypes,
    clean_empty_lists_and_strings,
)

__all__ = [
    "__version__",
    "__author__",
    "__license__",

    "LibraryManager",

    "clean_spectrum",
    "standardize_spectrum",
    "neutral_loss_spectrum",
    "spectral_entropy",
    "batch_clean",
    "batch_standardize",
    "batch_neutral_loss",
    "batch_spectral_entropy",
    "drop_msms_precursor",
    "batch_drop_msms_precursor",
    "normalized_spectral_entropy",
    "batch_normalized_spectral_entropy",

    "score_similarity",
    "batch_score_similarity",

    "categorize_molecule",
    "categorize_molecules",
    "categories_to_sparse",
    "get_fingerprint_ensemble",
    "combine_with_structural_features",

    "mass_to_mz",
    "mz_to_mass",
    "batch_adduct_mz",
    "add_adduct_mz_to_df",

    "compute_isotopic_distribution",
    "batch_isotopic_distribution",
    "add_isotopic_distribution_to_df",

    "pubchem_search",
    "pubchem_parallel_search",

    "isnull",
    "notnull",
    "isnull_or_empty",
    "enforce_columns",
    "reorder_columns",
    "enforce_dtypes",
    "clean_empty_lists_and_strings",
]
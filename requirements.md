# SpectralLibrarian Dependencies

This document summarizes the runtime dependencies used by `SpectralLibrarian`. The authoritative dependency list is the `[project].dependencies` section of `pyproject.toml`; `requirements.txt` is kept synchronized for users who prefer requirements-file installation.

## Runtime dependencies

| Package | Minimum Version | Purpose | Required? |
|---|---:|---|---|
| numpy | ≥2.2.6, <2.3 | Core arrays and numerical operations | Yes |
| pandas | ≥2.3.3 | DataFrame-based data handling | Yes |
| scipy | ≥1.15.3 | Scientific computing utilities | Yes |
| matchms | ≥0.33.0 | MS/MS spectral processing tools | Yes |
| ms_entropy | ≥1.5.0 | Spectral entropy and similarity scoring | Yes |
| pyteomics | ≥5.0 | mzML and mass spectrometry file handling | Yes |
| psims | ≥1.3.6 | mzML/psi-ms support required by pyteomics workflows | Yes |
| rdkit | ≥2026.3.2 | Chemoinformatics and molecular fingerprints | Yes |
| molmass | ≥2025.4.14 | Molecular formula and mass calculations | Yes |
| pubchempy | ≥1.0.5 | PubChem metadata queries | Yes |
| openpyxl | ≥3.1.5 | Excel file I/O | Yes |
| xmltodict | ≥1.0.4 | XML parsing helpers | Yes |
| matplotlib | ≥3.10.9 | Plotting and visualization | Yes |
| seaborn | ≥0.13.2 | Plotting and visualization helpers | Yes |
| python-socketio[client] | ≥5.16.2 | CTS/websocket client wrapper | Yes |
| joblib | ≥1.5.3 | Parallel processing utilities | Yes |
| tqdm | ≥4.67.3 | Progress bars | Yes |

## Development dependencies

Development dependencies are listed in `requirements-dev.txt` and in the `dev` optional dependency group in `pyproject.toml`.

Examples include:

- black
- ruff
- mypy
- pytest
- pytest-cov
- pre-commit
- sphinx and related documentation tools
- jupyterlab / ipywidgets / notebook for interactive development

## Install

For normal package development, use:

```bash
pip install -e .
```

For development tools as well, use:

```bash
pip install -e ".[dev,jupyter]"
```

Alternatively, using requirements files:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

or for development:

```bash
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

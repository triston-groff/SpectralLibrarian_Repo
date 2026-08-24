# SpectralLibrarian Dependencies

Aligned with `pyproject.toml` on `main`.

| Package | Version constraint | Purpose | Required? |
|---------|-------------------|---------|-----------|
| joblib | ≥1.5.0 | Parallel processing | Yes |
| matchms | ==0.33.1 | Neutral loss & spectral tools | Yes (pinned) |
| ms_entropy | ≥1.3.9 | Spectral entropy metrics | Yes |
| numpy | ≥2.0.0,<2.3 | Core arrays & math | Yes |
| pandas | ≥2.2.3 | DataFrames | Yes |
| psutil | ≥7.0.0 | Memory monitoring | Yes |
| pubchempy | ≥1.0.5 | PubChem metadata queries | Yes |
| pyteomics | ≥4.7.5 | Formula / MS file helpers | Yes |
| rdkit | ≥2024.09.1 | Chemistry & fingerprints | Yes |
| scikit-learn | ≥1.5 | ML compatibility | Yes |
| scipy | ≥1.15.3 | Sparse matrices, assignment, stats | Yes |
| SpectralEntropy | git fork @ pinned commit | Custom similarity scoring | Yes |
| tqdm | ≥4.67.1 | Progress bars | Yes |
| dask | ==2025.11.0 | Parallel bags / scaling in SimilarityTools | Yes |
| pyarrow | ≥14.0.0 | Arrow/parquet support for Dask & pandas | Yes |
| packaging | ≥21.3,<25.1 | Version/packaging utilities | Yes |
| pytz | ≥2021.1,<2026 | Timezone support | Yes |

## Optional extras (`pyproject.toml`)

| Extra | Packages |
|-------|----------|
| `plot` | matplotlib≥3.9, seaborn≥0.13 |
| `jupyter` | jupyterlab≥4.4, ipywidgets≥8.1, notebook≥7.4 |
| `dev` | black, ruff, mypy, pytest, pytest-cov, pre-commit, sphinx, sphinx-rtd-theme, sphinx-autodoc-typehints, sphinx-copybutton |

## Install

```bash
# Editable package (reads pyproject.toml) — preferred
pip install -e .

# Editable + development tooling
pip install -e ".[dev]"

# Full local environment
pip install -e ".[dev,jupyter,plot]"

# Requirements files (approx. mirror; pyproject is source of truth)
pip install -r requirements.txt
pip install -r requirements-dev.txt
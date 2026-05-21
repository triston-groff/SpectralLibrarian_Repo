# SpectralLibrarian Dependencies

| Package            | Minimum Version | Purpose                                      | Required? |
|--------------------|-----------------|----------------------------------------------|-----------|
| joblib             | ≥1.5.0          | Parallel processing                          | Yes       |
| matchms            | ≥0.31.0         | Neutral loss & spectral tools                | Yes       |
| ms_entropy         | ≥1.3.9          | Modified cosine & spectral entropy           | Yes       |
| numpy              | ≥2.3.5          | Core arrays & math                           | Yes       |
| pandas             | ≥2.2.3          | DataFrames                                   | Yes       |
| psutil             | ≥7.0.0          | Memory monitoring                            | Yes       |
| pubchempy          | ≥1.0.5          | PubChem metadata queries                     | Yes       |
| pyteomics          | ≥4.7.5          | mzML and MS file handling                    | Yes       |
| rdkit              | ≥2025.9.1       | Chemistry & fingerprints                     | Yes       |
| scikit-learn       | ≥1.7.2          | ML compatibility                             | Yes       |
| scipy              | ≥1.16.1         | Sparse matrices, stats                       | Yes       |
| SpectralEntropy    | git fork        | Custom similarity scoring algorithms         | Yes       |
| tqdm               | ≥4.67.1         | Progress bars                                | Yes       |

**Optional extras** (installed via `.[plot,jupyter,dev]`):
- `plot`: matplotlib, seaborn
- `jupyter`: jupyterlab, ipywidgets, notebook
- `dev`: black, ruff, mypy, pytest, etc.

## Install
```bash
# Full development environment (recommended)
pip install -e ".[dev,jupyter,plot]"
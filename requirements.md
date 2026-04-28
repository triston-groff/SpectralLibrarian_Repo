# MSAnalyzer Dependencies (November 18, 2025)

| Package         | Version       | Purpose                                         | Required? |
|-----------------|---------------|-------------------------------------------------|-----------|
| numpy           | ≥2.1.3        | Core arrays & math                              | Yes       |
| pandas          | ≥2.2.3        | DataFrames for library viewing & modification   | Yes       |
| rdkit           | 2024.09.1+    | All chemistry (SMILES, fingerprints, etc.)      | Yes       |
| scipy           | ≥1.16         | Sparse matrices, stats                          | Yes       |
| pyteomics       | ≥4.7.5        | mzML and other MS functionalities               | Yes       |
| scikit-learn    | ≥1.5.2        | ML compatibility                                | Yes       |
| ms_entropy      | ≥1.0.2        | Modified cosine & spectral entropy              | Yes       |
| SpectralEntropy | ≥1.0.0        | Various similarity scoring algorithms           | Yes       |
| matchms         | ≥0.31.0       | Neutral loss & other spectral tools             | Yes       |
| pubchempy       | ≥1.0.5        | Query meta data                                 | Yes       |
| joblib          | latest        | Parallel processing                             | Yes       |
| psutil          | latest        | Memory monitoring                               | Yes       |
| matplotlib      | ≥3.9          | Plotting (optional)                             | Plot      |
| seaborn         | ≥0.13         | Beautiful plots                                 | Plot      |
| jupyterlab      | ≥4.4          | Interactive development                         | Dev       |

## Install

```bash
# Production (minimal)
pip install -r requirements.txt

# Full development environment
pip install -e ".[dev,jupyter,plot]"

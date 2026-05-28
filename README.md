# SpectralLibrarian

**State-of-the-art mass spectrometry & chemoinformatics toolkit.**

<p align="left">
  <img src="assets/SpectralLibrarian_LOGO.png" alt="SpectralLibrarian" width="490" />
</p>

A comprehensive Python library for adduct analysis, spectral similarity computation, spectral library management, **transformation product prediction**, and advanced chemoinformatics workflows.

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

### ⚠️ Important: Windows Users Only (One-Time Setup)

**SpectralLibrarian** depends on **SpectralEntropy**, which contains a performance-critical Cython extension.  
If you see “Microsoft Visual C++ 14.0 or greater is required”, install the **Desktop development with C++** workload from the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

macOS and Linux users do **not** need this step.

### For users (once published to PyPI)

```bash
pip install SpectralLibrarian
```

### For developers / editable install

```bash
# Full development environment (recommended)
pip install -e ".[dev,jupyter,plot,ord]"
```

---

## Transformation Product Prediction
**`SpectralLibrarian.PredictionTools.PredictTransformation`** lets you predict possible transformation products (metabolites, environmental degradation products, biodegradation products, etc.) from SMILES strings.

### Features
- **Exact lookup mode** – Fast lookup of *observed* products from the Open Reaction Database (ORD)
- **enviPath mode** – Expert-curated biotransformation rules (environmental / microbial focus)
- Both modes can be combined (`mode="both"`)

### Important Caveats

| Feature              | Requirement                                      | Size / Time                  | Notes |
|----------------------|--------------------------------------------------|------------------------------|-------|
| **Exact ORD lookup** | Download & process ORD data (`orderly`)         | ~10–20 GB download, 30–60+ min first time | One-time step. Use `PredictTransformation.download_ord_data()` |
| **enviPath rules**   | Free account + login at [envipath.org](https://envipath.org) | Very small                   | Optional but recommended for best results |

### Quick Start

```python
from SpectralLibrarian import PredictTransformation

# Quick test with enviPath only (no ORD data needed)
predictor = PredictTransformation(
    ord_parquet_path=None,
    use_exact_lookup=False,
    use_enviPath=True,
    enviPath_username="your_username",      # ← create free account at envipath.org
    enviPath_password="your_password",
)

df = predictor.predict(
    smiles_list=["CCO", "c1ccccc1O", "CC(=O)O", "ClCCCl"],
    mode="both",                    # "exact", "both", or just enviPath
    max_products_per_mol=15
)

print(df)
df.to_csv("transformation_products.csv", index=False)
```

### Download ORD data (for exact lookup)

```python
# One-time command (run once)
parquet_path = PredictTransformation.download_ord_data(output_dir="ord_cleaned")
```

Then use it with full functionality:

```python
predictor = PredictTransformation(
    ord_parquet_path=parquet_path,
    use_exact_lookup=True,
    use_enviPath=True,
    # enviPath credentials are optional
)
```

---

## Optional Extras

```bash
pip install -e ".[ord]"          # adds orderly (required for full ORD exact lookup)
pip install -e ".[dev,jupyter,plot,ord]"   # everything
```

See `requirements.md` for the full dependency list.

---

**Enjoy the new transformation product prediction capabilities!**  
Questions or issues? Open an issue on the repository.


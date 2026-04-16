# MSAnalyzer Package Requirements

This file lists all Python dependencies for the MSAnalyzer package, formatted as a Markdown table for readability. Generated on Sep 08, 2025, for the unified MSPLibraryManager and ms2parser codebase. Install via `pip install -r requirements.txt` (after converting this to txt for pip) or manually with `pip install <package>==<version>`.

| Package             | Version | Purpose                                                                 |
|---------------------|---------|-------------------------------------------------------------------------|
| pandas             | 2.2.3   | DataFrame handling for MSP parsing, metadata, and similarity results     |
| numpy              | 2.1.1   | Array operations for mz/intensity, vectorized similarity calculations    |
| rdkit              | 2024.3.5| SMILES/InChI parsing for molecule analysis                              |
| scikit-learn       | 1.5.2   | MultiLabelBinarizer for functional groups; ML for analysis (future)      |
| seaborn            | 0.13.2  | Visualization of spectral data and similarity metrics                   |
| matplotlib         | 3.9.2   | Plotting spectra (mirror/stacked) and graphs                            |
| networkx           | 3.5     | Graph-based analysis of compound relationships                          |
| thermo             | 0.2.27  | Functional group detection via thermo.functional_groups                |
| pybatchclassyfire  | 0.1.6   | Batch ClassyFire API calls for compound classification                  |
| ms_entropy         | 0.9.4   | Spectral similarity metrics (cosine, entropy, etc.)                     |
| psutil             | latest  | RAM monitoring for chunked similarity calculations                      |
| scipy              | latest  | Statistical functions (e.g., fisher_exact, linear_sum_assignment)       |
| ipywidgets         | 8.1.5   | Interactive notebook widgets (optional for testing in Jupyter)          |
| IPython            | latest  | Notebook display utilities (optional for testing)                       |
| jupyter            | 1.1.1   | JupyterLab for interactive analysis (optional for development)          |
| requests-cache     | 1.2.1   | Caching for pybatchclassyfire API to handle rate limits                 |
| joblib             | latest  | Parallel processing for sklearn and similarity calculations             |
| pytest             | latest  | Unit testing for package (to be added in src/MSAnalyzer/tests/)         |

## Installation Instructions
1. Activate virtual environment:
   ```bash
   cd ~/Projects/MSAnalyzer
   pyenv activate msanalyzer-env  # Auto-activates if .python-version set to 3.11.9
   ```
2. Install dependencies:
   ```bash
   pip install pandas==2.2.3 numpy==2.1.1 rdkit==2024.3.5 scikit-learn==1.5.2 seaborn==0.13.2 matplotlib==3.9.2 networkx==3.5 pybatchclassyfire==0.1.6 ms_entropy==0.9.4 psutil scipy ipywidgets==8.1.5 IPython jupyter==1.1.1 requests-cache==1.2.1 joblib pytest thermo==0.2.27
   ```
3. Verify:
   ```bash
   python -c "import pandas, numpy, rdkit, sklearn, seaborn, matplotlib, networkx, thermo, pybatchclassyfire, ms_entropy, psutil, scipy, ipywidgets, IPython, jupyter, requests_cache, joblib, pytest; print('All imports successful')"
   ```
4. Convert to `requirements.txt` for pip (if needed):
   ```bash
   pip freeze > requirements.txt
   ```

## Notes
- Versions pinned where known (from MSAnalyzer_ProjectArchitectureSummary.md or verified installs); others use `latest` for compatibility. Update after install with `pip freeze`.
- `ms_entropy==0.9.4` corrects earlier `spectral_entropy` typo (PyPI package for spectral similarity).
- `pybatchclassyfire==0.1.6` is latest PyPI version (0.2 was a typo).
- `thermo==0.2.27` added (latest PyPI version; was omitted in prior install).
- `rdkit` installs natively on Mac M4 (arm64); Windows may need Visual Studio Build Tools or conda-forge fallback (prefer pip per instructions).
- Standard library modules (`argparse`, `json`, `os`, `shutil`, `itertools`, `copy`) are not listed as they require no install.
- Data files (e.g., `msms-hexose-phosphates.msp`) should be in `~/Data/MSAnalyzer/`; symlink to `src/MSAnalyzer/tests/data/` for tests.
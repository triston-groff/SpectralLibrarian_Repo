#!/bin/bash
# setup_env.sh
# Script to set up the MSAnalyzer virtual environment and install dependencies.
# Run from ~/Projects/MSAnalyzer: `bash setup_env.sh`
# Updated: Sep 11, 2025
# Changes: Removed pyproject.toml creation; updated verification to import LibraryManager (not MSPLibraryManager).
# Reasoning: Ensures setup.py is used exclusively (per user instruction); aligns with class rename.

set -e  # Exit on error

# Define paths
PROJECT_DIR="$HOME/Projects/MSAnalyzer"
DATA_DIR="$HOME/Data/MSAnalyzer"
VENV_NAME="msanalyzer-env"
PYTHON_VERSION="3.11.9"

# Check pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv not found. Install via: brew install pyenv"
    exit 1
fi

# Set up virtual environment
cd "$PROJECT_DIR" || { echo "Error: Directory $PROJECT_DIR not found"; exit 1; }
if ! pyenv versions | grep -q "$VENV_NAME"; then
    echo "Creating virtualenv $VENV_NAME with Python $PYTHON_VERSION..."
    pyenv install -s "$PYTHON_VERSION"  # Install Python if not present
    pyenv virtualenv "$PYTHON_VERSION" "$VENV_NAME"
fi
pyenv local "$VENV_NAME"  # Set local env (writes .python-version)

# Activate environment
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
source "$HOME/.pyenv/versions/$VENV_NAME/bin/activate"

# Install dependencies with error handling
echo "Installing dependencies..."
pip install --upgrade pip || { echo "Error: Failed to upgrade pip"; exit 1; }
pip install pandas==2.2.3 numpy==2.1.1 rdkit==2024.3.5 scikit-learn==1.5.2 seaborn==0.13.2 matplotlib==3.9.2 networkx==3.5 pybatchclassyfire==0.1.6 ms_entropy==0.9.4 psutil scipy ipywidgets==8.1.5 IPython jupyter==1.1.1 requests-cache==1.2.1 joblib pytest thermo==0.2.27 || { echo "Error: Failed to install dependencies"; exit 1; }

# Ensure setup.py exists for package installation
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found in $PROJECT_DIR"
    exit 1
fi

# Install MSAnalyzer package in editable mode
echo "Installing MSAnalyzer package in editable mode..."
pip install -e . || { echo "Error: Failed to install MSAnalyzer package. Check setup.py and src layout."; exit 1; }

# Create data directory and move MSP/config files
echo "Setting up data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"
# Move MSP and config files from temp or ms2parser dirs (if exist)
for dir in "$PROJECT_DIR/temp" "$PROJECT_DIR/ms2parser" "$PROJECT_DIR/ms2parser_keep"; do
    if [ -d "$dir" ]; then
        find "$dir" -type f \( -name "*.msp" -o -name "*.txt" \) -exec mv {} "$DATA_DIR/" \;
    fi
done
# Symlink for tests
mkdir -p "$PROJECT_DIR/src/MSAnalyzer/tests/data"
ln -sf "$DATA_DIR"/*.msp "$DATA_DIR"/*.txt "$PROJECT_DIR/src/MSAnalyzer/tests/data/"

# Verify imports (including package)
echo "Verifying imports..."
python -c "from MSAnalyzer import LibraryManager; from MSAnalyzer.ms2parser import group_msp, plot_spectra, CompoundContainer, get_spectra_indices; import pandas, numpy, rdkit, sklearn, seaborn, matplotlib, networkx, thermo, pybatchclassyfire, ms_entropy, psutil, scipy, ipywidgets, IPython, jupyter, requests_cache, joblib, pytest; print('All imports successful')" || { echo "Error: Import verification failed"; exit 1; }

# Export requirements.txt for pip compatibility
echo "Exporting requirements.txt..."
pip freeze > "$PROJECT_DIR/requirements.txt"

echo "Setup complete. Activate env with: cd $PROJECT_DIR; pyenv activate $VENV_NAME"
echo "Data files moved to $DATA_DIR; symlinked to src/MSAnalyzer/tests/data/"
echo "Package installed: from MSAnalyzer import LibraryManager should work."
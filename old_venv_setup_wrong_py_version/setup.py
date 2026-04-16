from setuptools import setup, find_packages

setup(
    name='MSAnalyzer',
    version='0.1.0',
    description='Unified package for MSPLibraryManager and ms2parser mass spectrometry analysis.',
    author='Triston Groff',
    author_email='tristongroff@mac.dhcp.wustl.edu',
    license='MIT',
    python_requires='>=3.11',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[],  # Deps handled separately via requirements.txt
)
# Example usage:
# From root: pip install -e .
# Then: import MSAnalyzer as msa; msa.LibraryManager
# Note: For editable mode, changes in src/ reflect immediately without reinstall.
# Rationale: Legacy setup.py avoids pyproject.toml; supports src layout via find_packages(where='src').
import os
import sys

# Make Sphinx find your package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

project = 'SpectralLibrarian'
copyright = '2026, Triston Groff and Yunwon Kang'
author = 'Triston Groff, Yunwon Kang'
release = '0.2.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Autodoc settings
autoclass_content = 'both'
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
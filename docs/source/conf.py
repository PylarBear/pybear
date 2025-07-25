# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pybear

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pybear'
copyright = '2025, PylarBear'
author = 'PylarBear'
release = pybear.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc'
]

autosummary_generate = True

autodoc_typehints = "none"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": False,
    "show-inheritance": False,
}

numpydoc_show_class_members = True
numpydoc_class_members_toctree = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'   #   'alabaster' # 
html_static_path = ['_static']
html_theme_options = {
    "logo": {"text": "pybear"},
    "navigation_with_keys": True,
    "secondary_sidebar_items": [],
    "switcher": True
}
html_context = {
   # ...
   "default_mode": "light"
}

html_meta = {"google-site-verification":"Vfm-HV9ibdA4ubmnyo7bO3KJ4LgmB48Lxp634CmYAPI"}




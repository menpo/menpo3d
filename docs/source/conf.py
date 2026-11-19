# Sphinx configuration for the menpo3d documentation.
import os
import sys

# Mayavi must not try to open a window while autodoc imports the package
os.environ.setdefault("ETS_TOOLKIT", "null")

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
sys.path.insert(0, os.path.abspath(".."))  # xref_map and sphinxext
if not on_rtd:
    # use the checkout when building locally; on Read the Docs the installed package is used
    sys.path.insert(0, os.path.abspath("../../"))

import menpo3d
from xref_map import xref_map

needs_sphinx = "5.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxext.ref_prettify",
    "sphinxext.theme_hack",
    "sphinxmapxrefrole",
]

# :map:`TriMesh` and friends: short names to the full path of the class or function
xref_mapping_dict = xref_map

# the classes of menpo that menpo3d builds on link to menpo's own documentation
intersphinx_mapping = {
    "menpo": ("https://menpo.readthedocs.io/en/stable/", None),
}

templates_path = ["_templates"]
html_static_path = ["_static"]

napoleon_include_special_with_doc = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# the "References" sections of the docstrings are footnotes that the text does not cite
suppress_warnings = ["ref.footnote"]

master_doc = "index"
autodoc_member_order = "groupwise"

project = "Menpo3D"
authors = (
    "Joan Alabort-i-Medina, Epameinondas Antonakos, James Booth,"
    " Patrick Snape, and Stefanos Zafeiriou"
)
copyright = "2014, " + authors
version = menpo3d.__version__
release = menpo3d.__version__

exclude_patterns = ["_build", "**/test/**"]
html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "Menpo3Ddoc"

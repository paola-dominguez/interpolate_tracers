import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = "interpolate_tracers"
copyright = "2026, Paola Dominguez Fernandez"
author = "Paola Dominguez Fernandez"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

# autodoc imports every module to introspect it; these dependencies aren't
# needed to build the docs and may not be installed in the docs environment
autodoc_mock_imports = ["yt"]

myst_enable_extensions = ["colon_fence"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# README.md's plain GitHub-relative links (e.g. to LICENSE, a non-.md file)
# render fine on GitHub but aren't real Sphinx documents, so myst can't
# resolve them as cross-references -- expected, not worth failing the build.
suppress_warnings = ["myst.xref_missing"]

html_theme = "furo"
html_title = "interpolate_tracers"

"""Sphinx configuration of Wenuray's documentation: sphinx-build -W -b html docs docs/_build/html"""
from pathlib import Path

from setuptools_scm import get_version

ROOT = Path(__file__).resolve().parents[1]

project = "wenuray"
author = "Miguel Cárcamo"
copyright = "Miguel Cárcamo"
try:
    release = get_version(root=str(ROOT))
except LookupError:  # no git metadata, e.g. a source archive
    release = "unknown"
version = release

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]

# The API reference, from the source: Wenuray isn't imported, so CASA isn't needed
autoapi_dirs = [str(ROOT / "src" / "wenuray")]
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
autoapi_ignore = ["*/_pyralysis_worker.py", "*/_version.py"]
autoapi_member_order = "bysource"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

myst_heading_anchors = 3
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

html_theme = "sphinx_book_theme"
html_title = "wenuray {0}".format(release)
html_theme_options = {
    "repository_url": "https://github.com/miguelcarcamov/wenuray",
    "repository_branch": "development",
    "use_repository_button": True,
    "use_issues_button": True,
}

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import re
import sys
import tomllib as toml
from datetime import datetime
from pathlib import Path

CONF_FILE = Path(__file__).resolve()
DOCS_DIR = CONF_FILE.parent
ROOT_DIR = DOCS_DIR.parent
ON_RTD = os.environ.get("READTHEDOCS") == "True"
ON_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"
USE_COMMITTED_GALLERY = ON_RTD or ON_GITHUB_ACTIONS or os.environ.get("BIOSYM_USE_COMMITTED_GALLERY") == "1"

sys.path.insert(0, str(DOCS_DIR))
sys.path.insert(0, str(ROOT_DIR))


URL = "https://github.com/biomac-group/biosym"

# -- Project information -----------------------------------------------------

# Info from pyproject.toml config:
with (ROOT_DIR / "pyproject.toml").open("rb") as pyproject_file:
    info = toml.load(pyproject_file)["project"]

project = info["name"]
author = "biosym contributors"  # No authors field in project config
release = info["version"]

copyright = f"2021 - {datetime.now().year}, MaD Lab, FAU"

# -- Copy the README and Changelog and fix image path --------------------------------------
with (ROOT_DIR / "README.md").open() as f:
    out = f.read()
with (DOCS_DIR / "README.md").open("w+") as f:
    f.write(out)

with (ROOT_DIR / "CHANGELOG.md").open() as f:
    out = f.read()
with (DOCS_DIR / "CHANGELOG.md").open("w+") as f:
    f.write(out)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    # "sphinx.ext.linkcode",  # Temporarily disabled due to Constants.actuator_model issue
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.imgconverter",
    "sphinx_gallery.gen_gallery",
    "recommonmark",
]

if not ON_RTD:
    extensions.append("numpydoc")

autodoc_mock_imports = []
if ON_RTD:
    autodoc_mock_imports = [
        "absl",
        "absl.logging",
        "cloudpickle",
        "cyipopt",
        "dash",
        "dash_ag_grid",
        "flax",
        "flatbuffers",
        "jax",
        "jax.export",
        "jax.numpy",
        "matplotlib",
        "matplotlib.pyplot",
        "numpy",
        "numpy.linalg",
        "pandas",
        "sympy",
        "sympy.physics",
        "sympy.physics.mechanics",
        "tqdm",
        "yaml",
    ]

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Taken from sklearn config
# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "special_members": True,
}
# autodoc_typehints = 'description'  # Does not work as expected. Maybe try at future date again

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "templates"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Activate the theme.
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": URL,
    "show_prev_next": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# -- Options for extensions --------------------------------------------------
# Intersphinx

# intersphinx configuration
intersphinx_module_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:25.0) Gecko/20100101 Firefox/25.0"

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    **intersphinx_module_mapping,
}

gallery_reference_url = {
    "biosym": None,
    **{k: v[0] for k, v in intersphinx_module_mapping.items()},
}
gallery_doc_module = ("biosym",)

if USE_COMMITTED_GALLERY:
    # RTD and GitHub docs builds reuse committed gallery artefacts, so skip the
    # post-build code-link embedding step that relies on writable dbm caches.
    gallery_reference_url = {}
    gallery_doc_module = ()

# Sphinx Gallary
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["./auto_examples"],
    "reference_url": gallery_reference_url,
    # 'default_thumb_file': 'fig/logo.png',
    "backreferences_dir": "modules/generated/backreferences",
    "doc_module": gallery_doc_module,
    "filename_pattern": re.escape(os.sep),
    "plot_gallery": not USE_COMMITTED_GALLERY,
    "remove_config_comments": True,
    "show_memory": True,
}


from sphinxext.githublink import make_linkcode_resolve

linkcode_resolve = make_linkcode_resolve(
    "biosym",
    "https://github.com/biomac-group/biosym/blob/{revision}/{package}/{path}#L{lineno}",
)

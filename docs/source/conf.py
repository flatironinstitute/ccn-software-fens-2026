# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import glob
import pathlib
project = 'CCN software workshop, FENS 2026'
copyright = '2025, Edoardo Balzani, Billy Broderick, Sarah Jo Venditto, Guillaume Viejo'
author = 'Edoardo Balzani, Billy Broderick, Sarah Jo Venditto, Guillaume Viejo'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_togglebutton',
    'sphinx_design',
]

templates_path = []
exclude_patterns = []

nitpicky = True

sphinxemoji_style = 'twemoji'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# max time (in secs) per notebook cell. here, we disable this
nb_execution_timeout = -1
nb_execution_raise_on_error = True
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_favicon = '_static/ccn_small.png'
html_sourcelink_suffix = ""
myst_enable_extensions = ["colon_fence", "dollarmath", "attrs_inline"]
html_theme_options = {
    "home_page_in_toc": True,
    "github_url": "https://github.com/flatironinstitute/ccn-software-fens-2026",
    "repository_url": "https://github.com/flatironinstitute/ccn-software-fens-2026",
    "logo": {
        "alt_text": "Home",
        "image_light": "_static/01-FI-primary-logo-color.png",
        "image_dark": "_static/03-FI-primary-logo-white.png",
    },
    "use_download_button": True,
    "use_repository_button": True,
    "icon_links": [
        {
            "name": "Workshops home",
            "url": "https://flatironinstitute.github.io/neurorse-workshops/",
            "type": "fontawesome",
            "icon": "fa-solid fa-house",
        },
        {
            "name": "Binder",
            "url": "https://binder.flatironinstitute.org/v2/user/wbroderick/fens2026?labpath=notebooks/",
            "type": "url",
            "icon": "https://mybinder.org/badge_logo.svg",
        },
    ],
}
nb_execution_mode = "cache"

if run_nb := os.environ.get("RUN_NB"):
    all_nbs = glob.glob("full/**/*md", recursive=True)
    all_nbs = [pathlib.Path(n).stem for n in all_nbs]
    run_globs = [f"*{n}*" for n in run_nb.split(",")]
    nb_execution_excludepatterns = [
        f"*{n}*"
        for n in all_nbs
        if not any([glob.fnmatch.fnmatch(n, g) for g in run_globs])
    ]
    print(f"Excluding notebooks: {nb_execution_excludepatterns}")
else:
    nb_execution_excludepatterns = []
    print("Running all notebooks")

nb_execution_excludepatterns += ['*-users*', '*-presenters*']

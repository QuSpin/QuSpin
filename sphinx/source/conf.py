# -*- coding: utf-8 -*-
#
# QuSpin documentation build configuration file, created by
# sphinx-quickstart on Mon Jul 24 11:07:25 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("quspin_modules/"))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '7.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    #'sphinx.ext.autosectionlabel', # doesn't seem to work
    "sphinxtogithub",
    "sphinx.ext.napoleon",
    # "sphinxcontrib.googleanalytics",
    "sphinx_rtd_size", # allows to fix width of rtd sphinx theme
]


# -- General configuration ------------------------------------------------
# autoclass_content = ""
# autodoc_default_flags = [
#         # Make sure that any autodoc declarations show the right members
#         "members",
#         "inherited-members",
#         "show-inheritance",
#         "no-special-members",
#         "no-private-members",
#         "no-undoc-members"
# ]

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "no-special-members": True,
    "no-private-members": True,
    "no-undoc-members": True,
    #  'member-order': 'bysource',
    "undoc-members": True,
    "exclude-members": "__init__",
}

autosummary_generate = True  # Make _autosummary files and include them

# Only the class' docstring is inserted.
autoclass_content = "class"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# remove duplicates
numpydoc_show_class_members = False

# # NumPydoc settings
# numpydoc_show_class_members = True
# numpydoc_class_members_toctree = True
# numpydoc_show_inherited_class_members = True

# Napoleon settings
# napoleon_use_rtype = False  # More legible
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = True
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True


# autodoc_mock_imports = ["numpy","scipy","scipy.sparse","scipy.linalg","multiprocessing","joblib","six"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "QuSpin"
copyright = "2024, Phillip Weinberg, Markus Schmitt, and Marin Bukov"
author = "Phillip Weinberg, Markus Schmitt, and Marin Bukov"
rst_prolog = open("global.rst", "r").read()

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
# The full version, including alpha/beta/rc tags.
release = "1.0.0"


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
# exclude_patterns =  [] #['_build', 'Thumbs.db', '.DS_Store', '.git']
# exclude_patterns = ['links.rst', 'README.rst']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "classic"
# html_theme_options = {
#     "body_max_width": "fill-available",
#     "linkcolor": "blue",
#     "externalrefs": "true",
# }



## google analytics
# googleanalytics_id = 'UA-110543543-1' # old tag, deprecated since July 2023
googleanalytics_id = "G-6885KZ7NH6"

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'analytics_id': googleanalytics_id,  #  Provided by Google in your dashboard
    'analytics_anonymize_ip': False,
    #'logo_only': True,
    #'display_version': True,
    #'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    #'style_nav_header_background': 'blue',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
sphinx_rtd_size_width = "100%"


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


# These folders are copied to the documentation's HTML output
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/py_class_property_fix.css',
]


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "QuSpindoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "QuSpin.tex",
        "QuSpin Documentation",
        "Phillip Weinberg, Markus Schmitt and Marin Bukov",
        "manual",
    ),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "quspin", "QuSpin Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "QuSpin",
        "QuSpin Documentation",
        author,
        "QuSpin",
        "One line description of project.",
        "Miscellaneous",
    ),
]



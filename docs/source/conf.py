import toml
import sys
import os
from datetime import datetime

with open('../../pyproject.toml') as f:
    pyproject = toml.load(f)
    __version__ = pyproject['tool']['poetry']['version']

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'eq-detection'
copyright = f'2022-{datetime.now().year}, Matteo Spanio'
author = 'Matteo Spanio'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

numfig = True

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.duration',
    'sphinxcontrib.bibtex',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.duration',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.graphviz',
    'sphinx_gallery.gen_gallery',
    'matplotlib.sphinxext.plot_directive',
]

#autodoc_default_options = {"members": True} #, "undoc-members": True}

if 'html' in sys.argv[1:]:
    extensions.append("sphinx_copybutton")

bibtex_bibfiles = ['references.bib']

templates_path = ['_templates']
exclude_patterns = []

todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "_static/img/MPAI-logo.png"
html_theme = 'furo'
html_static_path = ['_static']

# -- Options for LaTeX output ------------------------------------------------

LATEX_CONTENT_FOLDER = '_static/latex'

with open(os.path.join(LATEX_CONTENT_FOLDER, 'preamble.tex'), 'r') as f:
    latex_preamble = f.read()
with open(os.path.join(LATEX_CONTENT_FOLDER, 'titlepage.tex'), 'r') as f:
    maketitle = f.read()

latex_elements = {
    'papersize': 'a4paper',
    'fontpkg': '\\usepackage{noto}',
    'preamble': latex_preamble,
    'maketitle': maketitle,
}

latex_logo = html_logo
latex_show_pagerefs = True
# latex_show_urls = 'inline'

# -- Options for sphinx-gallery ---------------------------------------------
sphinx_gallery_conf = {
    'examples_dirs'        : [os.path.join('..', '..', 'examples', 'gallery')],
    'gallery_dirs'         : ['auto_examples'],
    'backreferences_dir'   : 'gen_modules/backreferences',
    'capture_repr'         : ('_repr_html_', '__repr__', '__str__'),
    'doc_module'           : ('audiohandler', 'ml'),
    'exclude_implicit_doc' : {},
    'ignore_repr_types'    : r'matplotlib[text, axes]',
    'promote_jupyter_magic': True,
    'min_reported_time'    : 2,
    'show_memory'          : False if 'pdf' in sys.argv[1:] else True,
    'show_signature'       : False,
}

autosummary_generate = True

# -- Options for sphix-intersphinx
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable', None),
    'python': ('https://docs.python.org/3', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'seaborn': ('https://seaborn.pydata.org', None),
}

# -- Options for napoleon ---------------------------------------------------
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# -- Options for matplotlib plot directive ----------------------------------
if 'html' in sys.argv[1:]:
    with open('_templates/plot_template.rst', 'r') as f:
        plot_template = f.read()
else:
    plot_template = """
{{ source_code }}

.. only:: not html

   {% for img in images %}
   .. _{{ img.basename }}:
   .. figure:: {{ build_dir }}/{{ img.basename }}.*
      {% for option in options -%}
      {{ option }}
      {% endfor -%}

      {{ caption }}  {# appropriate leading whitespace added beforehand #}
   {% endfor %}
"""

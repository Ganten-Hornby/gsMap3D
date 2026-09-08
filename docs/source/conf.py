project = "gsMap"
copyright = "2026, JianYang-Lab"
author = "Liyang, Wenhao"
# release = gsMap.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "nbsphinx",
    "myst_parser",
    # "sphinx_charts.charts",
    "sphinxcontrib.jquery",
    "sphinx_inline_tabs",
    "sphinxcontrib.mermaid",
]

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'classic'
# html_theme = 'sphinx_rtd_theme'
# html_theme = "pydata_sphinx_theme"
html_theme = "furo"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_show_sourcelink = False
html_css_files = ["custom.css"]

# One Mermaid theme for all documentation diagrams. Semantic node classes in
# individual flowcharts distinguish work, buffers and data, not page themes.
mermaid_light_theme = "base"
mermaid_dark_theme = "base"
mermaid_height = "auto"
mermaid_fullscreen = True
mermaid_fullscreen_button = "Expand"
mermaid_init_config = {
    "startOnLoad": False,
    "theme": "base",
    "themeVariables": {
        "fontFamily": "Arial, sans-serif",
        "fontSize": "14px",
        "primaryColor": "#edf4fc",
        "primaryTextColor": "#19324d",
        "primaryBorderColor": "#517ba5",
        "secondaryColor": "#fff4df",
        "tertiaryColor": "#eef2f6",
        "lineColor": "#64748b",
        "textColor": "#19324d",
        "clusterBkg": "#f8fafc",
        "clusterBorder": "#cbd5e1",
        "titleColor": "#19324d",
        "edgeLabelBackground": "#ffffff",
    },
    "flowchart": {
        "wrappingWidth": 320,
        "nodeSpacing": 20,
        "rankSpacing": 28,
        "subGraphTitleMargin": {"top": 8, "bottom": 12},
    },
}

html_theme_options = {
    # "light_css_variables": {
    #     "color-brand-primary": "#7C4DFF",
    #     "color-brand-content": "#7C4DFF",
    #     "color-code-background": "#f5f5f5",
    # },
}

# add plotly.js to the build
# html_js_files = [
#     "https://cdn.plot.ly/plotly-latest.min.js",
# ]

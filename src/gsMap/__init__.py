"""Genetically informed spatial mapping of cells for complex traits."""

import logging
from importlib.metadata import version

import pandas as pd

# Prevent pandas from using PyArrow-backed string arrays, which are
# incompatible with anndata's h5ad writer (h5py cannot serialise
# ArrowStringArray).  This must be set before any DataFrame is created.
pd.options.future.infer_string = False

# Package name and version
package_name = "gsMap3D"
__version__ = version(package_name)

# Libraries should not alter application/root logging during import. The CLI and
# public pipeline entry points install project-aware handlers when execution starts.
logger = logging.getLogger("gsMap")
logger.addHandler(logging.NullHandler())

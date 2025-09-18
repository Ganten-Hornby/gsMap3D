"""Genetically informed spatial mapping of cells for complex traits."""

import logging
from importlib.metadata import version

# Package name and version
package_name = "gsMap"
__version__ = version(package_name)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gsMap")
logger.propagate = False

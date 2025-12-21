"""
gsMap configuration module.

This module provides:
- Configuration dataclasses for all gsMap commands
- Base classes with automatic path generation
- Decorators for CLI integration and resource tracking
"""

from collections import OrderedDict
from collections import namedtuple

# Base classes and utilities
from .base import ConfigWithAutoPaths, ensure_path_exists

# Decorators
from .decorators import dataclass_typer, track_resource_usage, show_banner

# Create a legacy registry for backward compatibility with main.py
cli_function_registry = OrderedDict()
subcommand = namedtuple("subcommand", ["name", "func", "add_args_function", "description"])

# Configuration dataclasses
from .dataclasses import (
    RunAllModeConfig,
    CreateSliceMeanConfig,
    FormatSumstatsConfig,
    DiagnosisConfig,
    VisualizeConfig,
    ThreeDCombineConfig,
    RunLinkModeConfig,
    gsMapPipelineConfig
)

# Migrated module configurations
from .find_latent_config import FindLatentRepresentationsConfig
from .latent2gene_config import LatentToGeneConfig, DatasetType, MarkerScoreCrossSliceStrategy
from .spatial_ldsc_config import SpatialLDSCConfig
from .report_config import ReportConfig
from .quick_mode_config import QuickModeConfig
from .ldscore_config import LDScoreConfig, GenerateLDScoreConfig
from .cauchy_config import CauchyCombinationConfig

__all__ = [
    # Base classes
    'ConfigWithAutoPaths',
    'ensure_path_exists',
    
    # Decorators
    'dataclass_typer',
    'track_resource_usage',
    'show_banner',
    
    # Legacy compatibility
    'cli_function_registry',
    
    # Configurations
    'RunAllModeConfig',
    'QuickModeConfig',
    'FindLatentRepresentationsConfig',
    'LatentToGeneConfig',
    'DatasetType',
    'MarkerScoreCrossSliceStrategy',
    'SpatialLDSCConfig',
    'ReportConfig',
    'GenerateLDScoreConfig',
    'CauchyCombinationConfig',
    'CreateSliceMeanConfig',
    'FormatSumstatsConfig',
    'DiagnosisConfig',
    'VisualizeConfig',
    'ThreeDCombineConfig',
    'RunLinkModeConfig',
]
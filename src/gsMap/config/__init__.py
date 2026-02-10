"""
gsMap configuration module.

This module provides:
- Configuration dataclasses for all gsMap commands
- Base classes with automatic path generation
- Decorators for CLI integration and resource tracking
"""

from collections import OrderedDict, namedtuple

# Base classes and utilities
from .base import BaseConfig, ConfigWithAutoPaths, ensure_path_exists

# Configuration dataclasses
from .cauchy_config import CauchyCombinationConfig
from .dataclasses import (
    CreateSliceMeanConfig,
    DiagnosisConfig,
    RunLinkModeConfig,
    ThreeDCombineConfig,
    VisualizeConfig,
    gsMapPipelineConfig,
)

# Decorators
from .decorators import dataclass_typer, show_banner, track_resource_usage

# Migrated module configurations
from .find_latent_config import FindLatentRepresentationsConfig
from .format_sumstats_config import FormatSumstatsConfig
from .latent2gene_config import DatasetType, LatentToGeneConfig, MarkerScoreCrossSliceStrategy
from .ldscore_config import GenerateLDScoreConfig, LDScoreConfig
from .quick_mode_config import QuickModeConfig
from .report_config import ReportConfig
from .spatial_ldsc_config import SpatialLDSCConfig

# Create a legacy registry for backward compatibility with main.py
cli_function_registry = OrderedDict()
subcommand = namedtuple("subcommand", ["name", "func", "add_args_function", "description"])

__all__ = [
    # Base classes
    "BaseConfig",
    "ConfigWithAutoPaths",
    "ensure_path_exists",
    # Decorators
    "dataclass_typer",
    "track_resource_usage",
    "show_banner",
    # Legacy compatibility
    "cli_function_registry",
    # Configurations
    "QuickModeConfig",
    "FindLatentRepresentationsConfig",
    "LatentToGeneConfig",
    "DatasetType",
    "MarkerScoreCrossSliceStrategy",
    "SpatialLDSCConfig",
    "ReportConfig",
    "GenerateLDScoreConfig",
    "CauchyCombinationConfig",
    "CreateSliceMeanConfig",
    "FormatSumstatsConfig",
    "DiagnosisConfig",
    "VisualizeConfig",
    "ThreeDCombineConfig",
    "RunLinkModeConfig",
]

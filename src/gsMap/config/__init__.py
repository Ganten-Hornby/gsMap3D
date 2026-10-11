"""Public configuration API with lazy imports.

Importing one lightweight command configuration previously imported every
gsMap configuration and their transitive scientific stacks. Lazy attribute
resolution preserves the public API while avoiding that startup penalty.
"""

from __future__ import annotations

from collections import OrderedDict, namedtuple
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

cli_function_registry = OrderedDict()
subcommand = namedtuple("subcommand", ["name", "func", "add_args_function", "description"])

_LAZY_IMPORTS = {
    "BaseConfig": (".base", "BaseConfig"),
    "ConfigWithAutoPaths": (".base", "ConfigWithAutoPaths"),
    "DataPreparationConfig": (".data_preparation_config", "DataPreparationConfig"),
    "ensure_path_exists": (".base", "ensure_path_exists"),
    "dataclass_typer": (".decorators", "dataclass_typer"),
    "track_resource_usage": (".decorators", "track_resource_usage"),
    "show_banner": (".decorators", "show_banner"),
    "CauchyCombinationConfig": (".cauchy_config", "CauchyCombinationConfig"),
    "FindLatentRepresentationsConfig": (
        ".find_latent_config",
        "FindLatentRepresentationsConfig",
    ),
    "FormatSumstatsConfig": (".format_sumstats_config", "FormatSumstatsConfig"),
    "CrossSliceSelectionMode": (".latent2gene_config", "CrossSliceSelectionMode"),
    "DatasetType": (".latent2gene_config", "DatasetType"),
    "HomogeneousSelectionMode": (".latent2gene_config", "HomogeneousSelectionMode"),
    "LatentToGeneConfig": (".latent2gene_config", "LatentToGeneConfig"),
    "MarkerScoreCrossSliceStrategy": (
        ".latent2gene_config",
        "MarkerScoreCrossSliceStrategy",
    ),
    "GenerateLDScoreConfig": (".ldscore_config", "GenerateLDScoreConfig"),
    "LDScoreConfig": (".ldscore_config", "LDScoreConfig"),
    "QuickModeConfig": (".quick_mode_config", "QuickModeConfig"),
    "ReportConfig": (".report_config", "ReportConfig"),
    "SpatialLDSCConfig": (".spatial_ldsc_config", "SpatialLDSCConfig"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


__all__ = [
    "BaseConfig",
    "CauchyCombinationConfig",
    "ConfigWithAutoPaths",
    "DataPreparationConfig",
    "CrossSliceSelectionMode",
    "DatasetType",
    "FindLatentRepresentationsConfig",
    "FormatSumstatsConfig",
    "GenerateLDScoreConfig",
    "HomogeneousSelectionMode",
    "LDScoreConfig",
    "LatentToGeneConfig",
    "MarkerScoreCrossSliceStrategy",
    "QuickModeConfig",
    "ReportConfig",
    "SpatialLDSCConfig",
    "cli_function_registry",
    "dataclass_typer",
    "ensure_path_exists",
    "show_banner",
    "track_resource_usage",
]

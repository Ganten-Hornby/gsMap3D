"""
Configuration for Quick Mode pipeline.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Annotated, List, Literal, OrderedDict, Dict, Any

import typer
import logging

from gsMap.config.base import ConfigWithAutoPaths
# Use relative imports to avoid circular dependency
from .find_latent_config import FindLatentRepresentationsConfig, FindLatentCoreConfig
from .latent2gene_config import LatentToGeneConfig, LatentToGeneCoreConfig, LatentToGeneComputeConfig
from .spatial_ldsc_config import SpatialLDSCConfig, SpatialLDSCCoreConfig, SpatialLDSCComputeConfig
from .report_config import ReportConfig
from .cauchy_config import CauchyCombinationConfig

logger = logging.getLogger("gsMap.config")

@dataclass
class QuickModeConfig(ReportConfig, SpatialLDSCConfig, LatentToGeneConfig, FindLatentRepresentationsConfig, ConfigWithAutoPaths):
    """Quick Mode Pipeline Configuration"""
    __core_only__ = True

    # ------------------------------------------------------------------------
    # Pipeline Control
    # ------------------------------------------------------------------------
    start_step: Annotated[str, typer.Option(
        help="Step to start execution from (find_latent, latent2gene, spatial_ldsc, cauchy, report)",
        case_sensitive=False
    )] = "find_latent"

    stop_step: Annotated[Optional[str], typer.Option(
        help="Step to stop execution at (inclusive)",
        case_sensitive=False
    )] = None


    def __post_init__(self):

        ConfigWithAutoPaths.__post_init__(self)

        # Unify high quality qc cell flag in find_latent and latent2gene when both steps run
        if self.is_both_latent_and_gene_running:
            self.high_quality_neighbor_filter = self.high_quality_cell_qc

        # Process sumstats inputs
        self.sumstats_config_dict = self.process_sumstats_inputs()

    @property
    def is_both_latent_and_gene_running(self) -> bool:
        """Check if both find_latent and latent2gene are in the execution range."""
        steps = ["find_latent", "latent2gene", "spatial_ldsc", "cauchy", "report"]
        try:
            start_idx = steps.index(self.start_step)
            stop_idx = steps.index(self.stop_step) if self.stop_step else len(steps) - 1
            return start_idx <= 0 and stop_idx >= 1
        except ValueError:
            return False

    @property
    def find_latent_config(self) -> FindLatentRepresentationsConfig:
        return FindLatentRepresentationsConfig(**{
            f.name: getattr(self, f.name) for f in fields(FindLatentRepresentationsConfig) if f.init
        })

    @property
    def latent2gene_config(self) -> LatentToGeneConfig:
        params = {f.name: getattr(self, f.name) for f in fields(LatentToGeneConfig) if f.init}

        # If both steps run, clear explicit h5ad inputs to trigger auto-detection in LatentToGeneConfig
        if self.is_both_latent_and_gene_running:
            params['h5ad_path'] = None
            params['h5ad_yaml'] = None
            params['h5ad_list_file'] = None
            params['sample_h5ad_dict'] = None
        return LatentToGeneConfig(**params)

    @property
    def spatial_ldsc_config(self) -> SpatialLDSCConfig:
        return SpatialLDSCConfig(**{
            f.name: getattr(self, f.name) for f in fields(SpatialLDSCConfig) if f.init
        })

    @property
    def report_config(self) -> ReportConfig:
        return ReportConfig(**{
            f.name: getattr(self, f.name) for f in fields(ReportConfig) if f.init and hasattr(self, f.name)
        })

    @property
    def cauchy_config(self) -> CauchyCombinationConfig:
        return CauchyCombinationConfig(**{
            f.name: getattr(self, f.name) for f in fields(CauchyCombinationConfig) if f.init and hasattr(self, f.name)
        })

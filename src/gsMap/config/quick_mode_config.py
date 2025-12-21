"""
Configuration for Quick Mode pipeline.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Annotated, List, Literal, OrderedDict, Dict, Any

import typer
import logging
import yaml

from gsMap.config.base import ConfigWithAutoPaths
# Use relative imports to avoid circular dependency
from .find_latent_config import FindLatentRepresentationsConfig
from .latent2gene_config import LatentToGeneConfig
from .ldscore_config import GenerateLDScoreConfig
from .spatial_ldsc_config import SpatialLDSCConfig
from .report_config import ReportConfig
from .cauchy_config import CauchyCombinationConfig
from .compute_config import LatentToGeneComputeConfig, SpatialLDSCComputeConfig
from gsMap.config.utils import process_h5ad_inputs

logger = logging.getLogger("gsMap.config")

@dataclass
class QuickModeConfig(FindLatentRepresentationsConfig, LatentToGeneConfig, SpatialLDSCConfig):
    """Configuration for running the complete gsMap pipeline in a single command.
    
    Inherits fields from all major sub-configs to provide a unified interface.
    """

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

    # ------------------------------------------------------------------------
    # Report Parameters
    # ------------------------------------------------------------------------
    top_corr_genes: Annotated[int, typer.Option(
        help="Number of top correlated genes to display",
        min=1,
        max=500
    )] = 50

    selected_genes: Annotated[Optional[str], typer.Option(
        help="Comma-separated list of specific genes to include"
    )] = None


    def __post_init__(self):
        # We don't call super().__post_init__() to avoid mandatory validation errors
        # in base classes (e.g., SpatialLDSCConfig requiring sumstats) which might
        # not be needed if the user only runs partial steps.
        # Instead we call ConfigWithAutoPaths.__post_init__ directly.
        ConfigWithAutoPaths.__post_init__(self)

        # Unify high quality cell flag
        self.high_quality_neighbor_filter = self.high_quality_cell_qc

        # Process h5ad inputs if provided (for FindLatent step)
        input_options = {
            'h5ad_yaml': ('h5ad_yaml', 'yaml'),
            'h5ad_path': ('h5ad_path', 'list'),
            'h5ad_list_file': ('h5ad_list_file', 'file'),
        }
        has_input = any(getattr(self, k) is not None for k in ['h5ad_path', 'h5ad_yaml', 'h5ad_list_file'])
        if has_input:
            self.sample_h5ad_dict = process_h5ad_inputs(self, input_options)
            
        # Process sumstats inputs
        if self.sumstats_file is None and self.sumstats_config_file is None:
            # We don't raise error immediately because user might only run find_latent or latent2gene
             pass
        elif self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError(
                "Only one of sumstats_file and sumstats_config_file must be provided."
            )
        elif self.sumstats_file is not None:
             if self.trait_name is None:
                 raise ValueError("trait_name must be provided if sumstats_file is provided.")
             self.sumstats_config_dict = {self.trait_name: self.sumstats_file}
        elif self.sumstats_config_file is not None:
             if self.trait_name is not None:
                 raise ValueError("trait_name must not be provided if sumstats_config_file is provided.")
             with open(self.sumstats_config_file) as f:
                config_loaded = yaml.load(f, Loader=yaml.FullLoader)
                for t_name, s_file in config_loaded.items():
                    self.sumstats_config_dict[t_name] = Path(s_file)

    @property
    def find_latent_config(self) -> FindLatentRepresentationsConfig:
        return FindLatentRepresentationsConfig(**{
            f.name: getattr(self, f.name) for f in fields(FindLatentRepresentationsConfig)
        })

    @property
    def latent2gene_config(self) -> LatentToGeneConfig:
        params = {f.name: getattr(self, f.name) for f in fields(LatentToGeneConfig)}
        # Unify high quality cell flag: QuickMode's high_quality_cell_qc maps to
        # LatentToGeneConfig's high_quality_neighbor_filter
        params['high_quality_neighbor_filter'] = self.high_quality_cell_qc
        return LatentToGeneConfig(**params)

    @property
    def spatial_ldsc_config(self) -> SpatialLDSCConfig:
        return SpatialLDSCConfig(**{
            f.name: getattr(self, f.name) for f in fields(SpatialLDSCConfig)
        })

    def get_report_config(self, trait_name: str, sumstats_file: Path) -> ReportConfig:
        return ReportConfig(
            workdir=self.workdir,
            project_name=self.project_name,
            sample_name=self.project_name,
            trait_name=trait_name,
            annotation=self.annotation, 
            sumstats_file=sumstats_file,
            top_corr_genes=self.top_corr_genes,
            selected_genes=self.selected_genes,
        )

    def get_cauchy_config(self, trait_name: str) -> CauchyCombinationConfig:
        return CauchyCombinationConfig(
            workdir=self.workdir,
            project_name=self.project_name,
            trait_name=trait_name,
            annotation=self.annotation,
            sample_name_list=[self.project_name],
        )

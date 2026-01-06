"""
Configuration for generating reports.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Annotated
import logging
import typer

from . import LatentToGeneConfig
from .cauchy_config import CauchyCombinationConfig
from .spatial_ldsc_config import SpatialLDSCCoreConfig, SpatialLDSCConfig
from .base import ConfigWithAutoPaths, ensure_path_exists

logger = logging.getLogger("gsMap.config")

@dataclass
class ReportConfig(CauchyCombinationConfig,ConfigWithAutoPaths):
    """Report Generation Configuration"""
    
    downsampling_n_spots: Annotated[int, typer.Option(
        help="Number of spots to downsample for PCC calculation if n_spots > this value",
        min=1000,
        max=100000
    )] = 10000

    top_corr_genes: Annotated[int, typer.Option(
        help="Number of top correlated genes to display",
        min=1,
        max=500
    )] = 50

    # Advanced visualization parameters
    single_sample_multi_trait_max_cols: int = 5
    subsample_n_points: Optional[int] = None
    single_sample_multi_trait_subplot_width_inches: float = 4.0
    single_sample_multi_trait_dpi: int = 300
    enable_pdf_output: bool = True
    hover_text_list: Optional[list] = None
    single_trait_multi_sample_max_cols: int = 8
    single_trait_multi_sample_subplot_width_inches: float = 4.0
    single_trait_multi_sample_scaling_factor: float = 1.0
    single_trait_multi_sample_dpi: int = 300
    share_coords: bool = False

    # Compatibility properties for visualization paths
    @property
    @ensure_path_exists
    def visualization_result_dir(self) -> Path:
        return self.project_dir / "report" / self.project_name / (self.trait_name or "multi_trait")


    def __post_init__(self):
        CauchyCombinationConfig.__post_init__(self)
        self.show_config(ReportConfig)


def check_report_done(config: ReportConfig, trait_name: str) -> bool:
    """
    Check if report step is done for a specific trait.
    """
    report_file = config.get_gsMap_report_file(trait_name)
    return report_file.exists()

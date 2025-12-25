"""
Configuration for generating reports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated
import logging
import typer

from .cauchy_config import CauchyCombinationConfig
from .base import ConfigWithAutoPaths

logger = logging.getLogger("gsMap.config")

@dataclass
class ReportConfig(CauchyCombinationConfig):
    """Configuration for generating reports."""
    
    sumstats_file: Annotated[Optional[Path], typer.Option(
        help="Path to GWAS summary statistics file (optional)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

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
    
    selected_genes: Annotated[Optional[str], typer.Option(
        help="Comma-separated list of specific genes to include"
    )] = None
    
    fig_width: Annotated[Optional[int], typer.Option(
        help="Width of the generated figures in pixels"
    )] = None
    
    fig_height: Annotated[Optional[int], typer.Option(
        help="Height of the generated figures in pixels"
    )] = None
    
    point_size: Annotated[Optional[int], typer.Option(
        help="Point size for the figures"
    )] = None
    
    fig_style: Annotated[str, typer.Option(
        help="Style of the generated figures",
        case_sensitive=False
    )] = "light"

    plot_type: Annotated[str, typer.Option(
        help="Type of plots to generate: 'gsMap', 'manhattan', 'GSS', or 'all'",
        case_sensitive=False
    )] = "all"

    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm storing spatial coordinates"
    )] = "spatial"

    memmap_tmp_dir: Annotated[Optional[Path], typer.Option(
        help="Temporary directory for memory-mapped files"
    )] = None

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
    def visualization_result_dir(self) -> Path:
        return self.project_dir / "report" / self.project_name / (self.trait_name or "multi_trait")


    @property
    def customize_fig(self) -> bool:
        """Check if figure customization is requested."""
        return any([self.fig_width, self.fig_height, self.point_size])

    # Settings for the viewer
    port: Annotated[int, typer.Option(help="Port to serve the interactive report on")] = 5006
    browser: Annotated[bool, typer.Option(help="Whether to open the browser automatically")] = True

    def __post_init__(self):
        super().__post_init__()
        self.show_config("Report Generation Configuration")


def check_report_done(config: ReportConfig, trait_name: str) -> bool:
    """
    Check if report step is done for a specific trait.
    """
    report_file = config.get_gsMap_report_file(trait_name)
    return report_file.exists()

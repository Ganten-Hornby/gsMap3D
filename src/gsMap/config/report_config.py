"""
Configuration for generating reports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated
import logging
import typer

from gsMap.config.base import ConfigWithAutoPaths

logger = logging.getLogger("gsMap.config")

@dataclass
class ReportConfig(ConfigWithAutoPaths):
    """Configuration for generating reports."""
    
    # Required from parent
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]
    
    trait_name: Annotated[Optional[str], typer.Option(
        help="Name of the trait (optional, will include all traits if not provided)"
    )] = None
    
    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation layer name"
    )] = None
    
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

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
    
    sample_name: Annotated[str, typer.Option(
        help="Name of the sample"
    )]
    
    trait_name: Annotated[str, typer.Option(
        help="Name of the trait to generate the report for"
    )]
    
    annotation: Annotated[str, typer.Option(
        help="Annotation layer name"
    )]
    
    sumstats_file: Annotated[Path, typer.Option(
        help="Path to GWAS summary statistics file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )]
    

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
    
    # Hidden parameter
    plot_type: str = "all"

    def __post_init__(self):
        super().__post_init__()
        self.show_config("Report Generation Configuration")


def check_report_done(config: ReportConfig, trait_name: str) -> bool:
    """
    Check if report step is done for a specific trait.
    """
    report_file = config.get_gsMap_report_file(trait_name)
    return report_file.exists()

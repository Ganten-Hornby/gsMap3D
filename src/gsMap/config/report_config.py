"""
Configuration for generating reports.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from .base import ConfigWithAutoPaths, ensure_path_exists
from .cauchy_config import CauchyCombinationConfig

logger = logging.getLogger("gsMap.config")


@dataclass
class ReportConfig(CauchyCombinationConfig, ConfigWithAutoPaths):
    """Report Generation Configuration"""

    downsampling_n_spots_pcc: Annotated[
        int,
        typer.Option(
            help="Number of spots to downsample for PCC calculation if n_spots > this value",
            min=1000,
            max=100000,
        ),
    ] = 20000

    downsampling_n_spots_3d: Annotated[
        int,
        typer.Option(
            help="Number of spots to downsample for 3D visualization if n_spots > this value",
            min=1000,
            max=2000000,
        ),
    ] = 1000000

    downsampling_n_spots_2d: Annotated[
        int,
        typer.Option(
            help="Max spots per sample for 2D distribution plots. Samples with more spots will be randomly downsampled.",
            min=10000,
            max=500000,
        ),
    ] = 250000

    top_corr_genes: Annotated[
        int, typer.Option(help="Number of top correlated genes to display", min=1, max=500)
    ] = 50

    # Advanced visualization parameters
    single_sample_multi_trait_max_cols: int = 5
    subsample_n_points: int | None = None
    single_sample_multi_trait_subplot_width_inches: float = 4.0
    single_sample_multi_trait_dpi: int = 300
    enable_pdf_output: bool = True
    hover_text_list: list | None = None
    single_trait_multi_sample_max_cols: int = 8
    single_trait_multi_sample_subplot_width_inches: float = 4.0
    single_trait_multi_sample_scaling_factor: float = 1.0
    single_trait_multi_sample_dpi: int = 300
    share_coords: bool = False

    # Weather to generate single-feature multi-sample plots (LDSC, annotation, and gene diagnostic plots)
    generate_multi_sample_plots: bool = False

    # Plot origin for spatial plots ('upper' or 'lower')
    plot_origin: Annotated[
        str,
        typer.Option(
            help="Plot origin for spatial plots ('upper' or 'lower'). 'upper' will flip the y-axis (standard for images)."
        ),
    ] = "upper"

    # Legend marker size for categorical plots
    legend_marker_size: float = 10.0

    # Force re-run of report generation even if results exist
    force_report_re_run: bool = False

    # Compatibility properties for visualization paths
    @property
    @ensure_path_exists
    def visualization_result_dir(self) -> Path:
        return self.project_dir / "report" / self.project_name / (self.trait_name or "multi_trait")

    def __post_init__(self):
        CauchyCombinationConfig.__post_init__(self)
        self.show_config(ReportConfig)

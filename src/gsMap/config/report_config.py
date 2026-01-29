"""
Configuration for generating reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Annotated, List, Tuple, TYPE_CHECKING
import logging
import typer

if TYPE_CHECKING:
    from .quick_mode_config import QuickModeConfig
from .latent2gene_config import LatentToGeneConfig, DatasetType
from .cauchy_config import CauchyCombinationConfig
from .spatial_ldsc_config import SpatialLDSCCoreConfig, SpatialLDSCConfig
from .base import ConfigWithAutoPaths, ensure_path_exists

logger = logging.getLogger("gsMap.config")

@dataclass
class ReportConfig(CauchyCombinationConfig,ConfigWithAutoPaths):
    """Report Generation Configuration"""
    
    downsampling_n_spots_pcc: Annotated[int, typer.Option(
        help="Number of spots to downsample for PCC calculation if n_spots > this value",
        min=1000,
        max=100000
    )] = 20000

    downsampling_n_spots_3d: Annotated[int, typer.Option(
        help="Number of spots to downsample for 3D visualization if n_spots > this value",
        min=1000,
        max=2000000
    )] = 1000000

    downsampling_n_spots_2d: Annotated[int, typer.Option(
        help="Max spots per sample for 2D distribution plots. Samples with more spots will be randomly downsampled.",
        min=10000,
        max=500000
    )] = 250000

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

    # Weather to generate single-feature multi-sample plots (LDSC, annotation, and gene diagnostic plots)
    generate_multi_sample_plots: bool = False 

    # Plot origin for spatial plots ('upper' or 'lower')
    plot_origin: Annotated[str, typer.Option(
        help="Plot origin for spatial plots ('upper' or 'lower'). 'upper' will flip the y-axis (standard for images)."
    )] = "upper"

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


def check_report_done(config: QuickModeConfig, verbose: bool = False) -> bool:

    missing_data_files, missing_web_files = get_report_missing_files(config)
    missing_files = missing_data_files + missing_web_files

    if missing_files and verbose:
        logger.info(f"Report incomplete. Missing {len(missing_files)} files:")
        for f in missing_files[:10]:  # Show first 10
            logger.info(f"  - {f}")
        if len(missing_files) > 10:
            logger.info(f"  ... and {len(missing_files) - 10} more")

    return len(missing_files) == 0


def get_report_missing_files(config: QuickModeConfig) -> Tuple[List[Path], List[Path]]:
    """
    Get lists of missing report files, categorized by type.

    Returns:
        Tuple of (missing_data_files, missing_web_files)
    """
    missing_data_files = []
    missing_web_files = []

    # Get dataset type (default to spatial2D if not found)
    dataset_type = config.dataset_type

    # === Web Report Files ===
    web_report_dir = config.web_report_dir
    js_data_dir = web_report_dir / "js_data"

    core_web_files = [
        web_report_dir / "index.html",
        web_report_dir / "report_meta.json",
        js_data_dir / "sample_index.js",
        js_data_dir / "report_meta.js",
        js_data_dir / "cauchy_results.js",
    ]

    for f in core_web_files:
        if not f.exists():
            missing_web_files.append(f)

    # === Data Files ===
    report_data_dir = config.report_data_dir

    core_data_files = [
        report_data_dir / "spot_metadata.csv",
        report_data_dir / "gene_list.csv",
        report_data_dir / "cauchy_results.csv",
    ]

    for f in core_data_files:
        if not f.exists():
            missing_data_files.append(f)

    # Per-trait files
    traits = config.trait_name_list
    annotation_list = config.annotation_list

    for trait in traits:
        trait_gss_csv = report_data_dir / "gss_stats" / f"gene_trait_correlation_{trait}.csv"
        trait_manhattan_csv = report_data_dir / "manhattan_data" / f"{trait}_manhattan.csv"

        if not trait_gss_csv.exists():
            missing_data_files.append(trait_gss_csv)
        if not trait_manhattan_csv.exists():
            missing_data_files.append(trait_manhattan_csv)

        trait_gss_js = js_data_dir / "gss_stats" / f"gene_trait_correlation_{trait}.js"
        trait_manhattan_js = js_data_dir / f"manhattan_{trait}.js"

        if not trait_gss_js.exists():
            missing_web_files.append(trait_gss_js)
        if not trait_manhattan_js.exists():
            missing_web_files.append(trait_manhattan_js)

    # Per-sample spatial JS files (Spatial only)
    if dataset_type in (DatasetType.SPATIAL_2D, DatasetType.SPATIAL_3D):
        sample_h5ad_dict = config.sample_h5ad_dict
        if sample_h5ad_dict:
            for sample_name in sample_h5ad_dict.keys():
                safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
                sample_js = js_data_dir / f"sample_{safe_name}_spatial.js"
                if not sample_js.exists():
                    missing_web_files.append(sample_js)

        # Gene diagnostic plots directory (Spatial only)
        gene_plot_dir = web_report_dir / "gene_diagnostic_plots"
        if not gene_plot_dir.exists():
            missing_web_files.append(gene_plot_dir)

    # 3D specific files (Spatial 3D only)
    if dataset_type == DatasetType.SPATIAL_3D:
        three_d_data_dir = report_data_dir / "spatial_3d"
        three_d_web_dir = web_report_dir / "spatial_3d"

        # 3D H5AD file
        h5ad_3d = three_d_data_dir / "spatial_3d.h5ad"
        if not h5ad_3d.exists():
            missing_data_files.append(h5ad_3d)

        # 3D HTML plot files (one per trait/annotation)
        for trait in traits:
            safe_trait = "".join(c if c.isalnum() else "_" for c in trait)
            trait_3d_html = three_d_web_dir / f"spatial_3d_trait_{safe_trait}.html"
            if not trait_3d_html.exists():
                missing_web_files.append(trait_3d_html)

        for anno in annotation_list:
            safe_anno = "".join(c if c.isalnum() else "_" for c in anno)
            anno_3d_html = three_d_web_dir / f"spatial_3d_anno_{safe_anno}.html"
            if not anno_3d_html.exists():
                missing_web_files.append(anno_3d_html)

    # Multi-sample plots (Spatial only, and if enabled)
    if dataset_type != DatasetType.SCRNA_SEQ and config.generate_multi_sample_plots:
        spatial_plot_dir = web_report_dir / "spatial_plots"
        annotation_plot_dir = web_report_dir / "annotation_plots"

        for trait in traits:
            plot_path = spatial_plot_dir / f"ldsc_{trait}.png"
            if not plot_path.exists():
                missing_web_files.append(plot_path)

        for anno in annotation_list:
            plot_path = annotation_plot_dir / f"anno_{anno}.png"
            if not plot_path.exists():
                missing_web_files.append(plot_path)

    # UMAP data (optional)
    concat_adata_path = config.concatenated_latent_adata_path
    if concat_adata_path and concat_adata_path.exists():
        umap_file = report_data_dir / "umap_data.csv"
        umap_js = js_data_dir / "umap_data.js"
        if not umap_file.exists():
            missing_data_files.append(umap_file)
        if not umap_js.exists():
            missing_web_files.append(umap_js)

    return missing_data_files, missing_web_files

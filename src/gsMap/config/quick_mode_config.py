"""
Configuration for Quick Mode pipeline.
"""

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Annotated

import typer

from gsMap.config.base import ConfigWithAutoPaths

from .cauchy_config import CauchyCombinationConfig

# Use relative imports to avoid circular dependency
from .find_latent_config import FindLatentRepresentationsConfig
from .latent2gene_config import DatasetType, LatentToGeneConfig
from .report_config import ReportConfig
from .spatial_ldsc_config import SpatialLDSCConfig

logger = logging.getLogger("gsMap.config")


@dataclass
class QuickModeConfig(
    ReportConfig,
    SpatialLDSCConfig,
    LatentToGeneConfig,
    FindLatentRepresentationsConfig,
    ConfigWithAutoPaths,
):
    """Quick Mode Pipeline Configuration"""

    __core_only__ = True

    # ------------------------------------------------------------------------
    # Pipeline Control
    # ------------------------------------------------------------------------
    start_step: Annotated[
        str,
        typer.Option(
            help="Step to start execution from (find_latent, latent2gene, spatial_ldsc, cauchy, report)",
            case_sensitive=False,
        ),
    ] = "find_latent"

    stop_step: Annotated[
        str | None,
        typer.Option(help="Step to stop execution at (inclusive)", case_sensitive=False),
    ] = None

    def __post_init__(self):
        ConfigWithAutoPaths.__post_init__(self)

        self._init_sumstats()
        self._init_annotation_list()

        if self.is_both_latent_and_gene_running:
            self.high_quality_neighbor_filter = self.high_quality_cell_qc

            # Use dual embeddings if both steps are running
            if self.latent_representation_niche is None:
                self.latent_representation_niche = "emb_niche"
            if self.latent_representation_cell is None:
                self.latent_representation_cell = "emb_cell"
        self.show_config(QuickModeConfig)

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
        return FindLatentRepresentationsConfig(
            **{
                f.name: getattr(self, f.name)
                for f in fields(FindLatentRepresentationsConfig)
                if f.init
            }
        )

    @property
    def latent2gene_config(self) -> LatentToGeneConfig:
        params = {f.name: getattr(self, f.name) for f in fields(LatentToGeneConfig) if f.init}
        return LatentToGeneConfig(**params)

    @property
    def spatial_ldsc_config(self) -> SpatialLDSCConfig:
        return SpatialLDSCConfig(
            **{f.name: getattr(self, f.name) for f in fields(SpatialLDSCConfig) if f.init}
        )

    @property
    def report_config(self) -> ReportConfig:
        return ReportConfig(
            **{
                f.name: getattr(self, f.name)
                for f in fields(ReportConfig)
                if f.init and hasattr(self, f.name)
            }
        )

    @property
    def cauchy_config(self) -> CauchyCombinationConfig:
        return CauchyCombinationConfig(
            **{
                f.name: getattr(self, f.name)
                for f in fields(CauchyCombinationConfig)
                if f.init and hasattr(self, f.name)
            }
        )


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


def get_report_missing_files(config: QuickModeConfig) -> tuple[list[Path], list[Path]]:
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

"""
Report Data Preparation Module

This module prepares data for the Alpine.js + Tailwind CSS + Plotly.js report.
All data is exported as JS files with window global variables to bypass CORS restrictions
when opening index.html via file:// protocol.
"""

import gc
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import plotly
import scanpy as sc
import scipy.sparse as sp

from gsMap.config import QuickModeConfig
from gsMap.config.latent2gene_config import DatasetType
from gsMap.find_latent.st_process import setup_data_layer
from gsMap.report.diagnosis import convert_z_to_p, filter_snps, load_gwas_data
from gsMap.report.three_d_plot.three_d_plots import three_d_plot, three_d_plot_save
from gsMap.report.visualize import (
    estimate_plotly_point_size,
)
from gsMap.spatial_ldsc.io import load_marker_scores_memmap_format

logger = logging.getLogger(__name__)


def _available_cpu_count() -> int:
    """Return the CPU count assigned to this process, including under SLURM."""

    for variable in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(variable)
        if value:
            try:
                count = int(value)
            except ValueError:
                continue
            if count > 0:
                return count

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _report_worker_count(report_config: QuickModeConfig, ceiling: int) -> int:
    """Bound report workers by both runtime visibility and configured CPU intent."""

    configured_hints = [
        int(value)
        for value in (
            getattr(report_config, "ldsc_read_workers", 0),
            getattr(report_config, "ldsc_compute_workers", 0),
        )
        if isinstance(value, int) and value > 0
    ]
    configured_limit = max(configured_hints, default=ceiling)
    return max(1, min(ceiling, configured_limit, _available_cpu_count()))


@dataclass(frozen=True)
class _ManhattanWeightContext:
    """Read-only weight resources shared by Manhattan worker threads."""

    snp_index: pd.Index
    chromosome: np.ndarray
    base_pair: np.ndarray
    matrix: object
    target_gene_indices: np.ndarray
    target_genes: np.ndarray


class ReportDataManager:
    def __init__(self, report_config: QuickModeConfig):
        self.report_config = report_config

        # Two separate directories: data files vs web report
        self.report_data_dir = report_config.report_data_dir
        self.web_report_dir = report_config.web_report_dir
        self.js_data_dir = self.web_report_dir / "js_data"

        # Create directories
        self.report_data_dir.mkdir(parents=True, exist_ok=True)
        self.web_report_dir.mkdir(parents=True, exist_ok=True)
        self.js_data_dir.mkdir(exist_ok=True)

        if self.report_config.sample_h5ad_dict is None:
            self.report_config._resolve_h5ad_inputs()

        self.force_re_run = getattr(report_config, "force_report_re_run", False)

        # Internal state
        self.ldsc_results = None
        self.traits = []
        self.sample_names = []
        self.coords = None
        self.is_3d = False
        self.z_axis = None
        self.common_spots = None
        self.analysis_spots = None
        self.gene_stats = None
        self.metadata = None
        self.gss_adata = None
        self.chrom_tick_positions = None
        self.umap_info = None
        self.spatial_3d_html = None

    def close(self):
        """Release resources and close file handles."""
        if self.gss_adata is not None:
            logger.info("Closing GSS AnnData and memory maps...")
            if "memmap_manager" in self.gss_adata.uns:
                self.gss_adata.uns["memmap_manager"].close()
            self.gss_adata = None
            gc.collect()

    def _is_step_complete(self, files: list[Path]) -> bool:
        if self.force_re_run:
            return False
        return all(f.is_file() and f.stat().st_size > 0 for f in files)

    def run(self):
        """Orchestrate the report data preparation."""
        logger.info("Starting report data preparation...")
        step_timings: dict[str, float] = {}
        try:
            steps = (
                ("base metadata", self.prepare_base_metadata),
                ("GSS statistics", self.prepare_gss_stats),
                ("spot metadata", self.prepare_spot_metadata),
                ("Manhattan data", self.prepare_manhattan_data),
                ("static plots", self.render_static_plots),
                ("Cauchy results", self.collect_cauchy_results),
                ("UMAP data", self.prepare_umap_data),
                ("3D visualization", self.prepare_3d_visualization),
                ("final assets", self.finalize_report),
            )
            for step_name, step_function in steps:
                started = time.perf_counter()
                logger.info(f"Report step started: {step_name}")
                step_function()
                elapsed = time.perf_counter() - started
                step_timings[step_name] = elapsed
                logger.info(f"Report step completed: {step_name} in {elapsed:.2f}s")

            logger.info("Report data preparation complete.")
            logger.info(
                "Report timing summary: "
                + ", ".join(f"{name}={seconds:.2f}s" for name, seconds in step_timings.items())
            )
            logger.info(f"  Data files: {self.report_data_dir}")
            logger.info(f"  Web report: {self.web_report_dir}")
        finally:
            self.close()

        return self.web_report_dir

    def prepare_base_metadata(self):
        """Load LDSC results and coordinates."""
        # 1. Load LDSC results
        if self.ldsc_results is None:
            self.ldsc_results, self.traits, self.sample_names = _load_ldsc_results(
                self.report_config
            )

        # 2. Load coordinates
        if self.coords is None:
            self.coords, self.is_3d, self.z_axis = _load_coordinates(self.report_config)

        # Export base metadata as JS if needed (usually handled in finalizing or per-sample)
        # However, _export_per_sample_spatial_js is what we use now.

    def prepare_gss_stats(self):
        """Load GSS data, calculate PCC per trait, and split results."""
        gss_dir = self.report_data_dir / "gss_stats"
        gss_dir.mkdir(exist_ok=True)

        gene_list_file = self.report_data_dir / "gene_list.csv"

        # Determine which traits need PCC calculation
        traits_to_run = []
        for trait in self.traits:
            csv_path = self.report_config.get_gene_diagnostic_info_save_path(trait)
            js_path = self.js_data_dir / "gss_stats" / f"gene_trait_correlation_{trait}.js"
            if not self._is_step_complete([csv_path, js_path]):
                traits_to_run.append(trait)

        if not traits_to_run and self._is_step_complete([gene_list_file]):
            logger.info("GSS statistics (PCC) already complete for all traits. Skipping.")
            return

        logger.info(f"Processing GSS statistics for {len(traits_to_run)} traits...")

        # Load GSS and common spots
        if self.gss_adata is None:
            self.common_spots, self.gss_adata, self.gene_stats, self.analysis_spots = (
                _load_gss_and_calculate_stats_base(
                    self.report_config, self.ldsc_results, self.coords, self.report_data_dir
                )
            )

        # Pre-filter to high expression genes and pre-calculate centered matrix
        exp_frac = pd.read_parquet(self.report_config.mean_frac_path)
        high_expr_genes = exp_frac[exp_frac["frac"] > 0.01].index.tolist()

        # Use analysis_spots (subsampled) for PCC calculation
        gss_adata_sub = self.gss_adata[self.analysis_spots, high_expr_genes]
        gss_matrix = gss_adata_sub.X
        if hasattr(gss_matrix, "toarray"):
            gss_matrix = gss_matrix.toarray()

        gss_mean = gss_matrix.mean(axis=0).astype(np.float32)
        gss_centered = (gss_matrix - gss_mean).astype(np.float32)
        gss_ssq = np.sum(gss_centered**2, axis=0)
        gene_names = gss_adata_sub.var_names.tolist()

        # Calculate PCC for each missing trait
        for trait in self.traits:
            # We check both the CSV in gss_stats and the JS in js_data/gss_stats
            csv_path = self.report_config.get_gene_diagnostic_info_save_path(trait)
            js_path = self.js_data_dir / "gss_stats" / f"gene_trait_correlation_{trait}.js"

            if self._is_step_complete([csv_path, js_path]):
                continue

            trait_pcc = _calculate_pcc_for_single_trait_fast(
                trait,
                self.ldsc_results,
                self.analysis_spots,
                gss_centered,
                gss_ssq,
                gene_names,
                self.gene_stats,
                self.report_config,
                self.report_data_dir,
                gss_dir,
            )
            if trait_pcc is not None:
                # Export to JS immediately after CSV creation
                self._export_single_pcc_js(trait, trait_pcc)

    def prepare_spot_metadata(self):
        """Save spot metadata and coordinates."""
        metadata_file = self.report_data_dir / "spot_metadata.csv"
        if self._is_step_complete([metadata_file]):
            self.metadata = pd.read_csv(metadata_file)
            required_columns = {
                "spot",
                "sample_name",
                *self.traits,
                *self.report_config.annotation_list,
            }
            if self.report_config.dataset_type in (
                DatasetType.SPATIAL_2D,
                DatasetType.SPATIAL_3D,
            ):
                required_columns.update(("sx", "sy"))
            missing_columns = sorted(required_columns.difference(self.metadata.columns))
            if missing_columns:
                logger.warning(
                    f"Cached spot_metadata.csv is missing required columns: {missing_columns}. "
                    "Regenerating it."
                )
                self.metadata = _save_metadata(
                    self.ldsc_results, self.coords, self.report_data_dir
                )
            else:
                logger.info("Spot metadata already exists. Skipping.")
            if self.common_spots is None:
                self.common_spots = self.metadata["spot"].values
            return

        self.metadata = _save_metadata(self.ldsc_results, self.coords, self.report_data_dir)
        if self.common_spots is None:
            self.common_spots = self.metadata["spot"].values

    def _export_single_pcc_js(self, trait, df):
        """Export single trait PCC results to JS in gss_stats subfolder."""
        gss_js_dir = self.js_data_dir / "gss_stats"
        gss_js_dir.mkdir(exist_ok=True)

        data_json = df.to_json(orient="records")
        var_name = (
            f"GSMAP_GENE_TRAIT_CORRELATION_{''.join(c if c.isalnum() else '_' for c in trait)}"
        )
        js_content = f"window.{var_name} = {data_json};"
        _atomic_write_text(gss_js_dir / f"gene_trait_correlation_{trait}.js", js_content)

    def prepare_manhattan_data(self):
        """Prepare Manhattan data for all traits."""
        manhattan_dir = self.report_data_dir / "manhattan_data"
        manhattan_dir.mkdir(exist_ok=True)

        # Determine which traits need Manhattan data
        traits_to_run = []
        chrom_tick_positions = {}
        for trait in self.traits:
            csv_path = manhattan_dir / f"{trait}_manhattan.csv"
            js_path = self.js_data_dir / f"manhattan_{trait}.js"
            if not self._is_step_complete([csv_path, js_path]):
                traits_to_run.append(trait)
            else:
                ticks = _load_chrom_tick_positions(csv_path)
                if ticks:
                    chrom_tick_positions[trait] = ticks
                else:
                    logger.warning(
                        f"Cached Manhattan output for {trait} has no valid chromosome "
                        "coordinates. Regenerating it."
                    )
                    traits_to_run.append(trait)

        if not traits_to_run:
            logger.info("Manhattan data already complete for all traits. Skipping.")
            self.chrom_tick_positions = chrom_tick_positions
            return

        logger.info(f"Processing Manhattan data for {len(traits_to_run)} traits...")

        # Load weights
        logger.info(f"Loading weights from {self.report_config.snp_gene_weight_adata_path}")
        weight_adata = ad.read_h5ad(self.report_config.snp_gene_weight_adata_path)

        # Load gene ref
        genes_file = self.report_data_dir / "gene_list.csv"
        gene_names_ref = pd.read_csv(genes_file)["gene"].tolist() if genes_file.exists() else []

        gene_names_ref_set = set(gene_names_ref)
        target_gene_indices = np.flatnonzero(
            weight_adata.var_names.isin(gene_names_ref_set)
            & (weight_adata.var_names != "unmapped")
        )
        target_genes = weight_adata.var_names.to_numpy(dtype=str)[target_gene_indices]
        weight_matrix = weight_adata.X
        if sp.issparse(weight_matrix):
            weight_matrix = weight_matrix.tocsr(copy=False)

        weight_context = _ManhattanWeightContext(
            snp_index=pd.Index(weight_adata.obs_names.astype(str)),
            chromosome=pd.to_numeric(weight_adata.obs["CHR"], errors="coerce").to_numpy(),
            base_pair=pd.to_numeric(weight_adata.obs["BP"], errors="coerce").to_numpy(),
            matrix=weight_matrix,
            target_gene_indices=target_gene_indices,
            target_genes=target_genes,
        )

        max_workers = min(_report_worker_count(self.report_config, ceiling=4), len(traits_to_run))
        logger.info(
            f"Preparing Manhattan traits with {max_workers} worker(s); "
            f"GWAS parser=pyarrow with pandas fallback"
        )

        def prepare_trait(trait: str) -> tuple[str, dict]:
            started = time.perf_counter()
            trait_pcc_file = self.report_config.get_gene_diagnostic_info_save_path(trait)
            trait_pcc_df = pd.read_csv(trait_pcc_file) if trait_pcc_file.exists() else None
            ticks = _prepare_manhattan_for_trait(
                self.report_config,
                trait,
                weight_context,
                trait_pcc_df,
                manhattan_dir,
            )
            _export_single_manhattan_js(
                trait, manhattan_dir / f"{trait}_manhattan.csv", self.js_data_dir
            )
            logger.info(
                f"Manhattan trait complete: {trait} in {time.perf_counter() - started:.2f}s"
            )
            return trait, ticks

        failures: list[tuple[str, Exception]] = []
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="report-manhattan"
        ) as executor:
            futures = {executor.submit(prepare_trait, trait): trait for trait in traits_to_run}
            for future in as_completed(futures):
                trait = futures[future]
                try:
                    completed_trait, ticks = future.result()
                    chrom_tick_positions[completed_trait] = ticks
                except Exception as error:
                    failures.append((trait, error))
                    logger.exception(f"Failed to prepare Manhattan data for {trait}")

        del weight_adata, weight_matrix
        gc.collect()

        if failures:
            failed_traits = ", ".join(trait for trait, _ in failures)
            raise RuntimeError(
                f"Failed to prepare Manhattan data for {len(failures)}/"
                f"{len(traits_to_run)} traits: {failed_traits}"
            ) from failures[0][1]

        missing_outputs = []
        for trait in self.traits:
            csv_path = manhattan_dir / f"{trait}_manhattan.csv"
            js_path = self.js_data_dir / f"manhattan_{trait}.js"
            if not (csv_path.is_file() and csv_path.stat().st_size > 0):
                missing_outputs.append(str(csv_path))
            if not (js_path.is_file() and js_path.stat().st_size > 0):
                missing_outputs.append(str(js_path))
        if missing_outputs:
            raise RuntimeError(
                "Manhattan preparation completed with missing or empty outputs: "
                + ", ".join(missing_outputs)
            )

        self.chrom_tick_positions = chrom_tick_positions

    def render_static_plots(self):
        """Render static plots (LDSC, Annotation, Gene Diagnostics)."""
        dataset_type = getattr(self.report_config, "dataset_type", DatasetType.SPATIAL_2D)
        is_spatial = dataset_type in (DatasetType.SPATIAL_2D, DatasetType.SPATIAL_3D)

        if not is_spatial:
            logger.info("Skipping static plot rendering for non-spatial dataset type")
            return

        from .visualize import VisualizeRunner

        visualizer = VisualizeRunner(self.report_config)

        n_samples = len(self.sample_names)
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols

        obs_data = self.metadata.copy()
        if "sample" not in obs_data.columns:
            obs_data["sample"] = obs_data["sample_name"]

        if self.report_config.generate_multi_sample_plots:
            # Render LDSC plots
            spatial_plot_dir = self.web_report_dir / "spatial_plots"
            spatial_plot_dir.mkdir(exist_ok=True)
            for trait in self.traits:
                trait_plot_path = spatial_plot_dir / f"ldsc_{trait}.png"
                if not self._is_step_complete([trait_plot_path]):
                    logger.info(f"Rendering LDSC plot for {trait}...")
                    visualizer._create_single_trait_multi_sample_matplotlib_plot(
                        obs_ldsc_merged=obs_data,
                        trait_abbreviation=trait,
                        sample_name_list=self.sample_names,
                        output_png_path=trait_plot_path,
                        n_rows=n_rows,
                        n_cols=n_cols,
                        subplot_width_inches=5.0,
                    )

            # Render annotation plots
            anno_dir = self.web_report_dir / "annotation_plots"
            anno_dir.mkdir(exist_ok=True)
            for anno in self.report_config.annotation_list:
                anno_plot_path = anno_dir / f"anno_{anno}.png"
                if not self._is_step_complete([anno_plot_path]):
                    logger.info(f"Rendering plot for annotation: {anno}...")
                    fig = visualizer._create_multi_sample_annotation_plot(
                        obs_ldsc_merged=obs_data,
                        annotation=anno,
                        sample_names_list=self.sample_names,
                        output_dir=None,
                        n_rows=n_rows,
                        n_cols=n_cols,
                        fig_width=5 * n_cols,
                        fig_height=5 * n_rows,
                    )
                    import matplotlib.pyplot as plt

                    fig.savefig(anno_plot_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)

        # Gene diagnostic plots
        # Ensure base metadata exists (needed for common_spots and ldsc_results)
        if self.ldsc_results is None or self.coords is None:
            self.prepare_base_metadata()

        if self.metadata is None:
            self.prepare_spot_metadata()

        def load_gss_for_plots():
            if self.gss_adata is None:
                self.common_spots, self.gss_adata, self.gene_stats, self.analysis_spots = (
                    _load_gss_and_calculate_stats_base(
                        self.report_config,
                        self.ldsc_results,
                        self.coords,
                        self.report_data_dir,
                    )
                )
            return self.gss_adata

        _render_gene_diagnostic_plots_refactored(
            self.report_config,
            self.metadata,
            self.common_spots,
            self.gss_adata,
            n_rows,
            n_cols,
            self.web_report_dir,
            self.report_data_dir,
            self._is_step_complete,
            load_gss_for_plots,
        )

    def collect_cauchy_results(self):
        """Collect and save Cauchy combination results."""
        cauchy_file = self.report_data_dir / "cauchy_results.csv"
        cauchy_js = self.js_data_dir / "cauchy_results.js"
        # Since this combines all traits/annotations, we check for the final csv
        if self._is_step_complete([cauchy_file, cauchy_js]):
            # Validate all configured traits are covered in cached results
            df = pd.read_csv(cauchy_file)
            cached_traits = set(df["trait"].unique()) if "trait" in df.columns else set()
            cached_annotations = (
                set(df["annotation_name"].unique()) if "annotation_name" in df.columns else set()
            )
            missing_traits = sorted(set(self.traits) - cached_traits)
            missing_annotations = sorted(
                set(self.report_config.annotation_list) - cached_annotations
            )
            if not missing_traits and not missing_annotations:
                logger.info("Cauchy results already collected. Skipping.")
                return
            logger.warning(
                f"Cached Cauchy results are incomplete; missing traits={missing_traits}, "
                f"missing annotations={missing_annotations}. Regenerating them."
            )

        _collect_cauchy_results(self.report_config, self.traits, self.report_data_dir)
        _export_cauchy_js(self.report_data_dir, self.js_data_dir)

    def prepare_umap_data(self):
        """Prepare UMAP data from embeddings."""
        umap_file = self.report_data_dir / "umap_data.csv"
        umap_js = self.js_data_dir / "umap_data.js"
        if self._is_step_complete([umap_file, umap_js]):
            report_meta_path = self.web_report_dir / "report_meta.json"
            if report_meta_path.is_file():
                try:
                    with report_meta_path.open() as handle:
                        self.umap_info = json.load(handle).get("umap_info")
                except (OSError, ValueError) as error:
                    logger.warning(f"Could not restore cached UMAP metadata: {error}")
            if self.umap_info:
                logger.info("UMAP data already exists. Skipping.")
                return
            logger.warning(
                "Cached UMAP data are missing their metadata; regenerating the UMAP export."
            )

        self.umap_info = _prepare_umap_data(
            self.report_config, self.metadata, self.report_data_dir
        )
        _export_umap_js(self.report_data_dir, self.js_data_dir, {"umap_info": self.umap_info})

    def prepare_3d_visualization(self):
        """Prepare 3D visualization if applicable."""
        if not self.is_3d:
            self.spatial_3d_html = None
            return

        self.spatial_3d_html = _prepare_3d_visualization(
            self.report_config,
            self.metadata,
            self.traits,
            self.report_data_dir,
            self.web_report_dir,
        )

    def finalize_report(self):
        """Save report metadata and final JS assets."""
        # 1. Save standard report metadata
        meta = _save_report_metadata(
            self.report_config,
            self.traits,
            self.sample_names,
            self.web_report_dir,
            getattr(self, "chrom_tick_positions", None),
            getattr(self, "umap_info", None),
            is_3d=self.is_3d,
            spatial_3d_path=getattr(self, "spatial_3d_html", None),
        )

        # 2. Copy JS libraries
        _copy_js_assets(self.web_report_dir)

        # 3. Export per-sample spatial JS (highly important for performance)
        if not self.force_re_run and _per_sample_spatial_outputs_current(
            self.report_data_dir, self.js_data_dir, meta
        ):
            logger.info("Per-sample spatial JS is current. Skipping export.")
        else:
            _export_per_sample_spatial_js(self.report_data_dir, self.js_data_dir, meta)
        _export_report_meta_js(self.web_report_dir, self.js_data_dir, meta)


def _render_single_sample_gene_plot_task(task_data: dict):
    """
    Parallel task to render single-sample Expression or GSS plot.
    Creates a single plot for one gene in one sample.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    gene = task_data["gene"]
    sample_name = task_data["sample_name"]
    plot_type = task_data["plot_type"]
    coords = task_data["coords"]
    values = task_data["values"]
    output_path = Path(task_data["output_path"])
    fig_width = task_data.get("fig_width", 6.0)
    dpi = task_data.get("dpi", 150)

    if coords is None or values is None or len(coords) == 0:
        raise ValueError(f"No data for {gene} in {sample_name}")

    custom_colors = [
        "#313695",
        "#4575b4",
        "#74add1",
        "#abd9e9",
        "#e0f3f8",
        "#fee090",
        "#fdae61",
        "#f46d43",
        "#d73027",
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", custom_colors)

    fig, ax = plt.subplots(figsize=(fig_width, fig_width))

    from gsMap.report.visualize import estimate_matplotlib_scatter_marker_size

    point_size = estimate_matplotlib_scatter_marker_size(ax, coords)
    point_size = min(max(point_size, 1), 200)

    vmin = 0
    with np.errstate(all="ignore"):
        non_nan_values = values[np.isfinite(values)] if len(values) > 0 else np.array([])
        vmax = np.percentile(non_nan_values, 99.5) if len(non_nan_values) > 0 else 1.0

    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap=custom_cmap,
        s=point_size,
        vmin=vmin,
        vmax=vmax,
        marker="o",
        edgecolors="none",
        rasterized=True,
    )

    if task_data.get("plot_origin", "upper") == "upper":
        ax.invert_yaxis()

    ax.axis("off")
    ax.set_aspect("equal")

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Expression" if plot_type == "exp" else "GSS", fontsize=10)
    ax.set_title(
        f"{gene} - {'Expression' if plot_type == 'exp' else 'GSS'}",
        fontsize=12,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.png")
    temporary.unlink(missing_ok=True)
    try:
        fig.savefig(temporary, dpi=dpi, bbox_inches="tight", facecolor="white")
        os.replace(temporary, output_path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    return True


def _render_single_sample_gene_pair_task(task_data: dict):
    """Render expression and GSS once for a unique sample-gene pair."""

    common = {
        "gene": task_data["gene"],
        "sample_name": task_data["sample_name"],
        "coords": task_data["coords"],
        "plot_origin": task_data.get("plot_origin", "upper"),
        "fig_width": task_data.get("fig_width", 6.0),
        "dpi": task_data.get("dpi", 150),
    }
    _render_single_sample_gene_plot_task(
        {
            **common,
            "plot_type": "exp",
            "values": task_data["exp_values"],
            "output_path": task_data["exp_output_path"],
        }
    )
    _render_single_sample_gene_plot_task(
        {
            **common,
            "plot_type": "gss",
            "values": task_data["gss_values"],
            "output_path": task_data["gss_output_path"],
        }
    )
    gc.collect()
    return True


# =============================================================================
# UMAP Calculation Functions
# =============================================================================


def _detect_z_axis(coords_3d: np.ndarray, sample_names: pd.Series) -> int:
    """
    Detect which dimension is the Z axis (stacking dimension) in 3D coordinates.

    The Z axis is identified as the dimension with the least within-sample variance,
    since samples are stacked along Z and should have minimal variation in that dimension.

    Args:
        coords_3d: 3D coordinates array (n_spots, 3)
        sample_names: Series of sample names for each spot

    Returns:
        Index of the Z axis (0, 1, or 2)
    """
    # Calculate within-sample variance for each dimension
    within_sample_vars = []

    for dim in range(3):
        dim_values = coords_3d[:, dim]

        # Calculate variance within each sample
        sample_vars = []
        for sample in sample_names.unique():
            mask = sample_names == sample
            sample_values = dim_values[mask]
            if len(sample_values) > 1:
                sample_vars.append(np.var(sample_values))

        # Average within-sample variance for this dimension
        avg_within_var = np.mean(sample_vars) if sample_vars else 0
        within_sample_vars.append(avg_within_var)

    # Z axis has the least within-sample variance
    z_axis = int(np.argmin(within_sample_vars))
    logger.info(
        f"Detected Z axis: dimension {z_axis} (within-sample variances: {within_sample_vars})"
    )

    return z_axis


def _reorder_coords_for_3d(coords_3d: np.ndarray, z_axis: int) -> tuple[np.ndarray, list[str]]:
    """
    Reorder 3D coordinates so that Z axis is the last dimension.

    Returns:
        Tuple of (reordered coords, column names)
    """
    # Create axis order: put z_axis last
    axis_order = [i for i in range(3) if i != z_axis] + [z_axis]
    reordered = coords_3d[:, axis_order]

    # Column names reflect the reordering
    col_names = ["3d_x", "3d_y", "3d_z"]

    return reordered, col_names


def _stratified_subsample(
    spot_names: np.ndarray, sample_names: pd.Series, n_samples: int, random_state: int = 42
) -> np.ndarray:
    """
    Stratified subsampling that ensures representation from all samples.

    Args:
        spot_names: Array of spot identifiers
        sample_names: Series mapping spot names to sample names
        n_samples: Target number of samples
        random_state: Random seed for reproducibility

    Returns:
        Array of selected spot names
    """
    np.random.seed(random_state)

    # Get sample counts
    sample_counts = sample_names.value_counts()
    n_samples_total = len(spot_names)

    if n_samples >= n_samples_total:
        return spot_names

    # Calculate proportional samples per group
    selected_spots = []
    for sample_name, count in sample_counts.items():
        sample_spots = spot_names[sample_names == sample_name]
        # Proportional allocation
        n_select = max(1, int(np.ceil(n_samples * count / n_samples_total)))
        n_select = min(n_select, len(sample_spots))

        selected = np.random.choice(sample_spots, n_select, replace=False)
        selected_spots.extend(selected)

    # If we have too many, randomly remove some
    selected_spots = np.array(selected_spots)
    if len(selected_spots) > n_samples:
        selected_spots = np.random.choice(selected_spots, n_samples, replace=False)

    return selected_spots


def _calculate_umap_from_embeddings(
    adata: ad.AnnData,
    embedding_key: str,
    umap_key: str = "X_umap",
) -> np.ndarray:
    if umap_key in adata.obsm:
        logger.info(f"Using pre-computed '{umap_key}' from adata.obsm, skipping neighbors + UMAP.")
        return adata.obsm[umap_key]

    logger.info(f"Calculating UMAP for {embedding_key} with {adata.n_obs} spots using scanpy...")
    sc.pp.neighbors(adata, use_rep=embedding_key)
    sc.tl.umap(adata)

    return adata.obsm["X_umap"]


def _prepare_umap_data(
    report_config: QuickModeConfig, metadata: pd.DataFrame, report_dir: Path
) -> dict | None:
    """
    Prepare UMAP data from cell and niche embeddings.

    Returns dict with umap_cell, umap_niche (optional), and metadata for visualization.
    """
    logger.info("Preparing UMAP data from embeddings...")

    # Load concatenated adata
    adata_path = report_config.concatenated_latent_adata_path
    if not adata_path.exists():
        logger.warning(f"Concatenated adata not found at {adata_path}")
        return None

    adata = ad.read_h5ad(adata_path)

    # Get embedding keys from config
    cell_emb_key = getattr(report_config, "latent_representation_cell", "emb_cell")
    niche_emb_key = getattr(report_config, "latent_representation_niche", None)

    # Check if embeddings exist
    if cell_emb_key not in adata.obsm:
        logger.warning(f"Cell embedding '{cell_emb_key}' not found in adata.obsm")
        return None

    has_niche = niche_emb_key is not None and niche_emb_key in adata.obsm

    # Stratified subsampling
    n_subsample = getattr(report_config, "downsampling_n_spots", 20000)
    spot_names = adata.obs_names.values
    sample_names = adata.obs["sample_name"]

    if len(spot_names) > n_subsample:
        logger.info(f"Stratified subsampling from {len(spot_names)} to {n_subsample} spots...")
        selected_spots = _stratified_subsample(spot_names, sample_names, n_subsample)
        adata_sub = adata[selected_spots].copy()
    else:
        adata_sub = adata.copy()
        selected_spots = spot_names

    logger.info(f"Using {len(selected_spots)} spots for UMAP calculation")

    # Calculate UMAPs
    umap_cell = _calculate_umap_from_embeddings(adata_sub, cell_emb_key, umap_key="X_umap")

    # Estimate point size for UMAP cell
    _, point_size_cell = estimate_plotly_point_size(umap_cell, DEFAULT_PIXEL_WIDTH=600)

    umap_niche = None
    point_size_niche = None
    if has_niche:
        umap_niche = _calculate_umap_from_embeddings(
            adata_sub, niche_emb_key, umap_key="X_umap_niche"
        )
        _, point_size_niche = estimate_plotly_point_size(umap_niche, DEFAULT_PIXEL_WIDTH=600)

    # Prepare metadata for the subsampled spots
    umap_metadata = pd.DataFrame(
        {
            "spot": adata_sub.obs_names,
            "sample_name": adata_sub.obs["sample_name"].values,
            "umap_cell_x": umap_cell[:, 0],
            "umap_cell_y": umap_cell[:, 1],
        }
    )

    if umap_niche is not None:
        umap_metadata["umap_niche_x"] = umap_niche[:, 0]
        umap_metadata["umap_niche_y"] = umap_niche[:, 1]

    # Add all annotation columns
    for anno in report_config.annotation_list:
        if anno in adata_sub.obs.columns:
            # Fill NaN with 'NaN' and convert to string to avoid sorting errors
            umap_metadata[anno] = adata_sub.obs[anno].astype(str).fillna("NaN").values

    # Add trait -log10(p) values from metadata if available (vectorized join)
    traits = report_config.trait_name_list
    available_traits = [t for t in traits if t in metadata.columns]
    if available_traits:
        # Prepare trait data: ensure 'spot' is a column and is string type
        trait_data = metadata[available_traits].copy()
        trait_data["spot"] = metadata["spot"].astype(str)
        umap_metadata = umap_metadata.merge(trait_data, on="spot", how="left")

        # Keep 1 decimal precision for trait values and handle non-numeric data
        for trait in available_traits:
            if trait in umap_metadata.columns:
                # Convert to numeric in case values were strings, then round
                umap_metadata[trait] = pd.to_numeric(umap_metadata[trait], errors="coerce").round(
                    1
                )
            logger.info(f"Added trait {trait} to UMAP data with 1 decimal precision")

    # Save to CSV
    _atomic_to_csv(umap_metadata, report_dir / "umap_data.csv", index=False)
    logger.info(f"UMAP data saved with {len(umap_metadata)} points")

    return {
        "has_niche": has_niche,
        "cell_emb_key": cell_emb_key,
        "niche_emb_key": niche_emb_key,
        "n_points": len(umap_metadata),
        "point_size_cell": float(point_size_cell),
        "point_size_niche": float(point_size_niche) if point_size_niche is not None else None,
        "default_opacity": 0.7,
    }


def _prepare_3d_visualization(
    report_config: QuickModeConfig,
    metadata: pd.DataFrame,
    traits: list[str],
    report_data_dir: Path,
    web_report_dir: Path,
) -> str | None:
    """
    Create 3D visualization widget using spatialvista and save as HTML.

    Args:
        report_config: QuickModeConfig with annotation_list and other settings
        metadata: DataFrame with 3d_x, 3d_y, 3d_z coordinates and annotations
        traits: List of trait names (for continuous values)
        report_data_dir: Directory for data files (h5ad)
        web_report_dir: Directory for web report files (HTML)

    Returns:
        Path to saved HTML file, or None if failed
    """
    try:
        import pyvista  # noqa: F401
    except ImportError:
        logger.warning("pyvista not installed. Skipping 3D visualization.")
        return None

    # Check if we have 3D coordinates
    if (
        "3d_x" not in metadata.columns
        or "3d_y" not in metadata.columns
        or "3d_z" not in metadata.columns
    ):
        logger.warning("3D coordinates not found in metadata. Skipping 3D visualization.")
        return None

    logger.info("Creating 3D visualization...")

    # 1. Create adata_vis using ALL spots
    n_spots_all = len(metadata)
    adata_vis_full = ad.AnnData(
        X=sp.csr_matrix((n_spots_all, 1), dtype=np.float32), obs=metadata.set_index("spot")
    )
    # Set coordinates
    adata_vis_full.obsm["spatial_3d"] = metadata[["3d_x", "3d_y", "3d_z"]].values
    adata_vis_full.obsm["spatial"] = adata_vis_full.obsm["spatial_3d"]
    if "sx" in metadata.columns and "sy" in metadata.columns:
        adata_vis_full.obsm["spatial_2d"] = metadata[["sx", "sy"]].values

    # Ensure trait columns are float32
    adata_vis_full.obs = adata_vis_full.obs.astype(
        {trait: np.float32 for trait in traits if trait in adata_vis_full.obs.columns}
    )

    # Create 3D visualization directories
    three_d_data_dir = report_data_dir / "spatial_3d"
    three_d_data_dir.mkdir(exist_ok=True)
    three_d_web_dir = web_report_dir / "spatial_3d"
    three_d_web_dir.mkdir(exist_ok=True)

    # 2. Save the full adata_vis to data directory
    h5ad_path = three_d_data_dir / "spatial_3d.h5ad"
    adata_vis_full.write_h5ad(h5ad_path)
    logger.info(f"Full 3D visualization data saved to {h5ad_path}")

    # 3. Stratified subsampling for HTML visualization (limit to reasonable size)
    n_max_points = getattr(report_config, "downsampling_n_spots_3d", 1000000)
    if len(metadata) > n_max_points:
        sample_names = metadata["sample_name"]
        selected_idx = _stratified_subsample(metadata.index.values, sample_names, n_max_points)
        adata_vis = adata_vis_full[selected_idx].copy()
        logger.info(f"Subsampled to {len(adata_vis)} spots for HTML 3D visualization")
    else:
        adata_vis = adata_vis_full.copy()

    # Add traits as continuous values
    continuous_cols = traits

    # Add annotations (categorical)
    annotation_cols = []
    for anno in report_config.annotation_list:
        if anno in adata_vis.obs.columns:
            annotation_cols.append(anno)

    logger.info(f"3D visualization: annotations={annotation_cols}, traits={continuous_cols}")

    # Create 3D plots
    try:
        from gsMap.report.visualize import _create_color_map

        # P-value color scale (Red-Blue)
        P_COLOR = [
            "#313695",
            "#4575b4",
            "#74add1",
            "#fee090",
            "#fdae61",
            "#f46d43",
            "#d73027",
            "#a50026",
        ]

        # Calculate appropriate point size based on coverage ratio (20-30%)
        # Formula: S = sqrt(W * H * k / N)
        win_w, win_h = 1200, 1008
        k_coverage = 0.25
        n_points = len(adata_vis)
        point_size = np.sqrt((win_w * win_h * k_coverage) / n_points)
        point_size = max(0.5, min(5.0, point_size))  # Clamp between 0.5 and 5.0
        logger.info(f"Estimated 3D point size: {point_size:.2f} for {n_points} spots")

        # Shared plotting settings
        text_kwargs = dict(text_font_size=15, text_loc="upper_edge")

        # 1. Save all Annotation 3D plots
        categorical_legend_kwargs = dict(categorical_legend_loc="center right")
        for anno in annotation_cols:
            logger.info(f"Generating 3D plot for annotation: {anno}")

            # Use same colormap logic as distribution plots
            if pd.api.types.is_numeric_dtype(adata_vis.obs[anno]):
                color_map = P_COLOR
            else:
                unique_vals = adata_vis.obs[anno].unique()
                color_map = _create_color_map(unique_vals, hex=True, rng=42)

            safe_anno = "".join(c if c.isalnum() else "_" for c in anno)
            anno_plot_name = three_d_web_dir / f"spatial_3d_anno_{safe_anno}"

            plotter_anno = three_d_plot(
                adata=adata_vis,
                spatial_key="spatial",
                keys=[anno],
                cmaps=[color_map],
                point_size=point_size,
                texts=[anno],
                off_screen=True,
                jupyter=False,
                background="white",
                legend_kwargs=categorical_legend_kwargs,
                text_kwargs=text_kwargs,
                window_size=(win_w, win_h),
            )
            three_d_plot_save(plotter_anno, filename=str(anno_plot_name))

        # 2. Save each Trait 3D plot
        legend_kwargs = dict(scalar_bar_title_size=30, scalar_bar_label_size=30, fmt="%.1e")

        for trait in continuous_cols:
            logger.info(f"Generating 3D plot for trait: {trait}")

            # Calculate opacity based on LogP (exponential scaling)
            trait_values = adata_vis.obs[trait].fillna(0).values
            bins = np.linspace(trait_values.min(), trait_values.max(), 5)
            # Handle case where min == max to avoid bins error
            if bins[0] == bins[-1]:
                opacity_show = 1.0
            else:
                alpha = np.exp(np.linspace(0.1, 1.0, num=(len(bins) - 1))) - 1
                alpha = alpha / max(alpha)
                opacity_show = pd.cut(
                    trait_values, bins=bins, labels=alpha, include_lowest=True
                ).tolist()

            # Set the clim (median of top 20 spots)
            sorted_vals = np.sort(trait_values)[::-1]
            max_v = (
                np.round(np.median(sorted_vals[0:20]))
                if len(sorted_vals) >= 20
                else sorted_vals[0]
            )
            if max_v <= 0:
                max_v = 1.0

            safe_trait = "".join(c if c.isalnum() else "_" for c in trait)
            trait_plot_name = three_d_web_dir / f"spatial_3d_trait_{safe_trait}"

            plotter_trait = three_d_plot(
                adata=adata_vis,
                spatial_key="spatial",
                keys=[trait],
                cmaps=[P_COLOR],
                point_size=point_size,
                opacity=opacity_show,
                clim=[0, max_v],
                scalar_bar_titles=["-log10(p)"],
                texts=[trait],
                off_screen=True,
                jupyter=False,
                background="white",
                legend_kwargs=legend_kwargs,
                text_kwargs=text_kwargs,
                window_size=(win_w, win_h),
            )
            three_d_plot_save(plotter_trait, filename=str(trait_plot_name))

        # Return the relative path of the first available plot
        if annotation_cols:
            safe_first = "".join(c if c.isalnum() else "_" for c in annotation_cols[0])
            return f"spatial_3d/spatial_3d_anno_{safe_first}.html"
        elif continuous_cols:
            safe_first_trait = "".join(c if c.isalnum() else "_" for c in continuous_cols[0])
            return f"spatial_3d/spatial_3d_trait_{safe_first_trait}.html"

        return None

    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to create 3D visualization: {e}")
        import traceback

        traceback.print_exc()
        return None


# =============================================================================
# Data Loading Functions
# =============================================================================


def _load_ldsc_results(
    report_config: QuickModeConfig,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load combined LDSC results and extract traits/samples."""
    logger.info(f"Loading combined LDSC results from {report_config.ldsc_combined_parquet_path}")

    if not report_config.ldsc_combined_parquet_path.exists():
        raise FileNotFoundError(
            f"Combined LDSC parquet not found at {report_config.ldsc_combined_parquet_path}. "
            "Please run Cauchy combination first."
        )

    ldsc_results = pd.read_parquet(report_config.ldsc_combined_parquet_path)
    if "spot" in ldsc_results.columns:
        ldsc_results.set_index("spot", inplace=True)

    assert "sample_name" in ldsc_results.columns, (
        "LDSC combined results must have 'sample_name' column."
    )

    traits = report_config.trait_name_list

    # Explicitly use sample order from report_config.sample_h5ad_dict
    sample_names = list(report_config.sample_h5ad_dict.keys())
    actual_samples_in_data = set(ldsc_results["sample_name"].unique())

    # Assert all samples in config are actually in the data
    missing = [s for s in sample_names if s not in actual_samples_in_data]
    assert not missing, f"Samples {missing} in config are not present in LDSC results."

    return ldsc_results, traits, sample_names


def _load_coordinates(report_config: QuickModeConfig) -> tuple[pd.DataFrame, bool, int | None]:
    """
    Load spatial coordinates from concatenated adata.

    Returns:
        Tuple of (coords DataFrame, is_3d flag, z_axis index if 3D)
    """
    from gsMap.config import DatasetType

    logger.info(f"Loading coordinates from {report_config.concatenated_latent_adata_path}")

    adata_concat = ad.read_h5ad(report_config.concatenated_latent_adata_path, backed="r")
    sample_info = adata_concat.obs[["sample_name"]]

    # For scRNA-seq, spatial_key is None — no spatial coordinates available
    if report_config.spatial_key is None:
        logger.info("scRNA-seq dataset: spatial_key is None, skipping spatial coordinate loading")
        coords = pd.DataFrame(index=adata_concat.obs_names)
        coords = pd.concat([coords, sample_info], axis=1)
        if adata_concat.isbacked:
            adata_concat.file.close()
        return coords, False, None

    assert report_config.spatial_key in adata_concat.obsm, (
        f"spatial_key '{report_config.spatial_key}' not found in adata.obsm"
    )
    coords_data = adata_concat.obsm[report_config.spatial_key]

    # Check if dataset type is 3D
    is_3d_type = (
        hasattr(report_config, "dataset_type")
        and report_config.dataset_type == DatasetType.SPATIAL_3D
    )
    has_3d_coords = coords_data.shape[1] >= 3

    z_axis = None
    if is_3d_type and has_3d_coords:
        # True 3D coordinates
        logger.info("Detected 3D spatial coordinates")
        coords_3d = coords_data[:, :3]

        # Detect Z axis
        z_axis = _detect_z_axis(coords_3d, adata_concat.obs["sample_name"])

        # Reorder coordinates so Z is last
        reordered_coords, col_names = _reorder_coords_for_3d(coords_3d, z_axis)
        coords = pd.DataFrame(reordered_coords, columns=col_names, index=adata_concat.obs_names)

        # Also keep 2D coords for compatibility (use x and y from reordered)
        coords["sx"] = coords["3d_x"]
        coords["sy"] = coords["3d_y"]

    elif is_3d_type and not has_3d_coords:
        # 3D dataset type but only 2D coordinates - create pseudo Z axis
        logger.info(
            "3D dataset type with 2D coordinates - creating pseudo Z axis based on sample_name"
        )
        coords_2d = coords_data[:, :2]
        coords = pd.DataFrame(coords_2d, columns=["sx", "sy"], index=adata_concat.obs_names)

        # Calculate the span for pseudo Z axis
        sx_span = coords["sx"].max() - coords["sx"].min()
        sy_span = coords["sy"].max() - coords["sy"].min()
        z_span = max(sx_span, sy_span)

        # Assign Z values based on sample order from report_config
        sample_names_all = adata_concat.obs["sample_name"]
        actual_samples_in_data = set(sample_names_all.unique())

        # Explicitly use report_config order
        ordered_samples = list(report_config.sample_h5ad_dict.keys())
        missing = [s for s in ordered_samples if s not in actual_samples_in_data]
        assert not missing, f"Samples {missing} in config are not present in coordinates data."

        n_samples = len(ordered_samples)

        # Create evenly spaced Z values for each sample in the specific order
        if n_samples > 1:
            z_values_per_sample = {
                sample: z_span * i / (n_samples - 1) for i, sample in enumerate(ordered_samples)
            }
        else:
            z_values_per_sample = {ordered_samples[0]: 0.0}

        # Assign Z values to each spot
        pseudo_z = sample_names_all.map(z_values_per_sample).values

        coords["3d_x"] = coords["sx"]
        coords["3d_y"] = coords["sy"]
        coords["3d_z"] = pseudo_z
        z_axis = 2  # Pseudo Z is always the last dimension

        logger.info(f"Created pseudo Z axis with span {z_span:.2f} for {n_samples} samples")

    else:
        # 2D coordinates (non-3D dataset type)
        coords_2d = coords_data[:, :2]
        coords = pd.DataFrame(coords_2d, columns=["sx", "sy"], index=adata_concat.obs_names)

    coords = pd.concat([coords, sample_info], axis=1)

    # Close backed file
    if adata_concat.isbacked:
        adata_concat.file.close()

    # Return is_3d as True if dataset type is 3D (regardless of coord dimensions)
    return coords, is_3d_type, z_axis


def _load_gss_and_calculate_stats_base(
    report_config: QuickModeConfig,
    ldsc_results: pd.DataFrame,
    coords: pd.DataFrame,
    report_dir: Path,
) -> tuple[np.ndarray, ad.AnnData, pd.DataFrame, np.ndarray]:
    """Load GSS data, calculate general stats, and return analysis results."""
    logger.info("Loading GSS data...")

    gss_adata = load_marker_scores_memmap_format(report_config)
    common_spots = np.intersect1d(gss_adata.obs_names, ldsc_results.index)
    logger.info(f"Common spots (gss & ldsc): {len(common_spots)}")
    assert len(common_spots) > 0, "No common spots found between GSS and LDSC results."

    # Stratified subsample if requested
    analysis_spots = common_spots
    if (
        report_config.downsampling_n_spots_pcc
        and len(common_spots) > report_config.downsampling_n_spots_pcc
    ):
        sample_names = ldsc_results.loc[common_spots, "sample_name"]
        analysis_spots = _stratified_subsample(
            common_spots, sample_names, report_config.downsampling_n_spots_pcc
        )
        logger.info(f"Stratified subsampled to {len(analysis_spots)} spots for PCC calculation.")

    # Filter to high expression genes
    exp_frac = pd.read_parquet(report_config.mean_frac_path)
    high_expr_genes = exp_frac[exp_frac["frac"] > 0.01].index.tolist()
    logger.info(f"Using {len(high_expr_genes)} high expression genes for PCC calculation.")

    gss_adata_sub = gss_adata[analysis_spots, high_expr_genes]
    gss_matrix = gss_adata_sub.X
    gene_names = gss_adata_sub.var_names.tolist()
    _atomic_to_csv(pd.DataFrame({"gene": gene_names}), report_dir / "gene_list.csv", index=False)

    # Calculate gene annotation stats
    adata_concat = ad.read_h5ad(report_config.concatenated_latent_adata_path, backed="r")
    anno_col = report_config.annotation_list[0]
    annotations = adata_concat.obs.loc[analysis_spots, anno_col]

    if hasattr(gss_matrix, "toarray"):
        gss_matrix = gss_matrix.toarray()

    gss_df_temp = pd.DataFrame(gss_matrix, index=analysis_spots, columns=gene_names)
    grouped_gss = gss_df_temp.groupby(annotations).median()

    gene_stats = pd.DataFrame(
        {
            "gene": grouped_gss.idxmax().index,
            "Annotation": grouped_gss.idxmax().values,
            "Median_GSS": grouped_gss.max().values,
        }
    )
    gene_stats.dropna(subset=["Median_GSS"], inplace=True)

    return common_spots, gss_adata, gene_stats, analysis_spots


def _calculate_pcc_for_single_trait_fast(
    trait: str,
    ldsc_results: pd.DataFrame,
    analysis_spots: np.ndarray,
    gss_centered: np.ndarray,
    gss_ssq: np.ndarray,
    gene_names: list[str],
    gene_stats: pd.DataFrame,
    report_config: QuickModeConfig,
    report_dir: Path,
    gss_dir: Path,
) -> pd.DataFrame | None:
    """Calculate PCC for a single trait using pre-calculated centered GSS matrix."""
    if trait not in ldsc_results.columns:
        logger.warning(
            f"Trait {trait} not found in LDSC combined results. Skipping PCC calculation."
        )
        return None

    def fast_corr(centered_matrix, ssq_matrix, vector):
        v_centered = vector - vector.mean()
        numerator = np.dot(v_centered, centered_matrix)
        denominator = np.sqrt(np.sum(v_centered**2) * ssq_matrix)
        return numerator / (denominator + 1e-12)

    logger.info(f"Processing PCC for trait: {trait}")

    logp_vec = ldsc_results.loc[analysis_spots, trait].values.astype(np.float32)
    assert not np.any(np.isnan(logp_vec)), f"NaN values found in LDSC results for trait {trait}."
    pccs = fast_corr(gss_centered, gss_ssq, logp_vec)

    trait_pcc = pd.DataFrame({"gene": gene_names, "PCC": pccs, "trait": trait})

    if gene_stats is not None:
        trait_pcc = trait_pcc.merge(gene_stats, on="gene", how="left")

    trait_pcc_sorted = trait_pcc.sort_values("PCC", ascending=False)

    # Save to gss_stats subfolder
    # Save result CSV
    diag_info_path = report_config.get_gene_diagnostic_info_save_path(trait)
    _atomic_to_csv(trait_pcc_sorted, diag_info_path, index=False)

    return trait_pcc_sorted


def _export_single_manhattan_js(trait: str, csv_path: Path, js_data_dir: Path):
    """Export single trait Manhattan data to JS."""
    df = pd.read_csv(csv_path)
    if "P" in df.columns and "logp" not in df.columns:
        df["logp"] = -np.log10(df["P"] + 1e-300)

    data_struct = {
        "x": df["BP_cum"].tolist(),
        "y": df["logp"].tolist(),
        "gene": df["GENE"].fillna("").tolist(),
        "chr": df["CHR"].astype(int).tolist(),
        "snp": df["SNP"].tolist() if "SNP" in df.columns else [],
        "is_top": df["is_top_pcc"].astype(int).tolist() if "is_top_pcc" in df.columns else [],
        "bp": df["BP"].astype(int).tolist() if "BP" in df.columns else [],
    }

    json_str = json.dumps(data_struct, separators=(",", ":"))
    safe_trait = "".join(c if c.isalnum() else "_" for c in trait)
    js_content = f"window.GSMAP_MANHATTAN_{safe_trait} = {json_str};"

    _atomic_write_text(js_data_dir / f"manhattan_{trait}.js", js_content)


# =============================================================================
# Data Preparation Functions
# =============================================================================


def _save_metadata(
    ldsc_results: pd.DataFrame, coords: pd.DataFrame, report_dir: Path
) -> pd.DataFrame:
    """Save metadata and coordinates to CSV."""
    logger.info("Saving metadata and coordinates...")

    common_indices = ldsc_results.index.intersection(coords.index)
    ldsc_subset = ldsc_results.loc[common_indices]
    cols_to_use = ldsc_subset.columns.difference(coords.columns)
    metadata = pd.concat([coords.loc[common_indices], ldsc_subset[cols_to_use]], axis=1)

    metadata.index.name = "spot"
    metadata = metadata.reset_index()
    metadata = metadata.loc[:, ~metadata.columns.duplicated()]
    _atomic_to_csv(metadata, report_dir / "spot_metadata.csv", index=False)

    return metadata


def _prepare_manhattan_for_trait(
    report_config: QuickModeConfig,
    trait: str,
    weight_context: _ManhattanWeightContext,
    trait_pcc_df: pd.DataFrame | None,
    manhattan_dir: Path,
) -> dict:
    """Prepare Manhattan data for a single trait."""
    from gsMap.utils.manhattan_plot import _ManhattanPlot

    sumstats_file = report_config.sumstats_config_dict.get(trait)
    if not sumstats_file or not Path(sumstats_file).exists():
        return {}

    logger.info(f"Processing Manhattan for {trait}...")
    gwas_data = load_gwas_data(
        sumstats_file,
        calculate_p=False,
    )
    if gwas_data.empty:
        raise ValueError(f"GWAS file contains no rows: {sumstats_file}")

    if gwas_data["SNP"].duplicated().any():
        duplicate_count = int(gwas_data["SNP"].duplicated().sum())
        logger.warning(
            f"Dropping {duplicate_count:,} duplicate SNP rows before Manhattan processing for "
            f"{trait}"
        )
        gwas_data = gwas_data.drop_duplicates("SNP", keep="first")

    gwas_snp_index = pd.Index(gwas_data["SNP"])
    gwas_positions = gwas_snp_index.get_indexer(weight_context.snp_index)
    common_mask = gwas_positions >= 0
    weight_rows = np.flatnonzero(common_mask)
    if weight_rows.size == 0:
        raise ValueError(f"No common SNPs between GWAS and weight data for {trait}")

    selected_gwas_positions = gwas_positions[common_mask]
    z_values = pd.to_numeric(gwas_data["Z"], errors="coerce").to_numpy()[selected_gwas_positions]
    finite_mask = np.isfinite(z_values)
    weight_rows = weight_rows[finite_mask]
    selected_gwas_positions = selected_gwas_positions[finite_mask]
    z_values = z_values[finite_mask]
    if weight_rows.size == 0:
        raise ValueError(f"No finite GWAS Z-scores remain for {trait}")

    gwas_subset = gwas_data.iloc[selected_gwas_positions].reset_index(drop=True).copy()
    gwas_subset = gwas_subset.drop(columns=[c for c in ["CHR", "BP", "P"] if c in gwas_subset])
    # The legacy merge placed the join key first even when a source file stored SNP
    # later in its schema. Preserve that stable report CSV contract.
    gwas_subset = gwas_subset[["SNP", *[c for c in gwas_subset.columns if c != "SNP"]]]
    gwas_subset["SNP"] = weight_context.snp_index.to_numpy()[weight_rows]
    gwas_subset["Z"] = z_values
    gwas_subset = convert_z_to_p(gwas_subset)
    gwas_subset["CHR"] = weight_context.chromosome[weight_rows]
    gwas_subset["BP"] = weight_context.base_pair[weight_rows]
    gwas_subset["_weight_row"] = weight_rows

    snps2plot_ids = filter_snps(gwas_subset.sort_values("P"), SUBSAMPLE_SNP_NUMBER=50000)
    gwas_plot_data = gwas_subset[gwas_subset["SNP"].isin(snps2plot_ids)].copy()

    gwas_plot_data["CHR"] = pd.to_numeric(gwas_plot_data["CHR"], errors="coerce")
    gwas_plot_data["BP"] = pd.to_numeric(gwas_plot_data["BP"], errors="coerce")
    gwas_plot_data = gwas_plot_data.dropna(subset=["CHR", "BP"])

    # Gene assignment
    if weight_context.target_genes.size:
        selected_weight_rows = gwas_plot_data["_weight_row"].to_numpy(dtype=np.int64)
        weights_matrix = weight_context.matrix[selected_weight_rows][
            :, weight_context.target_gene_indices
        ]

        if sp.issparse(weights_matrix):
            max_idx = np.array(weights_matrix.argmax(axis=1)).ravel()
            max_val = np.array(weights_matrix.max(axis=1).toarray()).ravel()
        else:
            max_idx = np.argmax(weights_matrix, axis=1)
            max_val = np.max(weights_matrix, axis=1)

        gene_map = np.where(max_val > 1, weight_context.target_genes[max_idx], "None")
        gwas_plot_data["GENE"] = gene_map

        if trait_pcc_df is not None:
            top_n = getattr(report_config, "top_corr_genes", 50)
            trait_top_genes = (
                trait_pcc_df.sort_values("PCC", ascending=False).head(top_n)["gene"].tolist()
            )
            gwas_plot_data["is_top_pcc"] = gwas_plot_data["GENE"].isin(trait_top_genes)
        else:
            gwas_plot_data["is_top_pcc"] = False

    if "GENE" not in gwas_plot_data.columns:
        gwas_plot_data["GENE"] = "None"
    if "is_top_pcc" not in gwas_plot_data.columns:
        gwas_plot_data["is_top_pcc"] = False

    gwas_plot_data = gwas_plot_data.drop(columns="_weight_row")

    # Calculate cumulative positions
    chrom_ticks = {}
    try:
        mp_helper = _ManhattanPlot(gwas_plot_data)
        gwas_plot_data["BP_cum"] = mp_helper.data["POSITION"].values
        gwas_plot_data["CHR_INDEX"] = mp_helper.data["INDEX"].values

        chrom_groups = gwas_plot_data.groupby("CHR")["BP_cum"]
        chrom_ticks = {int(chrom): float(positions.median()) for chrom, positions in chrom_groups}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to calculate Manhattan coordinates: {e}")
        gwas_plot_data["BP_cum"] = np.arange(len(gwas_plot_data))
        gwas_plot_data["CHR_INDEX"] = gwas_plot_data["CHR"] % 2

    _atomic_to_csv(gwas_plot_data, manhattan_dir / f"{trait}_manhattan.csv", index=False)
    return chrom_ticks


def _load_chrom_tick_positions(csv_path: Path) -> dict[int, float]:
    """Restore chromosome ticks when cached Manhattan outputs are reused."""

    try:
        data = pd.read_csv(csv_path, usecols=["CHR", "BP_cum"])
        data["CHR"] = pd.to_numeric(data["CHR"], errors="coerce")
        data["BP_cum"] = pd.to_numeric(data["BP_cum"], errors="coerce")
        data = data.dropna(subset=["CHR", "BP_cum"])
        return {
            int(chromosome): float(positions.median())
            for chromosome, positions in data.groupby("CHR")["BP_cum"]
        }
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        logger.warning(f"Could not restore chromosome ticks from {csv_path}: {error}")
        return {}


# =============================================================================
# Plot Rendering Functions
# =============================================================================


def _materialize_plot_alias(source: Path, target: Path) -> None:
    """Create a portable trait-specific alias for an identical plot."""

    if source == target:
        return
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Cannot alias missing or empty diagnostic plot: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _render_gene_diagnostic_plots_refactored(
    report_config: QuickModeConfig,
    metadata: pd.DataFrame,
    common_spots: np.ndarray,
    gss_adata: ad.AnnData | None,
    n_rows: int,
    n_cols: int,
    web_report_dir: Path,
    report_data_dir: Path,
    is_step_complete_func,
    load_gss_func=None,
):
    """Render gene expression and GSS diagnostic plots with completeness check."""
    started = time.perf_counter()
    gene_plot_dir = web_report_dir / "gene_diagnostic_plots"
    gene_plot_dir.mkdir(exist_ok=True)

    if not report_config.sample_h5ad_dict:
        logger.warning("Skipping gene diagnostic plots: missing h5ad dict")
        return

    top_n = report_config.top_corr_genes
    # 1. Identify all genes that need to be plotted from per-trait PCC files
    gss_stats_dir = report_data_dir / "gss_stats"
    trait_top_genes = {}
    traits = sorted(
        p.name.replace("gene_trait_correlation_", "").replace(".csv", "")
        for p in gss_stats_dir.glob("gene_trait_correlation_*.csv")
    )

    for trait in traits:
        pcc_path = report_config.get_gene_diagnostic_info_save_path(trait)
        if pcc_path.exists():
            group = pd.read_csv(pcc_path)
            genes = group.sort_values("PCC", ascending=False).head(top_n)["gene"].tolist()
            trait_top_genes[trait] = genes

    sample_names_sorted = list(report_config.sample_h5ad_dict.keys())

    metadata_common = metadata[metadata["spot"].isin(common_spots)]

    logger.info("Checking completeness for diagnostic plots...")

    force_re_run = getattr(report_config, "force_report_re_run", False)
    targets_by_pair: dict[tuple[str, str], list[tuple[Path, Path]]] = {}
    expected_paths: list[Path] = []
    for sample_name in sample_names_sorted:
        safe_sample = "".join(c if c.isalnum() else "_" for c in sample_name)
        for trait in traits:
            for gene in trait_top_genes.get(trait, []):
                exp_path = gene_plot_dir / f"gene_{trait}_{gene}_{safe_sample}_exp.png"
                gss_path = gene_plot_dir / f"gene_{trait}_{gene}_{safe_sample}_gss.png"
                targets_by_pair.setdefault((sample_name, gene), []).append((exp_path, gss_path))
                expected_paths.extend((exp_path, gss_path))

    requests_by_sample: dict[str, list[dict]] = {}
    alias_only_pairs = 0
    for (sample_name, gene), targets in targets_by_pair.items():
        complete_targets = [
            (exp_path, gss_path)
            for exp_path, gss_path in targets
            if exp_path.is_file()
            and exp_path.stat().st_size > 0
            and gss_path.is_file()
            and gss_path.stat().st_size > 0
        ]
        if not force_re_run and complete_targets:
            source_exp, source_gss = complete_targets[0]
            for exp_path, gss_path in targets:
                if not is_step_complete_func([exp_path]):
                    _materialize_plot_alias(source_exp, exp_path)
                if not is_step_complete_func([gss_path]):
                    _materialize_plot_alias(source_gss, gss_path)
            alias_only_pairs += 1
            continue

        source_exp, source_gss = targets[0]
        requests_by_sample.setdefault(sample_name, []).append(
            {
                "sample_name": sample_name,
                "gene": gene,
                "source_exp": source_exp,
                "source_gss": source_gss,
                "targets": targets,
            }
        )

    trait_gene_pairs = sum(len(targets) for targets in targets_by_pair.values())
    unique_pairs = len(targets_by_pair)
    render_pairs = sum(len(requests) for requests in requests_by_sample.values())
    logger.info(
        f"Diagnostic plot plan: {trait_gene_pairs:,} trait-gene-sample pairs, "
        f"{unique_pairs:,} unique sample-gene pairs, {render_pairs:,} pairs to render, "
        f"{alias_only_pairs:,} pairs reused; expected PNG files={len(expected_paths):,}"
    )

    if render_pairs == 0:
        logger.info("All gene diagnostic plots already exist. Skipping.")
        return

    if gss_adata is None:
        if load_gss_func is None:
            raise ValueError("GSS data are required to render gene diagnostic plots")
        gss_adata = load_gss_func()

    max_workers = min(_report_worker_count(report_config, ceiling=20), render_pairs)
    logger.info(
        f"Rendering {render_pairs:,} unique sample-gene pairs "
        f"({render_pairs * 2:,} PNGs) with {max_workers} process worker(s)..."
    )

    all_futures: dict = {}
    futures_lock = threading.Lock()
    missing_genes: list[tuple[str, str]] = []
    missing_genes_lock = threading.Lock()

    max_pending_tasks = max_workers * 4
    task_semaphore = threading.Semaphore(max_pending_tasks)
    max_loading_threads = min(4, len(sample_names_sorted))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        def process_sample(sample_name):
            sample_requests = requests_by_sample.get(sample_name, [])
            if not sample_requests:
                return

            h5ad_path = report_config.sample_h5ad_dict[sample_name]
            sample_metadata = metadata_common[metadata_common["sample_name"] == sample_name]
            if sample_metadata.empty:
                raise ValueError(f"No report metadata rows found for sample {sample_name}")

            sample_spots = sample_metadata["spot"].values
            coords = sample_metadata[["sx", "sy"]].values

            adata_rep = ad.read_h5ad(h5ad_path)
            suffix = f"|{sample_name}"
            if adata_rep.n_obs and not str(adata_rep.obs_names[0]).endswith(suffix):
                adata_rep.obs_names = adata_rep.obs_names.astype(str) + suffix

            if "log1p" in adata_rep.uns and adata_rep.X is not None:
                is_count = False
            else:
                is_count, _ = setup_data_layer(adata_rep, report_config.data_layer, verbose=False)

            if is_count:
                sc.pp.normalize_total(adata_rep, target_sum=1e4)
                sc.pp.log1p(adata_rep)

            unique_sample_genes = sorted({request["gene"] for request in sample_requests})
            available_genes = [gene for gene in unique_sample_genes if gene in adata_rep.var_names]
            unavailable_genes = sorted(set(unique_sample_genes).difference(available_genes))
            if unavailable_genes:
                with missing_genes_lock:
                    missing_genes.extend((sample_name, gene) for gene in unavailable_genes)

            if not available_genes:
                del adata_rep
                gc.collect()
                return

            adata_sample_exp = adata_rep[sample_spots, available_genes].copy()
            adata_sample_gss = gss_adata[sample_spots, available_genes].to_memory()
            del adata_rep
            gc.collect()

            exp_matrix = adata_sample_exp.X
            gss_matrix = adata_sample_gss.X
            gene_positions = {gene: index for index, gene in enumerate(available_genes)}

            for request in sample_requests:
                gene = request["gene"]
                if gene not in available_genes:
                    continue

                gene_index = gene_positions[gene]
                exp_column = exp_matrix[:, gene_index]
                gss_column = gss_matrix[:, gene_index]
                exp_vals = (
                    np.asarray(exp_column.toarray() if sp.issparse(exp_column) else exp_column)
                    .ravel()
                    .astype(np.float32, copy=False)
                )
                gss_vals = (
                    np.asarray(gss_column.toarray() if sp.issparse(gss_column) else gss_column)
                    .ravel()
                    .astype(np.float32, copy=False)
                )

                task_semaphore.acquire()
                try:
                    future = executor.submit(
                        _render_single_sample_gene_pair_task,
                        {
                            "gene": gene,
                            "sample_name": sample_name,
                            "coords": coords.copy(),
                            "exp_values": exp_vals,
                            "gss_values": gss_vals,
                            "exp_output_path": request["source_exp"],
                            "gss_output_path": request["source_gss"],
                            "plot_origin": report_config.plot_origin,
                            "fig_width": 6.0,
                            "dpi": 150,
                        },
                    )
                    future.add_done_callback(lambda _: task_semaphore.release())
                    with futures_lock:
                        all_futures[future] = request
                except Exception:
                    task_semaphore.release()
                    raise

            del adata_sample_exp, adata_sample_gss
            gc.collect()

        with ThreadPoolExecutor(max_workers=max_loading_threads) as loader:
            list(loader.map(process_sample, requests_by_sample))

        logger.info(f"Waiting for {len(all_futures):,} unique diagnostic tasks to complete...")
        completed = 0
        progress_interval = max(1, len(all_futures) // 10)
        for future in as_completed(all_futures):
            request = all_futures[future]
            future.result()
            source_exp = request["source_exp"]
            source_gss = request["source_gss"]
            for exp_path, gss_path in request["targets"]:
                _materialize_plot_alias(source_exp, exp_path)
                _materialize_plot_alias(source_gss, gss_path)
            completed += 1
            if completed % progress_interval == 0 or completed == len(all_futures):
                elapsed = time.perf_counter() - started
                logger.info(
                    f"Diagnostic plot progress: {completed:,}/{len(all_futures):,} unique "
                    f"pairs ({completed / max(elapsed, 1e-9):.1f} pairs/s)"
                )

    if missing_genes:
        preview = ", ".join(f"{sample}:{gene}" for sample, gene in missing_genes[:10])
        raise ValueError(
            f"Could not render {len(missing_genes)} sample-gene pairs because genes were absent: "
            f"{preview}"
        )

    missing_outputs = [
        path for path in expected_paths if not path.is_file() or path.stat().st_size == 0
    ]
    if missing_outputs:
        raise RuntimeError(
            f"Diagnostic rendering left {len(missing_outputs)} missing or empty PNG files; "
            f"first missing output: {missing_outputs[0]}"
        )

    elapsed = time.perf_counter() - started
    logger.info(
        f"Successfully prepared {len(expected_paths):,} diagnostic PNG files from "
        f"{render_pairs:,} unique renders in {elapsed:.2f}s"
    )


# =============================================================================
# Results Collection Functions
# =============================================================================


def _collect_cauchy_results(report_config: QuickModeConfig, traits: list[str], report_dir: Path):
    """Collect and save Cauchy combination results."""
    logger.info("Collecting Cauchy combination results...")
    all_cauchy = []

    for trait in traits:
        for annotation in report_config.annotation_list:
            # Aggregated results
            cauchy_file_all = report_config.get_cauchy_result_file(
                trait, annotation=annotation, all_samples=True
            )
            if cauchy_file_all.exists():
                try:
                    df = pd.read_csv(cauchy_file_all)
                    df["trait"] = trait
                    df["annotation_name"] = annotation
                    df["type"] = "aggregated"
                    if "sample_name" not in df.columns:
                        df["sample_name"] = "All Samples"

                    # Convert to -log10 scale for report
                    df["mlog10_p_cauchy"] = -np.log10(df["p_cauchy"].clip(lower=1e-300))
                    df["mlog10_p_median"] = -np.log10(df["p_median"].clip(lower=1e-300))

                    all_cauchy.append(df)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"Failed to load aggregated Cauchy result {cauchy_file_all}: {e}"
                    )

            # Per-sample results
            cauchy_file = report_config.get_cauchy_result_file(
                trait, annotation=annotation, all_samples=False
            )
            if cauchy_file.exists():
                try:
                    df = pd.read_csv(cauchy_file)
                    df["trait"] = trait
                    df["annotation_name"] = annotation
                    df["type"] = "sample"

                    # Convert to -log10 scale for report
                    df["mlog10_p_cauchy"] = -np.log10(df["p_cauchy"].clip(lower=1e-300))
                    df["mlog10_p_median"] = -np.log10(df["p_median"].clip(lower=1e-300))

                    all_cauchy.append(df)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to load sample Cauchy result {cauchy_file}: {e}")

    if all_cauchy:
        combined_cauchy = pd.concat(all_cauchy, ignore_index=True)
        if "sample" in combined_cauchy.columns and "sample_name" not in combined_cauchy.columns:
            combined_cauchy = combined_cauchy.rename(columns={"sample": "sample_name"})

        cauchy_save_path = report_dir / "cauchy_results.csv"
        _atomic_to_csv(combined_cauchy, cauchy_save_path, index=False)
        logger.info(f"Saved {len(combined_cauchy)} Cauchy results to {cauchy_save_path}")
    else:
        _atomic_to_csv(
            pd.DataFrame(
                columns=[
                    "trait",
                    "annotation_name",
                    "mlog10_p_cauchy",
                    "mlog10_p_median",
                    "top_95_quantile",
                    "type",
                    "sample_name",
                ]
            ),
            report_dir / "cauchy_results.csv",
            index=False,
        )
        logger.warning("No Cauchy results found to save.")


def _save_report_metadata(
    report_config: QuickModeConfig,
    traits: list[str],
    sample_names: list[str],
    report_dir: Path,
    chrom_tick_positions: dict | None = None,
    umap_info: dict | None = None,
    is_3d: bool = False,
    spatial_3d_path: str | None = None,
):
    """Save report configuration metadata as JSON."""
    logger.info("Saving report configuration metadata...")
    report_meta = report_config.to_dict_with_paths_as_strings()

    report_meta["traits"] = traits
    report_meta["samples"] = sample_names
    report_meta["annotations"] = report_config.annotation_list
    report_meta["top_corr_genes"] = report_config.top_corr_genes
    report_meta["plot_origin"] = report_config.plot_origin
    report_meta["legend_marker_size"] = report_config.legend_marker_size
    report_meta["downsampling_n_spots_2d"] = getattr(
        report_config, "downsampling_n_spots_2d", 250000
    )

    # Add chromosome tick positions for Manhattan plot
    if chrom_tick_positions:
        report_meta["chrom_tick_positions"] = chrom_tick_positions

    # Add UMAP info
    if umap_info:
        report_meta["umap_info"] = umap_info

    # Add 3D visualization info
    report_meta["is_3d"] = is_3d
    report_meta["has_3d_widget"] = spatial_3d_path is not None
    report_meta["spatial_3d_widget_path"] = spatial_3d_path

    # Add dataset type info for conditional section rendering
    dataset_type_value = report_config.dataset_type
    if hasattr(dataset_type_value, "value"):
        dataset_type_value = dataset_type_value.value
    report_meta["dataset_type"] = dataset_type_value
    report_meta["dataset_type_label"] = {
        "spatial3D": "Spatial 3D",
        "spatial2D": "Spatial 2D",
        "scRNA": "Single Cell RNA-seq",
    }.get(dataset_type_value, dataset_type_value)

    _write_text_if_changed(
        report_dir / "report_meta.json", json.dumps(report_meta, sort_keys=True)
    )
    return report_meta


def _copy_js_assets(report_dir: Path):
    """Copy bundled JS assets for local usage (no network required)."""
    logger.info("Copying JS assets for local usage...")
    js_lib_dir = report_dir / "js_lib"
    js_lib_dir.mkdir(exist_ok=True)

    # Copy bundled Alpine.js and Tailwind.js from static folder
    static_js_dir = Path(__file__).parent / "static" / "js_lib"
    if static_js_dir.exists():
        for js_file in static_js_dir.glob("*.js"):
            dest = js_lib_dir / js_file.name
            if not dest.exists():
                shutil.copy2(js_file, dest)
                logger.info(f"Copied {js_file.name}")

    # Copy plotly.min.js from installed plotly Python package
    plotly_js_src = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    plotly_dest = js_lib_dir / "plotly.min.js"
    if plotly_js_src.exists() and not plotly_dest.exists():
        shutil.copy2(plotly_js_src, plotly_dest)
        logger.info("Copied plotly.min.js from plotly package")


# =============================================================================
# JS Export Functions
# =============================================================================


def _atomic_write_text(path: Path, content: str) -> None:
    """Write a text output through an atomic same-directory replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_to_csv(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write a CSV through an atomic same-directory replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_csv(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_if_changed(path: Path, content: str) -> bool:
    """Atomically update a small text output only when its content changed."""

    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == content:
                return False
        except OSError:
            pass

    _atomic_write_text(path, content)
    return True


def _sample_spatial_export_signature(data_dir: Path, meta: dict) -> dict:
    """Build a cheap, stable signature for the per-sample export inputs."""

    metadata_file = data_dir / "spot_metadata.csv"
    stat = metadata_file.stat()
    relevant_meta = {
        key: meta.get(key)
        for key in ("samples", "traits", "annotations", "downsampling_n_spots_2d")
    }
    meta_digest = hashlib.sha256(
        json.dumps(relevant_meta, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {
        "metadata_inode": stat.st_ino,
        "metadata_size": stat.st_size,
        "metadata_mtime_ns": stat.st_mtime_ns,
        "meta_sha256": meta_digest,
    }


def _per_sample_spatial_outputs_current(data_dir: Path, js_data_dir: Path, meta: dict) -> bool:
    """Return whether per-sample JS outputs match a completed export manifest."""

    metadata_file = data_dir / "spot_metadata.csv"
    manifest_path = js_data_dir / "sample_spatial_manifest.json"
    if not metadata_file.is_file() or metadata_file.stat().st_size == 0:
        return False
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    if manifest.get("version") != 1:
        return False
    if manifest.get("signature") != _sample_spatial_export_signature(data_dir, meta):
        return False

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return False
    return all(
        (js_data_dir / filename).is_file() and (js_data_dir / filename).stat().st_size > 0
        for filename in outputs
    )


def export_data_as_js_modules(data_dir: Path):
    """
    Convert the CSV data in the report directory into JavaScript modules (.js files)
    that assign the data to window global variables.
    This allows loading data via <script> tags locally without CORS issues.
    """
    logger.info("Exporting data as JS modules...")
    js_data_dir = data_dir / "js_data"
    js_data_dir.mkdir(exist_ok=True)

    meta = {}
    meta_file = data_dir / "report_meta.json"
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load report_meta.json: {e}")

    # Use per-sample export for efficient on-demand loading (instead of monolithic _export_metadata_js)
    _export_per_sample_spatial_js(data_dir, js_data_dir, meta)
    _export_cauchy_js(data_dir, js_data_dir)
    _export_manhattan_js(data_dir, js_data_dir)
    _export_umap_js(data_dir, js_data_dir, meta)
    _export_report_meta_js(data_dir, js_data_dir, meta)

    logger.info(f"JS modules exported to {js_data_dir}")


def _export_per_sample_spatial_js(data_dir: Path, js_data_dir: Path, meta: dict):
    """
    Export spatial metadata as per-sample JS files for efficient on-demand loading.

    This creates:
    - sample_index.js: Lightweight index mapping sample names to file info
    - sample_{name}_spatial.js: Per-sample data with coordinates, traits, annotations

    This approach is much more efficient than loading all samples at once,
    especially for datasets with millions of spots.
    """
    metadata_file = data_dir / "spot_metadata.csv"
    if not metadata_file.exists():
        logger.warning("spot_metadata.csv not found, skipping per-sample spatial export")
        return

    logger.info("Exporting per-sample spatial data as JS modules...")
    df = pd.read_csv(metadata_file)

    samples = meta.get("samples", [])
    traits = meta.get("traits", [])
    annotations = meta.get("annotations", [])

    if not samples:
        # Fallback to unique sample names from data
        samples = df["sample_name"].unique().tolist() if "sample_name" in df.columns else []

    # Detect whether spatial coordinates are available (absent for scRNA-seq)
    has_spatial_coords = "sx" in df.columns and "sy" in df.columns

    sample_index = {}
    output_files = []
    max_spots_2d = meta.get("downsampling_n_spots_2d", 250000)

    for sample_name in samples:
        sample_df = df[df["sample_name"] == sample_name]

        if len(sample_df) == 0:
            raise ValueError(f"No spot metadata found for configured sample: {sample_name}")

        # Track original count before downsampling
        n_spots_original = len(sample_df)

        # Downsample if needed for 2D visualization performance
        if has_spatial_coords and len(sample_df) > max_spots_2d:
            logger.info(
                f"Downsampling {sample_name} from {len(sample_df):,} to {max_spots_2d:,} spots for 2D visualization"
            )
            sample_df = sample_df.sample(n=max_spots_2d, random_state=42)

        # Calculate point size for this sample (spatial only)
        if has_spatial_coords:
            _, point_size = estimate_plotly_point_size(
                sample_df[["sx", "sy"]], DEFAULT_PIXEL_WIDTH=560
            )
        else:
            point_size = 5.0  # default point size for non-spatial datasets

        # Build columnar data structure for efficient ScatterGL rendering
        data_struct = {
            "spot": sample_df["spot"].tolist(),
            "point_size": round(point_size, 2),
        }

        # Add spatial coordinates only when available
        if has_spatial_coords:
            data_struct["sx"] = sample_df["sx"].tolist()
            data_struct["sy"] = sample_df["sy"].tolist()

        # Add 3D coordinates if present
        for coord in ["3d_x", "3d_y", "3d_z"]:
            if coord in sample_df.columns:
                data_struct[coord] = sample_df[coord].tolist()

        # Add all annotation columns
        for anno in annotations:
            if anno in sample_df.columns:
                data_struct[anno] = sample_df[anno].tolist()

        # Add all trait columns (round to 1 decimal to reduce file size)
        for trait in traits:
            if trait in sample_df.columns:
                data_struct[trait] = [
                    round(v, 1) if pd.notna(v) else None for v in sample_df[trait]
                ]

        # Create safe filename (replace non-alphanumeric chars)
        safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
        var_name = f"GSMAP_SAMPLE_{safe_name}"
        file_name = f"sample_{safe_name}_spatial.js"

        # Write per-sample JS file
        js_content = f"window.{var_name} = {json.dumps(data_struct, separators=(',', ':'))};"
        _atomic_write_text(js_data_dir / file_name, js_content)
        output_files.append(file_name)

        # Track in sample index
        sample_index[sample_name] = {
            "var_name": var_name,
            "file": file_name,
            "n_spots": len(sample_df),
            "n_spots_original": n_spots_original,
            "point_size": round(point_size, 2),
        }

        if n_spots_original > len(sample_df):
            logger.info(
                f"  Exported {sample_name}: {len(sample_df):,} spots (downsampled from {n_spots_original:,}) -> {file_name}"
            )
        else:
            logger.info(f"  Exported {sample_name}: {len(sample_df):,} spots -> {file_name}")

    # Export lightweight sample index (loaded upfront)
    js_content = f"window.GSMAP_SAMPLE_INDEX = {json.dumps(sample_index, separators=(',', ':'))};"
    _atomic_write_text(js_data_dir / "sample_index.js", js_content)
    output_files.append("sample_index.js")

    manifest = {
        "version": 1,
        "signature": _sample_spatial_export_signature(data_dir, meta),
        "outputs": sorted(output_files),
    }
    _atomic_write_text(
        js_data_dir / "sample_spatial_manifest.json",
        json.dumps(manifest, separators=(",", ":"), sort_keys=True),
    )

    logger.info(f"Exported {len(sample_index)} per-sample spatial JS files + sample_index.js")


def _export_cauchy_js(data_dir: Path, js_data_dir: Path):
    """Export Cauchy results as JS module."""
    cauchy_file = data_dir / "cauchy_results.csv"
    if cauchy_file.exists():
        df = pd.read_csv(cauchy_file)
        data_json = df.to_json(orient="records")
        js_content = f"window.GSMAP_CAUCHY = {data_json};"
        _atomic_write_text(js_data_dir / "cauchy_results.js", js_content)


def _export_manhattan_js(data_dir: Path, js_data_dir: Path):
    """Export Manhattan data as JS modules (one per trait)."""
    manhattan_dir = data_dir / "manhattan_data"
    if not manhattan_dir.exists():
        return

    for csv_file in manhattan_dir.glob("*_manhattan.csv"):
        trait = csv_file.name.replace("_manhattan.csv", "")
        try:
            df = pd.read_csv(csv_file)
            if "P" in df.columns and "logp" not in df.columns:
                df["logp"] = -np.log10(df["P"] + 1e-300)

            data_struct = {
                "x": df["BP_cum"].tolist(),
                "y": df["logp"].tolist(),
                "gene": df["GENE"].fillna("").tolist(),
                "chr": df["CHR"].astype(int).tolist(),
                "snp": df["SNP"].tolist() if "SNP" in df.columns else [],
                "is_top": df["is_top_pcc"].astype(int).tolist()
                if "is_top_pcc" in df.columns
                else [],
                "bp": df["BP"].astype(int).tolist() if "BP" in df.columns else [],
            }

            json_str = json.dumps(data_struct, separators=(",", ":"))
            safe_trait = "".join(c if c.isalnum() else "_" for c in trait)
            js_content = f"window.GSMAP_MANHATTAN_{safe_trait} = {json_str};"

            _atomic_write_text(js_data_dir / f"manhattan_{trait}.js", js_content)

        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to export Manhattan JS for {trait}: {e}")


def _export_report_meta_js(data_dir: Path, js_data_dir: Path, meta: dict):
    """Export report metadata as JS module."""
    if meta:
        js_content = (
            "window.GSMAP_REPORT_META = "
            f"{json.dumps(meta, separators=(',', ':'), sort_keys=True)};"
        )
        _write_text_if_changed(js_data_dir / "report_meta.js", js_content)


def _export_umap_js(data_dir: Path, js_data_dir: Path, meta: dict):
    """Export UMAP data as JS module."""
    umap_file = data_dir / "umap_data.csv"
    if umap_file.exists():
        df = pd.read_csv(umap_file)

        # Build efficient columnar structure for ScatterGL
        data_struct = {
            "spot": df["spot"].tolist(),
            "sample_name": df["sample_name"].tolist(),
            "umap_cell_x": df["umap_cell_x"].tolist(),
            "umap_cell_y": df["umap_cell_y"].tolist(),
            "point_size_cell": meta.get("umap_info", {}).get("point_size_cell", 4),
            "default_opacity": meta.get("umap_info", {}).get("default_opacity", 0.8),
        }

        # Add niche UMAP if available
        if "umap_niche_x" in df.columns:
            data_struct["umap_niche_x"] = df["umap_niche_x"].tolist()
            data_struct["umap_niche_y"] = df["umap_niche_y"].tolist()
            data_struct["point_size_niche"] = meta.get("umap_info", {}).get("point_size_niche", 4)
            data_struct["has_niche"] = True
        else:
            data_struct["has_niche"] = False

        # Add all annotation columns (excluding known columns)
        known_cols = {
            "spot",
            "sample_name",
            "umap_cell_x",
            "umap_cell_y",
            "umap_niche_x",
            "umap_niche_y",
        }
        annotation_cols = [c for c in df.columns if c not in known_cols]
        data_struct["annotation_columns"] = annotation_cols

        from gsMap.report.visualize import _create_color_map

        P_COLOR = [
            "#313695",
            "#4575b4",
            "#74add1",
            "#fee090",
            "#fdae61",
            "#f46d43",
            "#d73027",
            "#a50026",
        ]
        color_maps = {}

        for col in annotation_cols:
            data_struct[col] = df[col].tolist()
            # Generate color map for each column
            if pd.api.types.is_numeric_dtype(df[col]):
                color_maps[col] = P_COLOR
            else:
                # Convert to string before sorting to avoid TypeError with mixed float/str (e.g. NaN)
                unique_vals = sorted(df[col].astype(str).unique().tolist())
                color_maps[col] = _create_color_map(unique_vals, hex=True, rng=42)

        data_struct["color_maps"] = color_maps

        js_content = f"window.GSMAP_UMAP = {json.dumps(data_struct, separators=(',', ':'))};"
        _atomic_write_text(js_data_dir / "umap_data.js", js_content)
        logger.info(f"Exported UMAP data with {len(df)} points")

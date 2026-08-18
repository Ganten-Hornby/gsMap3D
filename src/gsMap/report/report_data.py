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
from gsMap.latent2gene.marker_layout import open_marker_obs_indices
from gsMap.latent2gene.memmap_io import MemMapDense
from gsMap.report.diagnosis import convert_z_to_p, filter_snps, load_gwas_data
from gsMap.report.spatial_payload import (
    SPATIAL_PAYLOAD_VERSION,
    export_sample_spatial_payloads,
    export_spatial_3d_payloads,
)
from gsMap.report.visualize import (
    estimate_plotly_point_size,
)
from gsMap.spatial_ldsc.io import (
    load_marker_scores_memmap_format,
    resolve_marker_scores_memmap_path,
)

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


@dataclass(frozen=True)
class _GssPccSample:
    """Marker-score rows and aligned metadata used by report PCC calculations."""

    common_spots: np.ndarray
    analysis_spots: np.ndarray
    gene_names: list[str]
    marker_scores: np.ndarray
    annotations: pd.Series


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
        self.spatial_3d_coordinate_info = None
        self.common_spots = None
        self.analysis_spots = None
        self.gene_stats = None
        self.metadata = None
        self.gss_adata = None
        self.chrom_tick_positions = None
        self.umap_info = None
        self.spatial_3d_info = None

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
            self.spatial_3d_coordinate_info = self.coords.attrs.get("spatial_3d_coordinate_info")

        # Export base metadata as JS if needed (usually handled in finalizing or per-sample)
        # However, _export_per_sample_spatial_js is what we use now.

    def prepare_gss_stats(self):
        """Load one sampled GSS matrix and calculate all missing trait PCCs."""
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

        if not traits_to_run:
            exp_frac = pd.read_parquet(self.report_config.mean_frac_path)
            gene_names = exp_frac.index[exp_frac["frac"] > 0.01].astype(str).tolist()
            _atomic_to_csv(pd.DataFrame({"gene": gene_names}), gene_list_file, index=False)
            logger.info(f"Rebuilt the report gene list with {len(gene_names):,} genes.")
            return

        logger.info(f"Processing GSS statistics for {len(traits_to_run)} traits...")
        sample = _load_gss_pcc_sample(
            self.report_config,
            self.ldsc_results,
            self.report_data_dir,
        )
        self.common_spots = sample.common_spots
        self.analysis_spots = sample.analysis_spots

        stats_started = time.perf_counter()
        self.gene_stats = _calculate_gene_annotation_stats(
            sample.marker_scores,
            sample.analysis_spots,
            sample.gene_names,
            sample.annotations,
        )
        logger.info(
            f"Calculated gene annotation medians in {time.perf_counter() - stats_started:.2f}s"
        )

        center_started = time.perf_counter()
        gss_mean = sample.marker_scores.mean(axis=0).astype(np.float32)
        gss_centered = (sample.marker_scores - gss_mean).astype(np.float32)
        gss_ssq = np.einsum(
            "ij,ij->j",
            gss_centered,
            gss_centered,
            dtype=np.float32,
            optimize=True,
        )
        logger.info(
            f"Centered the sampled marker matrix in {time.perf_counter() - center_started:.2f}s"
        )

        pcc_matrix = _calculate_pcc_for_traits_fast(
            traits_to_run,
            self.ldsc_results,
            sample.analysis_spots,
            gss_centered,
            gss_ssq,
        )
        del gss_centered

        for trait_index, trait in enumerate(traits_to_run):
            trait_pcc = _save_pcc_for_trait(
                trait,
                pcc_matrix[:, trait_index],
                sample.gene_names,
                self.gene_stats,
                self.report_config,
            )
            self._export_single_pcc_js(trait, trait_pcc)

    def prepare_spot_metadata(self):
        """Save spot metadata and coordinates."""
        metadata_file = self.report_data_dir / "spot_metadata.csv"
        coordinate_info_file = self.report_data_dir / "spatial_3d_coordinate_info.json"
        coordinate_info_is_current = _spatial_coordinate_info_is_current(
            coordinate_info_file, self.spatial_3d_coordinate_info
        )
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
            if self.report_config.dataset_type == DatasetType.SPATIAL_3D:
                required_columns.update(("3d_x", "3d_y", "3d_z"))
            missing_columns = sorted(required_columns.difference(self.metadata.columns))
            if missing_columns or not coordinate_info_is_current:
                reasons = []
                if missing_columns:
                    reasons.append(f"missing columns: {missing_columns}")
                if not coordinate_info_is_current:
                    reasons.append("3D coordinate provenance or scaling changed")
                logger.warning(
                    f"Cached spot_metadata.csv is stale ({'; '.join(reasons)}). Regenerating it."
                )
                self.metadata = _save_metadata(
                    self.ldsc_results, self.coords, self.report_data_dir
                )
                _write_spatial_coordinate_info(
                    coordinate_info_file, self.spatial_3d_coordinate_info
                )
            else:
                logger.info("Spot metadata already exists. Skipping.")
            if self.common_spots is None:
                self.common_spots = self.metadata["spot"].values
            return

        self.metadata = _save_metadata(self.ldsc_results, self.coords, self.report_data_dir)
        _write_spatial_coordinate_info(coordinate_info_file, self.spatial_3d_coordinate_info)
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
                self.common_spots, self.gss_adata = _load_gss_for_report_plots(
                    self.report_config,
                    self.ldsc_results,
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
            self.spatial_3d_info = None
            return

        self.spatial_3d_info = _prepare_3d_visualization(
            self.report_config,
            self.metadata,
            self.traits,
            self.report_data_dir,
            self.web_report_dir,
            coordinate_info=self.spatial_3d_coordinate_info,
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
            spatial_3d_info=getattr(self, "spatial_3d_info", None),
        )

        # 2. Copy JS libraries
        _copy_js_assets(self.web_report_dir)

        # 3. Export per-sample spatial JS (highly important for performance)
        if not self.force_re_run and _per_sample_spatial_outputs_current(
            self.report_data_dir, self.js_data_dir, meta
        ):
            logger.info("Per-sample spatial JS is current. Skipping export.")
        else:
            _export_per_sample_spatial_js(
                self.report_data_dir,
                self.js_data_dir,
                meta,
                metadata=self.metadata,
            )
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
    coordinate_info: dict | None = None,
) -> dict | None:
    """Export a shared Plotly 3D geometry and lazy trait/annotation payloads."""

    return export_spatial_3d_payloads(
        metadata=metadata,
        traits=traits,
        annotations=report_config.annotation_list,
        report_data_dir=report_data_dir,
        web_report_dir=web_report_dir,
        max_points=getattr(report_config, "downsampling_n_spots_3d", 100000),
        force=getattr(report_config, "force_report_re_run", False),
        coordinate_info=coordinate_info,
    )


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
    coordinate_info = None
    if is_3d_type and has_3d_coords:
        # True 3D coordinates
        logger.info("Detected 3D spatial coordinates")
        z_axis = int(getattr(report_config, "cross_slice_z_coordinate", -1))
        if z_axis < 0:
            z_axis += coords_data.shape[1]
        if not 0 <= z_axis < coords_data.shape[1]:
            raise IndexError(
                f"cross_slice_z_coordinate is outside the {coords_data.shape[1]} "
                "spatial coordinate columns"
            )
        if z_axis < 2:
            raise ValueError(
                "cross_slice_z_coordinate must identify a Z column after the first "
                "two X/Y spatial coordinate columns"
            )

        report_coordinates = np.asarray(coords_data[:, [0, 1, z_axis]], dtype=np.float64)
        z_scale = float(getattr(report_config, "cross_slice_z_scale", 1.0))
        report_coordinates[:, 2] *= z_scale
        coords = pd.DataFrame(
            report_coordinates,
            columns=["3d_x", "3d_y", "3d_z"],
            index=adata_concat.obs_names,
        )
        coordinate_info = {
            "schema_version": 2,
            "z_source": "registered_coordinate",
            "z_axis_input": z_axis,
            "z_scale": z_scale,
            "z_is_physical": True,
            "default_aspect_mode": "physical",
            "axis_titles": ["X", "Y", "Z"],
            "note": (f"Z uses input coordinate axis {z_axis} with configured scale {z_scale:g}."),
        }

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

        # Assign Z values based on sample order from report_config
        sample_names_all = adata_concat.obs["sample_name"]
        actual_samples_in_data = set(sample_names_all.unique())

        # Explicitly use report_config order
        ordered_samples = list(report_config.sample_h5ad_dict.keys())
        missing = [s for s in ordered_samples if s not in actual_samples_in_data]
        assert not missing, f"Samples {missing} in config are not present in coordinates data."

        n_samples = len(ordered_samples)
        z_values_per_sample = {
            sample: float(index) for index, sample in enumerate(ordered_samples)
        }

        # Assign Z values to each spot
        pseudo_z = sample_names_all.map(z_values_per_sample).values

        coords["3d_x"] = coords["sx"]
        coords["3d_y"] = coords["sy"]
        coords["3d_z"] = pseudo_z
        z_axis = 2  # Pseudo Z is always the last dimension
        coordinate_info = {
            "schema_version": 2,
            "z_source": "ordinal_section_order",
            "z_axis_input": None,
            "z_scale": None,
            "z_is_physical": False,
            "default_aspect_mode": "expanded",
            "axis_titles": ["X", "Y", "Section order"],
            "sample_z_positions": z_values_per_sample,
            "note": (
                "No measured Z coordinate was provided; Z encodes input section order only. "
                "Section thickness, missing sections, and unequal spacing are not represented."
            ),
        }

        logger.warning(
            f"Created an ordinal (non-physical) Z axis for {n_samples} samples. "
            "Provide a three-column spatial key for measured 3D distances."
        )

    else:
        # 2D coordinates (non-3D dataset type)
        coords_2d = coords_data[:, :2]
        coords = pd.DataFrame(coords_2d, columns=["sx", "sy"], index=adata_concat.obs_names)

    coords = pd.concat([coords, sample_info], axis=1)
    if coordinate_info is not None:
        coords.attrs["spatial_3d_coordinate_info"] = coordinate_info

    # Close backed file
    if adata_concat.isbacked:
        adata_concat.file.close()

    # Return is_3d as True if dataset type is 3D (regardless of coord dimensions)
    return coords, is_3d_type, z_axis


def _load_gss_for_report_plots(
    report_config: QuickModeConfig,
    ldsc_results: pd.DataFrame,
) -> tuple[np.ndarray, ad.AnnData]:
    """Load the full marker wrapper only when diagnostic plots require it."""

    logger.info("Loading the full GSS wrapper for diagnostic plots...")
    gss_adata = load_marker_scores_memmap_format(report_config)
    common_spots = np.intersect1d(gss_adata.obs_names, ldsc_results.index)
    if len(common_spots) == 0:
        raise ValueError("No common spots found between GSS and LDSC results.")
    logger.info(f"Common spots (GSS and LDSC): {len(common_spots):,}")
    return common_spots, gss_adata


def _read_marker_score_rows(
    data_path: Path,
    shape: tuple[int, int],
    dtype: np.dtype,
    marker_rows: np.ndarray,
    gene_positions: np.ndarray,
    n_workers: int,
) -> np.ndarray:
    """Read selected dense marker rows concurrently while preserving row order."""

    marker_rows = np.asarray(marker_rows, dtype=np.int64)
    gene_positions = np.asarray(gene_positions, dtype=np.int64)
    output = np.empty((len(marker_rows), len(gene_positions)), dtype=dtype)
    if len(marker_rows) == 0:
        return output

    io_order = np.argsort(marker_rows, kind="stable")
    sorted_marker_rows = marker_rows[io_order]
    row_bytes = shape[1] * dtype.itemsize
    source_bytes = len(marker_rows) * row_bytes
    task_rows = 256
    tasks = [
        (start, min(start + task_rows, len(marker_rows)))
        for start in range(0, len(marker_rows), task_rows)
    ]
    progress_interval = max(10_000, len(marker_rows) // 10)
    next_progress = progress_interval
    started = time.perf_counter()
    file_descriptor = os.open(data_path, os.O_RDONLY)

    def read_task(bounds: tuple[int, int]) -> int:
        start, stop = bounds
        source_rows = sorted_marker_rows[start:stop]
        destination_rows = io_order[start:stop]
        local_index = 0
        while local_index < len(source_rows):
            run_stop = local_index + 1
            while (
                run_stop < len(source_rows)
                and source_rows[run_stop] == source_rows[run_stop - 1] + 1
            ):
                run_stop += 1

            first_row = int(source_rows[local_index])
            run_length = run_stop - local_index
            raw = os.pread(
                file_descriptor,
                run_length * row_bytes,
                first_row * row_bytes,
            )
            expected_bytes = run_length * row_bytes
            if len(raw) != expected_bytes:
                raise OSError(
                    f"Short marker-score read at row {first_row:,}: "
                    f"received {len(raw):,}/{expected_bytes:,} bytes"
                )
            block = np.frombuffer(raw, dtype=dtype).reshape(run_length, shape[1])
            output[destination_rows[local_index:run_stop]] = block[:, gene_positions]
            local_index = run_stop
        return stop - start

    completed = 0
    try:
        with ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="report-gss-reader",
        ) as executor:
            for completed_rows in executor.map(read_task, tasks):
                completed += completed_rows
                if completed >= next_progress or completed == len(marker_rows):
                    elapsed = time.perf_counter() - started
                    logger.info(
                        f"Sampled marker read: {completed:,}/{len(marker_rows):,} rows "
                        f"({completed / max(elapsed, 1e-9):,.0f} rows/s)"
                    )
                    while next_progress <= completed:
                        next_progress += progress_interval
    finally:
        os.close(file_descriptor)

    elapsed = time.perf_counter() - started
    logger.info(
        f"Read {source_bytes / 1024**3:.2f} GiB of selected marker rows from {data_path} "
        f"in {elapsed:.2f}s ({source_bytes / 1024**2 / max(elapsed, 1e-9):.1f} MiB/s); "
        f"sampled matrix={output.nbytes / 1024**3:.2f} GiB"
    )
    return output


def _load_gss_pcc_sample(
    report_config: QuickModeConfig,
    ldsc_results: pd.DataFrame,
    report_dir: Path,
) -> _GssPccSample:
    """Load exactly one deterministic marker-score sample for report PCC."""

    started = time.perf_counter()
    marker_path = Path(report_config.marker_scores_memmap_path)
    is_complete, marker_metadata = MemMapDense.check_complete(marker_path)
    if not is_complete or marker_metadata is None:
        raise ValueError(
            f"Marker score at {marker_path} is incomplete or corrupted. Please recompute."
        )

    source_adata = ad.read_h5ad(report_config.concatenated_latent_adata_path, backed="r")
    row_order = None
    try:
        shape = (source_adata.n_obs, source_adata.n_vars)
        if tuple(marker_metadata["shape"]) != shape:
            raise ValueError(
                f"Marker score shape {tuple(marker_metadata['shape'])} does not match "
                f"AnnData shape {shape}"
            )

        common_spots = np.intersect1d(source_adata.obs_names, ldsc_results.index)
        if len(common_spots) == 0:
            raise ValueError("No common spots found between GSS and LDSC results.")
        logger.info(f"Common spots (GSS and LDSC): {len(common_spots):,}")

        analysis_spots = common_spots
        requested_spots = report_config.downsampling_n_spots_pcc
        if requested_spots and len(common_spots) > requested_spots:
            analysis_spots = _stratified_subsample(
                common_spots,
                ldsc_results.loc[common_spots, "sample_name"],
                requested_spots,
            )
            logger.info(
                f"Stratified subsampled {len(analysis_spots):,}/{len(common_spots):,} "
                "spots for PCC calculation."
            )

        exp_frac = pd.read_parquet(report_config.mean_frac_path)
        gene_names = exp_frac.index[exp_frac["frac"] > 0.01].astype(str).tolist()
        gene_positions = source_adata.var_names.get_indexer(gene_names)
        if np.any(gene_positions < 0):
            missing = np.asarray(gene_names, dtype=object)[gene_positions < 0]
            raise ValueError(
                f"{len(missing):,} high-expression genes are absent from marker scores; "
                f"first missing gene: {missing[0]}"
            )
        _atomic_to_csv(
            pd.DataFrame({"gene": gene_names}),
            report_dir / "gene_list.csv",
            index=False,
        )
        logger.info(f"Using {len(gene_names):,} high-expression genes for PCC calculation.")

        original_rows = source_adata.obs_names.get_indexer(analysis_spots)
        if np.any(original_rows < 0):
            raise RuntimeError("A sampled report spot is absent from the source AnnData.")

        row_order = open_marker_obs_indices(
            marker_path,
            marker_metadata,
            expected_obs=source_adata.n_obs,
        )
        if row_order is None:
            marker_rows = original_rows
        else:
            inverse_row_order = np.empty(source_adata.n_obs, dtype=np.int64)
            inverse_row_order[np.asarray(row_order)] = np.arange(
                source_adata.n_obs,
                dtype=np.int64,
            )
            marker_rows = inverse_row_order[original_rows]

        annotation = report_config.annotation_list[0]
        annotations = source_adata.obs.iloc[original_rows][annotation].copy()
    finally:
        if row_order is not None:
            row_order._mmap.close()
        if source_adata.isbacked:
            source_adata.file.close()

    effective_marker_path = resolve_marker_scores_memmap_path(
        report_config,
        marker_metadata,
        populate_cache=False,
    )
    data_path = effective_marker_path.with_suffix(".dat")
    n_workers = _report_worker_count(report_config, ceiling=32)
    logger.info(
        f"Reading the report marker sample with {n_workers} I/O workers from {data_path}; "
        "the complete marker matrix will not be staged solely for PCC."
    )
    marker_scores = _read_marker_score_rows(
        data_path=data_path,
        shape=shape,
        dtype=np.dtype(marker_metadata["dtype"]),
        marker_rows=marker_rows,
        gene_positions=gene_positions,
        n_workers=n_workers,
    )
    logger.info(f"Prepared the PCC marker sample in {time.perf_counter() - started:.2f}s")
    return _GssPccSample(
        common_spots=common_spots,
        analysis_spots=analysis_spots,
        gene_names=gene_names,
        marker_scores=marker_scores,
        annotations=annotations,
    )


def _calculate_gene_annotation_stats(
    marker_scores: np.ndarray,
    analysis_spots: np.ndarray,
    gene_names: list[str],
    annotations: pd.Series,
) -> pd.DataFrame:
    """Assign each gene to the annotation with the largest median GSS."""

    gss_frame = pd.DataFrame(marker_scores, index=analysis_spots, columns=gene_names)
    grouped_gss = gss_frame.groupby(annotations).median()
    gene_stats = pd.DataFrame(
        {
            "gene": grouped_gss.columns,
            "Annotation": grouped_gss.idxmax().to_numpy(),
            "Median_GSS": grouped_gss.max().to_numpy(),
        }
    )
    gene_stats.dropna(subset=["Median_GSS"], inplace=True)
    return gene_stats


def _calculate_pcc_for_traits_fast(
    traits: list[str],
    ldsc_results: pd.DataFrame,
    analysis_spots: np.ndarray,
    gss_centered: np.ndarray,
    gss_ssq: np.ndarray,
) -> np.ndarray:
    """Calculate every requested gene-trait PCC in one BLAS matrix multiply."""

    missing_traits = [trait for trait in traits if trait not in ldsc_results.columns]
    if missing_traits:
        raise KeyError(f"Traits are absent from combined LDSC results: {missing_traits}")

    trait_matrix = ldsc_results.loc[analysis_spots, traits].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    if np.isnan(trait_matrix).any():
        affected = [
            trait for index, trait in enumerate(traits) if np.isnan(trait_matrix[:, index]).any()
        ]
        raise ValueError(f"NaN values found in LDSC results for traits: {affected}")

    trait_matrix -= trait_matrix.mean(axis=0, keepdims=True)
    trait_ssq = np.einsum(
        "ij,ij->j",
        trait_matrix,
        trait_matrix,
        dtype=np.float32,
        optimize=True,
    )
    started = time.perf_counter()
    pcc_matrix = gss_centered.T @ trait_matrix
    pcc_matrix /= np.sqrt(gss_ssq[:, None] * trait_ssq[None, :]) + 1e-12
    elapsed = time.perf_counter() - started
    logger.info(
        f"Calculated {len(traits):,} trait correlations together for "
        f"{len(analysis_spots):,} spots and {gss_centered.shape[1]:,} genes in {elapsed:.2f}s"
    )
    return pcc_matrix


def _save_pcc_for_trait(
    trait: str,
    pccs: np.ndarray,
    gene_names: list[str],
    gene_stats: pd.DataFrame | None,
    report_config: QuickModeConfig,
) -> pd.DataFrame:
    """Attach gene metadata, sort, and atomically save one trait's PCC results."""

    trait_pcc = pd.DataFrame({"gene": gene_names, "PCC": pccs, "trait": trait})
    if gene_stats is not None:
        trait_pcc = trait_pcc.merge(gene_stats, on="gene", how="left")
    trait_pcc.sort_values("PCC", ascending=False, inplace=True)
    _atomic_to_csv(
        trait_pcc,
        report_config.get_gene_diagnostic_info_save_path(trait),
        index=False,
    )
    return trait_pcc


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


def _spatial_coordinate_info_is_current(path: Path, coordinate_info: dict | None) -> bool:
    """Return whether cached Spatial3D coordinates use the current provenance/scaling."""

    if coordinate_info is None:
        return True
    if not path.is_file():
        return False
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return cached == coordinate_info


def _write_spatial_coordinate_info(path: Path, coordinate_info: dict | None) -> None:
    if coordinate_info is None:
        return
    _write_text_if_changed(
        path,
        json.dumps(coordinate_info, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )


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
    spatial_3d_info: dict | None = None,
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
    report_meta["has_3d_widget"] = spatial_3d_info is not None
    report_meta["spatial_3d"] = spatial_3d_info
    report_meta["spatial_3d_widget_path"] = None

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

    if manifest.get("version") != SPATIAL_PAYLOAD_VERSION:
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


def _export_per_sample_spatial_js(
    data_dir: Path,
    js_data_dir: Path,
    meta: dict,
    metadata: pd.DataFrame | None = None,
):
    """Export compact per-sample geometry and lazy trait payloads."""
    metadata_file = data_dir / "spot_metadata.csv"
    if not metadata_file.exists():
        logger.warning("spot_metadata.csv not found, skipping per-sample spatial export")
        return

    logger.info("Exporting compact per-sample spatial payloads...")
    if metadata is None:
        metadata = pd.read_csv(metadata_file)

    samples = meta.get("samples", [])
    traits = meta.get("traits", [])
    annotations = meta.get("annotations", [])

    if not samples:
        samples = (
            metadata["sample_name"].drop_duplicates().tolist() if "sample_name" in metadata else []
        )

    js_data_dir.mkdir(parents=True, exist_ok=True)
    sample_index, output_files = export_sample_spatial_payloads(
        metadata=metadata,
        samples=samples,
        traits=traits,
        annotations=annotations,
        js_data_dir=js_data_dir,
        max_points=int(meta.get("downsampling_n_spots_2d", 250000)),
    )

    # Export lightweight sample index (loaded upfront)
    js_content = (
        "window.GSMAP_SAMPLE_INDEX="
        + json.dumps(sample_index, ensure_ascii=False, separators=(",", ":"))
        + ";"
    )
    _atomic_write_text(js_data_dir / "sample_index.js", js_content)
    output_files.append("sample_index.js")

    manifest = {
        "version": SPATIAL_PAYLOAD_VERSION,
        "signature": _sample_spatial_export_signature(data_dir, meta),
        "outputs": sorted(output_files),
    }
    _atomic_write_text(
        js_data_dir / "sample_spatial_manifest.json",
        json.dumps(manifest, separators=(",", ":"), sort_keys=True),
    )

    logger.info(
        f"Exported {len(sample_index)} compact sample bases and "
        f"{sum(len(info['trait_files']) for info in sample_index.values())} lazy trait payloads"
    )


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

"""Bounded streaming engine for spatial LDSC."""

import logging
import os
import time
from collections import deque
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from gsMap.config import SpatialLDSCConfig
from gsMap.config.spatial_ldsc_config import resolve_spatial_ldsc_batch_size
from gsMap.spatial_ldsc.backends.cpu_sparse import available_cpu_count
from gsMap.spatial_ldsc.data.chunks import SpatialLDChunkSource
from gsMap.spatial_ldsc.data.loading import prepare_trait_data
from gsMap.spatial_ldsc.data.traits import prepare_snp_data_for_blocks
from gsMap.spatial_ldsc.data.writer import ResultWriter
from gsMap.spatial_ldsc.execution.pipeline import LDScoreComputePool, LDScoreStream
from gsMap.spatial_ldsc.kernels.jax_regression import process_spatial_ldsc_batch

logger = logging.getLogger("gsMap.spatial_ldsc.engine")

_MIN_COMMON_GENE_OVERLAP = 200
_UNMAPPED_WEIGHT_COLUMN = "Unmapped"
_FEATURE_NAME_EXAMPLE_LIMIT = 5


def _align_marker_score_genes(
    marker_gene_names,
    weight_gene_names,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Align marker-score features to mapped SNP-weight genes.

    Feature matching is intentionally exact and case-sensitive. More than 200
    common genes are required before spatial LDSC proceeds to block-jackknife
    estimation; a smaller overlap usually indicates incompatible feature-name
    conventions rather than a usable biological gene set.
    """

    marker_gene_index = pd.Index(np.asarray(marker_gene_names, dtype=str))
    weight_gene_index = pd.Index(np.asarray(weight_gene_names, dtype=str))
    mapped_weight_gene_index = weight_gene_index[weight_gene_index != _UNMAPPED_WEIGHT_COLUMN]

    common_genes_mask = marker_gene_index.isin(mapped_weight_gene_index)
    common_genes = marker_gene_index[common_genes_mask].to_numpy(dtype=str)
    if common_genes.size <= _MIN_COMMON_GENE_OVERLAP:
        marker_only = marker_gene_index[~common_genes_mask][:_FEATURE_NAME_EXAMPLE_LIMIT].tolist()
        weight_only = mapped_weight_gene_index[~mapped_weight_gene_index.isin(marker_gene_index)][
            :_FEATURE_NAME_EXAMPLE_LIMIT
        ].tolist()
        raise ValueError(
            "Insufficient feature-name overlap between the marker-score file and "
            f"SNP-gene weight matrix: found {common_genes.size} exactly matching genes, "
            f"but more than {_MIN_COMMON_GENE_OVERLAP} are required for spatial-LDSC "
            "block-jackknife estimation. Feature matching is exact and case-sensitive; "
            "check gene-symbol capitalization, identifier type, species conversion, and "
            "Ensembl version suffixes. "
            f"Marker-score features={marker_gene_index.size}, mapped weight-matrix "
            f"features={mapped_weight_gene_index.size}. Example marker-only features: "
            f"{marker_only or ['<none>']}. Example weight-only features: "
            f"{weight_only or ['<none>']}."
        )

    weight_gene_indices = [weight_gene_index.get_loc(gene) for gene in common_genes]
    return common_genes_mask, common_genes, weight_gene_indices


def _canonicalize_sparse_weight_matrix(matrix):
    """Return canonical CSR required by unchecked PyTorch sparse kernels.

    Reordering the gene columns to marker-score order generally leaves SciPy's
    CSR column indices unsorted.  SciPy accepts that representation, but a
    ``torch.sparse_csr_tensor`` constructed with invariant checks disabled can
    segfault during GPU multiplication.  Canonicalize once per trait before
    handing the matrix to either backend.
    """

    if not hasattr(matrix, "tocsr"):
        return matrix
    matrix = matrix.tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


class SpatialLDSCEngine:
    """Bounded spatial-LDSC source, compute, and output engine."""

    def __init__(
        self,
        config: SpatialLDSCConfig,
        output_dir: Path,
        marker_score_adata: ad.AnnData | None,
        snp_gene_weight_adata: ad.AnnData | None,
        baseline_ld: pd.DataFrame,
        w_ld: pd.DataFrame,
        ldscore_adata: ad.AnnData | None = None,
    ):
        """Initialize the spatial-LDSC engine.

        Args:
            config: Configuration object
            output_dir: Output directory for results
            marker_score_adata: AnnData with marker scores in .X (weight-matrix mode).
                                 Pass None when using ldscore_adata.
            snp_gene_weight_adata: AnnData with SNP-gene weights (weight-matrix mode).
                                    Pass None when using ldscore_adata.
            baseline_ld: Baseline LD scores common to all traits
            w_ld: Weights common to all traits
            ldscore_adata: Pre-computed SNP × Cell LD score AnnData (obs=SNPs, var=cells).
                           When provided, activates direct ldscore mode and marker_score_adata /
                           snp_gene_weight_adata are ignored.
        """
        self.config = config
        self.output_dir = output_dir
        self.baseline_ld = baseline_ld
        self.w_ld = w_ld
        # Trait specific state
        self.trait_name = None
        self.data_truncated = None

        self.result_writer = None
        self.chunk_source = None
        self.defer_ldscore_to_compute = False

        if ldscore_adata is not None:
            # ── Direct ldscore mode ──────────────────────────────────────────
            self.mode = "direct_ldscore"
            self.ldscore_adata = ldscore_adata
            self.marker_score_adata = None
            self.snp_gene_weight_adata = None
            # Cells are var (columns) in the ldscore h5ad
            self.n_spots = ldscore_adata.n_vars
            self.spot_names_all = np.array(ldscore_adata.var_names)
            self.marker_score_gene_indices = None
            self.weight_gene_indices = None
            logger.info(
                f"Direct ldscore mode: {self.n_spots} cells, "
                f"{ldscore_adata.n_obs} SNPs in ldscore h5ad"
            )
        else:
            # ── Weight-matrix mode ──────────────────────────────────────────
            self.mode = "weight_matrix"
            self.ldscore_adata = None
            self.marker_score_adata = marker_score_adata
            self.snp_gene_weight_adata = snp_gene_weight_adata
            self.n_spots = marker_score_adata.n_obs
            self.spot_names_all = np.array(self.marker_score_adata.obs_names)

            logger.info(
                f"Detected Marker scores shape: (n_spots={self.n_spots}, n_genes={self.marker_score_adata.n_vars})"
            )

            if self.snp_gene_weight_adata is None:
                raise ValueError("snp_gene_weight_adata must be provided in weight-matrix mode")

            # Find common genes and initialize indices. Fail before allocating
            # trait-specific matrices when the feature namespaces disagree.
            common_genes_mask, common_genes, self.weight_gene_indices = _align_marker_score_genes(
                self.marker_score_adata.var_names,
                self.snp_gene_weight_adata.var_names,
            )
            self.marker_score_gene_indices = np.where(common_genes_mask)[0]

            logger.info(f"Found {len(common_genes)} common genes")

        # Filter by sample if specified
        if self.mode == "direct_ldscore" and self.config.sample_filter:
            raise ValueError(
                "sample_filter is not supported in direct ldscore mode. "
                "Use cell_indices_range to process a subset of cells."
            )

        if self.config.sample_filter:
            logger.info(f"Filtering spots by sample: {self.config.sample_filter}")
            sample_info = self.marker_score_adata.obs.get(
                "sample", self.marker_score_adata.obs.get("sample_name", None)
            )

            if sample_info is None:
                raise ValueError("No 'sample' or 'sample_name' column found in obs")

            sample_info = sample_info.to_numpy()
            self.spot_indices = np.where(sample_info == self.config.sample_filter)[0]
            if self.spot_indices.size == 0:
                raise ValueError(f"No spots found for sample '{self.config.sample_filter}'")
            self.spot_names_filtered = self.spot_names_all[self.spot_indices]
            logger.info(
                f"Found {len(self.spot_indices)} spots for sample '{self.config.sample_filter}'"
            )
        else:
            self.spot_indices = np.arange(self.n_spots)
            self.spot_names_filtered = self.spot_names_all

        self.n_spots_filtered = len(self.spot_indices)

        # Resolve automatic sizing on the actual execution device, including a
        # requested GPU falling back to CPU. Multi-GPU coordinators must not
        # initialize JAX: their workers are GPU-only and enforce that contract.
        requested_batch = self.config.spots_per_chunk_quick_mode
        if requested_batch is None:
            if getattr(self.config, "ldsc_num_gpus", 1) > 1:
                platform = "gpu"
            else:
                import jax

                platform = jax.default_backend()
            batch_source = f"auto: {platform.upper()}"
        else:
            platform = ""  # An explicit override needs no device discovery.
            batch_source = "explicit override"
        self.chunk_size = resolve_spatial_ldsc_batch_size(requested_batch, platform)
        logger.info("Spatial LDSC batch size: %d spots/chunk (%s)", self.chunk_size, batch_source)

        # Handle cell indices range if specified
        if self.config.cell_indices_range:
            start_cell, end_cell = self.config.cell_indices_range
            # Adjust for filtered spots
            start_cell = max(0, start_cell)
            end_cell = min(end_cell, self.n_spots_filtered)
            if start_cell >= end_cell:
                raise ValueError(
                    f"cell_indices_range [{start_cell}, {end_cell}) is empty after "
                    f"clamping to {self.n_spots_filtered} available cells"
                )
            self.chunk_starts = list(range(start_cell, end_cell, self.chunk_size))
            self._cell_end = end_cell  # cap for the last (possibly partial) chunk
            logger.info(f"Processing cell range [{start_cell}, {end_cell})")
            self.total_cells_to_process = end_cell - start_cell
        else:
            if self.n_spots_filtered == 0:
                raise ValueError("No cells are available for spatial LDSC")
            self.chunk_starts = list(range(0, self.n_spots_filtered, self.chunk_size))
            self._cell_end = self.n_spots_filtered
            self.total_cells_to_process = self.n_spots_filtered

        self.total_chunks = len(self.chunk_starts)
        logger.info(f"Total chunks to process: {self.total_chunks}")

    def setup_trait(self, trait_name: str, sumstats_file: str):
        """
        Setup processor for a new trait.
        Loads sumstats, prepares data, and initializes trait-specific components.
        """
        self.trait_name = trait_name
        logger.info(f"Setting up processor for trait: {trait_name}")

        # Prepare trait data — pass the appropriate reference adata for snp_positions
        snp_ref = (
            self.ldscore_adata if self.mode == "direct_ldscore" else self.snp_gene_weight_adata
        )
        data, common_snps = prepare_trait_data(
            self.config,
            trait_name,
            sumstats_file,
            self.baseline_ld,
            self.w_ld,
            snp_ref,
        )

        # Prepare blocks
        self.data_truncated = prepare_snp_data_for_blocks(data, self.config.n_blocks)

        logger.info(f"Initializing trait-specific components for trait: {self.trait_name}")
        snp_positions = self.data_truncated.get("snp_positions", None)
        if snp_positions is None:
            raise ValueError("snp_positions not found in data_truncated")

        if self.mode == "direct_ldscore":
            # Store SNP positions for direct slicing from ldscore_adata
            self.ldscore_snp_positions = snp_positions
            logger.info(
                f"Direct ldscore: {len(snp_positions)} SNP positions stored for trait {trait_name}"
            )
        else:
            # Extract SNP-gene weight matrix for this trait's SNPs
            self.snp_gene_weight_sparse = self.snp_gene_weight_adata.X[snp_positions, :][
                :, self.weight_gene_indices
            ]

            self.snp_gene_weight_sparse = _canonicalize_sparse_weight_matrix(
                self.snp_gene_weight_sparse
            )

            logger.info(f"SNP-gene weight matrix shape: {self.snp_gene_weight_sparse.shape}")

        # Reset results
        self.result_writer = ResultWriter(
            config=self.config,
            trait_name=self.trait_name,
            output_dir=self.output_dir,
            n_spots=self.n_spots,
            total_chunks=self.total_chunks,
            expected_cells=self.total_cells_to_process,
        )
        self.chunk_source = SpatialLDChunkSource(
            mode=self.mode,
            chunk_starts=self.chunk_starts,
            chunk_size=self.chunk_size,
            cell_end=self._cell_end,
            spot_indices=self.spot_indices,
            spot_names_filtered=self.spot_names_filtered,
            ldscore_adata=self.ldscore_adata,
            ldscore_snp_positions=getattr(self, "ldscore_snp_positions", None),
            marker_score_adata=self.marker_score_adata,
            marker_score_gene_indices=self.marker_score_gene_indices,
            snp_gene_weight_sparse=getattr(self, "snp_gene_weight_sparse", None),
        )

    def _fetch_ldscore_chunk(self, chunk_index: int) -> tuple[np.ndarray, pd.Index, int, int]:
        """
        Fetch LD score chunk for given index.

        Returns:
            Tuple of (ldscore_array, spot_names, absolute_start, absolute_end)
        """
        if self.chunk_source is None:
            raise RuntimeError("Chunk source is not initialized; call setup_trait first")
        return self.chunk_source.fetch_chunk(
            chunk_index=chunk_index,
            defer_ldscore_to_compute=self.defer_ldscore_to_compute,
        )

    def _compute_ldscore_from_marker_chunk(self, marker_score_chunk: np.ndarray) -> np.ndarray:
        """Compute SNP x spot LD scores from a gene x spot marker-score chunk on CPU."""
        if self.chunk_source is None:
            raise RuntimeError("Chunk source is not initialized; call setup_trait first")
        return self.chunk_source.compute_ldscore_from_marker_chunk(marker_score_chunk)

    def _resolve_compute_workers(self, backend: str) -> int:
        """Decide how many compute (regression) worker threads to run.

        JAX parallelizes each regression internally. CPU workers allow concurrent
        batches, but share cores and memory bandwidth with JAX and sparse transforms;
        a blocked Python worker does not reserve a dedicated core. Cap dispatch
        concurrency at half the affinity/allocation budget (at least one), retaining
        the low-core safeguard tested in CI. GPU uses one asynchronous dispatcher
        to avoid cuSolver handle contention and excess device-memory pressure.

        ``GSMAP_LDSC_FORCE_COMPUTE_WORKERS`` overrides the heuristic for power users
        and benchmarking (use with care — it can reintroduce the deadlock on few-core
        machines).
        """
        configured = self.config.ldsc_compute_workers

        forced = os.environ.get("GSMAP_LDSC_FORCE_COMPUTE_WORKERS")
        if forced is not None:
            n = max(1, int(forced))
            logger.warning(
                f"GSMAP_LDSC_FORCE_COMPUTE_WORKERS={forced}: forcing {n} compute "
                f"worker(s), bypassing the safety cap (backend={backend})"
            )
            return n

        if backend == "gpu":
            cap = 1
        else:
            cap = max(1, available_cpu_count() // 2)

        effective = min(configured, cap)
        if effective != configured:
            logger.info(
                f"Capping ldsc_compute_workers {configured} -> {effective} for "
                f"{backend} backend; JAX parallelizes each regression internally "
                f"(override with GSMAP_LDSC_FORCE_COMPUTE_WORKERS)"
            )
        return effective

    def process_all_chunks(self) -> Path:
        """Run a trait and discard partial output on any initialization/worker failure."""
        if self.result_writer is None or self.chunk_source is None:
            raise RuntimeError("Trait is not initialized; call setup_trait first")
        try:
            return self._process_all_chunks()
        except BaseException:
            self.result_writer.abort()
            raise

    def _process_all_chunks(self) -> Path:
        """Process all chunks through the production streaming pipeline.

        Returns:
            Atomically published compressed result path
        """
        import jax as _jax

        # Create a bounded source-compute pipeline. The source is started only
        # after the consumer exists, so neither bounded queue can deadlock
        # during initialization.
        source_stream = LDScoreStream(
            processor=self,
            num_workers=self.config.ldsc_read_workers,
        )

        backend = _jax.default_backend()
        effective_compute_workers = self._resolve_compute_workers(backend)

        # GPU dispatches ahead without a per-batch device barrier. In the normal
        # sparse-GPU path only marker batches are uploaded; LD scores stay on the
        # device through DLPack. CPU runs concurrent blocking regression workers.
        pipeline_depth = 0
        if backend == "gpu":
            pipeline_depth = max(0, int(os.environ.get("GSMAP_LDSC_GPU_PIPELINE_DEPTH", "2")))

        computer = LDScoreComputePool(
            processor=self,
            regression_fn=process_spatial_ldsc_batch,
            num_workers=effective_compute_workers,
            input_queue=source_stream.result_queue,
            pipeline_depth=pipeline_depth,
        )

        start_time = time.time()
        try:
            source_stream.start()

            # Build description with sample name and range info
            desc_parts = [
                f"Processing {self.total_chunks:,} chunks ({self.total_cells_to_process:,} cells)"
            ]

            if hasattr(self.config, "sample_filter") and self.config.sample_filter:
                desc_parts.append(f"Sample: {self.config.sample_filter}")

            if self.config.cell_indices_range:
                start_cell, end_cell = self.config.cell_indices_range
                desc_parts.append(f"Range: [{start_cell:,}-{end_cell:,})")

            description = " | ".join(desc_parts)

            # Rich Console auto-detects terminal/Jupyter environments
            console = Console(soft_wrap=True)

            # Log interval for non-interactive mode (every 10% or at least every 60 seconds)
            log_interval_pct = 10
            last_log_pct = 0
            last_log_time = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TextColumn("[bold green]{task.fields[speed]} cells/s"),
                TextColumn("[dim]I/O: {task.fields[io_speed]} MiB/s"),
                TextColumn("[dim]S→C: {task.fields[r_to_c_queue]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=2,
            ) as progress:
                task = progress.add_task(
                    description,
                    total=self.total_cells_to_process,
                    speed="0",
                    io_speed="-",
                    r_to_c_queue="0",
                )

                last_update_time = start_time
                last_chunks_processed = 0

                # For 10s moving average
                speed_window = deque()  # Stores (timestamp, n_cells_processed)

                while last_chunks_processed < self.total_chunks:
                    # Check for errors
                    source_stream.check_errors()
                    computer.check_errors()
                    self.result_writer.check_errors()

                    # Get current stats from computer
                    n_cells_processed, n_chunks_processed = computer.get_stats()

                    # Update progress periodically
                    current_time = time.time()
                    if current_time - last_update_time > 0.5:  # Update every 0.5 seconds
                        # Get queue sizes
                        r_pending, r_to_c = source_stream.get_queue_sizes()
                        io_speed = (
                            f"{source_stream.sequential_io_throughput.mib_per_second:,.1f}"
                            if source_stream.uses_sequential_windows
                            else "-"
                        )

                        # Calculate speed in cells/s (10s moving average)
                        speed_window.append((current_time, n_cells_processed))

                        # Remove entries older than 10s
                        while speed_window and current_time - speed_window[0][0] > 10.0:
                            speed_window.popleft()

                        if len(speed_window) > 1:
                            t_diff = speed_window[-1][0] - speed_window[0][0]
                            c_diff = speed_window[-1][1] - speed_window[0][1]
                            speed_10s = c_diff / t_diff if t_diff > 0 else 0
                        else:
                            # Fallback to cumulative speed if not enough data for window
                            elapsed_total = current_time - start_time
                            speed_10s = (
                                n_cells_processed / elapsed_total if elapsed_total > 0 else 0
                            )

                        # Update progress bar (only effective in terminal mode)
                        progress.update(
                            task,
                            completed=n_cells_processed,
                            speed=f"{speed_10s:,.0f}",
                            io_speed=io_speed,
                            r_to_c_queue=f"{r_to_c}",
                        )

                        # Log progress when output is not interactive (redirected to file)
                        if not console.is_interactive:
                            current_pct = (
                                (n_cells_processed / self.total_cells_to_process) * 100
                                if self.total_cells_to_process > 0
                                else 0
                            )
                            time_since_last_log = current_time - last_log_time

                            # Log every 10% or every 60 seconds
                            if (
                                current_pct >= last_log_pct + log_interval_pct
                                or time_since_last_log >= 60
                            ):
                                elapsed = current_time - start_time
                                logger.info(
                                    f"Progress: {n_cells_processed:,}/{self.total_cells_to_process:,} cells "
                                    f"({current_pct:.1f}%) | {n_chunks_processed}/{self.total_chunks} chunks | "
                                    f"Throughput: {speed_10s:,.0f} cell-traits/s (1 trait) | "
                                    f"Source I/O: {io_speed} MiB/s | "
                                    f"Source pending: {r_pending:,} | Ready queue: {r_to_c:,} | "
                                    f"Elapsed: {elapsed:.1f}s"
                                )
                                last_log_pct = (current_pct // log_interval_pct) * log_interval_pct
                                last_log_time = current_time

                        last_update_time = current_time

                    # Update last processed count
                    if n_chunks_processed > last_chunks_processed:
                        last_chunks_processed = n_chunks_processed

                    # Small sleep to prevent busy waiting
                    time.sleep(0.5)

        except BaseException:
            # Unblock compute producers before joining their worker threads.
            self.result_writer.abort()
            raise
        finally:
            # Clean up resources
            source_stream.close()
            computer.close()

            if source_stream.uses_sequential_windows:
                io_stats = source_stream.sequential_io_throughput
                logger.info(
                    "Sequential marker-score I/O: "
                    f"{io_stats.total_windows:,} windows, "
                    f"{io_stats.total_cells:,} cells, "
                    f"{io_stats.mib_per_second:,.0f} MiB/s "
                    f"({io_stats.cells_per_second:,.0f} cells/s while reading)"
                )
            source_label = (
                f"Marker-score transform throughput ({computer.gpu_ldscore_backend} GPU LD-score)"
                if getattr(computer, "use_gpu_ldscore", False)
                else (
                    "LD-score construction throughput"
                    if source_stream.uses_sequential_windows
                    else "LD-score read/construct throughput"
                )
            )
            logger.info(
                f"{source_label}: {source_stream.throughput.total_batches} chunks, "
                f"avg {source_stream.throughput.average_time:.3f}s/chunk"
            )
            logger.info(
                "LDSC regression throughput: "
                f"{computer.throughput.total_batches} chunks, "
                f"avg {computer.throughput.average_time:.3f}s/chunk"
            )

        # Validate completeness and atomically publish the streamed result.
        output_path = self._validate_merge_and_save()
        total_elapsed = time.time() - start_time
        overall_speed = self.total_cells_to_process / total_elapsed if total_elapsed > 0 else 0
        logger.info(
            f"Processing complete: {self.total_cells_to_process:,} cells in "
            f"{total_elapsed:.2f}s ({overall_speed:,.0f} cells/s, including output finalization)"
        )
        return output_path

    def _add_chunk_result(
        self,
        chunk_idx: int,
        betas: np.ndarray,
        ses: np.ndarray,
        spot_names: pd.Index,
        abs_start: int,
        abs_end: int,
    ):
        """Add a processed chunk to the bounded ordered output stream."""
        if self.result_writer is None:
            raise RuntimeError("Result writer is not initialized; call setup_trait first")
        self.result_writer.add_chunk_result(
            chunk_idx=chunk_idx,
            betas=betas,
            ses=ses,
            spot_names=spot_names,
            abs_start=abs_start,
            abs_end=abs_end,
        )

    def _validate_merge_and_save(self) -> Path:
        """Validate completeness and publish the streamed result."""
        if self.result_writer is None:
            raise RuntimeError("Result writer is not initialized; call setup_trait first")
        return self.result_writer.finalize()

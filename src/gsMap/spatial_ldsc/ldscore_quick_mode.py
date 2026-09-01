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

from .chunk_source import SpatialLDChunkSource
from .io import prepare_trait_data
from .pipeline import LDScoreComputePool, LDScoreStream
from .regression_kernel import process_spatial_ldsc_batch
from .result_writer import ResultWriter
from .trait_data import prepare_snp_data_for_blocks

logger = logging.getLogger("gsMap.spatial_ldsc.engine")


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

            # Find common genes and initialize indices
            gene_names_from_adata = np.array(self.marker_score_adata.var_names)

            if self.snp_gene_weight_adata is None:
                raise ValueError("snp_gene_weight_adata must be provided in weight-matrix mode")

            marker_score_genes_series = pd.Series(gene_names_from_adata)
            common_genes_mask = marker_score_genes_series.isin(
                self.snp_gene_weight_adata.var.index
            )
            common_genes = gene_names_from_adata[common_genes_mask]

            self.marker_score_gene_indices = np.where(common_genes_mask)[0]
            self.weight_gene_indices = [
                self.snp_gene_weight_adata.var.index.get_loc(g) for g in common_genes
            ]

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

        # Set up chunking
        self.chunk_size = self.config.spots_per_chunk_quick_mode

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

            if hasattr(self.snp_gene_weight_sparse, "tocsr"):
                self.snp_gene_weight_sparse = self.snp_gene_weight_sparse.tocsr()

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

        Each compute worker dispatches a jitted regression and then blocks in
        ``block_until_ready()``. JAX/XLA already parallelizes a *single* regression
        across cores (CPU intra-op pool) or on-device (GPU), so the workers buy
        throughput only by overlapping Python-side overhead (host<->device array
        conversion, the lock-guarded result accumulation) with JAX compute — not by
        parallelizing the math itself. The right count is therefore backend- and
        core-dependent:

        * **GPU** — a single on-device queue already saturates the device; extra
          workers add cuSolver handle-pool races and device-memory pressure. Pin to 1.
        * **CPU** — extra workers help on many-core hosts, but each one occupies a
          core while blocked. If the worker count approaches/exceeds the available
          cores, XLA's bounded intra-op pool is starved and the pipeline deadlocks
          (observed on 2-core CI). So cap at ``cpu_count // 2`` (>=1), which leaves
          headroom for the intra-op pool and source/transform threads, and naturally collapses
          to 1 on a 2-core box.

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
            try:
                n_cores = len(os.sched_getaffinity(0))
            except AttributeError:  # not available on all platforms
                n_cores = os.cpu_count() or 1
            cap = max(1, n_cores // 2)

        effective = min(configured, cap)
        if effective != configured:
            logger.info(
                f"Capping ldsc_compute_workers {configured} -> {effective} for "
                f"{backend} backend; JAX parallelizes each regression internally "
                f"(override with GSMAP_LDSC_FORCE_COMPUTE_WORKERS)"
            )
        return effective

    def process_all_chunks(self) -> Path:
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

        # On GPU a single stream already saturates the device and extra compute
        # threads only add cuSolver-handle contention, so instead of more workers
        # we double-buffer: one worker dispatches `pipeline_depth` chunks ahead and
        # blocks lazily, overlapping the (large) host->device LD-score upload and
        # result copy-back with on-device compute. Benchmarked ~1.25x over the
        # block-after-dispatch path. CPU keeps the multi-worker threads (depth 0),
        # which win there by running concurrent regressions across idle cores.
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
                                console.log(
                                    f"Progress: {n_cells_processed:,}/{self.total_cells_to_process:,} cells "
                                    f"({current_pct:.1f}%) | {n_chunks_processed}/{self.total_chunks} chunks | "
                                    f"Speed: {speed_10s:,.0f} cells/s | "
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

                        # This would block all threads, so we avoid it
                        # # Periodic memory check
                        # if n_chunks_processed % 100 == 0:
                        #     gc.collect()
                    # Small sleep to prevent busy waiting
                    time.sleep(0.5)

        except BaseException:
            if self.result_writer is not None:
                self.result_writer.abort()
            raise
        finally:
            # Clean up resources
            source_stream.close()
            computer.close()

            # Log overall speed
            total_elapsed = time.time() - start_time
            n_cells_total, _ = computer.get_stats()
            overall_speed = n_cells_total / total_elapsed if total_elapsed > 0 else 0
            logger.info(f"Processing complete. Overall speed: {overall_speed:,.0f} cells/s")
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
        return self._validate_merge_and_save()

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

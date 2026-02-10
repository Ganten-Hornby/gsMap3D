"""
Marker score calculation using homogeneous neighbors
Implements weighted geometric mean calculation in log space with JAX acceleration
"""

import gc
import json
import logging
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import anndata as ad
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from jax import jit

# Progress bar imports
from rich.console import Console, Group
from rich.panel import Panel
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
from rich.table import Table

from gsMap.config import DatasetType, MarkerScoreCrossSliceStrategy

from .connectivity import ConnectivityMatrixBuilder
from .memmap_io import (
    ComponentThroughput,
    MemMapDense,
    ParallelMarkerScoreWriter,
    ParallelRankReader,
)
from .row_ordering_jax import optimize_row_order_jax

logger = logging.getLogger(__name__)


class ParallelMarkerScoreComputer:
    """Multi-threaded computer pool for marker score calculation (reusable across cell types)"""

    def __init__(
        self,
        global_log_gmean: np.ndarray,
        global_expr_frac: np.ndarray,
        homogeneous_neighbors: int,
        num_workers: int = 4,
        input_queue: queue.Queue = None,
        output_queue: queue.Queue = None,
        cross_slice_strategy: str = None,
        n_slices: int = 1,
        num_homogeneous_per_slice: int = None,
        no_expression_fraction: bool = False,
    ):
        """
        Initialize computer pool

        Args:
            global_log_gmean: Global log geometric mean
            global_expr_frac: Global expression fraction
            homogeneous_neighbors: Number of homogeneous neighbors
            num_workers: Number of compute workers
            input_queue: Optional input queue (from reader)
            output_queue: Optional output queue (to writer)
            cross_slice_strategy: Strategy for 3D ('per_slice_pool' or 'hierarchical_pool')
            n_slices: Number of slices for 3D data
            num_homogeneous_per_slice: Neighbors per slice for 3D strategies
            no_expression_fraction: Skip expression fraction filtering if True
        """
        self.num_workers = num_workers
        self.homogeneous_neighbors = homogeneous_neighbors
        self.cross_slice_strategy = cross_slice_strategy
        self.n_slices = n_slices
        self.num_homogeneous_per_slice = num_homogeneous_per_slice or homogeneous_neighbors
        self.no_expression_fraction = no_expression_fraction

        # Store global statistics as JAX arrays
        self.global_log_gmean = jnp.array(global_log_gmean)
        self.global_expr_frac = jnp.array(global_expr_frac)

        # Queues for communication
        self.compute_queue = input_queue if input_queue else queue.Queue(maxsize=num_workers * 2)
        self.result_queue = output_queue if output_queue else queue.Queue(maxsize=num_workers * 2)

        # Throughput tracking
        self.throughput = ComponentThroughput()
        self.throughput_lock = threading.Lock()

        # Current processing context
        self.neighbor_weights = None
        self.cell_indices_sorted = None
        self.batch_size = None
        self.n_cells = None

        # Exception handling
        self.exception_queue = queue.Queue()
        self.has_error = threading.Event()

        # Start worker threads
        self.workers = []
        self.stop_workers = threading.Event()
        self.active_cell_type = None  # Track current cell type being processed
        self._start_workers()

    def _start_workers(self):
        """Start compute worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._compute_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.num_workers} compute workers")

    def _compute_worker(self, worker_id: int):
        """Compute worker thread"""
        logger.info(f"Compute worker {worker_id} started")

        while not self.stop_workers.is_set():
            try:
                # Get data from reader
                item = self.compute_queue.get(timeout=1)
                if item is None:
                    break

                # Unpack data from reader
                batch_idx, rank_data, rank_indices, original_shape, batch_metadata = item

                # Track timing
                start_time = time.time()

                # Extract batch parameters from metadata
                # These should always be provided, but check for safety
                if self.batch_size is None or self.neighbor_weights is None:
                    logger.error(f"Compute worker {worker_id}: batch context not set")
                    raise RuntimeError("Batch context must be set before processing")

                batch_start = batch_metadata["batch_start"]
                batch_end = batch_metadata["batch_end"]
                actual_batch_size = batch_end - batch_start

                # Verify shape
                assert original_shape == (
                    actual_batch_size,
                    self.homogeneous_neighbors * self.n_slices,
                ), (
                    f"Unexpected rank data shape: {original_shape}, expected ({actual_batch_size}, {self.homogeneous_neighbors * self.n_slices})"
                )

                # Get batch-specific data
                batch_weights = self.neighbor_weights[batch_start:batch_end]
                batch_cell_indices = self.cell_indices_sorted[batch_start:batch_end]

                # Convert to JAX for efficient computation
                rank_data_jax = jnp.array(rank_data)
                rank_indices_jax = jnp.array(rank_indices)

                # Use JAX fancy indexing
                batch_ranks = rank_data_jax[rank_indices_jax]

                # Compute marker scores using appropriate strategy
                if self.cross_slice_strategy == "hierarchical_pool":
                    # Use hierarchical pooling (per-slice marker score average) for 3D data
                    marker_scores = compute_marker_scores_3d_hierarchical_pool_jax(
                        batch_ranks,
                        batch_weights,
                        actual_batch_size,
                        self.n_slices,
                        self.num_homogeneous_per_slice,
                        self.global_log_gmean,
                        self.global_expr_frac,
                        self.no_expression_fraction,
                    )
                else:
                    # Use standard computation (includes mean pooling via weights)
                    marker_scores = compute_marker_scores_jax(
                        batch_ranks,
                        batch_weights,
                        actual_batch_size,
                        self.homogeneous_neighbors * self.n_slices,
                        self.global_log_gmean,
                        self.global_expr_frac,
                        self.no_expression_fraction,
                    )

                # Convert back to numpy as float16 for memory efficiency
                marker_scores_np = np.array(marker_scores, dtype=np.float16)

                # Track throughput
                elapsed = time.time() - start_time
                with self.throughput_lock:
                    self.throughput.record_batch(elapsed)

                # Put result directly to writer queue
                self.result_queue.put((batch_idx, marker_scores_np, batch_cell_indices))
                self.compute_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:  # noqa: BLE001
                error_trace = traceback.format_exc()
                logger.error(f"Compute worker {worker_id} error: {e}\nTraceback:\n{error_trace}")
                self.exception_queue.put((worker_id, e, error_trace))
                self.has_error.set()
                self.stop_workers.set()  # Signal all workers to stop
                break

        logger.info(f"Compute worker {worker_id} stopped")

    def set_batch_context(
        self,
        neighbor_weights: np.ndarray,
        cell_indices_sorted: np.ndarray,
        batch_size: int,
        n_cells: int,
    ):
        """Set context for processing batches of current cell type"""
        self.neighbor_weights = jnp.asarray(neighbor_weights)  # transfer to jax array only once
        self.cell_indices_sorted = cell_indices_sorted
        self.batch_size = batch_size
        self.n_cells = n_cells

    def reset_for_cell_type(self, cell_type: str):
        """Reset for processing a new cell type"""
        self.active_cell_type = cell_type
        self.neighbor_weights = None
        self.cell_indices_sorted = None
        self.batch_size = None
        self.n_cells = None
        with self.throughput_lock:
            self.throughput = ComponentThroughput()
        logger.debug(f"Reset computer throughput for {cell_type}")

    def get_queue_sizes(self):
        """Get current queue sizes for progress tracking"""
        return self.compute_queue.qsize(), self.result_queue.qsize()

    def check_errors(self):
        """Check if any worker encountered an error"""
        if self.has_error.is_set():
            try:
                worker_id, exception, error_trace = self.exception_queue.get_nowait()
                raise RuntimeError(
                    f"Compute worker {worker_id} failed: {exception}\nOriginal traceback:\n{error_trace}"
                ) from exception
            except queue.Empty:
                raise RuntimeError("Compute worker failed with unknown error") from None

    def close(self):
        """Close compute pool"""
        self.stop_workers.set()
        for _ in range(self.num_workers):
            self.compute_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5)
        logger.info("Compute pool closed")


@dataclass
class PipelineStats:
    """Statistics for pipeline monitoring"""

    total_batches: int
    completed_reads: int = 0
    completed_computes: int = 0
    completed_writes: int = 0
    pending_compute: int = 0
    pending_write: int = 0
    pending_read: int = 0
    start_time: float = 0

    # Component throughput tracking
    reader_throughput: ComponentThroughput = None
    computer_throughput: ComponentThroughput = None
    writer_throughput: ComponentThroughput = None

    def __post_init__(self):
        self.start_time = time.time()
        self.reader_throughput = ComponentThroughput()
        self.computer_throughput = ComponentThroughput()
        self.writer_throughput = ComponentThroughput()

    @property
    def elapsed_time(self):
        return time.time() - self.start_time

    @property
    def throughput(self):
        if self.elapsed_time > 0:
            return self.completed_writes / self.elapsed_time
        return 0


class MarkerScoreMessageQueue:
    """Streamlined pipeline for marker score calculation (reusable across cell types)"""

    def __init__(
        self,
        reader: ParallelRankReader,
        computer: ParallelMarkerScoreComputer,
        writer: ParallelMarkerScoreWriter,
        batch_size: int,
    ):
        """
        Initialize pipeline with shared pools

        Args:
            reader: Rank reader pool
            computer: Marker score computer pool
            writer: Marker score writer pool
            batch_size: Size of each batch
        """
        self.reader = reader
        self.computer = computer
        self.writer = writer
        self.batch_size = batch_size

        # Cell type specific parameters (set via reset_for_cell_type)
        self.n_batches = None
        self.n_cells = None
        self.active_cell_type = None
        self.stats = None

    def reset_for_cell_type(self, cell_type: str, n_cells: int):
        """Reset the queue for processing a new cell type

        Args:
            cell_type: Name of the cell type
            n_cells: Total number of cells for this cell type
        """
        # Reset all components for new cell type
        self.reader.reset_for_cell_type(cell_type)
        self.computer.reset_for_cell_type(cell_type)
        self.writer.reset_for_cell_type(cell_type)

        self.active_cell_type = cell_type
        self.n_cells = n_cells
        self.n_batches = (n_cells + self.batch_size - 1) // self.batch_size
        self.stats = PipelineStats(total_batches=self.n_batches)
        logger.debug(
            f"Reset MarkerScoreMessageQueue for {cell_type}: {self.n_batches} batches, {n_cells} cells"
        )

    def _submit_batches(
        self,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        cell_indices_sorted: np.ndarray,
    ):
        """Submit all batches to the reader"""
        # Set context for computer
        self.computer.set_batch_context(
            neighbor_weights=neighbor_weights,
            cell_indices_sorted=cell_indices_sorted,
            batch_size=self.batch_size,
            n_cells=self.n_cells,
        )

        # Submit all batches with metadata
        for batch_idx in range(self.n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, self.n_cells)
            batch_neighbors = neighbor_indices[batch_start:batch_end]

            # Include metadata for computer
            batch_metadata = {
                "batch_start": batch_start,
                "batch_end": batch_end,
                "batch_idx": batch_idx,
            }

            self.reader.submit_batch(batch_idx, batch_neighbors, batch_metadata)

    def _update_stats(self):
        """Update pipeline statistics"""
        # Update completed counts
        self.stats.completed_writes = self.writer.get_completed_count()

        # Update queue sizes
        read_pending, read_ready = self.reader.get_queue_sizes()
        self.stats.pending_read = read_pending

        compute_pending, compute_ready = self.computer.get_queue_sizes()
        self.stats.pending_compute = compute_pending

        self.stats.pending_write = self.writer.get_queue_size()

        # Update throughput stats
        self.stats.reader_throughput = self.reader.throughput
        self.stats.computer_throughput = self.computer.throughput
        self.stats.writer_throughput = self.writer.throughput

        # Estimate completed reads and computes based on queue states
        self.stats.completed_reads = self.n_batches - read_pending
        self.stats.completed_computes = self.stats.completed_writes + self.stats.pending_write

    def start(
        self,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        cell_indices_sorted: np.ndarray,
        enable_profiling: bool = False,
    ):
        """Run pipeline with rich progress display

        Args:
            neighbor_indices: Neighbor indices for each cell
            neighbor_weights: Weights for each neighbor
            cell_indices_sorted: Sorted cell indices
            enable_profiling: Whether to enable profiling
        """

        # Ensure the queue has been reset for this cell type
        if self.active_cell_type is None or self.stats is None:
            raise RuntimeError(
                "MarkerScoreMessageQueue must be reset before starting. Call reset_for_cell_type first."
            )

        # Optional profiling
        tracer = None
        if enable_profiling:
            import viztracer  # type: ignore[import-untyped]

            tracer = viztracer.VizTracer(
                output_file=f"marker_score_{self.active_cell_type}_{self.n_batches}.json",
                max_stack_depth=10,
            )
            tracer.start()

        console = Console()

        # Define queue color mapping based on queue size
        def get_queue_color(size: int) -> str:
            """Get color based on queue size"""
            if size == 0:
                return "dim"
            elif size < 5:
                return "green"
            elif size < 10:
                return "yellow"
            elif size < 20:
                return "bright_yellow"
            else:
                return "red"

        try:
            # Create progress bars
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=10,
            ) as progress:
                # Add single task for pipeline
                pipeline_task = progress.add_task(
                    f"[bold]{self.active_cell_type}[/bold]", total=self.n_batches
                )

                # Submit all batches to start the pipeline
                self._submit_batches(neighbor_indices, neighbor_weights, cell_indices_sorted)

                # Monitor progress
                while self.stats.completed_writes < self.n_batches:
                    # Check for errors in any component
                    self.reader.check_errors()
                    self.computer.check_errors()
                    self.writer.check_errors()

                    self._update_stats()

                    # Update progress based on completed writes (final stage)
                    progress.update(pipeline_task, completed=self.stats.completed_writes)

                    # Color code queue sizes based on fullness
                    compute_color = get_queue_color(self.stats.pending_compute)
                    write_color = get_queue_color(self.stats.pending_write)

                    # Update description with queue information
                    progress.update(
                        pipeline_task,
                        description=(
                            f"[bold]{self.active_cell_type}[/bold] | "
                            f"Queues: [{compute_color}]R→C:{self.stats.pending_compute}[/{compute_color}] "
                            f"[{write_color}]C→W:{self.stats.pending_write}[/{write_color}]"
                        ),
                    )

                    time.sleep(0.1)

                # Final update
                progress.update(pipeline_task, completed=self.n_batches)

        except Exception as e:
            logger.error(f"Pipeline failed for {self.active_cell_type}: {e}")
            # Stop all workers to prevent hanging
            self.stop()
            raise

        finally:
            # Stop profiling if enabled
            if tracer is not None:
                tracer.stop()
                tracer.save()
                logger.info(
                    f"Profiling saved to marker_score_{self.active_cell_type}_{self.n_batches}.json"
                )

        # No threads to wait for - processing happens in component worker threads

        # Print summary with component throughputs
        # Calculate effective pipeline throughput (limited by bottleneck)
        min(
            self.stats.reader_throughput.throughput * self.reader.num_workers
            if self.stats.reader_throughput.throughput > 0
            else float("inf"),
            self.stats.computer_throughput.throughput * self.computer.num_workers
            if self.stats.computer_throughput.throughput > 0
            else float("inf"),
            self.stats.writer_throughput.throughput * self.writer.num_workers
            if self.stats.writer_throughput.throughput > 0
            else float("inf"),
        )

        # Calculate cells per second for each component and pipeline
        pipeline_cells_per_sec = (
            self.stats.throughput * self.batch_size if self.stats.throughput > 0 else 0
        )
        reader_cells_per_sec = (
            self.stats.reader_throughput.throughput * self.batch_size * self.reader.num_workers
            if self.stats.reader_throughput.throughput > 0
            else 0
        )
        computer_cells_per_sec = (
            self.stats.computer_throughput.throughput * self.batch_size * self.computer.num_workers
            if self.stats.computer_throughput.throughput > 0
            else 0
        )
        writer_cells_per_sec = (
            self.stats.writer_throughput.throughput * self.batch_size * self.writer.num_workers
            if self.stats.writer_throughput.throughput > 0
            else 0
        )

        # Create summary table
        summary_table = Table(
            title=f"[bold green]✓ Completed {self.active_cell_type}[/bold green]", box=None
        )
        summary_table.add_column("Property", style="dim")
        summary_table.add_column("Value", style="green", justify="right")

        summary_table.add_row("Total Batches", str(self.n_batches))
        summary_table.add_row("Time Elapsed", f"{self.stats.elapsed_time:.2f}s")
        summary_table.add_row(
            "Pipeline Throughput",
            f"{self.stats.throughput:.2f} batches/s ({pipeline_cells_per_sec:.0f} cells/s)",
        )

        perf_table = Table(
            title="[bold]Component Performance (per worker)[/bold]",
            show_header=True,
            header_style="bold blue",
        )
        perf_table.add_column("Component")
        perf_table.add_column("Throughput", justify="right")
        perf_table.add_column("Workers", justify="right")
        perf_table.add_column("Total Throughput", justify="right", style="green")

        perf_table.add_row(
            "Reader",
            f"{self.stats.reader_throughput.throughput:.2f} b/s",
            str(self.reader.num_workers),
            f"{reader_cells_per_sec:.0f} c/s",
        )
        perf_table.add_row(
            "Computer",
            f"{self.stats.computer_throughput.throughput:.2f} b/s",
            str(self.computer.num_workers),
            f"{computer_cells_per_sec:.0f} c/s",
        )
        perf_table.add_row(
            "Writer",
            f"{self.stats.writer_throughput.throughput:.2f} b/s",
            str(self.writer.num_workers),
            f"{writer_cells_per_sec:.0f} c/s",
        )

        console.print(
            Panel(Group(summary_table, perf_table), title="Pipeline Summary", border_style="green")
        )

        # Final garbage collection
        import gc

        gc.collect()

        logger.info(f"✓ Completed processing {self.active_cell_type}")

    def stop(self):
        """Stop all workers in the pipeline components"""
        logger.info("Stopping MarkerScoreMessageQueue workers...")
        self.reader.stop_workers.set()
        self.computer.stop_workers.set()
        self.writer.stop_workers.set()
        logger.info("MarkerScoreMessageQueue workers stopped")


@partial(jit, static_argnums=(2, 3, 6))
def compute_marker_scores_jax(
    log_ranks: jnp.ndarray,  # (B*N) × G matrix
    weights: jnp.ndarray,  # B × N weight matrix
    batch_size: int,
    num_neighbors: int,
    global_log_gmean: jnp.ndarray,  # G-dimensional vector
    global_expr_frac: jnp.ndarray,  # G-dimensional vector
    no_expression_fraction: bool = False,  # Skip expression fraction filtering if True
) -> jnp.ndarray:
    """
    JAX-accelerated marker score computation

    Returns:
        B × G marker scores
    """
    n_genes = log_ranks.shape[1]

    # Reshape to batch format
    log_ranks_3d = log_ranks.reshape(batch_size, num_neighbors, n_genes)

    # Compute weighted geometric mean in log space
    weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize weights
    weighted_log_mean = jnp.einsum("bn,bng->bg", weights, log_ranks_3d)

    # Calculate marker score
    marker_score = jnp.exp(weighted_log_mean - global_log_gmean)
    marker_score = jnp.where(marker_score < 1.0, 0.0, marker_score)

    # Apply expression fraction filter (only if not disabled)
    if not no_expression_fraction:
        # Compute expression fraction (mean of is_expressed across neighbors)
        # Treat min log rank as non-expressed
        is_expressed = log_ranks_3d != log_ranks_3d.min(axis=-1, keepdims=True)

        # Create mask for valid neighbors (where weights > 0)
        valid_mask = weights > 0  # Shape: (batch_size, num_neighbors)

        # Apply mask and compute mean only for valid neighbors
        is_expressed_masked = jnp.where(valid_mask[:, :, None], is_expressed, 0)
        valid_counts = valid_mask.sum(axis=1, keepdims=True)  # Count of valid neighbors per cell

        # Compute mean only over valid neighbors (avoid division by zero)
        expr_frac = jnp.where(
            valid_counts > 0,
            is_expressed_masked.astype(jnp.float16).sum(axis=1) / valid_counts,
            0.0,
        )

        frac_mask = expr_frac > global_expr_frac
        marker_score = jnp.where(frac_mask, marker_score, 0.0)

    marker_score = jnp.exp(marker_score**1.5) - 1.0

    # Return as float16 for memory efficiency
    return marker_score.astype(jnp.float16)


@partial(jit, static_argnums=(2, 3, 4, 7))
def compute_marker_scores_3d_hierarchical_pool_jax(
    log_ranks: jnp.ndarray,  # (B*N) × G matrix where N = n_slices * num_homogeneous_per_slice
    weights: jnp.ndarray,  # B × N weight matrix
    batch_size: int,
    n_slices: int,
    num_homogeneous_per_slice: int,
    global_log_gmean: jnp.ndarray,  # G-dimensional vector
    global_expr_frac: jnp.ndarray,  # G-dimensional vector
    no_expression_fraction: bool = False,  # Skip expression fraction filtering if True
) -> jnp.ndarray:
    """
    JAX-accelerated marker score computation with hierarchical pooling for 3D spatial data.
    Computes marker scores independently for each slice and takes the average.

    Args:
        log_ranks: Flattened log ranks (batch_size * total_neighbors, n_genes)
        weights: Flattened weights (batch_size, total_neighbors)
        batch_size: Number of cells in batch
        n_slices: Number of slices (1 + 2 * n_adjacent_slices)
        num_homogeneous_per_slice: Number of homogeneous neighbors per slice
        global_log_gmean: Global log geometric mean
        global_expr_frac: Global expression fraction

    Returns:
        (batch_size, n_genes) marker scores using average cross slices
    """
    n_genes = log_ranks.shape[1]
    n_slices * num_homogeneous_per_slice

    # Reshape to separate slices: (batch_size, n_slices, num_homogeneous_per_slice, n_genes)
    log_ranks_4d = log_ranks.reshape(batch_size, n_slices, num_homogeneous_per_slice, n_genes)

    # Reshape weights: (batch_size, n_slices, num_homogeneous_per_slice)
    weights_3d = weights.reshape(batch_size, n_slices, num_homogeneous_per_slice)

    # Normalize weights within each slice (sum to 1 along num_homogeneous_per_slice axis)
    weights_sum = weights_3d.sum(axis=2, keepdims=True)  # Shape: (batch_size, n_slices, 1)
    weights_normalized = weights_3d / jnp.where(weights_sum > 0, weights_sum, 1.0)

    # Compute weighted geometric mean in log space for each slice
    # Result: (batch_size, n_slices, n_genes)
    weighted_log_mean = jnp.einsum("bsn,bsng->bsg", weights_normalized, log_ranks_4d)

    # Calculate marker score for each slice
    marker_score_per_slice = jnp.exp(weighted_log_mean - global_log_gmean[None, None, :])
    marker_score_per_slice = jnp.where(marker_score_per_slice < 1.0, 0.0, marker_score_per_slice)

    # Apply expression fraction filter for each slice (only if not disabled)
    if not no_expression_fraction:
        # Compute expression fraction for each slice
        # Treat min log rank as non-expressed
        min_log_rank = log_ranks_4d.min(axis=-1, keepdims=True)
        is_expressed = log_ranks_4d != min_log_rank

        # Create mask for valid neighbors within each slice (where weights > 0)
        valid_mask = weights_3d > 0  # Shape: (batch_size, n_slices, num_homogeneous_per_slice)

        # Apply mask and compute mean only for valid neighbors within each slice
        is_expressed_masked = jnp.where(valid_mask[:, :, :, None], is_expressed, 0)
        valid_counts = valid_mask.sum(axis=2, keepdims=True)  # Count of valid neighbors per slice

        # Compute mean only over valid neighbors (avoid division by zero)
        # Result: (batch_size, n_slices, n_genes)
        # valid_counts has shape (batch_size, n_slices, 1), need to broadcast properly
        expr_frac = jnp.where(
            valid_counts > 0,
            is_expressed_masked.astype(jnp.float16).sum(axis=2) / valid_counts,
            0.0,
        )

        frac_mask = expr_frac > global_expr_frac[None, None, :]
        marker_score_per_slice = jnp.where(frac_mask, marker_score_per_slice, 0.0)

    marker_score_per_slice = jnp.exp(marker_score_per_slice**1.5) - 1.0

    # Calculate median across slices instead of max pooling to reduce noise
    # Result: (batch_size, n_genes)

    # Get the invalid slice mask before calculating median.
    # These slices are invalid could be:
    # 1. no neighbors from this slice for the outermost slices
    # 2. the adjacent slices that do not have enough high quality cells which has been filtered out
    invalid_slice_mask = (weights_sum == 0).squeeze(-1)  # Shape: (batch_size, n_slices)

    # Set invalid slice scores to NaN, then calculate the median
    # NaN values will be ignored by jnp.nanmedian
    marker_score_per_slice = jnp.where(
        invalid_slice_mask[:, :, None],  # Broadcast to (batch_size, n_slices, n_genes)
        jnp.nan,
        marker_score_per_slice,
    )

    # Calculate median across slices (axis=1), ignoring NaN values
    marker_score = jnp.nanmean(marker_score_per_slice, axis=1)
    # marker_score = jnp.nanmedian(marker_score_per_slice, axis=1)

    # Handle cases where all slices are invalid (all NaN) - set to 0
    marker_score = jnp.where(jnp.isnan(marker_score), 0.0, marker_score)

    # Return as float16 for memory efficiency
    return marker_score.astype(jnp.float16)


class MarkerScoreCalculator:
    """Main class for calculating marker scores"""

    def __init__(self, config):
        """
        Initialize with configuration

        Args:
            config: LatentToGeneConfig object
        """
        self.config = config
        self.connectivity_builder = ConnectivityMatrixBuilder(config)

    def load_global_stats(self, mean_frac_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load pre-calculated global geometric mean and expression fraction from parquet"""

        logger.info("Loading global statistics from parquet...")
        parquet_path = Path(mean_frac_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Global stats file not found: {parquet_path}")

        # Load the dataframe
        mean_frac_df = pd.read_parquet(parquet_path)

        # Extract global log geometric mean and expression fraction
        global_log_gmean = mean_frac_df["G_Mean"].values.astype(np.float32)
        global_expr_frac = mean_frac_df["frac"].values.astype(np.float32)

        logger.info(f"Loaded global stats for {len(global_log_gmean)} genes")

        return global_log_gmean, global_expr_frac

    def _display_input_summary(self, adata, cell_types, n_cells, n_genes):
        """Display summary of input data and cell types"""
        table = Table(
            title="[bold cyan]Marker Score Input Summary[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Property", style="dim")
        table.add_column("Value", style="green")

        table.add_row("Total Cells", str(n_cells))
        table.add_row("Total Genes", str(n_genes))
        table.add_row("Cell Types", str(len(cell_types)))
        table.add_row("Dataset Type", str(self.config.dataset_type.value))
        table.add_row("Annotation Key", str(self.config.annotation))

        self.console.print(table)

        # Display cell type breakdown
        ct_table = Table(
            title="[bold cyan]Cell Type Breakdown[/bold cyan]",
            show_header=True,
            header_style="bold blue",
        )
        ct_table.add_column("Cell Type", style="dim")
        ct_table.add_column("Count", justify="right")

        ct_counts = adata.obs[self.config.annotation].value_counts()
        for ct in cell_types:
            count = ct_counts.get(ct, 0)
            style = "green" if count >= self.config.min_cells_per_type else "red"
            ct_table.add_row(ct, f"[{style}]{count}[/{style}]")

        self.console.print(ct_table)

    def _load_input_data(
        self, adata_path: str, rank_memmap_path: str, mean_frac_path: str
    ) -> tuple[ad.AnnData, MemMapDense, np.ndarray, np.ndarray, int, int, np.ndarray]:
        """Load input data: AnnData, rank memory map, global statistics, and high quality mask

        Returns:
            Tuple of (adata, rank_memmap, global_log_gmean, global_expr_frac, n_cells, n_genes, high_quality_mask)
        """
        # Load concatenated AnnData
        logger.info(f"Loading concatenated AnnData from {adata_path}")
        if not Path(adata_path).exists():
            raise FileNotFoundError(f"Concatenated AnnData not found: {adata_path}")
        adata = sc.read_h5ad(adata_path)

        # Load pre-calculated global statistics
        global_log_gmean, global_expr_frac = self.load_global_stats(mean_frac_path)

        # Open rank memory map and get dimensions
        rank_memmap_path = Path(rank_memmap_path)
        meta_path = rank_memmap_path.with_suffix(".meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        rank_memmap = MemMapDense(
            path=rank_memmap_path,
            shape=tuple(meta["shape"]),
            dtype=np.dtype(meta["dtype"]),
            mode="r",
            tmp_dir=self.config.memmap_tmp_dir,
        )

        logger.info(f"Opened rank memory map from {rank_memmap_path}")
        n_cells = adata.n_obs
        n_cells_rank = rank_memmap.shape[0]
        n_genes = rank_memmap.shape[1]

        logger.info(f"AnnData dimensions: {n_cells} cells × {adata.n_vars} genes")
        logger.info(f"Rank MemMap dimensions: {n_cells_rank} cells × {n_genes} genes")

        # Cells should match exactly since filtering is done before rank memmap creation
        assert n_cells == n_cells_rank, (
            f"Cell count mismatch: AnnData has {n_cells} cells, Rank MemMap has {n_cells_rank} cells. "
            f"This indicates the filtering was not applied consistently during rank calculation."
        )

        # Load high quality mask based on configuration
        if self.config.high_quality_neighbor_filter:
            if "High_quality" not in adata.obs.columns:
                raise ValueError(
                    "High_quality column not found in AnnData obs. Please ensure QC was applied during find_latent_representation step."
                )
            high_quality_mask = adata.obs["High_quality"].values.astype(bool)
            logger.info(
                f"Loaded high quality mask: {high_quality_mask.sum()}/{len(high_quality_mask)} cells marked as high quality"
            )
        else:
            # Create all-True mask when high quality filtering is disabled
            high_quality_mask = np.ones(n_cells, dtype=bool)
            logger.info("High quality filtering disabled - using all cells")

        return (
            adata,
            rank_memmap,
            global_log_gmean,
            global_expr_frac,
            n_cells,
            n_genes,
            high_quality_mask,
        )

    def _prepare_embeddings(
        self, adata: ad.AnnData
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
        """Prepare and normalize embeddings based on dataset type

        Returns:
            Tuple of (coords, emb_niche, emb_indv, slice_ids)
        """
        logger.info("Loading shared data structures...")

        coords = None
        emb_niche = None
        slice_ids = None

        if self.config.dataset_type in ["spatial2D", "spatial3D"]:
            # Load spatial coordinates for spatial datasets
            coords = adata.obsm[self.config.spatial_key]

            # Load slice IDs if provided (for both spatial2D and spatial3D)
            assert "slice_id" in adata.obs.columns
            slice_ids = adata.obs["slice_id"].values.astype(np.int32)

            # Try to load niche embeddings if they exist
            if self.config.latent_representation_niche in adata.obsm:
                emb_niche = adata.obsm[self.config.latent_representation_niche]

        # Load cell embeddings for all dataset types
        emb_indv = adata.obsm[self.config.latent_representation_cell].astype(np.float16)

        # --- ELEGANT FIX START ---
        # If emb_niche is missing (scRNA-seq or spatial without niche),
        # create a dummy (N, 1) array of ones.
        # Normalizing a vector of 1s results in 1.0, so cosine similarity will be 1.0 everywhere.
        if emb_niche is None:
            logger.info("No niche embeddings found. Using dummy embeddings (all ones).")
            emb_niche = np.ones((emb_indv.shape[0], 1), dtype=np.float32)
        # --- ELEGANT FIX END ---

        # Normalize embeddings
        logger.info("Normalizing embeddings...")

        # L2 normalize niche embeddings (always exists now)
        emb_niche_norm = np.linalg.norm(emb_niche, axis=1, keepdims=True)
        emb_niche = emb_niche / (emb_niche_norm + 1e-8)

        # L2 normalize individual embeddings
        emb_indv_norm = np.linalg.norm(emb_indv, axis=1, keepdims=True)
        emb_indv = emb_indv / (emb_indv_norm + 1e-8)

        return coords, emb_niche, emb_indv, slice_ids

    def _get_cell_types(self, adata: ad.AnnData) -> np.ndarray:
        """Get cell types from annotation key

        Returns:
            Array of unique cell types
        """
        annotation_key = self.config.annotation

        if annotation_key is not None:
            assert annotation_key in adata.obs.columns, (
                f"Annotation key '{annotation_key}' not found in adata.obs"
            )
            # Get unique cell types, excluding NaN values
            cell_types = adata.obs[annotation_key].dropna().unique()

            # Check if there are any NaN values and handle them
            nan_count = adata.obs[annotation_key].isna().sum()
            if nan_count > 0:
                logger.warning(
                    f"Found {nan_count} cells with NaN annotation in '{annotation_key}', these will be skipped"
                )
        else:
            logger.warning(
                f"Annotation {annotation_key} not found, processing all cells as one type"
            )
            cell_types = ["all"]
            adata.obs[annotation_key] = "all"

        logger.info(f"Processing {len(cell_types)} cell types")
        return cell_types

    def _initialize_pipeline(
        self,
        rank_memmap: MemMapDense,
        output_memmap: MemMapDense,
        global_log_gmean: np.ndarray,
        global_expr_frac: np.ndarray,
    ):
        """Initialize the processing pipeline with shared pools and queues"""
        logger.info("Initializing shared processing pools with direct queue connections...")

        # Create shared queues to connect components with configured sizes
        reader_to_computer_queue = queue.Queue(
            maxsize=self.config.mkscore_compute_workers * self.config.compute_input_queue_size
        )
        computer_to_writer_queue = queue.Queue(maxsize=self.config.writer_queue_size)

        self.reader = ParallelRankReader(
            rank_memmap,
            num_workers=self.config.rank_read_workers,
            output_queue=reader_to_computer_queue,  # Direct connection to computer
        )

        # Determine 3D strategy parameters
        cross_slice_strategy = None
        n_slices = 1
        num_homogeneous_per_slice = self.config.homogeneous_neighbors

        if (
            self.config.dataset_type == DatasetType.SPATIAL_3D
            and self.config.cross_slice_marker_score_strategy
            in [
                MarkerScoreCrossSliceStrategy.PER_SLICE_POOL,
                MarkerScoreCrossSliceStrategy.HIERARCHICAL_POOL,
            ]
        ):
            cross_slice_strategy = self.config.cross_slice_marker_score_strategy
            n_slices = 1 + 2 * self.config.n_adjacent_slices
            # For pooling strategies, num_homogeneous is per slice
            num_homogeneous_per_slice = self.config.homogeneous_neighbors

        self.computer = ParallelMarkerScoreComputer(
            global_log_gmean,
            global_expr_frac,
            self.config.homogeneous_neighbors,
            num_workers=self.config.mkscore_compute_workers,
            input_queue=reader_to_computer_queue,  # Input from reader
            output_queue=computer_to_writer_queue,  # Output to writer
            cross_slice_strategy=cross_slice_strategy,
            n_slices=n_slices,
            num_homogeneous_per_slice=num_homogeneous_per_slice,
            no_expression_fraction=self.config.no_expression_fraction,
        )

        self.writer = ParallelMarkerScoreWriter(
            output_memmap,
            num_workers=self.config.mkscore_write_workers,
            input_queue=computer_to_writer_queue,  # Input from computer
        )

        logger.info(
            f"Processing pools initialized: {self.config.rank_read_workers} readers, "
            f"{self.config.mkscore_compute_workers} computers, "
            f"{self.config.mkscore_write_workers} writers"
        )

        self.marker_score_queue = MarkerScoreMessageQueue(
            reader=self.reader,
            computer=self.computer,
            writer=self.writer,
            batch_size=self.config.mkscore_batch_size,
        )

    def _find_homogeneous_spots(
        self,
        adata: ad.AnnData,
        cell_type: str,
        annotation_key: str,
        coords: np.ndarray | None,
        emb_niche: np.ndarray,
        emb_indv: np.ndarray,
        slice_ids: np.ndarray | None,
        rank_shape: tuple[int, int],
        high_quality_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, int] | None:
        """
        Prepare batch data for a cell type

        Returns:
            Tuple of (neighbor_indices, cell_sims, niche_sims, cell_indices_sorted, n_cells)
            or None if cell type should be skipped
        """
        # Get cells of this type, excluding NaN values
        cell_mask = (adata.obs[annotation_key] == cell_type) & (adata.obs[annotation_key].notna())
        cell_indices = np.where(cell_mask)[0]
        n_cells = len(cell_indices)

        # Check minimum cells
        min_cells = self.config.min_cells_per_type
        if n_cells < min_cells:
            logger.warning(f"Skipping {cell_type}: only {n_cells} cells (min: {min_cells})")

        logger.info(f"Processing {cell_type}: {n_cells} cells")

        # Build connectivity matrix
        logger.info("Building connectivity matrix...")
        neighbor_indices, cell_sims, niche_sims = (
            self.connectivity_builder.build_connectivity_matrix(
                coords=coords,
                emb_niche=emb_niche,
                emb_indv=emb_indv,
                cell_mask=cell_mask,
                high_quality_mask=high_quality_mask,
                slice_ids=slice_ids,
                k_central=self.config.spatial_neighbors,
                k_adjacent=self.config.adjacent_slice_spatial_neighbors,
                n_adjacent_slices=self.config.n_adjacent_slices,
            )
        )
        gc.collect()

        # Validate neighbor indices
        max_valid_idx = rank_shape[0] - 1
        assert neighbor_indices.max() <= max_valid_idx, (
            f"Neighbor indices exceed bounds (max: {neighbor_indices.max()}, limit: {max_valid_idx})"
        )

        # Optimize row order using JAX implementation
        logger.info("Optimizing row order for cache efficiency...")
        row_order = optimize_row_order_jax(
            neighbor_indices=neighbor_indices[:, : self.config.homogeneous_neighbors],
            cell_indices=cell_indices,
            neighbor_weights=cell_sims[:, : self.config.homogeneous_neighbors],
        )

        neighbor_indices = neighbor_indices[row_order]
        cell_sims = cell_sims[row_order]
        if niche_sims is not None:
            niche_sims = niche_sims[row_order]
        cell_indices_sorted = cell_indices[row_order]

        # Save homogeneous neighbor data to adata
        has_real_niche_embedding = self.config.latent_representation_niche is not None
        self._save_homogeneous_data_to_adata(
            adata=adata,
            neighbor_indices=neighbor_indices,
            cell_sims=cell_sims,
            niche_sims=niche_sims,
            cell_indices_sorted=cell_indices_sorted,
            has_real_niche_embedding=has_real_niche_embedding,
        )

        # warning for cells not find homo neighbors
        homo_neighbor_count = np.count_nonzero(cell_sims > 0, axis=1)
        zero_homo_neighbor_mask = homo_neighbor_count <= 5
        if np.any(zero_homo_neighbor_mask):
            logger.warning(
                f"Cell type {cell_type}: {zero_homo_neighbor_mask.sum()} cells can't find enough homogeneous neighbors"
            )

        return neighbor_indices, cell_sims, niche_sims, cell_indices_sorted, n_cells

    def _save_homogeneous_data_to_adata(
        self,
        adata: ad.AnnData,
        neighbor_indices: np.ndarray,
        cell_sims: np.ndarray,
        niche_sims: np.ndarray | None,
        cell_indices_sorted: np.ndarray,
        has_real_niche_embedding: bool,
    ):
        """
        Save homogeneous neighbor data to adata obsm and obs.

        Args:
            adata: AnnData object to save data to
            neighbor_indices: (n_cells, k) array of neighbor indices
            cell_sims: (n_cells, k) array of cell similarity scores
            niche_sims: (n_cells, k) array of niche similarity scores or None
            cell_indices_sorted: (n_cells,) array of cell indices in sorted order
            has_real_niche_embedding: Whether real niche embedding was provided
        """
        # Initialize obsm matrices if they don't exist
        if "gsMap_homo_indices" not in adata.obsm.keys() or (
            adata.obsm["gsMap_homo_indices"].shape[1] != neighbor_indices.shape[1]
        ):
            adata.obsm["gsMap_homo_indices"] = np.zeros(
                (adata.n_obs, neighbor_indices.shape[1]), dtype=neighbor_indices.dtype
            )
        if "gsMap_homo_cell_sims" not in adata.obsm.keys() or (
            adata.obsm["gsMap_homo_cell_sims"].shape[1] != neighbor_indices.shape[1]
        ):
            adata.obsm["gsMap_homo_cell_sims"] = np.zeros(
                (adata.n_obs, cell_sims.shape[1]), dtype=cell_sims.dtype
            )

        # Store the neighbor indices and cell_sims for this cell type
        adata.obsm["gsMap_homo_indices"][cell_indices_sorted] = neighbor_indices
        adata.obsm["gsMap_homo_cell_sims"][cell_indices_sorted] = cell_sims

        # Only store niche_sims if niche embedding was provided (not dummy)
        if has_real_niche_embedding and niche_sims is not None:
            if "gsMap_homo_niche_sims" not in adata.obsm.keys() or (
                adata.obsm["gsMap_homo_niche_sims"].shape[1] != neighbor_indices.shape[1]
            ):
                adata.obsm["gsMap_homo_niche_sims"] = np.zeros(
                    (adata.n_obs, niche_sims.shape[1]), dtype=niche_sims.dtype
                )
            adata.obsm["gsMap_homo_niche_sims"][cell_indices_sorted] = niche_sims

        # Initialize obs columns if they don't exist
        if "gsMap_homo_neighbor_count" not in adata.obs.columns:
            adata.obs["gsMap_homo_neighbor_count"] = 0
        if "gsMap_homo_cell_sims_median" not in adata.obs.columns:
            adata.obs["gsMap_homo_cell_sims_median"] = np.nan
        if has_real_niche_embedding and "gsMap_homo_niche_sims_median" not in adata.obs.columns:
            adata.obs["gsMap_homo_niche_sims_median"] = np.nan

        # Calculate statistics for each cell (only consider valid neighbors where cell_sims > 0)
        valid_mask = cell_sims > 0  # (n_cells, k) boolean mask

        # Count of valid homogeneous neighbors
        homo_neighbor_count = valid_mask.sum(axis=1)
        adata.obs.loc[adata.obs.index[cell_indices_sorted], "gsMap_homo_neighbor_count"] = (
            homo_neighbor_count
        )

        # Median of cell similarity (only for valid neighbors)
        cell_sims_masked = np.where(valid_mask, cell_sims, np.nan)
        cell_sims_median = np.nanmedian(cell_sims_masked, axis=1)
        adata.obs.loc[adata.obs.index[cell_indices_sorted], "gsMap_homo_cell_sims_median"] = (
            cell_sims_median
        )

        # Median of niche similarity (only for valid neighbors, only if niche embedding provided)
        if has_real_niche_embedding and niche_sims is not None:
            niche_sims_masked = np.where(valid_mask, niche_sims, np.nan)
            niche_sims_median = np.nanmedian(niche_sims_masked, axis=1)
            adata.obs.loc[adata.obs.index[cell_indices_sorted], "gsMap_homo_niche_sims_median"] = (
                niche_sims_median
            )

    def _calculate_marker_scores_by_cell_type(
        self,
        adata: ad.AnnData,
        cell_type: str,
        coords: np.ndarray | None,
        emb_niche: np.ndarray,
        emb_indv: np.ndarray,
        annotation_key: str,
        slice_ids: np.ndarray | None = None,
        high_quality_mask: np.ndarray = None,
    ):
        """Process a single cell type with shared pools"""

        # Find homogeneous spots
        neighbor_indices, cell_sims, niche_sims, cell_indices_sorted, n_cells = (
            self._find_homogeneous_spots(
                adata,
                cell_type,
                annotation_key,
                coords,
                emb_niche,
                emb_indv,
                slice_ids,
                self.reader.shape,
                high_quality_mask,
            )
        )

        # Reset the message queue for this cell type
        self.marker_score_queue.reset_for_cell_type(cell_type, n_cells)

        # Run the marker_score_queue to compute marker score
        self.marker_score_queue.start(
            neighbor_indices=neighbor_indices,
            neighbor_weights=cell_sims,
            cell_indices_sorted=cell_indices_sorted,
            enable_profiling=self.config.enable_profiling,
        )

    def calculate_marker_scores(
        self,
        adata_path: str,
        rank_memmap_path: str,
        mean_frac_path: str,
        output_path: str | Path | None = None,
    ) -> str | Path:
        """
        Main execution function for marker score calculation

        Args:
            adata_path: Path to concatenated latent adata
            rank_memmap_path: Path to rank memory map
            mean_frac_path: Path to mean expression fraction parquet
            output_path: Optional output path for marker scores

        Returns:
            Path to output marker score memory map file
        """
        logger.info("Starting marker score calculation...")
        self.console = Console()

        self.console.print(
            Panel.fit(
                "[bold cyan]Marker Score Calculation Stage[/bold cyan]",
                subtitle="gsMap Stage 2",
                border_style="cyan",
            )
        )

        # Use config path if not specified
        if output_path is None:
            output_path = Path(self.config.marker_scores_memmap_path)
        else:
            output_path = Path(output_path)

        # Load all input data
        (
            adata,
            rank_memmap,
            global_log_gmean,
            global_expr_frac,
            n_cells,
            n_genes,
            high_quality_mask,
        ) = self._load_input_data(adata_path, rank_memmap_path, mean_frac_path)

        # Initialize output memory map
        output_memmap = MemMapDense(
            output_path,
            shape=(n_cells, n_genes),
            dtype=np.float16,  # Use float16 to save memory
            mode="w",
            num_write_workers=self.config.mkscore_write_workers,
            tmp_dir=self.config.memmap_tmp_dir,
        )

        # Get cell types to process
        cell_types = self._get_cell_types(adata)
        annotation_key = self.config.annotation

        # Display summary
        self._display_input_summary(adata, cell_types, n_cells, n_genes)

        # Prepare embeddings
        coords, emb_niche, emb_indv, slice_ids = self._prepare_embeddings(adata)

        # Initialize processing pipeline
        self._initialize_pipeline(rank_memmap, output_memmap, global_log_gmean, global_expr_frac)

        # Process each cell type
        for cell_type in cell_types:
            self._calculate_marker_scores_by_cell_type(
                adata,
                cell_type,
                coords,
                emb_niche,
                emb_indv,
                annotation_key,
                slice_ids,
                high_quality_mask,
            )

        # Save updated AnnData with neighbor matrices
        logger.info(f"Saving updated AnnData with homogeneous spot matrices to {adata_path}")
        adata.write_h5ad(adata_path)

        # Close all shared pools after all cell types are processed
        logger.info("Closing shared processing pools...")
        self.reader.close()
        self.computer.close()
        self.writer.close()

        # Close memory maps
        rank_memmap.close()
        output_memmap.close()

        self.console.print(
            Panel.fit(
                "[bold green]✓ Marker score calculation complete![/bold green]",
                border_style="green",
            )
        )

        logger.info(f"Results saved to {output_path}")

        return str(output_path)

"""
Marker score calculation using homogeneous neighbors
Implements weighted geometric mean calculation in log space with JAX acceleration
"""

import logging
import queue
import threading
import time
import json
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
from functools import partial
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import gc
from scipy.sparse import csr_matrix
import jax
import jax.numpy as jnp
from jax import jit

from .memmap_io import MemMapDense
from .connectivity import ConnectivityMatrixBuilder
from .row_ordering import optimize_row_order


logger = logging.getLogger(__name__)

# Progress bar imports
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn,
    TimeElapsedColumn
)
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
USE_RICH = True

class ParallelRankReader:
    """Multi-threaded reader for log-rank data from memory-mapped storage"""
    
    def __init__(
        self,
        rank_memmap: Union[MemMapDense, str],
        num_workers: int = 4,
        output_queue: queue.Queue = None,
        cache_size_mb: int = 1000
    ):
        # Store path and metadata for workers to open their own instances
        if isinstance(rank_memmap, str):
            self.memmap_path = Path(rank_memmap)
            meta_path = self.memmap_path.with_suffix('.meta.json')
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.shape = tuple(meta['shape'])
            self.dtype = np.dtype(meta['dtype'])
        else:
            # If MemMapDense instance, extract path and metadata
            assert hasattr(rank_memmap, 'path')
            self.memmap_path = rank_memmap.path
            self.shape = rank_memmap.shape
            self.dtype = rank_memmap.dtype

        self.num_workers = num_workers
        
        # Queues for communication
        self.read_queue = queue.Queue()
        # Use provided output queue or create own
        self.result_queue = output_queue if output_queue else queue.Queue(maxsize=self.num_workers * 4)
        
        # Throughput tracking
        self.throughput = ComponentThroughput()
        self.throughput_lock = threading.Lock()
        
        # Exception handling
        self.exception_queue = queue.Queue()
        self.has_error = threading.Event()
        
        # Start worker threads
        self.workers = []
        self.stop_workers = threading.Event()
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker(self, worker_id: int):
        """Worker thread for reading batches from memory map"""
        logger.info(f"Reader worker {worker_id} started")
        
        # Open worker's own memory map instance
        data_path = self.memmap_path.with_suffix('.dat')
        worker_memmap = np.memmap(
            data_path,
            dtype=self.dtype,
            mode='r',
            shape=self.shape
        )
        logger.info(f"Worker {worker_id} opened its own memory map at {data_path}")

        while not self.stop_workers.is_set():
            try:
                # Get batch request
                item = self.read_queue.get()
                if item is None:
                    break
                
                batch_id, neighbor_indices, batch_metadata = item
                
                # Track timing
                start_time = time.time()
                
                # Flatten and deduplicate indices for efficient reading
                flat_indices = np.unique(neighbor_indices.flatten())
                
                # Validate indices are within bounds
                max_idx = self.shape[0] - 1
                assert flat_indices.max() <= max_idx, \
                    f"Worker {worker_id}: Indices exceed bounds (max: {flat_indices.max()}, limit: {max_idx})"
                
                # Read from worker's own memory map (direct array access)
                # Memory map stores log-ranks directly
                rank_data = worker_memmap[flat_indices]
                
                # Ensure we have a numpy array
                if not isinstance(rank_data, np.ndarray):
                    rank_data = np.array(rank_data)
                
                # Create mapping for reconstruction
                idx_map = {idx: i for i, idx in enumerate(flat_indices)}
                
                # Map neighbor indices to rank_data indices
                flat_neighbors = neighbor_indices.flatten()
                rank_indices = np.array([idx_map[neighbor_idx] for neighbor_idx in flat_neighbors])
                
                # Track throughput
                elapsed = time.time() - start_time
                with self.throughput_lock:
                    self.throughput.record_batch(elapsed)
                
                # Put result with metadata for computer
                self.result_queue.put((batch_id, rank_data, rank_indices, neighbor_indices.shape, batch_metadata))
                self.read_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Reader worker {worker_id} error: {e}")
                self.exception_queue.put((worker_id, e))
                self.has_error.set()
                self.stop_workers.set()  # Signal all workers to stop
                break
        
        # Clean up worker's memory map if it was opened
        if self.memmap_path is not None and 'worker_memmap' in locals():
            del worker_memmap
            logger.info(f"Worker {worker_id} closed its memory map")
    
    def submit_batch(self, batch_id: int, neighbor_indices: np.ndarray, batch_metadata: dict = None):
        """Submit batch for reading with metadata"""
        self.read_queue.put((batch_id, neighbor_indices, batch_metadata or {}))
    
    def get_result(self):
        """Get next completed batch"""
        return self.result_queue.get()
    
    def get_queue_sizes(self):
        """Get current queue sizes for monitoring"""
        return self.read_queue.qsize(), self.result_queue.qsize()
    
    def check_errors(self):
        """Check if any worker encountered an error"""
        if self.has_error.is_set():
            try:
                worker_id, exception = self.exception_queue.get_nowait()
                raise RuntimeError(f"Reader worker {worker_id} failed: {exception}") from exception
            except queue.Empty:
                raise RuntimeError("Reader worker failed with unknown error")
    
    def reset_for_cell_type(self, cell_type: str):
        """Reset throughput tracking for new cell type"""
        with self.throughput_lock:
            self.throughput = ComponentThroughput()
        logger.debug(f"Reset reader throughput for {cell_type}")
    
    def close(self):
        """Clean up resources"""
        self.stop_workers.set()
        for _ in range(self.num_workers):
            self.read_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5)
        
        # No need to close individual worker memmaps as they're cleaned up in _worker
        # Only close if we have a shared rank_memmap (fallback mode)
        if hasattr(self, 'rank_memmap') and hasattr(self.rank_memmap, 'close'):
            self.rank_memmap.close()


class ParallelMarkerScoreComputer:
    """Multi-threaded computer pool for marker score calculation (reusable across cell types)"""
    
    def __init__(
        self,
        global_log_gmean: np.ndarray,
        global_expr_frac: np.ndarray,
        num_homogeneous: int,
        num_workers: int = 4,
        input_queue: queue.Queue = None,
        output_queue: queue.Queue = None
    ):
        """
        Initialize computer pool
        
        Args:
            global_log_gmean: Global log geometric mean
            global_expr_frac: Global expression fraction
            num_homogeneous: Number of homogeneous neighbors
            num_workers: Number of compute workers
            input_queue: Optional input queue (from reader)
            output_queue: Optional output queue (to writer)
        """
        self.num_workers = num_workers
        self.num_homogeneous = num_homogeneous
        
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
            worker = threading.Thread(
                target=self._compute_worker,
                args=(i,),
                daemon=True
            )
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
                    
                batch_start = batch_metadata['batch_start']
                batch_end = batch_metadata['batch_end']
                actual_batch_size = batch_end - batch_start
                
                # Verify shape
                assert original_shape == (actual_batch_size, self.num_homogeneous), \
                    f"Shape mismatch: expected {(actual_batch_size, self.num_homogeneous)}, got {original_shape}"
                
                # Get batch-specific data
                batch_weights = self.neighbor_weights[batch_start:batch_end]
                batch_cell_indices = self.cell_indices_sorted[batch_start:batch_end]
                
                # Convert to JAX for efficient computation
                rank_data_jax = jnp.array(rank_data)
                rank_indices_jax = jnp.array(rank_indices)
                
                # Use JAX fancy indexing
                batch_ranks = rank_data_jax[rank_indices_jax]
                
                # Compute marker scores using JAX
                marker_scores = compute_marker_scores_jax(
                    batch_ranks,
                    jnp.array(batch_weights),
                    actual_batch_size,  # Use actual batch size, not self.batch_size
                    self.num_homogeneous,
                    self.global_log_gmean,
                    self.global_expr_frac
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
            except Exception as e:
                logger.error(f"Compute worker {worker_id} error: {e}")
                self.exception_queue.put((worker_id, e))
                self.has_error.set()
                self.stop_workers.set()  # Signal all workers to stop
                break
        
        logger.info(f"Compute worker {worker_id} stopped")
    
    def set_batch_context(self, neighbor_weights: np.ndarray, cell_indices_sorted: np.ndarray, 
                         batch_size: int, n_cells: int):
        """Set context for processing batches of current cell type"""
        self.neighbor_weights = neighbor_weights
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
                worker_id, exception = self.exception_queue.get_nowait()
                raise RuntimeError(f"Compute worker {worker_id} failed: {exception}") from exception
            except queue.Empty:
                raise RuntimeError("Compute worker failed with unknown error")
    
    def close(self):
        """Close compute pool"""
        self.stop_workers.set()
        for _ in range(self.num_workers):
            self.compute_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5)
        logger.info("Compute pool closed")


class ParallelMarkerScoreWriter:
    """Multi-threaded writer pool for marker scores (reusable across cell types)"""
    
    def __init__(
        self,
        output_memmap: MemMapDense,
        num_workers: int = 4,
        input_queue: queue.Queue = None
    ):
        """
        Initialize writer pool
        
        Args:
            output_memmap: Output memory map
            num_workers: Number of writer threads
            input_queue: Optional input queue (from computer)
        """
        # Store path and metadata for workers to open their own instances
        self.memmap_path = output_memmap.path
        self.shape = output_memmap.shape
        self.dtype = output_memmap.dtype
        self.num_workers = num_workers
        
        # Queue for write requests
        self.write_queue = input_queue if input_queue else queue.Queue(maxsize=100)
        self.completed_count = 0
        self.completed_lock = threading.Lock()
        self.active_cell_type = None  # Track current cell type being processed
        
        # Throughput tracking
        self.throughput = ComponentThroughput()
        self.throughput_lock = threading.Lock()
        
        # Exception handling
        self.exception_queue = queue.Queue()
        self.has_error = threading.Event()
        
        # Start worker threads
        self.workers = []
        self.stop_workers = threading.Event()
        self._start_workers()
    
    def _start_workers(self):
        """Start writer worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._writer_worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.num_workers} writer threads")
    
    def _writer_worker(self, worker_id: int):
        """Writer worker thread"""
        logger.info(f"Writer worker {worker_id} started")
        
        # Open worker's own memory map instance
        data_path = self.memmap_path.with_suffix('.dat')
        worker_memmap = np.memmap(
            data_path,
            dtype=self.dtype,
            mode='r+',  # Read-write mode for writing
            shape=self.shape
        )
        logger.info(f"Writer worker {worker_id} opened its own memory map at {data_path}")
        
        while not self.stop_workers.is_set():
            try:
                # Get write request
                item = self.write_queue.get(timeout=1)
                if item is None:
                    break
                
                batch_idx, marker_scores, cell_indices = item
                
                # Track timing
                start_time = time.time()
                
                # Write directly to worker's memory map
                # cell_indices should be the absolute indices in the full matrix
                worker_memmap[cell_indices] = marker_scores
                
                # Track throughput
                elapsed = time.time() - start_time
                with self.throughput_lock:
                    self.throughput.record_batch(elapsed)
                
                # Update completed count
                with self.completed_lock:
                    self.completed_count += 1
                
                self.write_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Writer worker {worker_id} error: {e}")
                self.exception_queue.put((worker_id, e))
                self.has_error.set()
                self.stop_workers.set()  # Signal all workers to stop
                break
        
        # Final flush before closing
        worker_memmap.flush()
        # Clean up worker's memory map
        del worker_memmap
        logger.info(f"Writer worker {worker_id} closed its memory map")
    
    def reset_for_cell_type(self, cell_type: str):
        """Reset for processing a new cell type"""
        self.active_cell_type = cell_type
        with self.completed_lock:
            self.completed_count = 0
        with self.throughput_lock:
            self.throughput = ComponentThroughput()
        logger.debug(f"Reset writer throughput for {cell_type}")
    
    def get_completed_count(self):
        """Get number of completed writes"""
        with self.completed_lock:
            return self.completed_count
    
    def get_queue_size(self):
        """Get write queue size"""
        return self.write_queue.qsize()
    
    def check_errors(self):
        """Check if any worker encountered an error"""
        if self.has_error.is_set():
            try:
                worker_id, exception = self.exception_queue.get_nowait()
                raise RuntimeError(f"Writer worker {worker_id} failed: {exception}") from exception
            except queue.Empty:
                raise RuntimeError("Writer worker failed with unknown error")
    
    def close(self):
        """Close writer pool"""
        logger.info("Closing writer pool...")
        
        # Wait for queue to empty
        if not self.write_queue.empty():
            logger.info("Waiting for remaining writes...")
            self.write_queue.join()
        
        # Stop workers
        self.stop_workers.set()
        for _ in range(self.num_workers):
            self.write_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        logger.info("Writer pool closed")


@dataclass
class ComponentThroughput:
    """Track throughput for individual pipeline components"""
    total_batches: int = 0
    total_time: float = 0.0
    last_batch_time: float = 0.0
    
    def record_batch(self, elapsed_time: float):
        """Record a batch completion"""
        self.total_batches += 1
        self.total_time += elapsed_time
        self.last_batch_time = elapsed_time
    
    @property
    def average_time(self) -> float:
        """Average time per batch"""
        if self.total_batches > 0:
            return self.total_time / self.total_batches
        return 0.0
    
    @property
    def throughput(self) -> float:
        if self.average_time > 0:
            return 1.0 / self.average_time
        return 0.0


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


class MarkerScorePipeline:
    """Streamlined pipeline for marker score calculation"""
    
    def __init__(
        self,
        reader: ParallelRankReader,
        computer: ParallelMarkerScoreComputer,
        writer: ParallelMarkerScoreWriter,
        n_batches: int,
        batch_size: int,
        n_cells: int
    ):
        """
        Initialize pipeline
        
        Args:
            reader: Rank reader pool
            computer: Marker score computer pool
            writer: Marker score writer pool
            n_batches: Total number of batches
            batch_size: Size of each batch
            n_cells: Total number of cells
        """
        self.reader = reader
        self.computer = computer
        self.writer = writer
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.n_cells = n_cells
        self.stats = PipelineStats(total_batches=n_batches)
    
    def submit_batches(
        self,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        cell_indices_sorted: np.ndarray
    ):
        """Submit all batches to the reader"""
        # Set context for computer
        self.computer.set_batch_context(
            neighbor_weights=neighbor_weights,
            cell_indices_sorted=cell_indices_sorted,
            batch_size=self.batch_size,
            n_cells=self.n_cells
        )
        
        # Submit all batches with metadata
        for batch_idx in range(self.n_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, self.n_cells)
            batch_neighbors = neighbor_indices[batch_start:batch_end]
            
            # Include metadata for computer
            batch_metadata = {
                'batch_start': batch_start,
                'batch_end': batch_end,
                'batch_idx': batch_idx
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
    
    def run(
        self,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        cell_indices_sorted: np.ndarray,
        num_homogeneous: int,
        cell_type: str
    ):
        """Run pipeline with rich progress display"""
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
            refresh_per_second=10
        ) as progress:
            
            # Add single task for pipeline
            pipeline_task = progress.add_task(
                f"[bold]{cell_type}[/bold]", total=self.n_batches
            )
            
            # Submit all batches to start the pipeline
            self.submit_batches(neighbor_indices, neighbor_weights, cell_indices_sorted)
            
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
                        f"[bold]{cell_type}[/bold] | "
                        f"Queues: [{compute_color}]R→C:{self.stats.pending_compute}[/{compute_color}] "
                        f"[{write_color}]C→W:{self.stats.pending_write}[/{write_color}]"
                    )
                )
                
                time.sleep(0.1)
            
            # Final update
            progress.update(pipeline_task, completed=self.n_batches)
            
        # No threads to wait for - processing happens in component worker threads
        
        # Print summary with component throughputs
        # Calculate effective pipeline throughput (limited by bottleneck)
        bottleneck_throughput = min(
            self.stats.reader_throughput.throughput * self.reader.num_workers if self.stats.reader_throughput.throughput > 0 else float('inf'),
            self.stats.computer_throughput.throughput * self.computer.num_workers if self.stats.computer_throughput.throughput > 0 else float('inf'),
            self.stats.writer_throughput.throughput * self.writer.num_workers if self.stats.writer_throughput.throughput > 0 else float('inf')
        )

        # Calculate cells per second for each component and pipeline
        pipeline_cells_per_sec = self.stats.throughput * self.batch_size if self.stats.throughput > 0 else 0
        reader_cells_per_sec = self.stats.reader_throughput.throughput * self.batch_size * self.reader.num_workers if self.stats.reader_throughput.throughput > 0 else 0
        computer_cells_per_sec = self.stats.computer_throughput.throughput * self.batch_size * self.computer.num_workers if self.stats.computer_throughput.throughput > 0 else 0
        writer_cells_per_sec = self.stats.writer_throughput.throughput * self.batch_size * self.writer.num_workers if self.stats.writer_throughput.throughput > 0 else 0
        
        console.print(Panel.fit(
            f"[bold green]✓ Completed {cell_type}[/bold green]\n"
            f"Total batches: {self.n_batches}\n"
            f"Time elapsed: {self.stats.elapsed_time:.2f}s\n"
            f"Pipeline throughput: {self.stats.throughput:.2f} batches/s ({pipeline_cells_per_sec:.0f} cells/s)\n"
            # f"Theoretical max: {bottleneck_throughput:.2f} batches/s\n\n"
            f"[bold]Component Performance (per worker):[/bold]\n"
            f"  Reader:   {self.stats.reader_throughput.throughput:.2f} batches/s × {self.reader.num_workers} workers ({reader_cells_per_sec:.0f} cells/s)\n"
            f"  Computer: {self.stats.computer_throughput.throughput:.2f} batches/s × {self.computer.num_workers} workers ({computer_cells_per_sec:.0f} cells/s)\n"
            f"  Writer:   {self.stats.writer_throughput.throughput:.2f} batches/s × {self.writer.num_workers} workers ({writer_cells_per_sec:.0f} cells/s)",
            title="Pipeline Summary"
        ))



@partial(jit, static_argnums=(2, 3))
def compute_marker_scores_jax(
    log_ranks: jnp.ndarray,  # (B*N) × G matrix
    weights: jnp.ndarray,  # B × N weight matrix
    batch_size: int,
    num_neighbors: int,
    global_log_gmean: jnp.ndarray,  # G-dimensional vector
    global_expr_frac: jnp.ndarray  # G-dimensional vector
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
    weighted_log_mean = jnp.einsum('bn,bng->bg', weights, log_ranks_3d)
    
    # Compute expression fraction (mean of is_expressed across neighbors)
    # Treat min log rank as non-expressed
    is_expressed = (log_ranks_3d != log_ranks_3d.min(axis=-1, keepdims=True))
    expr_frac = is_expressed.astype(jnp.float16).mean(axis=1)  # Mean across neighbors (float16 for memory)

    # Calculate marker score
    marker_score = jnp.exp(weighted_log_mean - global_log_gmean)
    marker_score = jnp.where(marker_score < 1.0, 0.0, marker_score)

    # Apply expression fraction filter
    frac_mask = expr_frac > global_expr_frac
    marker_score = jnp.where(frac_mask, marker_score, 0.0)

    marker_score = jnp.exp(marker_score ** 1.5) - 1.0

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
        
    def load_global_stats(self, mean_frac_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-calculated global geometric mean and expression fraction from parquet"""
        
        logger.info("Loading global statistics from parquet...")
        parquet_path = Path(mean_frac_path)
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Global stats file not found: {parquet_path}")
        
        # Load the dataframe
        mean_frac_df = pd.read_parquet(parquet_path)
        
        # Extract global log geometric mean and expression fraction
        global_log_gmean = mean_frac_df['G_Mean'].values.astype(np.float32)
        global_expr_frac = mean_frac_df['frac'].values.astype(np.float32)
        
        logger.info(f"Loaded global stats for {len(global_log_gmean)} genes")
        
        return global_log_gmean, global_expr_frac

    def _find_homogeneous_spots(
        self,
        adata: ad.AnnData,
        cell_type: str,
        annotation_key: str,
        coords: Optional[np.ndarray],
        emb_gcn: Optional[np.ndarray],
        emb_indv: np.ndarray,
        slice_ids: Optional[np.ndarray],
        rank_shape: Tuple[int, int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """
        Prepare batch data for a cell type
        
        Returns:
            Tuple of (neighbor_indices, neighbor_weights, cell_indices_sorted, n_cells)
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
            return None
        
        logger.info(f"Processing {cell_type}: {n_cells} cells")
        
        # Build connectivity matrix
        logger.info("Building connectivity matrix...")
        neighbor_indices, neighbor_weights = self.connectivity_builder.build_connectivity_matrix(
            coords=coords,
            emb_gcn=emb_gcn,
            emb_indv=emb_indv,
            cell_mask=cell_mask,
            slice_ids=slice_ids,
            return_dense=True,
            k_central=self.config.num_neighbour_spatial,
            k_adjacent=self.config.k_adjacent if hasattr(self.config, 'k_adjacent') else 7,
            n_adjacent_slices=self.config.n_adjacent_slices if hasattr(self.config, 'n_adjacent_slices') else 1
        )
        gc.collect()

        # Validate neighbor indices
        max_valid_idx = rank_shape[0] - 1
        assert neighbor_indices.max() <= max_valid_idx, \
            f"Neighbor indices exceed bounds (max: {neighbor_indices.max()}, limit: {max_valid_idx})"
        assert neighbor_indices.min() >= 0, \
            f"Found negative neighbor indices (min: {neighbor_indices.min()})"
        
        # Optimize row order
        logger.info("Optimizing row order for cache efficiency...")
        row_order = optimize_row_order(
            neighbor_indices,
            cell_indices=cell_indices,
            method=None,
            neighbor_weights=neighbor_weights
        )
        
        neighbor_indices = neighbor_indices[row_order]
        neighbor_weights = neighbor_weights[row_order]
        cell_indices_sorted = cell_indices[row_order]
        
        # Save neighbor indices and weights to obsm matrices
        # Initialize matrices if they don't exist
        if 'gsMap_homo_indices' not in adata.obsm.keys():
            adata.obsm['gsMap_homo_indices'] = np.zeros((adata.n_obs, neighbor_indices.shape[1]), dtype=neighbor_indices.dtype)
        if 'gsMap_homo_weights' not in adata.obsm.keys():
            adata.obsm['gsMap_homo_weights'] = np.zeros((adata.n_obs, neighbor_weights.shape[1]), dtype=neighbor_weights.dtype)
        
        # Store the neighbor indices and weights for this cell type
        adata.obsm['gsMap_homo_indices'][cell_indices_sorted] = neighbor_indices
        adata.obsm['gsMap_homo_weights'][cell_indices_sorted] = neighbor_weights
        
        return neighbor_indices, neighbor_weights, cell_indices_sorted, n_cells

    def process_cell_type(
        self,
        adata: ad.AnnData,
        cell_type: str,
        output_memmap: MemMapDense,
        global_log_gmean: np.ndarray,
        global_expr_frac: np.ndarray,
        rank_memmap,
        reader: ParallelRankReader,
        computer: ParallelMarkerScoreComputer,
        writer: ParallelMarkerScoreWriter,
        coords: Optional[np.ndarray],
        emb_gcn: Optional[np.ndarray],
        emb_indv: np.ndarray,
        annotation_key: str,
        slice_ids: Optional[np.ndarray] = None
    ):
        """Process a single cell type with shared pools"""
        
        # Prepare batch data
        batch_data = self._find_homogeneous_spots(
            adata, cell_type, annotation_key, coords, emb_gcn,
            emb_indv, slice_ids, reader.shape
        )
        
        if batch_data is None:
            return
        
        neighbor_indices, neighbor_weights, cell_indices_sorted, n_cells = batch_data
        
        # Reset all components for new cell type
        reader.reset_for_cell_type(cell_type)
        computer.reset_for_cell_type(cell_type)
        writer.reset_for_cell_type(cell_type)
        
        # Calculate batch parameters
        batch_size = self.config.mkscore_batch_size
        n_batches = (n_cells + batch_size - 1) // batch_size
        
        # Set batch context for computer (will be set by pipeline.submit_batches)
        # Note: Pipeline will handle batch submission and context setting
        
        # Optional profiling
        use_profiling = self.config.enable_profiling
        if use_profiling:
            import viztracer
            tracer = viztracer.VizTracer(
                output_file=f"marker_score_{cell_type}_{n_batches}.json",
                max_stack_depth=10,

            )
            tracer.start()
        
        # Create and run pipeline
        pipeline = MarkerScorePipeline(
            reader=reader,
            computer=computer,
            writer=writer,
            n_batches=n_batches,
            batch_size=batch_size,
            n_cells=n_cells
        )
        
        try:
            pipeline.run(
                neighbor_indices=neighbor_indices,
                neighbor_weights=neighbor_weights,
                cell_indices_sorted=cell_indices_sorted,
                num_homogeneous=self.config.num_homogeneous,
                cell_type=cell_type
            )
        except Exception as e:
            logger.error(f"Pipeline failed for {cell_type}: {e}")
            # Stop all workers to prevent hanging
            reader.stop_workers.set()
            computer.stop_workers.set()
            writer.stop_workers.set()
            raise
        finally:
            # Stop profiling if enabled
            if use_profiling:
                tracer.stop()
                tracer.save()
                logger.info(f"Profiling saved to marker_score_{cell_type}_{n_batches}.json")
        
        # Final garbage collection
        import gc
        gc.collect()
        
        logger.info(f"✓ Completed processing {cell_type}")
    
    def calculate_marker_scores(
        self,
        adata_path: str,
        rank_memmap_path: str,
        mean_frac_path: str,
        output_path: Optional[str] = None
    ) -> Union[str, Path]:
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

        # Use config path if not specified
        if output_path is None:
            output_path = Path(self.config.marker_scores_memmap_path)
        else:
            output_path = Path(output_path)
        
        # Load concatenated AnnData
        logger.info(f"Loading concatenated AnnData from {adata_path}")
        
        if not Path(adata_path).exists():
            raise FileNotFoundError(f"Concatenated AnnData not found: {adata_path}")
        
        adata = sc.read_h5ad(adata_path)
        
        # Load pre-calculated global statistics
        global_log_gmean, global_expr_frac = self.load_global_stats(mean_frac_path)
        
        # Get annotation key
        annotation_key = self.config.annotation
        
        # Open rank memory map and get dimensions
        rank_memmap_path = Path(rank_memmap_path)
        
        # Get metadata to determine shape
        meta_path = rank_memmap_path.with_suffix('.meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        rank_memmap = MemMapDense(
            path=rank_memmap_path,
            shape=tuple(meta['shape']),
            dtype=np.dtype(meta['dtype']),
            mode='r'
        )
        
        logger.info(f"Opened rank memory map from {rank_memmap_path}")
        n_cells = adata.n_obs
        n_cells_rank = rank_memmap.shape[0]
        n_genes = rank_memmap.shape[1]
        
        logger.info(f"AnnData dimensions: {n_cells} cells × {adata.n_vars} genes")
        logger.info(f"Rank MemMap dimensions: {n_cells_rank} cells × {n_genes} genes")
        
        # Cells should match exactly since filtering is done before rank memmap creation
        assert n_cells == n_cells_rank, \
            f"Cell count mismatch: AnnData has {n_cells} cells, Rank MemMap has {n_cells_rank} cells. " \
            f"This indicates the filtering was not applied consistently during rank calculation."
        
        # Initialize output memory map with float16 for memory efficiency
        output_memmap = MemMapDense(
            output_path,
            shape=(n_cells, n_genes),
            dtype=np.float16,  # Use float16 to save memory
            mode='w',
            num_write_workers=self.config.mkscore_write_workers
        )
        
        # Process each cell type
        if annotation_key is not None:
            assert annotation_key in adata.obs.columns, f"Annotation key '{annotation_key}' not found in adata.obs"
            # Get unique cell types, excluding NaN values
            cell_types = adata.obs[annotation_key].dropna().unique()
            
            # Check if there are any NaN values and handle them
            nan_count = adata.obs[annotation_key].isna().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} cells with NaN annotation in '{annotation_key}', these will be skipped")
        else:
            logger.warning(f"Annotation {annotation_key} not found, processing all cells as one type")
            cell_types = ["all"]
            adata.obs[annotation_key] = "all"
        
        logger.info(f"Processing {len(cell_types)} cell types")
        
        # Load shared data structures once
        logger.info("Loading shared data structures...")
        
        # Load embeddings based on dataset type
        coords = None
        emb_gcn = None
        slice_ids = None
        
        if self.config.dataset_type in ['spatial2D', 'spatial3D']:
            # Load spatial coordinates for spatial datasets
            coords = adata.obsm[self.config.spatial_key]
            # Load niche embeddings for spatial datasets (float16 for memory efficiency)
            emb_gcn = adata.obsm[self.config.latent_representation_niche].astype(np.float16)
            
            # Load slice IDs if provided (for both spatial2D and spatial3D)
            if hasattr(self.config, 'slice_id_key') and self.config.slice_id_key:
                if self.config.slice_id_key in adata.obs.columns:
                    slice_ids = adata.obs[self.config.slice_id_key].values.astype(np.int32)
                    if self.config.dataset_type == 'spatial2D':
                        logger.info(f"Loading slice IDs from {self.config.slice_id_key} for 2D multi-slice data (no cross-slice search)")
                    else:  # spatial3D
                        logger.info(f"Loading slice IDs from {self.config.slice_id_key} for 3D neighbor search (with cross-slice search)")
                else:
                    logger.warning(f"Slice ID key '{self.config.slice_id_key}' not found in adata.obs")
        
        # Load cell embeddings for all dataset types
        emb_indv = adata.obsm[self.config.latent_representation_cell].astype(np.float16)
        
        # Normalize embeddings
        logger.info("Normalizing embeddings...")
        
        # L2 normalize niche embeddings (only for spatial datasets)
        if emb_gcn is not None:
            emb_gcn_norm = np.linalg.norm(emb_gcn, axis=1, keepdims=True)
            emb_gcn = emb_gcn / (emb_gcn_norm + 1e-8)
        
        # L2 normalize individual embeddings
        emb_indv_norm = np.linalg.norm(emb_indv, axis=1, keepdims=True)
        emb_indv = emb_indv / (emb_indv_norm + 1e-8)
        
        # Initialize shared pools with directly connected queues
        logger.info("Initializing shared processing pools with direct queue connections...")

        # Create shared queues to connect components with configured sizes
        reader_to_computer_queue = queue.Queue(maxsize=self.config.compute_workers * self.config.compute_input_queue_size)
        computer_to_writer_queue = queue.Queue(maxsize=self.config.writer_queue_size)
        
        reader = ParallelRankReader(
            rank_memmap,
            num_workers=self.config.rank_read_workers,
            output_queue=reader_to_computer_queue  # Direct connection to computer
        )
        
        computer = ParallelMarkerScoreComputer(
            global_log_gmean,
            global_expr_frac,
            self.config.num_homogeneous,
            num_workers=self.config.compute_workers,
            input_queue=reader_to_computer_queue,  # Input from reader
            output_queue=computer_to_writer_queue  # Output to writer
        )
        
        writer = ParallelMarkerScoreWriter(
            output_memmap,
            num_workers=self.config.mkscore_write_workers,
            input_queue=computer_to_writer_queue  # Input from computer
        )
        
        logger.info(f"Processing pools initialized: {self.config.rank_read_workers} readers, "
                   f"{self.config.compute_workers} computers, "
                   f"{self.config.mkscore_write_workers} writers")
        
        for cell_type in cell_types:
            self.process_cell_type(
                adata,
                cell_type,
                output_memmap,
                global_log_gmean,
                global_expr_frac,
                rank_memmap,
                reader,
                computer,
                writer,
                coords,
                emb_gcn,
                emb_indv,
                annotation_key,
                slice_ids
            )
        
        # Save updated AnnData with neighbor matrices
        logger.info(f"Saving updated AnnData with homogeneous spot matrices to {adata_path}")
        adata.write_h5ad(adata_path)
        
        # Close all shared pools after all cell types are processed
        logger.info("Closing shared processing pools...")
        reader.close()
        computer.close()
        writer.close()
        
        # Close rank memory map
        rank_memmap.close()
        
        output_memmap.close()
        logger.info("Marker score calculation complete!")
        
        # Save metadata
        metadata = {
            'n_cells': n_cells,
            'n_genes': n_genes,
            'config': {
                'dataset_type': self.config.dataset_type,
                'num_neighbour_spatial': self.config.num_neighbour_spatial if self.config.dataset_type != 'scRNA-seq' else None,
                'num_anchor': self.config.num_anchor if self.config.dataset_type != 'scRNA-seq' else None,
                'num_homogeneous': self.config.num_homogeneous,
                'similarity_threshold': self.config.similarity_threshold if hasattr(self.config, 'similarity_threshold') else 0.0,
                'k_adjacent': self.config.k_adjacent if hasattr(self.config, 'k_adjacent') else 7,
                'n_adjacent_slices': self.config.n_adjacent_slices if hasattr(self.config, 'n_adjacent_slices') else 1,
                'slice_id_key': self.config.slice_id_key if hasattr(self.config, 'slice_id_key') else None,
                'batch_size': self.config.mkscore_batch_size,
                'num_read_workers': self.config.rank_read_workers,
                'mkscore_write_workers': self.config.mkscore_write_workers
            },
            'global_log_gmean': global_log_gmean.tolist(),
            'global_expr_frac': global_expr_frac.tolist()
        }
        
        metadata_path = output_path.parent / f'{output_path.stem}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return str(output_path)
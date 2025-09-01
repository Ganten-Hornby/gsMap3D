"""
Marker score calculation using homogeneous neighbors
Implements weighted geometric mean calculation in log space with JAX acceleration
"""

import logging
import queue
import threading
import json
from pathlib import Path
from typing import Optional, Tuple, Union
from functools import partial

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
from scipy.sparse import csr_matrix
import jax
import jax.numpy as jnp
from jax import jit

from .memmap_io import MemMapDense
from .connectivity import ConnectivityMatrixBuilder
from .row_ordering import optimize_row_order

logger = logging.getLogger(__name__)


class ParallelRankReader:
    """Multi-threaded reader for log-rank data from memory-mapped storage"""
    
    def __init__(
        self,
        rank_memmap: Union[MemMapDense, str],
        num_workers: int = 4,
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
        self.result_queue = queue.Queue(maxsize=self.num_workers * 4)
        
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
                
                batch_id, neighbor_indices = item
                
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
                
                # Put result - send rank_data and rank_indices for main thread to combine
                self.result_queue.put((batch_id, rank_data, rank_indices, neighbor_indices.shape))
                self.read_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Reader worker {worker_id} error: {e}")
                raise
        
        # Clean up worker's memory map if it was opened
        if self.memmap_path is not None and 'worker_memmap' in locals():
            del worker_memmap
            logger.info(f"Worker {worker_id} closed its memory map")
    
    def submit_batch(self, batch_id: int, neighbor_indices: np.ndarray):
        """Submit batch for reading"""
        self.read_queue.put((batch_id, neighbor_indices))
    
    def get_result(self):
        """Get next completed batch"""
        return self.result_queue.get()
    
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
    """Multi-threaded computer pool for marker score calculation"""
    
    def __init__(
        self,
        global_log_gmean: np.ndarray,
        global_expr_frac: np.ndarray,
        num_homogeneous: int,
        num_workers: int = 4
    ):
        """
        Initialize computer pool
        
        Args:
            global_log_gmean: Global log geometric mean
            global_expr_frac: Global expression fraction
            num_homogeneous: Number of homogeneous neighbors
            num_workers: Number of compute workers
        """
        self.num_workers = num_workers
        self.num_homogeneous = num_homogeneous
        
        # Store global statistics as JAX arrays
        self.global_log_gmean = jnp.array(global_log_gmean)
        self.global_expr_frac = jnp.array(global_expr_frac)
        
        # Queues for communication
        self.compute_queue = queue.Queue(maxsize=num_workers * 2)
        self.result_queue = queue.Queue(maxsize=num_workers * 2)
        
        # Start worker threads
        self.workers = []
        self.stop_workers = threading.Event()
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
                # Get compute request
                item = self.compute_queue.get(timeout=1)
                if item is None:
                    break
                
                batch_idx, rank_data, rank_indices, neighbor_weights, cell_indices, batch_size = item
                
                # Convert to JAX for efficient computation
                rank_data_jax = jnp.array(rank_data)
                rank_indices_jax = jnp.array(rank_indices)
                
                # Use JAX fancy indexing
                batch_ranks = rank_data_jax[rank_indices_jax]
                
                # Compute marker scores using JAX
                marker_scores = compute_marker_scores_jax(
                    batch_ranks,
                    jnp.array(neighbor_weights),
                    batch_size,
                    self.num_homogeneous,
                    self.global_log_gmean,
                    self.global_expr_frac
                )
                
                # Convert back to numpy
                marker_scores_np = np.array(marker_scores)
                
                # Put result for writing
                self.result_queue.put((batch_idx, marker_scores_np, cell_indices))
                self.compute_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Compute worker {worker_id} error: {e}")
                raise
        
        logger.info(f"Compute worker {worker_id} stopped")
    
    def submit_batch(self, batch_idx: int, rank_data: np.ndarray, rank_indices: np.ndarray,
                    neighbor_weights: np.ndarray, cell_indices: np.ndarray, batch_size: int):
        """Submit batch for computation"""
        self.compute_queue.put((batch_idx, rank_data, rank_indices, neighbor_weights, cell_indices, batch_size))
    
    def get_result(self):
        """Get computed marker scores"""
        return self.result_queue.get()
    
    def get_queue_sizes(self):
        """Get current queue sizes for progress tracking"""
        return self.compute_queue.qsize(), self.result_queue.qsize()
    
    def close(self):
        """Close compute pool"""
        self.stop_workers.set()
        for _ in range(self.num_workers):
            self.compute_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5)
        logger.info("Compute pool closed")


class ParallelMarkerScoreWriter:
    """Multi-threaded writer pool for marker scores"""
    
    def __init__(
        self,
        output_memmap: MemMapDense,
        num_workers: int = 4
    ):
        """
        Initialize writer pool
        
        Args:
            output_memmap: Output memory map
            num_workers: Number of writer threads
        """
        self.output_memmap = output_memmap
        self.num_workers = num_workers
        
        # Queue for write requests
        self.write_queue = queue.Queue(maxsize=100)
        self.completed_count = 0
        self.completed_lock = threading.Lock()
        
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
        
        while not self.stop_workers.is_set():
            try:
                # Get write request
                item = self.write_queue.get(timeout=1)
                if item is None:
                    break
                
                batch_idx, marker_scores, cell_indices = item
                
                # Write to memory map
                self.output_memmap.write_batch(marker_scores, cell_indices)
                
                # Update completed count
                with self.completed_lock:
                    self.completed_count += 1
                
                self.write_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Writer worker {worker_id} error: {e}")
                raise
        
        logger.info(f"Writer worker {worker_id} stopped")
    
    def submit_batch(self, batch_idx: int, marker_scores: np.ndarray, cell_indices: np.ndarray):
        """Queue a batch for writing"""
        self.write_queue.put((batch_idx, marker_scores, cell_indices))
    
    def get_completed_count(self):
        """Get number of completed writes"""
        with self.completed_lock:
            return self.completed_count
    
    def get_queue_size(self):
        """Get write queue size"""
        return self.write_queue.qsize()
    
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

    return marker_score


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

    def process_cell_type(
        self,
        adata: ad.AnnData,
        cell_type: str,
        output_memmap: MemMapDense,
        global_log_gmean: np.ndarray,
        global_expr_frac: np.ndarray,
        rank_memmap,  # Now a memory map directly
        reader: ParallelRankReader,
        coords: np.ndarray,
        emb_gcn: np.ndarray,
        emb_indv: np.ndarray,
        annotation_key: str,
        slice_ids: Optional[np.ndarray] = None
    ):
        """Process a single cell type"""
        
        # Get cells of this type
        cell_mask = adata.obs[annotation_key] == cell_type
        cell_indices = np.where(cell_mask)[0]
        n_cells = len(cell_indices)
        
        min_cells = getattr(self.config, 'min_cells_per_type', 21)
        if n_cells < min_cells:
            logger.warning(f"Skipping {cell_type}: only {n_cells} cells (min: {min_cells})")
            return
        
        logger.info(f"Processing {cell_type}: {n_cells} cells")
        
        # Get rank memmap shape
        rank_memmap_shape = rank_memmap.shape if hasattr(rank_memmap, 'shape') else reader.shape
        
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
        
        # Validate neighbor indices are within bounds
        max_valid_idx = rank_memmap_shape[0] - 1
        assert neighbor_indices.max() <= max_valid_idx, \
            f"Neighbor indices exceed bounds (max: {neighbor_indices.max()}, limit: {max_valid_idx})"
        assert neighbor_indices.min() >= 0, \
            f"Found negative neighbor indices (min: {neighbor_indices.min()})"
        
        # Optimize row order (auto-selects best method)
        logger.info("Optimizing row order for cache efficiency...")
        row_order = optimize_row_order(
            neighbor_indices,
            cell_indices=cell_indices,
            method=None,  # Auto-select based on data
            neighbor_weights=neighbor_weights
        )
        neighbor_indices = neighbor_indices[row_order]
        neighbor_weights = neighbor_weights[row_order]
        cell_indices_sorted = cell_indices[row_order]
        
        # Process in batches
        batch_size = getattr(self.config, 'batch_size', 1000)
        n_batches = (n_cells + batch_size - 1) // batch_size
        
        # Submit all read requests
        logger.info(f"Submitting {n_batches} batches for reading...")
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_cells)
            
            batch_neighbors = neighbor_indices[batch_start:batch_end]
            reader.submit_batch(batch_idx, batch_neighbors)
        
        # Create compute and writer pools
        logger.info("Initializing compute and writer pools...")
        computer = ParallelMarkerScoreComputer(
            global_log_gmean,
            global_expr_frac,
            self.config.num_homogeneous,
            num_workers=getattr(self.config, 'compute_workers', 4)
        )
        
        writer = ParallelMarkerScoreWriter(
            output_memmap,
            num_workers=self.config.mkscore_write_workers
        )
        
        # Process pipeline with fancy progress bar
        logger.info("Starting processing pipeline...")
        
        # Optional profiling
        use_profiling = getattr(self.config, 'enable_profiling', False)
        if use_profiling:
            import viztracer
            tracer = viztracer.VizTracer(
                output_file=f"marker_score_trace_{cell_type}_{n_batches}_pipeline.json",
                max_stack_depth=10
            )
            tracer.start()
        
        # Create progress bar with custom format
        pbar = tqdm(
            total=n_batches,
            desc=f"Processing {cell_type}",
            bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] |{bar}| [Pending: C:{postfix[0]} W:{postfix[1]}]',
            postfix=[0, 0]
        )
        
        # Start a thread to move data from reader to computer
        def reader_to_computer():
            for _ in range(n_batches):
                # Get batch from reader
                result = reader.get_result()
                batch_idx, rank_data, rank_indices, original_shape = result
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_cells)
                actual_batch_size = batch_end - batch_start
                
                # Verify shape
                assert original_shape == (actual_batch_size, self.config.num_homogeneous), \
                    f"Shape mismatch: expected {(actual_batch_size, self.config.num_homogeneous)}, got {original_shape}"
                
                # Get batch weights and cell indices
                batch_weights = neighbor_weights[batch_start:batch_end]
                batch_cell_indices = cell_indices_sorted[batch_start:batch_end]
                
                # Submit to computer
                computer.submit_batch(
                    batch_idx, rank_data, rank_indices,
                    batch_weights, batch_cell_indices, actual_batch_size
                )
        
        # Start a thread to move data from computer to writer
        def computer_to_writer():
            for _ in range(n_batches):
                # Get computed results
                batch_idx, marker_scores, cell_indices = computer.get_result()
                
                # Submit to writer
                writer.submit_batch(batch_idx, marker_scores, cell_indices)
        
        # Start pipeline threads
        import threading
        reader_thread = threading.Thread(target=reader_to_computer, daemon=True)
        compute_thread = threading.Thread(target=computer_to_writer, daemon=True)
        reader_thread.start()
        compute_thread.start()
        
        # Monitor progress
        import time
        while writer.get_completed_count() < n_batches:
            # Update progress bar
            completed = writer.get_completed_count()
            compute_pending, compute_ready = computer.get_queue_sizes()
            write_pending = writer.get_queue_size()
            
            # Update postfix with pending counts
            pbar.postfix = [compute_pending + compute_ready, write_pending]
            pbar.n = completed
            pbar.refresh()
            
            time.sleep(0.1)
        
        # Final update
        pbar.n = n_batches
        pbar.postfix = [0, 0]
        pbar.refresh()
        pbar.close()
        
        # Wait for threads to complete
        reader_thread.join()
        compute_thread.join()
        
        # Clean up pools
        computer.close()
        writer.close()
        
        # Stop profiling if enabled
        if use_profiling:
            tracer.stop()
            tracer.save()
            logger.info(f"Profiling data saved to marker_score_trace_{cell_type}_{n_batches}_pipeline.json")
        
        logger.info(f"Completed processing {cell_type}")
    
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
        
        # Initialize output memory map
        output_memmap = MemMapDense(
            output_path,
            shape=(n_cells, n_genes),
            mode='w',
            num_write_workers=self.config.mkscore_write_workers
        )
        
        # Process each cell type
        if annotation_key and annotation_key in adata.obs.columns:
            cell_types = adata.obs[annotation_key].unique()
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
        
        # Initialize parallel reader once for all cell types
        logger.info("Initializing parallel reader...")

        reader = ParallelRankReader(
            rank_memmap,
            num_workers=self.config.rank_read_workers
        )
        
        for cell_type in cell_types:
            self.process_cell_type(
                adata,
                cell_type,
                output_memmap,
                global_log_gmean,
                global_expr_frac,
                rank_memmap,
                reader,
                coords,
                emb_gcn,
                emb_indv,
                annotation_key,
                slice_ids
            )
        
        # Close the shared reader after all cell types are processed
        reader.close()
        
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
                'batch_size': getattr(self.config, 'batch_size', 1000),
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
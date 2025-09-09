"""
Unified spatial LDSC processor combining chunk production, parallel loading, and result accumulation.
"""

import gc
import json
import logging
import os
import queue
import threading
import multiprocessing as mp
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn, TimeElapsedColumn

from ..config import SpatialLDSCConfig

logger = logging.getLogger("gsMap.spatial_ldsc_processor")


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
        """Batches per second"""
        if self.average_time > 0:
            return 1.0 / self.average_time
        return 0.0


class ParallelLDScoreReader:
    """Multi-threaded reader for fetching LD score chunks from memory-mapped marker scores"""
    
    def __init__(
        self,
        processor,  # Reference to SpatialLDSCProcessor for data access
        num_workers: int = 4,
        output_queue: queue.Queue = None
    ):
        """Initialize reader pool"""
        self.processor = processor
        self.num_workers = num_workers
        
        # Queues for communication
        self.read_queue = queue.Queue()
        self.result_queue = output_queue if output_queue else queue.Queue(maxsize=num_workers * 4)
        
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
        logger.info(f"Started {self.num_workers} reader threads")
    
    def _worker(self, worker_id: int):
        """Worker thread for reading LD score chunks"""
        logger.debug(f"Reader worker {worker_id} started")
        
        while not self.stop_workers.is_set():
            try:
                # Get chunk request
                item = self.read_queue.get(timeout=1)
                if item is None:
                    break
                
                chunk_idx = item
                
                # Track timing
                start_time = time.time()
                
                # Fetch the chunk using processor's method
                ldscore, spot_names, abs_start, abs_end = self.processor._fetch_ldscore_chunk(chunk_idx)
                
                # Truncate to match SNP data
                n_snps_used = self.processor.data_truncated.get('n_snps_used', ldscore.shape[0])
                ldscore = ldscore[:n_snps_used]
                
                # Track throughput
                elapsed = time.time() - start_time
                with self.throughput_lock:
                    self.throughput.record_batch(elapsed)
                
                # Put result for computer
                self.result_queue.put({
                    'chunk_idx': chunk_idx,
                    'ldscore': ldscore,
                    'spot_names': spot_names,
                    'abs_start': abs_start,
                    'abs_end': abs_end,
                    'worker_id': worker_id,
                    'success': True
                })
                self.read_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Reader worker {worker_id} error on chunk {chunk_idx}: {e}")
                self.exception_queue.put((worker_id, e))
                self.has_error.set()
                self.result_queue.put({
                    'chunk_idx': chunk_idx if 'chunk_idx' in locals() else -1,
                    'worker_id': worker_id,
                    'success': False,
                    'error': str(e)
                })
                break
        
        logger.debug(f"Reader worker {worker_id} stopped")
    
    def submit_chunk(self, chunk_idx: int):
        """Submit chunk for reading"""
        self.read_queue.put(chunk_idx)
    
    def get_result(self):
        """Get next completed chunk"""
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
    
    def close(self):
        """Clean up resources"""
        self.stop_workers.set()
        for _ in range(self.num_workers):
            self.read_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5)
        logger.info("Reader pool closed")


class ParallelLDScoreComputer:
    """Multi-threaded computer for processing LD scores with JAX"""
    
    def __init__(
        self,
        processor,  # Reference to SpatialLDSCProcessor
        process_chunk_jit_fn,  # JIT-compiled processing function
        num_workers: int = 4,
        input_queue: queue.Queue = None
    ):
        """Initialize computer pool"""
        self.processor = processor
        self.process_chunk_jit_fn = process_chunk_jit_fn
        self.num_workers = num_workers
        
        # Queues for communication
        self.compute_queue = input_queue if input_queue else queue.Queue(maxsize=num_workers * 2)
        
        # Throughput tracking
        self.throughput = ComponentThroughput()
        self.throughput_lock = threading.Lock()
        
        # Processing statistics
        self.total_cells_processed = 0
        self.total_chunks_processed = 0
        self.stats_lock = threading.Lock()
        
        # Exception handling
        self.exception_queue = queue.Queue()
        self.has_error = threading.Event()
        
        # Results storage
        self.results = []
        self.results_lock = threading.Lock()
        
        # Prepare static JAX arrays
        self._prepare_jax_arrays()
        
        # Start worker threads
        self.workers = []
        self.stop_workers = threading.Event()
        self._start_workers()
    
    def _prepare_jax_arrays(self):
        """Prepare static JAX arrays from data_truncated"""
        n_snps_used = self.processor.data_truncated['n_snps_used']
        
        baseline_ann = (self.processor.data_truncated['baseline_ld'].values.astype(np.float32) * 
                       self.processor.data_truncated['N'].reshape(-1, 1).astype(np.float32) / 
                       self.processor.data_truncated['Nbar'])
        baseline_ann = np.concatenate([baseline_ann, 
                                      np.ones((n_snps_used, 1), dtype=np.float32)], axis=1)
        
        # Convert to JAX arrays
        self.baseline_ld_sum_jax = jnp.asarray(self.processor.data_truncated['baseline_ld_sum'], dtype=jnp.float32)
        self.chisq_jax = jnp.asarray(self.processor.data_truncated['chisq'], dtype=jnp.float32)
        self.N_jax = jnp.asarray(self.processor.data_truncated['N'], dtype=jnp.float32)
        self.baseline_ann_jax = jnp.asarray(baseline_ann, dtype=jnp.float32)
        self.w_ld_jax = jnp.asarray(self.processor.data_truncated['w_ld'], dtype=jnp.float32)
        self.Nbar = self.processor.data_truncated['Nbar']
        
        del baseline_ann
        gc.collect()
    
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
        logger.debug(f"Compute worker {worker_id} started")
        
        while not self.stop_workers.is_set():
            try:
                # Get data from reader
                item = self.compute_queue.get(timeout=1)
                if item is None:
                    break
                
                # Skip failed chunks
                if not item.get('success', False):
                    logger.error(f"Skipping chunk {item.get('chunk_idx')} due to read error")
                    continue
                
                # Unpack data
                chunk_idx = item['chunk_idx']
                ldscore = item['ldscore']
                spot_names = item['spot_names']
                abs_start = item['abs_start']
                abs_end = item['abs_end']
                
                # Track timing
                start_time = time.time()
                
                # Convert to JAX and process
                spatial_ld_jax = jnp.asarray(ldscore, dtype=jnp.float32)
                
                # Process with JIT function
                batch_size = min(50, spot_names.shape[0])
                betas, ses = self.process_chunk_jit_fn(
                    self.processor.config.n_blocks,
                    batch_size,
                    spatial_ld_jax,
                    self.baseline_ld_sum_jax,
                    self.chisq_jax,
                    self.N_jax,
                    self.baseline_ann_jax,
                    self.w_ld_jax,
                    self.Nbar
                )
                
                # Ensure computation completes
                betas.block_until_ready()
                ses.block_until_ready()
                
                # Convert to numpy
                betas_np = np.array(betas)
                ses_np = np.array(ses)
                
                # Track throughput and statistics
                elapsed = time.time() - start_time
                n_cells_in_chunk = abs_end - abs_start
                
                with self.throughput_lock:
                    self.throughput.record_batch(elapsed)
                
                with self.stats_lock:
                    self.total_cells_processed += n_cells_in_chunk
                    self.total_chunks_processed += 1
                
                # Store result
                with self.results_lock:
                    self.processor._add_chunk_result(
                        chunk_idx, betas_np, ses_np, spot_names,
                        abs_start, abs_end
                    )
                
                # Clean up
                del spatial_ld_jax, betas, ses
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Compute worker {worker_id} error: {e}")
                self.exception_queue.put((worker_id, e))
                self.has_error.set()
                break
        
        logger.debug(f"Compute worker {worker_id} stopped")
    
    def get_queue_size(self):
        """Get compute queue size"""
        return self.compute_queue.qsize()
    
    def get_stats(self):
        """Get processing statistics"""
        with self.stats_lock:
            return self.total_cells_processed, self.total_chunks_processed
    
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


class SpatialLDSCProcessor:
    """
    Unified processor for spatial LDSC that combines:
    - ChunkProducer: Loading spatial LD chunks
    - ParallelChunkLoader: Managing parallel chunk loading with adjacent fetching
    - QuickModeLDScore: Handling memory-mapped marker scores
    - ResultAccumulator: Validating, merging and saving results
    """
    
    def __init__(self, 
                 config: SpatialLDSCConfig,
                 trait_name: str,
                 data_truncated: dict,
                 output_dir: Path,
                 n_loader_threads: int = 10):
        """
        Initialize the unified processor.
        
        Args:
            config: Configuration object
            trait_name: Name of the trait being processed
            data_truncated: Truncated SNP data dictionary
            output_dir: Output directory for results
            n_loader_threads: Number of parallel loader threads
        """
        self.config = config
        self.trait_name = trait_name
        self.data_truncated = data_truncated
        self.output_dir = output_dir
        self.n_loader_threads = n_loader_threads
        
        # Check marker score format
        if config.marker_score_format != "memmap":
            raise NotImplementedError(
                f"Only 'memmap' marker score format is supported. Got: {config.marker_score_format}"
            )
        
        # Initialize QuickModeLDScore components
        self._initialize_quick_mode()
        
        # Result accumulation
        self.results = []
        self.processed_chunks = set()
        self.min_spot_start = float('inf')
        self.max_spot_end = 0
        
    def _initialize_quick_mode(self):
        """Initialize quick mode components for memory-mapped marker scores."""
        logger.info("Initializing memory-mapped marker scores...")
        
        # Load memory-mapped marker scores
        mk_score_data_path = Path(self.config.marker_scores_memmap_path)
        mk_score_meta_path = mk_score_data_path.with_suffix('.meta.json')
        
        if not mk_score_meta_path.exists():
            raise FileNotFoundError(f"Marker scores metadata not found at {mk_score_meta_path}")
        
        # Load metadata
        with open(mk_score_meta_path, 'r') as f:
            meta = json.load(f)
        
        n_spots_memmap = meta['shape'][0]
        n_genes_memmap = meta['shape'][1]
        dtype = np.dtype(meta['dtype'])
        
        # Open memory-mapped array
        self.mkscore_memmap = np.memmap(
            mk_score_data_path,
            dtype=dtype,
            mode='r',
            shape=(n_spots_memmap, n_genes_memmap)
        )
        
        logger.info(f"Marker scores shape: (n_spots={n_spots_memmap}, n_genes={n_genes_memmap})")
        
        # Load concatenated latent adata for metadata
        concat_adata_path = Path(self.config.concatenated_latent_adata_path)
        concat_adata = ad.read_h5ad(concat_adata_path, backed='r')
        gene_names_from_adata = concat_adata.var_names.to_numpy()
        self.spot_names_all = concat_adata.obs_names.to_numpy()
        
        # Filter by sample if specified
        if self.config.sample_name:
            logger.info(f"Filtering spots by sample_name: {self.config.sample_name}")
            sample_info = concat_adata.obs.get('sample', concat_adata.obs.get('sample_name', None))
            
            if sample_info is None:
                concat_adata.file.close()
                raise ValueError("No 'sample' or 'sample_name' column found in obs")
            
            sample_info = sample_info.to_numpy()
            self.spot_indices = np.where(sample_info == self.config.sample_name)[0]
            
            # Verify spots are contiguous for efficient slicing
            expected_range = list(range(self.spot_indices[0], self.spot_indices[-1] + 1))
            if self.spot_indices.tolist() != expected_range:
                concat_adata.file.close()
                raise ValueError("Spot indices for sample must be contiguous")
            
            self.sample_start_offset = self.spot_indices[0]
            self.spot_names_filtered = self.spot_names_all[self.spot_indices]
            logger.info(f"Found {len(self.spot_indices)} spots for sample '{self.config.sample_name}'")
        else:
            self.spot_indices = np.arange(n_spots_memmap)
            self.spot_names_filtered = self.spot_names_all
            self.sample_start_offset = 0
        
        concat_adata.file.close()
        self.n_spots = n_spots_memmap
        self.n_spots_filtered = len(self.spot_indices)
        
        # Load SNP-gene weights
        snp_gene_weight_path = Path(self.config.snp_gene_weight_adata_path)
        if not snp_gene_weight_path.exists():
            raise FileNotFoundError(f"SNP-gene weight matrix not found at {snp_gene_weight_path}")
        
        snp_gene_weight_adata = ad.read_h5ad(snp_gene_weight_path)
        
        # Find common genes
        memmap_genes_series = pd.Series(gene_names_from_adata)
        common_genes_mask = memmap_genes_series.isin(snp_gene_weight_adata.var.index)
        common_genes = gene_names_from_adata[common_genes_mask]
        
        self.memmap_gene_indices = np.where(common_genes_mask)[0]
        snp_gene_indices = [snp_gene_weight_adata.var.index.get_loc(g) for g in common_genes]
        
        logger.info(f"Found {len(common_genes)} common genes")
        
        # Get SNP positions from data_truncated
        snp_positions = self.data_truncated.get('snp_positions', None)
        if snp_positions is None:
            raise ValueError("snp_positions not found in data_truncated")
        
        # Extract SNP-gene weight matrix
        self.snp_gene_weight_sparse = snp_gene_weight_adata[snp_positions, snp_gene_indices].X
        
        if hasattr(self.snp_gene_weight_sparse, 'tocsr'):
            self.snp_gene_weight_sparse = self.snp_gene_weight_sparse.tocsr()
        
        # Set up chunking
        self.chunk_size = self.config.spots_per_chunk_quick_mode
        
        # Handle cell indices range if specified
        if self.config.cell_indices_range:
            start_cell, end_cell = self.config.cell_indices_range
            # Adjust for filtered spots
            start_cell = max(0, start_cell)
            end_cell = min(end_cell, self.n_spots_filtered)
            self.chunk_starts = list(range(start_cell, end_cell, self.chunk_size))
            logger.info(f"Processing cell range [{start_cell}, {end_cell})")
            self.total_cells_to_process = end_cell - start_cell
        else:
            self.chunk_starts = list(range(0, self.n_spots_filtered, self.chunk_size))
            self.total_cells_to_process = self.n_spots_filtered
        
        self.total_chunks = len(self.chunk_starts)
        logger.info(f"Total chunks to process: {self.total_chunks}")
        
    def _fetch_ldscore_chunk(self, chunk_index: int) -> Tuple[np.ndarray, pd.Index, int, int]:
        """
        Fetch LD score chunk for given index.
        
        Returns:
            Tuple of (ldscore_array, spot_names, absolute_start, absolute_end)
        """
        if chunk_index >= len(self.chunk_starts):
            raise ValueError(f"Invalid chunk index {chunk_index}")
        
        start = self.chunk_starts[chunk_index]
        end = min(start + self.chunk_size, self.n_spots_filtered)
        
        # Calculate absolute positions in memmap
        memmap_start = self.sample_start_offset + start
        memmap_end = self.sample_start_offset + end
        
        # Load chunk from memmap
        mk_score_chunk = self.mkscore_memmap[memmap_start:memmap_end, self.memmap_gene_indices]
        mk_score_chunk = mk_score_chunk.T.astype(np.float32)
        
        # Compute LD scores via sparse matrix multiplication
        ldscore_chunk = self.snp_gene_weight_sparse @ mk_score_chunk
        
        if hasattr(ldscore_chunk, 'toarray'):
            ldscore_chunk = ldscore_chunk.toarray()
        
        # Get spot names
        spot_names = pd.Index(self.spot_names_filtered[start:end])
        
        # Calculate absolute positions in original data
        absolute_start = self.spot_indices[start] if start < len(self.spot_indices) else start
        absolute_end = self.spot_indices[end - 1] + 1 if end > 0 else absolute_start
        
        return ldscore_chunk.astype(np.float32, copy=False), spot_names, absolute_start, absolute_end
    
    
    def process_all_chunks(self, process_chunk_jit_fn) -> pd.DataFrame:
        """
        Process all chunks using parallel reader-computer pipeline.
        
        Args:
            process_chunk_jit_fn: JIT-compiled function for processing chunks
            
        Returns:
            Merged DataFrame with all results
        """
        # Create the reader-computer pipeline
        reader = ParallelLDScoreReader(
            processor=self,
            num_workers=self.config.num_read_workers,
        )
        
        computer = ParallelLDScoreComputer(
            processor=self,
            process_chunk_jit_fn=process_chunk_jit_fn,
            num_workers=self.config.ldsc_compute_workers,
            input_queue=reader.result_queue  # Connect reader output to computer input
        )
        
        try:
            # Submit all chunks to reader
            for chunk_idx in range(self.total_chunks):
                reader.submit_chunk(chunk_idx)
            
            # Build description with sample name and range info
            desc_parts = [f"Processing {self.total_chunks:,} chunks ({self.total_cells_to_process:,} cells)"]
            
            if hasattr(self.config, 'sample_name') and self.config.sample_name:
                desc_parts.append(f"Sample: {self.config.sample_name}")
            
            if self.config.cell_indices_range:
                start_cell, end_cell = self.config.cell_indices_range
                desc_parts.append(f"Range: [{start_cell:,}-{end_cell:,})")
            
            description = " | ".join(desc_parts)
            
            # Start JAX profiling if needed
            if hasattr(self.config, 'enable_jax_profiling') and self.config.enable_jax_profiling:
                jax.profiler.start_trace("/tmp/jax-trace-ldsc")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TextColumn("[bold green]{task.fields[speed]} cells/s"),
                TextColumn("[dim]R→C: {task.fields[r_to_c_queue]}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                refresh_per_second=2
            ) as progress:
                task = progress.add_task(
                    description, 
                    total=self.total_cells_to_process,
                    speed="0",
                    r_to_c_queue="0"
                )
                
                start_time = time.time()
                last_update_time = start_time
                last_chunks_processed = 0
                
                while last_chunks_processed < self.total_chunks:
                    # Check for errors
                    reader.check_errors()
                    computer.check_errors()
                    
                    # Get current stats from computer
                    n_cells_processed, n_chunks_processed = computer.get_stats()
                    
                    # Update progress periodically
                    current_time = time.time()
                    if current_time - last_update_time > 0.5:  # Update every 0.5 seconds
                        # Get queue sizes
                        r_pending, r_to_c = reader.get_queue_sizes()
                        
                        # Calculate speed in cells/s
                        elapsed_time = current_time - start_time
                        speed = n_cells_processed / elapsed_time if elapsed_time > 0 else 0
                        
                        # Update progress bar
                        progress.update(
                            task,
                            completed=n_cells_processed,
                            speed=f"{speed:,.0f}",
                            r_to_c_queue=f"{r_to_c}"
                        )
                        last_update_time = current_time
                    
                    # Update last processed count
                    if n_chunks_processed > last_chunks_processed:
                        last_chunks_processed = n_chunks_processed

                        # This would block all threads, so we avoid it
                        # # Periodic memory check
                        # if n_chunks_processed % 100 == 0:
                        #     gc.collect()
                    
                    # Small sleep to prevent busy waiting
                    time.sleep(0.1)
            
            if hasattr(self.config, 'enable_jax_profiling') and self.config.enable_jax_profiling:
                jax.profiler.stop_trace()
            logger.info("JAX profiling trace saved to /tmp/jax-trace-ldsc")
            
        finally:
            # Clean up resources
            reader.close()
            computer.close()
        
        # Validate and merge results
        return self._validate_merge_and_save()
    
    def _add_chunk_result(self, chunk_idx: int, betas: np.ndarray, ses: np.ndarray,
                         spot_names: pd.Index, abs_start: int, abs_end: int):
        """Add processed chunk result to accumulator."""
        # Update coverage tracking
        self.min_spot_start = min(self.min_spot_start, abs_start)
        self.max_spot_end = max(self.max_spot_end, abs_end)
        
        # Store result
        self.results.append({
            'chunk_idx': chunk_idx,
            'betas': betas,
            'ses': ses,
            'spot_names': spot_names,
            'abs_start': abs_start,
            'abs_end': abs_end
        })
        self.processed_chunks.add(chunk_idx)
    
    def _validate_merge_and_save(self) -> pd.DataFrame:
        """
        Validate completeness, merge results, and save with appropriate filename.
        
        Returns:
            Merged DataFrame with all results
        """
        if not self.results:
            raise ValueError("No results to merge")
        
        # Check completeness
        expected_chunks = set(range(self.total_chunks))
        missing_chunks = expected_chunks - self.processed_chunks
        
        if missing_chunks:
            logger.warning(f"Missing chunks: {sorted(missing_chunks)}")
            logger.warning(f"Processed {len(self.processed_chunks)}/{self.total_chunks} chunks")
        
        # Sort results by chunk index
        sorted_results = sorted(self.results, key=lambda x: x['chunk_idx'])
        
        # Merge all results
        dfs = []
        for result in sorted_results:
            betas = result['betas'].astype(np.float64)
            ses = result['ses'].astype(np.float64)
            
            # Calculate statistics
            z_scores = betas / ses
            p_values = norm.sf(z_scores)
            log10_p = -np.log10(np.maximum(p_values, 1e-300))
            
            chunk_df = pd.DataFrame({
                'spot': result['spot_names'],
                'beta': result['betas'],
                'se': result['ses'],
                'z': z_scores.astype(np.float32),
                'p': p_values,
                'neg_log10_p': log10_p
            })
            dfs.append(chunk_df)
        
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Generate filename with cell range information
        filename = self._generate_output_filename()
        output_path = self.output_dir / filename
        
        # Save results
        logger.info(f"Saving results to {output_path}")
        merged_df.to_csv(output_path, index=False, compression='gzip')
        
        # Log statistics
        self._log_statistics(merged_df, output_path)
        
        return merged_df
    
    def _generate_output_filename(self) -> str:
        """Generate output filename including cell range information."""
        base_name = f"{self.config.project_name}_{self.trait_name}"
        
        # If we have cell indices range, include it in filename
        if self.config.cell_indices_range:
            start_cell, end_cell = self.config.cell_indices_range
            # Adjust for actual processed range
            actual_start = max(self.min_spot_start, start_cell)
            actual_end = min(self.max_spot_end, end_cell)
            return f"{base_name}_cells_{actual_start}_{actual_end}.csv.gz"
        
        # Check if we have complete coverage
        if self.min_spot_start == 0 and self.max_spot_end == self.n_spots:
            return f"{base_name}.csv.gz"
        
        # Partial coverage without explicit range
        return f"{base_name}_start{self.min_spot_start}_end{self.max_spot_end}_total{self.n_spots}.csv.gz"
    
    def _log_statistics(self, df: pd.DataFrame, output_path: Path):
        """Log statistical summary of results."""
        n_spots = len(df)
        bonferroni_threshold = 0.05 / n_spots
        n_bonferroni_sig = (df['p'] < bonferroni_threshold).sum()
        
        # FDR correction
        _, fdr_corrected_pvals, _, _ = multipletests(
            df['p'], alpha=0.001, method='fdr_bh'
        )
        n_fdr_sig = fdr_corrected_pvals.sum()
        
        logger.info("=" * 70)
        logger.info("STATISTICAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total spots: {n_spots:,}")
        logger.info(f"Cell range processed: [{self.min_spot_start}, {self.max_spot_end})")
        logger.info(f"Max -log10(p): {df['neg_log10_p'].max():.2f}")
        logger.info("-" * 70)
        logger.info(f"Nominally significant (p < 0.05): {(df['p'] < 0.05).sum():,}")
        logger.info(f"Bonferroni threshold: {bonferroni_threshold:.2e}")
        logger.info(f"Bonferroni significant: {n_bonferroni_sig:,}")
        logger.info(f"FDR significant (alpha=0.001): {n_fdr_sig:,}")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {output_path}")
        
        # Warn if incomplete
        if len(self.processed_chunks) < self.total_chunks:
            logger.warning(f"WARNING: Only processed {len(self.processed_chunks)}/{self.total_chunks} chunks")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'mkscore_memmap'):
            del self.mkscore_memmap
            logger.debug("Cleaned up memory-mapped arrays")
        
        # Clean up other resources
        gc.collect()
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
"""
Memory-mapped I/O utilities for efficient large-scale data handling
Replaces Zarr-backed storage with NumPy memory maps for better performance
"""

import logging
import json
import queue
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import time

import numpy as np

logger = logging.getLogger(__name__)


class MemMapDense:
    """Dense matrix storage using NumPy memory maps with async multi-threaded writing"""

    def __init__(
        self,
        path: Union[str, Path],
        shape: Tuple[int, int],
        dtype=np.float32,
        mode: str = 'w',
        num_write_workers: int = 4,
        flush_interval: float = 30,
    ):
        """
        Initialize a memory-mapped dense matrix.

        Args:
            path: Path to the memory-mapped file (without extension)
            shape: Shape of the matrix (n_rows, n_cols)
            dtype: Data type of the matrix
            mode: 'w' for write (create/overwrite), 'r' for read, 'r+' for read/write
            num_write_workers: Number of worker threads for async writing
        """
        self.path = Path(path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.num_write_workers = num_write_workers
        self.flush_interval = flush_interval
        # File paths
        self.data_path = self.path.with_suffix('.dat')
        self.meta_path = self.path.with_suffix('.meta.json')

        # Initialize memory map
        if mode == 'w':
            self._create_memmap()
        elif mode == 'r':
            self._open_memmap_readonly()
        elif mode == 'r+':
            self._open_memmap_readwrite()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'w', 'r', or 'r+'")

        # Async writing setup (only for write modes)
        self.write_queue = queue.Queue(maxsize=100)
        self.writer_threads = []
        self.stop_writer = threading.Event()

        if mode in ('w', 'r+'):
            self._start_writer_threads()

    def _create_memmap(self):
        """Create a new memory-mapped file"""
        # Check if already exists and is complete
        if self.meta_path.exists():
            try:
                with open(self.meta_path, 'r') as f:
                    meta = json.load(f)
                if meta.get('complete', False):
                    raise ValueError(
                        f"MemMapDense at {self.path} already exists and is marked as complete. "
                        f"Please delete it manually if you want to overwrite: rm {self.data_path} {self.meta_path}"
                    )
                else:
                    logger.warning(f"MemMapDense at {self.path} exists but is incomplete. Recreating.")
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Invalid metadata at {self.meta_path}. Recreating.")

        # Create new memory map
        self.memmap = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode='w+',
            shape=self.shape
        )

        # # Initialize to zeros
        # self.memmap[:] = 0
        # self.memmap.flush()

        # Write metadata
        meta = {
            'shape': self.shape,
            'dtype': np.dtype(self.dtype).name,  # Use dtype.name for proper serialization
            'complete': False,
            'created_at': time.time()
        }
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Created MemMapDense at {self.data_path} with shape {self.shape}")

    def _open_memmap_readonly(self):
        """Open an existing memory-mapped file in read-only mode"""
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        # Read metadata
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)

        if not meta.get('complete', False):
            raise ValueError(f"MemMapDense at {self.path} is incomplete")

        # Validate shape and dtype
        if tuple(meta['shape']) != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape}, got {tuple(meta['shape'])}"
            )

        # Open memory map
        self.memmap = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode='r',
            shape=self.shape
        )

        logger.info(f"Opened MemMapDense at {self.data_path} in read-only mode")

    def _open_memmap_readwrite(self):
        """Open an existing memory-mapped file in read-write mode"""
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        # Read metadata
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)

        # Open memory map
        self.memmap = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode='r+',
            shape=tuple(meta['shape'])
        )

        logger.info(f"Opened MemMapDense at {self.data_path} in read-write mode")

    def _start_writer_threads(self):
        """Start multiple background writer threads"""
        def writer_worker(worker_id):
            last_flush_time = time.time()  # Track last flush time for worker 0

            while not self.stop_writer.is_set():
                try:
                    item = self.write_queue.get(timeout=1)
                    if item is None:
                        break
                    data, row_indices, col_slice = item

                    # Write data with thread safety
                    if isinstance(row_indices, slice):
                        self.memmap[row_indices, col_slice] = data
                    elif isinstance(row_indices, (int, np.integer)):
                        start_row = row_indices
                        end_row = start_row + data.shape[0]
                        self.memmap[start_row:end_row, col_slice] = data
                    else:
                        # Handle array of indices
                        self.memmap[row_indices, col_slice] = data

                    # Periodic flush every 1 second for worker 0
                    if worker_id == 0:
                        current_time = time.time()
                        if current_time - last_flush_time >= self.flush_interval:
                            self.memmap.flush()
                            last_flush_time = time.time()
                            logger.debug(f"Worker 0 flushed memmap at {last_flush_time:.2f}")

                    self.write_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Writer thread {worker_id} error: {e}")
                    raise

        # Start multiple writer threads
        for i in range(self.num_write_workers):
            thread = threading.Thread(target=writer_worker, args=(i,), daemon=True)
            thread.start()
            self.writer_threads.append(thread)
        logger.info(f"Started {self.num_write_workers} writer threads for MemMapDense")

    def write_batch(self, data: np.ndarray, row_indices: Union[int, slice, np.ndarray], col_slice=slice(None)):
        """Queue batch for async writing

        Args:
            data: Data to write
            row_indices: Either a single row index, slice, or array of row indices
            col_slice: Column slice (default: all columns)
        """
        if self.mode not in ('w', 'r+'):
            logger.warning("Cannot write to read-only MemMapDense")
            return

        self.write_queue.put((data, row_indices, col_slice))

    def read_batch(self, row_indices: Union[int, slice, np.ndarray], col_slice=slice(None)) -> np.ndarray:
        """Read batch of data

        Args:
            row_indices: Row indices to read
            col_slice: Column slice (default: all columns)

        Returns:
            NumPy array with the requested data
        """
        if isinstance(row_indices, (int, np.integer)):
            return self.memmap[row_indices:row_indices+1, col_slice].copy()
        else:
            return self.memmap[row_indices, col_slice].copy()

    def __getitem__(self, key):
        """Direct array access for compatibility"""
        return self.memmap[key]

    def __setitem__(self, key, value):
        """Direct array access for compatibility"""
        if self.mode not in ('w', 'r+'):
            raise ValueError("Cannot write to read-only MemMapDense")
        self.memmap[key] = value

    def mark_complete(self):
        """Mark the memory map as complete"""
        if self.mode in ('w', 'r+'):
            logger.info("Marking memmap as complete")
            # Ensure all writes are flushed
            if self.writer_threads and not self.write_queue.empty():
                logger.info("Waiting for remaining writes before marking complete...")
                self.write_queue.join()

            # Flush memory map to disk
            logger.info("Flushing memmap to disk...")
            self.memmap.flush()
            logger.info("Memmap flush complete")

            # Update metadata
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
            meta['complete'] = True
            meta['completed_at'] = time.time()
            # Ensure dtype is properly serialized
            if 'dtype' in meta and not isinstance(meta['dtype'], str):
                meta['dtype'] = np.dtype(self.dtype).name
            with open(self.meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(f"Marked MemMapDense at {self.path} as complete")

    @property
    def is_complete(self) -> bool:
        """Check if the memory map is marked as complete"""
        if not self.meta_path.exists():
            return False

        try:
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
            return meta.get('complete', False)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read metadata to check completion status: {e}")
            return False

    def close(self):
        """Clean up resources"""
        logger.info("MemMapDense.close() called")
        if self.writer_threads:
            logger.info("Closing MemMapDense: waiting for queued writes...")
            self.write_queue.join()
            logger.info("All queued writes have been processed")
            self.stop_writer.set()
            logger.info("Stop signal set for writer threads")

            # Send stop signal to all threads
            for _ in self.writer_threads:
                self.write_queue.put(None)
            logger.info("Stop sentinels queued for writer threads")

            # Wait for all threads to finish
            for thread in self.writer_threads:
                thread.join(timeout=5.0)

        # Final flush
        if self.mode in ('w', 'r+'):
            self.mark_complete()

    def __enter__(self):
        return self

    @property
    def attrs(self):
        """Compatibility property for accessing metadata"""
        if hasattr(self, '_attrs'):
            return self._attrs

        if self.meta_path.exists():
            with open(self.meta_path, 'r') as f:
                self._attrs = json.load(f)
        else:
            self._attrs = {}
        return self._attrs

    def delete(self):
        """Delete the memory-mapped files"""
        self.close()
        if self.data_path.exists():
            self.data_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        logger.info(f"Deleted MemMapDense files at {self.path}")


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
        logger.debug(f"Reader worker {worker_id} started")

        # Open worker's own memory map instance
        data_path = self.memmap_path.with_suffix('.dat')
        worker_memmap = np.memmap(
            data_path,
            dtype=self.dtype,
            mode='r',
            shape=self.shape
        )
        logger.debug(f"Worker {worker_id} opened its own memory map at {data_path}")

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
                error_trace = traceback.format_exc()
                logger.error(f"Reader worker {worker_id} error: {e}\nTraceback:\n{error_trace}")
                self.exception_queue.put((worker_id, e, error_trace))
                self.has_error.set()
                self.stop_workers.set()  # Signal all workers to stop
                break

        # Clean up worker's memory map if it was opened
        if self.memmap_path is not None and 'worker_memmap' in locals():
            del worker_memmap
            logger.debug(f"Worker {worker_id} closed its memory map")

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
                worker_id, exception, error_trace = self.exception_queue.get_nowait()
                raise RuntimeError(f"Reader worker {worker_id} failed: {exception}\nOriginal traceback:\n{error_trace}") from exception
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
        logger.debug(f"Started {self.num_workers} writer threads")

    def _writer_worker(self, worker_id: int):
        """Writer worker thread"""
        logger.debug(f"Writer worker {worker_id} started")

        # Open worker's own memory map instance
        data_path = self.memmap_path.with_suffix('.dat')
        worker_memmap = np.memmap(
            data_path,
            dtype=self.dtype,
            mode='r+',  # Read-write mode for writing
            shape=self.shape
        )
        logger.debug(f"Writer worker {worker_id} opened its own memory map at {data_path}")

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
                error_trace = traceback.format_exc()
                logger.error(f"Writer worker {worker_id} error: {e}\nTraceback:\n{error_trace}")
                self.exception_queue.put((worker_id, e, error_trace))
                self.has_error.set()
                self.stop_workers.set()  # Signal all workers to stop
                break

        # Final flush before closing
        worker_memmap.flush()
        # Clean up worker's memory map
        del worker_memmap
        logger.debug(f"Writer worker {worker_id} closed its memory map")

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
                worker_id, exception, error_trace = self.exception_queue.get_nowait()
                raise RuntimeError(f"Writer worker {worker_id} failed: {exception}\nOriginal traceback:\n{error_trace}") from exception
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

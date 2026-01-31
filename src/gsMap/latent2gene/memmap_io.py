"""
Memory-mapped I/O utilities for efficient large-scale data handling
Replaces Zarr-backed storage with NumPy memory maps for better performance
"""

import json
import logging
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class MemMapDense:
    """Dense matrix storage using NumPy memory maps with async multi-threaded writing"""

    def __init__(
        self,
        path: str | Path,
        shape: tuple[int, int],
        dtype=np.float16,
        mode: str = 'w',
        num_write_workers: int = 4,
        flush_interval: float = 30,
        tmp_dir: str | Path | None = None,
    ):
        """
        Initialize a memory-mapped dense matrix.

        Args:
            path: Path to the memory-mapped file (without extension)
            shape: Shape of the matrix (n_rows, n_cols)
            dtype: Data type of the matrix
            mode: 'w' for write (create/overwrite), 'r' for read, 'r+' for read/write
            num_write_workers: Number of worker threads for async writing
            tmp_dir: Optional temporary directory for faster I/O on slow filesystems.
                    If provided, files will be created/copied to tmp_dir for operations
                    and synced back to the original path on close.
        """
        self.original_path = Path(path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.num_write_workers = num_write_workers
        self.flush_interval = flush_interval
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None
        self.using_tmp = False
        self.tmp_path = None

        # Set up paths based on whether tmp_dir is provided
        if self.tmp_dir:
            self._setup_tmp_paths()
        else:
            self.path = self.original_path

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

    def _setup_tmp_paths(self):
        """Set up temporary paths for memory-mapped files"""
        # Create a unique subdirectory in tmp_dir to avoid conflicts
        unique_id = str(uuid.uuid4())[:8]
        self.tmp_subdir = self.tmp_dir / f"memmap_{unique_id}"
        self.tmp_subdir.mkdir(parents=True, exist_ok=True)

        # Create tmp path with same structure as original
        self.tmp_path = self.tmp_subdir / self.original_path.name
        self.path = self.tmp_path
        self.using_tmp = True

        logger.info(f"Using temporary directory for memmap: {self.tmp_subdir}")

        # If reading, copy existing files to tmp directory
        if self.mode in ('r', 'r+'):
            original_data_path = self.original_path.with_suffix('.dat')
            original_meta_path = self.original_path.with_suffix('.meta.json')

            if original_data_path.exists() and original_meta_path.exists():
                tmp_data_path = self.tmp_path.with_suffix('.dat')
                tmp_meta_path = self.tmp_path.with_suffix('.meta.json')

                logger.info("Copying memmap files to temporary directory for faster access...")
                shutil.copy2(original_data_path, tmp_data_path)
                shutil.copy2(original_meta_path, tmp_meta_path)
                logger.info(f"Memmap files copied to {self.tmp_subdir}")

    def _sync_tmp_to_original(self):
        """Sync temporary files back to original location"""
        if not self.using_tmp:
            return

        tmp_data_path = self.tmp_path.with_suffix('.dat')
        tmp_meta_path = self.tmp_path.with_suffix('.meta.json')
        original_data_path = self.original_path.with_suffix('.dat')
        original_meta_path = self.original_path.with_suffix('.meta.json')

        if tmp_data_path.exists():
            logger.info("Syncing memmap data from tmp to original location...")
            shutil.move(str(tmp_data_path), str(original_data_path))

        if tmp_meta_path.exists():
            shutil.move(str(tmp_meta_path), str(original_meta_path))

        logger.info(f"Memmap files synced to {self.original_path}")

    def _cleanup_tmp(self):
        """Clean up temporary directory"""
        if self.using_tmp and self.tmp_subdir and self.tmp_subdir.exists():
            try:
                shutil.rmtree(self.tmp_subdir)
                logger.debug(f"Cleaned up temporary directory: {self.tmp_subdir}")
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory {self.tmp_subdir}: {e}")

    def _create_memmap(self):
        """Create a new memory-mapped file"""
        # Check if already exists and is complete
        if self.meta_path.exists():
            try:
                with open(self.meta_path) as f:
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
        with open(self.meta_path) as f:
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
        with open(self.meta_path) as f:
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
        """Start multiple background writer threads sharing the same memmap object"""
        def writer_worker(worker_id):
            last_flush_time = time.time()  # Track last flush time for worker 0

            while not self.stop_writer.is_set():
                try:
                    item = self.write_queue.get(timeout=1)
                    if item is None:
                        break
                    data, row_indices, col_slice = item

                    # Write data with thread safety using shared memmap
                    if isinstance(row_indices, slice):
                        self.memmap[row_indices, col_slice] = data
                    elif isinstance(row_indices, int | np.integer):
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

    def write_batch(self, data: np.ndarray, row_indices: int | slice | np.ndarray, col_slice=slice(None)):
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

    def read_batch(self, row_indices: int | slice | np.ndarray, col_slice=slice(None)) -> np.ndarray:
        """Read batch of data

        Args:
            row_indices: Row indices to read
            col_slice: Column slice (default: all columns)

        Returns:
            NumPy array with the requested data
        """
        if isinstance(row_indices, int | np.integer):
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
            with open(self.meta_path) as f:
                meta = json.load(f)
            meta['complete'] = True
            meta['completed_at'] = time.time()
            # Ensure dtype is properly serialized
            if 'dtype' in meta and not isinstance(meta['dtype'], str):
                meta['dtype'] = np.dtype(self.dtype).name
            with open(self.meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(f"Marked MemMapDense at {self.path} as complete")

    @classmethod
    def check_complete(cls, memmap_path: str | Path, meta_path: str | Path | None = None) -> tuple[bool, dict | None]:
        """
        Check if a memory map file is complete without opening it.

        Args:
            memmap_path: Path to the memory-mapped file (without extension)
            meta_path: Optional path to metadata file. If not provided, will be derived from memmap_path

        Returns:
            Tuple of (is_complete, metadata_dict). metadata_dict is None if file doesn't exist or can't be read.
        """
        memmap_path = Path(memmap_path)

        if meta_path is None:
            # Derive metadata path from memmap path
            if memmap_path.suffix == '.dat':
                meta_path = memmap_path.with_suffix('.meta.json')
            elif memmap_path.suffix == '.meta.json':
                meta_path = memmap_path
            else:
                # Assume no extension, add .meta.json
                meta_path = memmap_path.with_suffix('.meta.json')
        else:
            meta_path = Path(meta_path)

        if not meta_path.exists():
            return False, None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get('complete', False), meta
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read metadata from {meta_path}: {e}")
            return False, None

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

            # Sync tmp files back to original location if using tmp
            if self.using_tmp:
                self._sync_tmp_to_original()

        self._cleanup_tmp()

    def __enter__(self):
        return self

    @property
    def attrs(self):
        """Compatibility property for accessing metadata"""
        if hasattr(self, '_attrs'):
            return self._attrs

        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self._attrs = json.load(f)
        else:
            self._attrs = {}
        return self._attrs


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
            rank_memmap: MemMapDense | str,
            num_workers: int = 4,
            output_queue: queue.Queue = None,
            cache_size_mb: int = 1000
    ):
        # Store shared memmap object if provided
        if isinstance(rank_memmap, MemMapDense):
            self.shared_memmap = rank_memmap.memmap
            self.memmap_path = rank_memmap.path
            self.shape = rank_memmap.shape
            self.dtype = rank_memmap.dtype
        else:
            # Fallback for string path: open a shared read-only memmap here
            self.memmap_path = Path(rank_memmap)
            meta_path = self.memmap_path.with_suffix('.meta.json')
            data_path = self.memmap_path.with_suffix('.dat')
            with open(meta_path) as f:
                meta = json.load(f)
            self.shape = tuple(meta['shape'])
            self.dtype = np.dtype(meta['dtype'])

            # Open the single shared memmap
            self.shared_memmap = np.memmap(
                data_path,
                dtype=self.dtype,
                mode='r',
                shape=self.shape
            )
            logger.info(f"Opened shared memmap for reading at {data_path}")

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
        """Worker thread for reading batches from shared memory map"""
        logger.debug(f"Reader worker {worker_id} started using shared memmap")

        # No need to open a new memmap; use self.shared_memmap directly.
        # Numpy releases the GIL during memmap array access, enabling parallelism.

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

                # Read from shared memory map (thread-safe, GIL released)
                rank_data = self.shared_memmap[flat_indices]

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

        # Do not close shared_memmap here if it was passed in (owned by caller)
        # If we opened it ourselves (str path), we could close it, but memmap doesn't strictly require close()
        pass


class ParallelMarkerScoreWriter:
    """Multi-threaded writer pool for marker scores using shared memmap"""

    def __init__(
            self,
            output_memmap: MemMapDense,
            num_workers: int = 4,
            input_queue: queue.Queue = None
    ):
        """
        Initialize writer pool

        Args:
            output_memmap: Output memory map wrapper object
            num_workers: Number of writer threads
            input_queue: Optional input queue (from computer)
        """
        # Store shared memmap object
        self.shared_memmap = output_memmap.memmap
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
        logger.debug(f"Started {self.num_workers} writer threads with shared memmap")

    def _writer_worker(self, worker_id: int):
        """Writer worker thread using shared memmap"""
        logger.debug(f"Writer worker {worker_id} started")

        while not self.stop_workers.is_set():
            try:
                # Get write request
                item = self.write_queue.get(timeout=1)
                if item is None:
                    break

                batch_idx, marker_scores, cell_indices = item

                # Track timing
                start_time = time.time()

                # Write directly to shared memory map
                # Thread-safe because workers process disjoint batches (indices)
                # cell_indices should be the absolute indices in the full matrix
                self.shared_memmap[cell_indices] = marker_scores

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

        # We do not close or delete the shared memmap here.
        # Flushing is handled by the main thread or MemMapDense wrapper.
        logger.debug(f"Writer worker {worker_id} stopping")

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

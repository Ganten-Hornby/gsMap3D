"""Threaded chunk reading and computation pipeline for spatial LDSC."""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .ldscore_backend import is_gpu_memory_error, select_gpu_ldscore_backend

logger = logging.getLogger("gsMap.spatial_ldsc.pipeline")


@dataclass
class ComponentThroughput:
    """Track throughput for individual pipeline components."""

    total_batches: int = 0
    total_time: float = 0.0
    last_batch_time: float = 0.0

    def record_batch(self, elapsed_time: float):
        self.total_batches += 1
        self.total_time += elapsed_time
        self.last_batch_time = elapsed_time

    @property
    def average_time(self) -> float:
        if self.total_batches > 0:
            return self.total_time / self.total_batches
        return 0.0

    @property
    def throughput(self) -> float:
        if self.average_time > 0:
            return 1.0 / self.average_time
        return 0.0


@dataclass
class SequentialIOThroughput:
    """Track the physical marker windows read from storage."""

    total_windows: int = 0
    total_cells: int = 0
    total_bytes: int = 0
    total_time: float = 0.0

    def record_window(self, elapsed_time: float, n_cells: int, n_bytes: int):
        self.total_windows += 1
        self.total_cells += n_cells
        self.total_bytes += n_bytes
        self.total_time += elapsed_time

    @property
    def cells_per_second(self) -> float:
        return self.total_cells / self.total_time if self.total_time > 0 else 0.0

    @property
    def mib_per_second(self) -> float:
        return self.total_bytes / 1024**2 / self.total_time if self.total_time > 0 else 0.0


class LDScoreStream:
    """Bounded source pipeline for LD-score regression batches.

    Marker memmaps and gene-block stores are consumed as large sequential
    windows by one source thread. Small regression batches are derived from the
    in-RAM window by a transform pool. Other input formats use one serial source
    thread; the old design in which many threads independently issued tiny
    storage reads is intentionally not retained.
    """

    def __init__(
        self,
        processor,
        num_workers: int = 4,
        output_queue: queue.Queue = None,
    ):
        self.processor = processor
        self.num_workers = num_workers
        self.result_queue = (
            output_queue
            if output_queue is not None
            else queue.Queue(maxsize=max(2, num_workers * 2))
        )
        self.throughput = ComponentThroughput()
        self.throughput_lock = threading.Lock()
        self.sequential_io_throughput = SequentialIOThroughput()
        self.exception_queue = queue.Queue()
        self.has_error = threading.Event()
        self.workers = []
        self.stop_workers = threading.Event()
        self.started = False
        chunk_source = getattr(self.processor, "chunk_source", None)
        self.total_chunks = int(
            getattr(
                self.processor,
                "total_chunks",
                len(getattr(chunk_source, "chunk_starts", ())),
            )
        )
        if self.total_chunks < 1:
            raise ValueError("LD-score stream requires at least one chunk")
        self.uses_sequential_windows = bool(
            chunk_source is not None and chunk_source.supports_sequential_marker_windows()
        )
        self.window_chunks = (
            chunk_source.recommended_marker_window_chunks() if self.uses_sequential_windows else 1
        )
        self.batch_queue = (
            queue.Queue(maxsize=max(self.window_chunks, self.num_workers * 2))
            if self.uses_sequential_windows
            else None
        )
        self.source_chunks_loaded = 0
        self.source_progress_lock = threading.Lock()
        self.publish_condition = threading.Condition()
        self.pending_transforms: dict[int, dict] = {}
        self.next_chunk_to_publish = 0
        self.max_pending_transforms = max(2, self.num_workers * 2)

    def start(self):
        """Start the bounded source only after the downstream consumer exists."""
        if self.started:
            raise RuntimeError("LD-score stream has already been started")
        self.started = True

        if self.uses_sequential_windows:
            direct_reads = self.processor.chunk_source.advise_sequential_marker_access()
            for i in range(self.num_workers):
                worker = threading.Thread(
                    target=self._window_transform_worker,
                    args=(i,),
                    name=f"ldsc-window-transform-{i}",
                    daemon=True,
                )
                worker.start()
                self.workers.append(worker)
            source_worker = threading.Thread(
                target=self._window_reader,
                name="ldsc-sequential-window-reader",
                daemon=True,
            )
            source_worker.start()
            self.workers.append(source_worker)
            row_mib = self.processor.chunk_source.marker_row_bytes() / 1024**2
            logger.info(
                f"Started sequential marker reader with {self.window_chunks} chunks/window "
                f"(~{self.window_chunks * self.processor.chunk_size * row_mib:.0f} MiB), "
                f"{self.num_workers} parallel transform workers, "
                f"direct positional reads={'enabled' if direct_reads else 'unavailable'}"
            )
            return

        source_worker = threading.Thread(
            target=self._serial_reader,
            name="ldsc-serial-source-reader",
            daemon=True,
        )
        source_worker.start()
        self.workers.append(source_worker)
        logger.info("Started one serial source reader for non-memmap LDSC input")

    def _serial_reader(self):
        """Read unsupported/non-memmap sources serially with bounded output."""
        logger.debug("Serial LD-score source reader started")
        chunk_idx = -1
        try:
            for chunk_idx in range(self.total_chunks):
                if self.stop_workers.is_set():
                    break
                start_time = time.time()
                fetched = self.processor._fetch_ldscore_chunk(chunk_idx)
                elapsed = time.time() - start_time
                with self.throughput_lock:
                    self.throughput.record_batch(elapsed)
                self._record_source_chunks(1)
                if not self._publish_fetched(chunk_idx, fetched, worker_id="serial-source"):
                    break
        except Exception as e:  # noqa: BLE001
            self._record_error("serial-source", chunk_idx, e)
        logger.debug("Serial LD-score source reader stopped")

    def _window_reader(self):
        """Coalesce queued compute chunks into large sequential memmap reads."""
        logger.debug("Sequential marker-window reader started")
        batch = []
        try:
            for first_chunk in range(0, self.total_chunks, self.window_chunks):
                if self.stop_workers.is_set():
                    break
                batch = list(
                    range(first_chunk, min(first_chunk + self.window_chunks, self.total_chunks))
                )
                start_time = time.perf_counter()
                window = self.processor.chunk_source.load_marker_window(batch)
                elapsed = time.perf_counter() - start_time
                with self.throughput_lock:
                    self.sequential_io_throughput.record_window(
                        elapsed,
                        n_cells=window.values.shape[0],
                        n_bytes=window.values.nbytes,
                    )
                self._record_source_chunks(len(batch))
                for chunk_idx in batch:
                    if not self._put_batch_until_stopped((window, chunk_idx)):
                        break
        except Exception as e:  # noqa: BLE001
            chunk_idx = batch[0] if batch else -1
            self._record_error("sequential-window", chunk_idx, e)
        finally:
            # The thread that owns positional reads also owns the descriptor.
            # This avoids closing/reusing a file descriptor underneath a slow
            # in-flight preadv during cancellation.
            self.processor.chunk_source.close_sequential_marker_access()
            logger.debug("Sequential marker-window reader stopped")

    def _window_transform_worker(self, worker_id: int):
        """Select genes and construct per-chunk LD scores from RAM windows."""
        logger.debug(f"Marker-window transform worker {worker_id} started")
        chunk_idx = -1

        while not self.stop_workers.is_set():
            item = None
            try:
                try:
                    item = self.batch_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if item is None:
                    break

                window, chunk_idx = item
                start_time = time.perf_counter()
                fetched = self.processor.chunk_source.fetch_chunk_from_marker_window(
                    window,
                    chunk_idx,
                    defer_ldscore_to_compute=self.processor.defer_ldscore_to_compute,
                )
                elapsed = time.perf_counter() - start_time
                with self.throughput_lock:
                    self.throughput.record_batch(elapsed)

                ldscore, spot_names, abs_start, abs_end, marker_score_chunk = self._unpack_fetched(
                    fetched
                )
                self._publish_transformed_in_order(
                    {
                        "chunk_idx": chunk_idx,
                        "ldscore": ldscore,
                        "spot_names": spot_names,
                        "abs_start": abs_start,
                        "abs_end": abs_end,
                        "marker_score_chunk": marker_score_chunk,
                        "worker_id": worker_id,
                        "success": True,
                    }
                )
            except Exception as e:  # noqa: BLE001
                self._record_error(worker_id, chunk_idx, e)
                break
            finally:
                if item is not None:
                    self.batch_queue.task_done()

        logger.debug(f"Marker-window transform worker {worker_id} stopped")

    @staticmethod
    def _unpack_fetched(fetched):
        marker_score_chunk = None
        if len(fetched) == 4:
            ldscore, spot_names, abs_start, abs_end = fetched
        else:
            ldscore, spot_names, abs_start, abs_end, marker_score_chunk = fetched
        return ldscore, spot_names, abs_start, abs_end, marker_score_chunk

    def _put_batch_until_stopped(self, item) -> bool:
        while not self.stop_workers.is_set():
            try:
                self.batch_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _publish_transformed_in_order(self, result: dict) -> bool:
        """Publish transform results in chunk order with bounded reordering.

        Parallel sparse transforms can finish out of order. Unboundedly sending
        later chunks downstream would let every subsequent regression finish
        while one early chunk is stalled, defeating the bounded-memory design.
        The earliest missing transform is always allowed into a full reorder
        buffer, so it can drain the gap without deadlocking.
        """
        chunk_idx = int(result["chunk_idx"])
        with self.publish_condition:
            while (
                len(self.pending_transforms) >= self.max_pending_transforms
                and chunk_idx != self.next_chunk_to_publish
                and not self.stop_workers.is_set()
            ):
                self.publish_condition.wait(timeout=0.1)

            if self.stop_workers.is_set():
                return False
            if chunk_idx < self.next_chunk_to_publish or chunk_idx in self.pending_transforms:
                raise RuntimeError(f"Duplicate transformed LDSC chunk {chunk_idx}")

            self.pending_transforms[chunk_idx] = result
            while self.next_chunk_to_publish in self.pending_transforms:
                ordered = self.pending_transforms.pop(self.next_chunk_to_publish)
                if not self._put_result_until_stopped(ordered):
                    return False
                self.next_chunk_to_publish += 1
                self.publish_condition.notify_all()
            return True

    def _publish_fetched(self, chunk_idx: int, fetched, worker_id) -> bool:
        ldscore, spot_names, abs_start, abs_end, marker_score_chunk = self._unpack_fetched(fetched)
        return self._put_result_until_stopped(
            {
                "chunk_idx": chunk_idx,
                "ldscore": ldscore,
                "spot_names": spot_names,
                "abs_start": abs_start,
                "abs_end": abs_end,
                "marker_score_chunk": marker_score_chunk,
                "worker_id": worker_id,
                "success": True,
            }
        )

    def _record_source_chunks(self, n_chunks: int) -> None:
        with self.source_progress_lock:
            self.source_chunks_loaded += n_chunks

    def _record_error(self, worker_id, chunk_idx: int, exception: Exception) -> None:
        error_trace = traceback.format_exc()
        logger.error(
            f"LD-score source worker {worker_id} failed on chunk {chunk_idx}: {exception}\n"
            f"Traceback:\n{error_trace}"
        )
        self.exception_queue.put((worker_id, exception, error_trace))
        self.has_error.set()
        self._put_result_until_stopped(
            {
                "chunk_idx": chunk_idx,
                "worker_id": worker_id,
                "success": False,
                "error": str(exception),
            }
        )

    def _put_result_until_stopped(self, result: dict) -> bool:
        """Publish a result without trapping the source behind a failed consumer."""
        while not self.stop_workers.is_set():
            try:
                self.result_queue.put(result, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def get_result(self):
        return self.result_queue.get()

    def get_queue_sizes(self):
        with self.source_progress_lock:
            pending_reads = max(0, self.total_chunks - self.source_chunks_loaded)
        if self.batch_queue is not None:
            pending_reads += self.batch_queue.qsize()
        with self.publish_condition:
            pending_reads += len(self.pending_transforms)
        return pending_reads, self.result_queue.qsize()

    def check_errors(self):
        if self.has_error.is_set():
            try:
                worker_id, exception, error_trace = self.exception_queue.get_nowait()
                raise RuntimeError(
                    f"Source worker {worker_id} failed: {exception}\n"
                    f"Original traceback:\n{error_trace}"
                ) from exception
            except queue.Empty:
                raise RuntimeError("Source worker failed with unknown error") from None

    def close(self):
        self.stop_workers.set()
        with self.publish_condition:
            self.publish_condition.notify_all()
        if self.batch_queue is not None:
            while True:
                try:
                    self.batch_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self.batch_queue.task_done()
        deadline = time.monotonic() + 5
        for worker in self.workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        live_workers = [worker.name for worker in self.workers if worker.is_alive()]
        if live_workers:
            logger.warning(f"Source workers did not stop within 5 seconds: {live_workers}")
        logger.info("LD-score source stream closed")


class LDScoreComputePool:
    """Bounded worker pool for LD-score construction and JAX regression."""

    def __init__(
        self,
        processor,
        regression_fn,
        num_workers: int = 4,
        input_queue: queue.Queue = None,
        pipeline_depth: int = 0,
    ):
        self.processor = processor
        self.regression_fn = regression_fn
        self.num_workers = num_workers
        # pipeline_depth > 0 => single-thread async double-buffering: dispatch up to
        # `depth` extra chunks ahead and block lazily, overlapping host<->device
        # transfer and result copy-back with on-device compute. Used on GPU, where
        # one stream saturates the device (extra threads only add cuSolver-handle
        # contention). 0 => the classic block-after-each path (used by CPU threads).
        self.pipeline_depth = pipeline_depth
        self.compute_queue = (
            input_queue if input_queue is not None else queue.Queue(maxsize=num_workers * 2)
        )
        self.throughput = ComponentThroughput()
        self.throughput_lock = threading.Lock()
        self.total_cells_processed = 0
        self.total_chunks_processed = 0
        self.stats_lock = threading.Lock()
        self.exception_queue = queue.Queue()
        self.has_error = threading.Event()
        self._prepare_jax_arrays()
        self._prepare_ldscore_backend()
        self.workers = []
        self.stop_workers = threading.Event()
        self._start_workers()

    def _prepare_jax_arrays(self):
        n_snps_used = self.processor.data_truncated["n_snps_used"]

        baseline_ann = (
            self.processor.data_truncated["baseline_ld"].values.astype(np.float32)
            * self.processor.data_truncated["N"].reshape(-1, 1).astype(np.float32)
            / self.processor.data_truncated["Nbar"]
        )
        baseline_ann = np.concatenate(
            [baseline_ann, np.ones((n_snps_used, 1), dtype=np.float32)], axis=1
        )

        self.baseline_ld_sum_jax = jnp.asarray(
            self.processor.data_truncated["baseline_ld_sum"], dtype=jnp.float32
        )
        self.chisq_jax = jnp.asarray(self.processor.data_truncated["chisq"], dtype=jnp.float32)
        self.N_jax = jnp.asarray(self.processor.data_truncated["N"], dtype=jnp.float32)
        self.baseline_ann_jax = jnp.asarray(baseline_ann, dtype=jnp.float32)
        self.w_ld_jax = jnp.asarray(self.processor.data_truncated["w_ld"], dtype=jnp.float32)
        self.Nbar = self.processor.data_truncated["Nbar"]

        del baseline_ann
        gc.collect()

    def _prepare_ldscore_backend(self):
        self.use_gpu_ldscore = False
        self.processor.defer_ldscore_to_compute = False
        self.gpu_ldscore_backend = None
        self.ldscore_backend = None
        self.ldscore_backend_lock = threading.Lock()

        if (
            self.processor.mode not in {"weight_matrix", "gene_block"}
            or jax.default_backend() != "gpu"
        ):
            return

        sparse_weight = self.processor.snp_gene_weight_sparse
        if sparse_weight is None:
            return

        self.ldscore_backend = select_gpu_ldscore_backend(sparse_weight)
        if self.ldscore_backend is not None:
            self.use_gpu_ldscore = True
            self.gpu_ldscore_backend = self.ldscore_backend.name
            self.processor.defer_ldscore_to_compute = True

    def _construct_spatial_ld_from_marker_chunk(self, marker_score_chunk: np.ndarray):
        backend = self.ldscore_backend
        if backend is not None:
            try:
                return backend.compute(marker_score_chunk)
            except Exception as e:
                if not is_gpu_memory_error(e):
                    raise
                with self.ldscore_backend_lock:
                    if self.ldscore_backend is backend:
                        logger.warning(
                            "GPU LD-score construction failed because GPU memory is insufficient. "
                            "Falling back to CPU/SciPy LD-score construction for remaining chunks "
                            "while keeping the shared --use-gpu JAX regression path. Reduce "
                            "--spots-per-chunk-quick-mode, set GSMAP_DISABLE_GPU_LDSCORE=1, "
                            "or use --no-gpu if regression also runs out of memory."
                        )
                        backend.close()
                        self.ldscore_backend = None
                        self.use_gpu_ldscore = False
                        self.gpu_ldscore_backend = None

        ldscore = self.processor._compute_ldscore_from_marker_chunk(marker_score_chunk)
        return jnp.asarray(ldscore, dtype=jnp.float32)

    def _start_workers(self):
        target = (
            self._compute_worker_pipelined if self.pipeline_depth > 0 else self._compute_worker
        )
        for i in range(self.num_workers):
            worker = threading.Thread(target=target, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        mode = (
            f"double-buffer depth={self.pipeline_depth}"
            if self.pipeline_depth > 0
            else "block-after-dispatch"
        )
        logger.info(f"Started {self.num_workers} compute worker(s) [{mode}]")

    def _dispatch_chunk(self, item):
        """Construct spatial LD and async-dispatch the regression (no block).

        Returns a pending tuple to be finalized later, or None if the upstream
        read failed for this chunk.
        """
        if not item.get("success", False):
            logger.error(f"Skipping chunk {item.get('chunk_idx')} due to read error")
            return None

        chunk_idx = item["chunk_idx"]
        spot_names = item["spot_names"]
        abs_start = item["abs_start"]
        abs_end = item["abs_end"]
        start_time = time.time()

        if item.get("marker_score_chunk") is not None:
            spatial_ld_jax = self._construct_spatial_ld_from_marker_chunk(
                item["marker_score_chunk"]
            )
        else:
            spatial_ld_jax = jnp.asarray(item["ldscore"], dtype=jnp.float32)

        betas, ses = self.regression_fn(
            self.processor.config.n_blocks,
            spatial_ld_jax,
            self.baseline_ld_sum_jax,
            self.chisq_jax,
            self.N_jax,
            self.baseline_ann_jax,
            self.w_ld_jax,
            self.Nbar,
        )
        return (chunk_idx, spot_names, abs_start, abs_end, betas, ses, spatial_ld_jax, start_time)

    def _finalize_chunk(self, pending):
        """Block on a dispatched chunk, copy back, accumulate, and update stats."""
        chunk_idx, spot_names, abs_start, abs_end, betas, ses, spatial_ld_jax, start_time = pending

        betas.block_until_ready()
        ses.block_until_ready()

        betas_np = np.array(betas)
        ses_np = np.array(ses)
        elapsed = time.time() - start_time
        # A sample-filtered chunk may contain non-contiguous source rows, so
        # the enclosing absolute range is not its cell count.
        n_cells_in_chunk = len(spot_names)

        with self.throughput_lock:
            self.throughput.record_batch(elapsed)

        self.processor._add_chunk_result(
            chunk_idx, betas_np, ses_np, spot_names, abs_start, abs_end
        )

        # Progress advances only after the ordered output sink accepted the
        # chunk. This prevents the monitor from declaring completion while a
        # worker is still blocked behind an earlier result gap.
        with self.stats_lock:
            self.total_cells_processed += n_cells_in_chunk
            self.total_chunks_processed += 1

        del spatial_ld_jax, betas, ses

    def _compute_worker(self, worker_id: int):
        logger.debug(f"Compute worker {worker_id} started")

        while not self.stop_workers.is_set():
            try:
                item = self.compute_queue.get(timeout=1)
                if item is None:
                    break

                pending = self._dispatch_chunk(item)
                if pending is None:
                    continue
                self._finalize_chunk(pending)

            except queue.Empty:
                continue
            except Exception as e:  # noqa: BLE001
                error_trace = traceback.format_exc()
                logger.error(f"Compute worker {worker_id} error: {e}\nTraceback:\n{error_trace}")
                self.exception_queue.put((worker_id, e, error_trace))
                self.has_error.set()
                break

        logger.debug(f"Compute worker {worker_id} stopped")

    def _compute_worker_pipelined(self, worker_id: int):
        """Single-thread async double-buffer: dispatch up to `pipeline_depth`
        chunks ahead, block lazily on the oldest. Overlaps host<->device transfer
        and result copy-back with on-device compute without extra threads (so no
        cuSolver-handle contention). Drains in-flight work when the input queue
        runs dry so the monitor's chunk counter can reach total_chunks.
        """
        logger.debug(f"Pipelined compute worker {worker_id} started (depth={self.pipeline_depth})")
        inflight: deque = deque()

        try:
            while not self.stop_workers.is_set():
                try:
                    item = self.compute_queue.get(timeout=0.1)
                except queue.Empty:
                    # Input drained: drain one in-flight chunk so stats advance
                    # (otherwise the last `depth` chunks would never be counted
                    # and the monitor loop would spin forever).
                    if inflight:
                        self._finalize_chunk(inflight.popleft())
                    continue

                if item is None:
                    break

                pending = self._dispatch_chunk(item)
                if pending is not None:
                    inflight.append(pending)
                if len(inflight) > self.pipeline_depth:
                    self._finalize_chunk(inflight.popleft())

            # Drain whatever is still in flight before exiting.
            while inflight:
                self._finalize_chunk(inflight.popleft())

        except Exception as e:  # noqa: BLE001
            error_trace = traceback.format_exc()
            logger.error(
                f"Pipelined compute worker {worker_id} error: {e}\nTraceback:\n{error_trace}"
            )
            self.exception_queue.put((worker_id, e, error_trace))
            self.has_error.set()

        logger.debug(f"Pipelined compute worker {worker_id} stopped")

    def get_queue_size(self):
        return self.compute_queue.qsize()

    def get_stats(self):
        with self.stats_lock:
            return self.total_cells_processed, self.total_chunks_processed

    def check_errors(self):
        if self.has_error.is_set():
            try:
                worker_id, exception, error_trace = self.exception_queue.get_nowait()
                raise RuntimeError(
                    f"Compute worker {worker_id} failed: {exception}\n"
                    f"Original traceback:\n{error_trace}"
                ) from exception
            except queue.Empty:
                raise RuntimeError("Compute worker failed with unknown error") from None

    def close(self):
        self.stop_workers.set()
        # A failed compute worker can leave the bounded source->compute queue
        # full. Blocking while adding sentinels would deadlock error cleanup;
        # discard pending work and let waiting workers observe stop_workers on
        # their bounded get timeout.
        while True:
            try:
                self.compute_queue.get_nowait()
            except queue.Empty:
                break
        deadline = time.monotonic() + 5
        for worker in self.workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        live_workers = [worker.name for worker in self.workers if worker.is_alive()]
        if live_workers:
            logger.warning(f"Compute workers did not stop within 5 seconds: {live_workers}")
        else:
            ldscore_backend = getattr(self, "ldscore_backend", None)
            if ldscore_backend is not None:
                ldscore_backend.close()
            self.ldscore_backend = None
        logger.info("Compute pool closed")

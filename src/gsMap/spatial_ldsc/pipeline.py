"""Threaded chunk reading and computation pipeline for spatial LDSC."""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time
import traceback
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


class ParallelLDScoreReader:
    """Multi-threaded reader for fetching LD-score chunks."""

    def __init__(
        self,
        processor,
        num_workers: int = 4,
        output_queue: queue.Queue = None,
    ):
        self.processor = processor
        self.num_workers = num_workers
        self.read_queue = queue.Queue()
        self.result_queue = output_queue if output_queue else queue.Queue(maxsize=num_workers * 8)
        self.throughput = ComponentThroughput()
        self.throughput_lock = threading.Lock()
        self.exception_queue = queue.Queue()
        self.has_error = threading.Event()
        self.workers = []
        self.stop_workers = threading.Event()
        self._start_workers()

    def _start_workers(self):
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.num_workers} reader threads")

    def _worker(self, worker_id: int):
        logger.debug(f"Reader worker {worker_id} started")

        while not self.stop_workers.is_set():
            try:
                item = self.read_queue.get(timeout=1)
                if item is None:
                    break

                chunk_idx = item
                start_time = time.time()

                fetched = self.processor._fetch_ldscore_chunk(chunk_idx)
                marker_score_chunk = None
                if len(fetched) == 4:
                    ldscore, spot_names, abs_start, abs_end = fetched
                else:
                    ldscore, spot_names, abs_start, abs_end, marker_score_chunk = fetched

                elapsed = time.time() - start_time
                with self.throughput_lock:
                    self.throughput.record_batch(elapsed)

                self.result_queue.put(
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
                self.read_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:  # noqa: BLE001
                error_trace = traceback.format_exc()
                logger.error(
                    f"Reader worker {worker_id} error on chunk {chunk_idx}: {e}\n"
                    f"Traceback:\n{error_trace}"
                )
                self.exception_queue.put((worker_id, e, error_trace))
                self.has_error.set()
                self.result_queue.put(
                    {
                        "chunk_idx": chunk_idx if "chunk_idx" in locals() else -1,
                        "worker_id": worker_id,
                        "success": False,
                        "error": str(e),
                    }
                )
                break

        logger.debug(f"Reader worker {worker_id} stopped")

    def submit_chunk(self, chunk_idx: int):
        self.read_queue.put(chunk_idx)

    def get_result(self):
        return self.result_queue.get()

    def get_queue_sizes(self):
        return self.read_queue.qsize(), self.result_queue.qsize()

    def check_errors(self):
        if self.has_error.is_set():
            try:
                worker_id, exception, error_trace = self.exception_queue.get_nowait()
                raise RuntimeError(
                    f"Reader worker {worker_id} failed: {exception}\n"
                    f"Original traceback:\n{error_trace}"
                ) from exception
            except queue.Empty:
                raise RuntimeError("Reader worker failed with unknown error") from None

    def close(self):
        self.stop_workers.set()
        for _ in range(self.num_workers):
            self.read_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5)
        logger.info("Reader pool closed")


class ParallelLDScoreComputer:
    """Multi-threaded computer for processing LD scores with JAX."""

    def __init__(
        self,
        processor,
        process_chunk_jit_fn,
        num_workers: int = 4,
        input_queue: queue.Queue = None,
    ):
        self.processor = processor
        self.process_chunk_jit_fn = process_chunk_jit_fn
        self.num_workers = num_workers
        self.compute_queue = input_queue if input_queue else queue.Queue(maxsize=num_workers * 2)
        self.throughput = ComponentThroughput()
        self.throughput_lock = threading.Lock()
        self.total_cells_processed = 0
        self.total_chunks_processed = 0
        self.stats_lock = threading.Lock()
        self.exception_queue = queue.Queue()
        self.has_error = threading.Event()
        self.results = []
        self.results_lock = threading.Lock()
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

        if self.processor.mode != "weight_matrix" or jax.default_backend() != "gpu":
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
            except Exception as e:  # noqa: BLE001
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
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._compute_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        logger.info(f"Started {self.num_workers} compute workers")

    def _compute_worker(self, worker_id: int):
        logger.debug(f"Compute worker {worker_id} started")

        while not self.stop_workers.is_set():
            try:
                item = self.compute_queue.get(timeout=1)
                if item is None:
                    break

                if not item.get("success", False):
                    logger.error(f"Skipping chunk {item.get('chunk_idx')} due to read error")
                    continue

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

                betas, ses = self.process_chunk_jit_fn(
                    self.processor.config.n_blocks,
                    spatial_ld_jax,
                    self.baseline_ld_sum_jax,
                    self.chisq_jax,
                    self.N_jax,
                    self.baseline_ann_jax,
                    self.w_ld_jax,
                    self.Nbar,
                )

                betas.block_until_ready()
                ses.block_until_ready()

                betas_np = np.array(betas)
                ses_np = np.array(ses)
                elapsed = time.time() - start_time
                n_cells_in_chunk = abs_end - abs_start

                with self.throughput_lock:
                    self.throughput.record_batch(elapsed)

                with self.stats_lock:
                    self.total_cells_processed += n_cells_in_chunk
                    self.total_chunks_processed += 1

                with self.results_lock:
                    self.processor._add_chunk_result(
                        chunk_idx, betas_np, ses_np, spot_names, abs_start, abs_end
                    )

                del spatial_ld_jax, betas, ses

            except queue.Empty:
                continue
            except Exception as e:  # noqa: BLE001
                error_trace = traceback.format_exc()
                logger.error(f"Compute worker {worker_id} error: {e}\nTraceback:\n{error_trace}")
                self.exception_queue.put((worker_id, e, error_trace))
                self.has_error.set()
                break

        logger.debug(f"Compute worker {worker_id} stopped")

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
        for _ in range(self.num_workers):
            self.compute_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5)
        logger.info("Compute pool closed")

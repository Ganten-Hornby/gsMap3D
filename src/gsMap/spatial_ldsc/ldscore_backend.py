"""LD-score construction backends for spatial LDSC.

The core weight-matrix operation is:

    spatial_ld = snp_gene_weight_sparse @ marker_score_chunk

where the sparse matrix is SNP x gene, the dense marker chunk is gene x cell,
and the returned LD-score chunk must be usable by the JAX regression kernel.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

import jax
import numpy as np

logger = logging.getLogger("gsMap.spatial_ldsc.ldscore_backend")

CPU_BACKEND_VALUES = {"0", "false", "off", "disable", "disabled", "cpu"}
GPU_BACKEND_VALUES = {"auto", "torch"}


def is_gpu_memory_error(exc: BaseException) -> bool:
    """Return True for common CUDA out-of-memory exceptions."""
    message = str(exc).lower()
    return "out of memory" in message or ("cuda" in message and "memory" in message)


class LDScoreBackend(ABC):
    """Backend that turns a CPU marker-score chunk into a JAX LD-score array."""

    name: str

    @abstractmethod
    def compute(self, marker_score_chunk: np.ndarray):
        """Return a JAX array with shape SNP x cell."""

    def close(self):
        """Release backend-owned GPU state."""


class TorchLDScoreBackend(LDScoreBackend):
    """PyTorch CSR backend.

    This is the preferred GPU path for gsMap GPU installs because PyTorch is
    already required and the benchmarked sparse CSR kernel is the fastest path
    for the production LD-score chunk shape.
    """

    name = "torch"

    def __init__(self, sparse_weight):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda is not available")

        self.torch = torch
        self.device = torch.device("cuda:0")
        self.sparse_weight = None
        self._last_output = None
        start_time = time.time()
        try:
            crow = torch.as_tensor(
                sparse_weight.indptr.astype(np.int64, copy=False), device=self.device
            )
            col = torch.as_tensor(
                sparse_weight.indices.astype(np.int64, copy=False), device=self.device
            )
            values = torch.as_tensor(
                sparse_weight.data.astype(np.float32, copy=False), device=self.device
            )
            self.sparse_weight = torch.sparse_csr_tensor(
                crow,
                col,
                values,
                size=sparse_weight.shape,
                dtype=torch.float32,
                device=self.device,
            )
            torch.cuda.synchronize()
        except Exception:
            self.close()
            raise
        logger.info(
            "Prepared PyTorch GPU SNP-gene sparse matrix for LD-score construction "
            f"in {time.time() - start_time:.2f}s"
        )

    def compute(self, marker_score_chunk: np.ndarray):
        marker_score_gpu = self.torch.as_tensor(
            marker_score_chunk,
            dtype=self.torch.float32,
            device=self.device,
        )
        ldscore_gpu = self.torch.sparse.mm(self.sparse_weight, marker_score_gpu)
        self.torch.cuda.synchronize()
        self._last_output = ldscore_gpu
        return jax.dlpack.from_dlpack(ldscore_gpu)

    def close(self):
        self._last_output = None
        self.sparse_weight = None
        if getattr(self, "torch", None) is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _try_backend(backend_cls: type[LDScoreBackend], sparse_weight) -> LDScoreBackend | None:
    try:
        return backend_cls(sparse_weight)
    except Exception as e:  # noqa: BLE001
        if is_gpu_memory_error(e):
            logger.warning(
                "Could not initialize %s LD-score backend because GPU memory is insufficient. "
                "Falling back to CPU/SciPy LD-score construction while keeping the shared "
                "--use-gpu JAX regression path. Reduce --spots-per-chunk-quick-mode, "
                "set GSMAP_DISABLE_GPU_LDSCORE=1, or use --no-gpu if regression also runs "
                "out of memory.",
                backend_cls.name,
            )
            return None
        logger.warning(
            "Could not initialize %s LD-score backend. Reason: %s: %s",
            backend_cls.name,
            type(e).__name__,
            e,
        )
        return None


def _backend_candidates(requested_backend: str) -> list[type[LDScoreBackend]]:
    if requested_backend == "auto":
        return [TorchLDScoreBackend]
    if requested_backend == "torch":
        return [TorchLDScoreBackend]
    return []


def select_gpu_ldscore_backend(sparse_weight) -> LDScoreBackend | None:
    """Select and initialize the GPU LD-score backend.

    Returns None when GPU LD-score construction is disabled or unavailable,
    leaving the caller on the CPU/SciPy construction path.
    """
    if os.environ.get("GSMAP_DISABLE_GPU_LDSCORE") == "1":
        logger.info(
            "GPU LD-score construction disabled by GSMAP_DISABLE_GPU_LDSCORE=1; "
            "using CPU sparse multiplication"
        )
        return None

    requested_backend = os.environ.get("GSMAP_GPU_LDSCORE_BACKEND", "auto").strip().lower()
    if requested_backend in CPU_BACKEND_VALUES:
        logger.info(
            "GPU LD-score construction disabled by GSMAP_GPU_LDSCORE_BACKEND=%r; "
            "using CPU sparse multiplication",
            requested_backend,
        )
        return None
    if requested_backend not in GPU_BACKEND_VALUES:
        logger.warning(
            "Unknown GSMAP_GPU_LDSCORE_BACKEND=%r; expected auto, torch, or cpu. Using auto.",
            requested_backend,
        )
        requested_backend = "auto"

    for backend_cls in _backend_candidates(requested_backend):
        backend = _try_backend(backend_cls, sparse_weight)
        if backend is not None:
            return backend

    if requested_backend == "auto":
        logger.warning(
            "No GPU LD-score backend could be initialized; using CPU sparse multiplication"
        )
    else:
        logger.warning(
            "Requested %s LD-score backend is unavailable; using CPU sparse multiplication",
            requested_backend,
        )
    return None

"""CPU sparse LD-score construction with bounded, GIL-free row accumulation."""

from __future__ import annotations

import os

import numpy as np
from numba import config as numba_config
from numba import get_num_threads, njit, prange, set_num_threads
from scipy import sparse


def available_cpu_count() -> int:
    """Bound concurrency by CPU affinity and any explicit Slurm task allocation."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    task_cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if task_cpus.isdigit() and int(task_cpus) > 0:
        count = min(count, int(task_cpus))
    return max(1, count)


@njit(nogil=True, cache=True)
def _csr_matmul_float32(indptr, indices, data, marker):
    """Accumulate each SNP row in its stored order, without fast-math.

    The single-thread fallback does not create a nested native thread pool.
    The streaming CPU path instead uses the row-parallel variant below.
    """
    result = np.empty((len(indptr) - 1, marker.shape[1]), np.float32)
    for row in range(len(indptr) - 1):
        for spot in range(marker.shape[1]):
            result[row, spot] = 0
        for entry in range(indptr[row], indptr[row + 1]):
            gene, weight = indices[entry], data[entry]
            for spot in range(marker.shape[1]):
                result[row, spot] += weight * marker[gene, spot]
    return result


@njit(nogil=True, parallel=True, cache=True)
def _csr_matmul_float32_parallel(indptr, indices, data, marker):
    """Independent SNP rows; preserve accumulation order within each row."""
    result = np.empty((len(indptr) - 1, marker.shape[1]), np.float32)
    for row in prange(len(indptr) - 1):
        for spot in range(marker.shape[1]):
            result[row, spot] = 0
        for entry in range(indptr[row], indptr[row + 1]):
            gene, weight = indices[entry], data[entry]
            for spot in range(marker.shape[1]):
                result[row, spot] += weight * marker[gene, spot]
    return result


def _validate_csr(matrix) -> None:
    """Validate once before handing CSR arrays to an unchecked native loop."""
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    if (
        indptr.ndim != 1
        or indices.ndim != 1
        or data.ndim != 1
        or len(indptr) != matrix.shape[0] + 1
        or indptr.dtype.kind not in "iu"
        or indices.dtype.kind not in "iu"
        or len(indices) != len(data)
    ):
        raise ValueError("Invalid CSR arrays for CPU LD-score construction")
    if indptr[0] != 0 or np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("CSR row pointers must start at zero and be nondecreasing")
    nnz = int(indptr[-1])
    if nnz > len(data):
        raise ValueError("CSR row pointers exceed the stored entries")
    used_indices = indices[:nnz]
    if nnz and (used_indices.min() < 0 or used_indices.max() >= matrix.shape[1]):
        raise ValueError("CSR gene indices are outside the matrix dimensions")


class CPUSparseLDScore:
    """One validated trait matrix shared by concurrent transform workers.

    Float32 CSR is the usual reference format. Other sparse formats, dense
    matrices and higher-precision inputs retain their native multiplication
    and accumulation semantics before the final float32 conversion.
    """

    def __init__(self, matrix):
        self.matrix = matrix
        self.threads = 1
        self.use_native = (
            sparse.issparse(matrix) and matrix.format == "csr" and matrix.dtype == np.float32
        )
        if self.use_native:
            _validate_csr(matrix)

    def configure_threads(self, requested: int) -> int:
        """Use the explicit transform-worker budget, capped by runtime limits."""
        self.threads = max(
            1, min(requested, numba_config.NUMBA_NUM_THREADS, available_cpu_count())
        )
        return self.threads

    def compute(self, marker: np.ndarray) -> np.ndarray:
        if marker.ndim != 2 or marker.shape[0] != self.matrix.shape[1]:
            raise ValueError("Marker gene dimension does not match the SNP-gene weight matrix")
        if self.use_native and isinstance(marker, np.ndarray) and marker.dtype == np.float32:
            matrix = self.matrix
            values = np.ascontiguousarray(marker)
            if self.threads == 1:
                return _csr_matmul_float32(matrix.indptr, matrix.indices, matrix.data, values)
            previous = get_num_threads()
            try:
                set_num_threads(self.threads)
                return _csr_matmul_float32_parallel(
                    matrix.indptr, matrix.indices, matrix.data, values
                )
            finally:
                set_num_threads(previous)
        result = self.matrix @ marker
        if sparse.issparse(result):
            result = result.toarray()
        return np.asarray(result, dtype=np.float32)

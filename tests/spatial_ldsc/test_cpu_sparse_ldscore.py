"""CPU-native multiplication preserves reference order and validates native inputs."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numba import get_num_threads
from scipy import sparse

from gsMap.spatial_ldsc.backends.cpu_sparse import CPUSparseLDScore


@pytest.mark.parametrize("batch", [0, 1, 7, 50, 100])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_native_matches_scipy_with_empty_rows_and_unsorted_duplicates(batch, index_dtype):
    matrix = sparse.csr_matrix(
        (
            np.array([1.5, -0.2, 0.0, 2, 3, -1], dtype=np.float32),
            np.array([3, 0, 3, 1, 1, 2], dtype=index_dtype),
            np.array([0, 3, 3, 5, 6, 6], dtype=index_dtype),
        ),
        shape=(5, 4),
    )
    matrix.indptr = matrix.indptr.astype(index_dtype)
    matrix.indices = matrix.indices.astype(index_dtype)
    marker = np.random.default_rng(24).normal(size=(4, batch)).astype(np.float32)
    before = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy(), marker.copy())
    backend = CPUSparseLDScore(matrix)
    assert backend.use_native
    for values in (marker, np.asfortranarray(marker), marker[:, ::-1]):
        actual = backend.compute(values)
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, matrix @ values)
    for actual, expected in zip(
        (matrix.data, matrix.indices, matrix.indptr, marker), before, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["csc", "dense", "float64", "csr_array"])
def test_other_representations_preserve_multiplication_semantics(kind):
    matrix = np.array([[0, 2, 0], [1, 0, -1]], dtype=np.float32)
    if kind == "float64":
        matrix = sparse.csr_matrix(matrix.astype(np.float64))
    elif kind == "csc":
        matrix = sparse.csc_matrix(matrix)
    elif kind == "csr_array":
        matrix = sparse.csr_array(matrix)
    marker = np.arange(12, dtype=np.float32).reshape(3, 4)
    backend = CPUSparseLDScore(matrix)
    np.testing.assert_array_equal(backend.compute(marker), (matrix @ marker).astype(np.float32))
    high_precision = marker.astype(np.float64) + 1e-9
    np.testing.assert_array_equal(
        backend.compute(high_precision), (matrix @ high_precision).astype(np.float32)
    )


def test_zero_rows_and_zero_gene_matrix():
    for shape in [(0, 3), (5, 0), (5, 3)]:
        matrix = sparse.csr_matrix(shape, dtype=np.float32)
        values = np.ones((shape[1], 7), dtype=np.float32)
        np.testing.assert_array_equal(CPUSparseLDScore(matrix).compute(values), matrix @ values)


@pytest.mark.parametrize("corruption", ["negative_gene", "large_gene", "bad_row", "past_end"])
def test_invalid_csr_rejected_before_native_execution(corruption):
    matrix = sparse.eye(3, format="csr", dtype=np.float32)
    if corruption == "negative_gene":
        matrix.indices[0] = -1
    elif corruption == "large_gene":
        matrix.indices[0] = 3
    elif corruption == "bad_row":
        matrix.indptr[0] = 1
    else:
        matrix.indptr[-1] = 10
    with pytest.raises(ValueError, match="CSR"):
        CPUSparseLDScore(matrix)


def test_dimension_check_and_concurrent_calls():
    rng = np.random.default_rng(24)
    matrix = sparse.random(300, 71, density=0.1, random_state=rng, dtype=np.float32, format="csr")
    backend = CPUSparseLDScore(matrix)
    values = [rng.normal(size=(71, n)).astype(np.float32) for n in [1, 7, 50, 100] * 3]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(backend.compute, values))
    for result, marker in zip(actual, values, strict=True):
        np.testing.assert_array_equal(result, matrix @ marker)
    with pytest.raises(ValueError, match="dimension"):
        backend.compute(np.zeros((70, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        backend.compute(np.zeros(71, dtype=np.float32))


def test_row_parallel_matches_serial_and_restores_thread_mask():
    rng = np.random.default_rng(25)
    matrix = sparse.random(511, 37, density=0.2, random_state=rng, dtype=np.float32, format="csr")
    backend = CPUSparseLDScore(matrix)
    old_threads = get_num_threads()
    backend.configure_threads(2)
    for width in [1, 7, 100]:
        marker = rng.normal(size=(37, width)).astype(np.float32)
        np.testing.assert_array_equal(backend.compute(marker), matrix @ marker)
        assert get_num_threads() == old_threads


def test_thread_budget_never_exceeds_explicit_job_or_affinity(monkeypatch):
    from gsMap.spatial_ldsc.backends import cpu_sparse

    monkeypatch.setattr(cpu_sparse.numba_config, "NUMBA_NUM_THREADS", 128)
    monkeypatch.setattr(cpu_sparse.os, "sched_getaffinity", lambda _: set(range(64)))
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "15")
    backend = CPUSparseLDScore(sparse.eye(3, dtype=np.float32, format="csr"))
    assert backend.configure_threads(120) == 15
    assert backend.configure_threads(4) == 4
    monkeypatch.setattr(cpu_sparse.os, "sched_getaffinity", lambda _: set(range(2)))
    assert backend.configure_threads(120) == 2
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "invalid")
    assert backend.configure_threads(120) == 2


def test_thread_budget_falls_back_when_affinity_query_fails(monkeypatch):
    from gsMap.spatial_ldsc.backends import cpu_sparse

    def unavailable(_pid):
        raise OSError("Affinity is unavailable")

    monkeypatch.setattr(cpu_sparse.os, "sched_getaffinity", unavailable)
    monkeypatch.setattr(cpu_sparse.os, "cpu_count", lambda: 64)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "15")
    assert cpu_sparse.available_cpu_count() == 15
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "invalid")
    assert cpu_sparse.available_cpu_count() == 64
    monkeypatch.setattr(cpu_sparse.os, "cpu_count", lambda: None)
    assert cpu_sparse.available_cpu_count() == 1


def test_cpu_stream_cooperates_on_one_batch_but_gpu_keeps_transform_pool(monkeypatch):
    from types import SimpleNamespace

    from gsMap.spatial_ldsc.execution import pipeline
    from gsMap.spatial_ldsc.execution.pipeline import LDScoreStream

    backend = CPUSparseLDScore(sparse.eye(3, dtype=np.float32, format="csr"))
    source = SimpleNamespace(
        cpu_ldscore_backend=backend,
        supports_sequential_marker_windows=lambda: False,
        chunk_starts=[0, 1, 2],
    )
    processor = SimpleNamespace(
        chunk_source=source, total_chunks=3, defer_ldscore_to_compute=False
    )
    monkeypatch.setattr(pipeline.jax, "default_backend", lambda: "cpu")
    stream = LDScoreStream(processor, num_workers=10)
    assert stream.num_workers == 1
    assert stream.result_queue.maxsize == 2
    assert 1 <= stream.cpu_sparse_threads <= 10
    monkeypatch.setattr(pipeline.jax, "default_backend", lambda: "gpu")
    stream = LDScoreStream(processor, num_workers=10)
    assert stream.num_workers == 10
    assert stream.result_queue.maxsize == 20
    assert stream.cpu_sparse_threads == 0

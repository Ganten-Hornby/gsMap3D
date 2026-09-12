import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gsMap.spatial_ldsc.backends import ldscore as ldscore_backend
from gsMap.spatial_ldsc.data.chunks import SpatialLDChunkSource
from gsMap.spatial_ldsc.execution import pipeline
from gsMap.spatial_ldsc.execution.pipeline import LDScoreComputePool, LDScoreStream


class _FakeTorchBackend:
    name = "torch"

    def __init__(self, sparse_weight):
        self.sparse_weight = sparse_weight


def test_torch_compute_uses_dlpack_object_without_device_barrier(monkeypatch):
    source = np.ones((2, 3), dtype=np.float32)
    producer_output = object()
    backend = ldscore_backend.TorchLDScoreBackend.__new__(ldscore_backend.TorchLDScoreBackend)
    backend.device = "cuda:0"
    backend.sparse_weight = SimpleNamespace(shape=(5, 2))
    calls = []

    def empty(shape, **kwargs):
        assert shape == (5, 3)
        assert kwargs == {"dtype": np.float32, "device": "cuda:0"}
        return producer_output

    def addmm(input_tensor, matrix, dense, *, beta, out):
        assert input_tensor is out is producer_output
        assert matrix is backend.sparse_weight
        assert dense is source
        assert beta == 0
        calls.append("addmm")
        return out

    # No cuda.synchronize method: compute must rely on the DLPack object protocol.
    backend.torch = SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda array, **kwargs: array,
        empty=empty,
        addmm=addmm,
    )
    imported = []
    monkeypatch.setattr(
        ldscore_backend.jax.dlpack,
        "from_dlpack",
        lambda tensor: imported.append(tensor) or "jax-result",
    )
    assert backend.compute(source) == "jax-result"
    assert imported == [producer_output]
    assert calls == ["addmm"]


@pytest.mark.parametrize("n_cells", [0, 1, 17, 100])
@pytest.mark.parametrize("all_zero", [False, True])
def test_torch_addmm_overwrites_poisoned_output_and_preserves_previous_batch(
    monkeypatch, n_cells, all_zero
):
    """Exercise the real CSR operation on CPU, including empty SNP rows.

    NaNs emulate arbitrary uninitialized output: beta=0 must not read them.
    GPU stream/lifetime behavior is covered by the CUDA test below.
    """
    torch = pytest.importorskip("torch")
    weight = np.array([[1, 0, 2], [0, 0, 0], [0, -3, 0]], dtype=np.float32)
    if all_zero:
        weight[:] = 0
    backend = ldscore_backend.TorchLDScoreBackend.__new__(ldscore_backend.TorchLDScoreBackend)
    backend.device = torch.device("cpu")
    backend.sparse_weight = torch.tensor(weight).to_sparse_csr()
    backend._last_output = None
    backend.torch = SimpleNamespace(
        float32=torch.float32,
        as_tensor=torch.as_tensor,
        addmm=torch.addmm,
        empty=lambda shape, **kwargs: torch.full(shape, float("nan"), **kwargs),
    )
    monkeypatch.setattr(ldscore_backend.jax.dlpack, "from_dlpack", lambda tensor: tensor)
    marker = np.arange(3 * n_cells, dtype=np.float32).reshape(3, n_cells)
    first = backend.compute(marker)
    second = backend.compute(marker + 1)
    assert first is not second
    np.testing.assert_allclose(first.numpy(), weight @ marker, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(second.numpy(), weight @ (marker + 1), rtol=2e-6, atol=2e-6)


def test_torch_dlpack_gpu_stream_dependency_and_lifetime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or ldscore_backend.jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation with both Torch and JAX GPU backends")
    rng = np.random.default_rng(42)
    weight = sp.random(500, 64, density=0.1, format="csr", random_state=rng, dtype=np.float32)
    backend = ldscore_backend.TorchLDScoreBackend(weight)
    stream = torch.cuda.Stream()
    results = []
    try:
        with torch.cuda.stream(stream):
            for i in range(24):
                marker = rng.random((64, 17 + i % 3), dtype=np.float32)
                # Delay the producer without blocking the host. Consumers must
                # wait on its stream, including after _last_output is replaced.
                torch.cuda._sleep(100_000)
                results.append((backend.compute(marker), weight @ marker))
        for actual, expected in results:
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-6, atol=2e-6)
        backend.close()
        for actual, expected in results:
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-6, atol=2e-6)
    finally:
        backend.close()


def test_select_gpu_ldscore_backend_respects_disable_env(monkeypatch):
    sparse_weight = sp.eye(2, format="csr", dtype=np.float32)

    def fail_if_called(_sparse_weight):
        raise AssertionError("backend should not be initialized when GPU LD-score is disabled")

    monkeypatch.setenv("GSMAP_DISABLE_GPU_LDSCORE", "1")
    monkeypatch.setattr(ldscore_backend, "_try_backend", fail_if_called)

    assert ldscore_backend.select_gpu_ldscore_backend(sparse_weight) is None


def test_select_gpu_ldscore_backend_auto_uses_torch_only(monkeypatch):
    sparse_weight = sp.eye(2, format="csr", dtype=np.float32)

    monkeypatch.delenv("GSMAP_DISABLE_GPU_LDSCORE", raising=False)
    monkeypatch.delenv("GSMAP_GPU_LDSCORE_BACKEND", raising=False)
    monkeypatch.setattr(ldscore_backend, "TorchLDScoreBackend", _FakeTorchBackend)

    backend = ldscore_backend.select_gpu_ldscore_backend(sparse_weight)

    assert isinstance(backend, _FakeTorchBackend)
    assert ldscore_backend._backend_candidates("auto") == [_FakeTorchBackend]
    assert ldscore_backend._backend_candidates("torch") == [_FakeTorchBackend]
    assert ldscore_backend._backend_candidates("jax") == []
    assert ldscore_backend._backend_candidates("cupy") == []


def test_select_gpu_ldscore_backend_falls_back_when_torch_setup_oom(monkeypatch):
    class OOMTorchBackend:
        name = "torch"

        def __init__(self, _sparse_weight):
            raise RuntimeError("CUDA out of memory while creating CSR")

    sparse_weight = sp.eye(2, format="csr", dtype=np.float32)

    monkeypatch.delenv("GSMAP_DISABLE_GPU_LDSCORE", raising=False)
    monkeypatch.delenv("GSMAP_GPU_LDSCORE_BACKEND", raising=False)
    monkeypatch.setattr(ldscore_backend, "TorchLDScoreBackend", OOMTorchBackend)
    warnings = []
    monkeypatch.setattr(
        ldscore_backend.logger,
        "warning",
        lambda message, *args, **_kwargs: warnings.append(message % args if args else message),
    )

    backend = ldscore_backend.select_gpu_ldscore_backend(sparse_weight)

    assert backend is None
    assert any("Falling back to CPU/SciPy LD-score construction" in msg for msg in warnings)


def test_chunk_source_can_defer_or_compute_cpu_ldscore():
    sparse_weight = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    marker_scores = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    marker_adata = type("MarkerAdata", (), {"X": marker_scores})()
    source = SpatialLDChunkSource(
        mode="weight_matrix",
        chunk_starts=[1],
        chunk_size=2,
        cell_end=3,
        spot_indices=np.array([0, 1, 2]),
        spot_names_filtered=np.array(["cell0", "cell1", "cell2"]),
        marker_score_adata=marker_adata,
        marker_score_gene_indices=np.array([0, 1, 2]),
        snp_gene_weight_sparse=sparse_weight,
    )

    deferred = source.fetch_chunk(0, defer_ldscore_to_compute=True)
    assert deferred[0] is None
    np.testing.assert_allclose(
        deferred[4],
        np.array([[4.0, 7.0], [5.0, 8.0], [6.0, 9.0]], dtype=np.float32),
    )

    computed = source.fetch_chunk(0, defer_ldscore_to_compute=False)
    expected = sparse_weight @ deferred[4]
    np.testing.assert_allclose(computed[0], expected.astype(np.float32))
    assert list(computed[1]) == ["cell1", "cell2"]
    assert computed[2:] == (1, 3)


def test_chunk_source_reads_noncontiguous_sample_rows_correctly():
    marker_scores = np.arange(15, dtype=np.float32).reshape(5, 3)
    marker_adata = type("MarkerAdata", (), {"X": marker_scores})()
    source = SpatialLDChunkSource(
        mode="weight_matrix",
        chunk_starts=[0],
        chunk_size=3,
        cell_end=3,
        spot_indices=np.array([0, 2, 4]),
        spot_names_filtered=np.array(["cell0", "cell2", "cell4"]),
        marker_score_adata=marker_adata,
        marker_score_gene_indices=np.array([0, 2]),
        snp_gene_weight_sparse=sp.eye(2, format="csr", dtype=np.float32),
    )

    deferred = source.fetch_chunk(0, defer_ldscore_to_compute=True)

    np.testing.assert_array_equal(deferred[4], marker_scores[[0, 2, 4]][:, [0, 2]].T)
    assert list(deferred[1]) == ["cell0", "cell2", "cell4"]
    assert deferred[2:4] == (0, 5)


def test_sequential_marker_window_matches_individual_memmap_reads(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    marker_scores = np.arange(35, dtype=np.float16).reshape(7, 5)
    marker_memmap = np.memmap(
        marker_path,
        dtype=np.float16,
        mode="w+",
        shape=marker_scores.shape,
    )
    marker_memmap[:] = marker_scores
    marker_memmap.flush()
    marker_adata = type("MarkerAdata", (), {"X": marker_memmap})()
    source = SpatialLDChunkSource(
        mode="weight_matrix",
        chunk_starts=[0, 2, 4, 6],
        chunk_size=2,
        cell_end=7,
        spot_indices=np.arange(7),
        spot_names_filtered=np.array([f"cell{i}" for i in range(7)]),
        marker_score_adata=marker_adata,
        marker_score_gene_indices=np.array([0, 2, 4]),
        snp_gene_weight_sparse=sp.eye(3, format="csr", dtype=np.float32),
    )

    try:
        assert source.supports_sequential_marker_windows()
        window = source.load_marker_window([0, 1, 2, 3])
        assert window.values.shape == marker_scores.shape

        for chunk_idx in range(4):
            expected = source.fetch_chunk(chunk_idx, defer_ldscore_to_compute=True)
            actual = source.fetch_chunk_from_marker_window(
                window,
                chunk_idx,
                defer_ldscore_to_compute=True,
            )
            assert actual[0] is expected[0] is None
            assert actual[1].equals(expected[1])
            assert actual[2:4] == expected[2:4]
            np.testing.assert_array_equal(actual[4], expected[4])
    finally:
        marker_memmap._mmap.close()


@pytest.mark.parametrize("rows", [[0, 2, 4], [2, 1, 4, 0, 2], [0, 2, 1, 3, 4]])
def test_noncontiguous_rows_retain_correct_fallback(tmp_path, rows):
    marker_path = tmp_path / "marker_scores.dat"
    marker_memmap = np.memmap(marker_path, dtype=np.float16, mode="w+", shape=(5, 3))
    marker_memmap[:] = np.arange(15, dtype=np.float16).reshape(5, 3)
    marker_memmap.flush()
    marker_adata = type("MarkerAdata", (), {"X": marker_memmap})()
    source = SpatialLDChunkSource(
        mode="weight_matrix",
        chunk_starts=list(range(0, len(rows), 2)),
        chunk_size=2,
        cell_end=len(rows),
        spot_indices=np.array(rows),
        spot_names_filtered=np.array([f"cell{i}" for i in rows]),
        marker_score_adata=marker_adata,
        marker_score_gene_indices=np.array([2, 0, 1]),
        snp_gene_weight_sparse=sp.eye(3, format="csr", dtype=np.float32),
    )

    try:
        assert not source.supports_sequential_marker_windows()
        assert source.recommended_marker_window_chunks() == 1
        for i in range(len(source.chunk_starts)):
            selected = rows[2 * i : 2 * i + 2]
            actual = source.fetch_chunk(i, defer_ldscore_to_compute=True)
            np.testing.assert_array_equal(actual[4], marker_memmap[selected][:, [2, 0, 1]].T)
            assert list(actual[1]) == [f"cell{j}" for j in selected]
    finally:
        source.close_sequential_marker_access()
        marker_memmap._mmap.close()


def test_ldscore_stream_uses_sequential_marker_windows(tmp_path, monkeypatch):
    marker_path = tmp_path / "marker_scores.dat"
    marker_scores = np.arange(24, dtype=np.float16).reshape(6, 4)
    marker_memmap = np.memmap(
        marker_path,
        dtype=np.float16,
        mode="w+",
        shape=marker_scores.shape,
    )
    marker_memmap[:] = marker_scores
    marker_memmap.flush()
    marker_adata = type("MarkerAdata", (), {"X": marker_memmap})()
    source = SpatialLDChunkSource(
        mode="weight_matrix",
        chunk_starts=[0, 2, 4],
        chunk_size=2,
        cell_end=6,
        spot_indices=np.arange(6),
        spot_names_filtered=np.array([f"cell{i}" for i in range(6)]),
        marker_score_adata=marker_adata,
        marker_score_gene_indices=np.array([0, 1, 3]),
        snp_gene_weight_sparse=sp.eye(3, format="csr", dtype=np.float32),
    )

    class Processor:
        chunk_source = source
        chunk_size = 2
        defer_ldscore_to_compute = True

    opened = []
    advise = source.advise_sequential_marker_access

    def record_open():
        result = advise()
        opened.append(source._matrix_reader is not None and source._matrix_reader.fd is not None)
        return result

    monkeypatch.setattr(source, "advise_sequential_marker_access", record_open)
    stream = LDScoreStream(Processor(), num_workers=2)
    try:
        assert stream.uses_sequential_windows
        assert not hasattr(stream, "read_queue")
        assert stream.batch_queue.maxsize < stream.total_chunks + 256
        stream.start()
        with pytest.raises(RuntimeError, match="already been started"):
            stream.start()

        results = [stream.get_result() for _ in range(3)]
        # The tiny source may already have finished and closed its descriptor
        # by the time start() returns; verify opening, not thread timing.
        assert opened == [True]
        assert [result["chunk_idx"] for result in results] == [0, 1, 2]
        for chunk_idx, result in enumerate(results):
            assert result["success"]
            assert result["ldscore"] is None
            expected_rows = marker_scores[chunk_idx * 2 : chunk_idx * 2 + 2]
            np.testing.assert_array_equal(
                result["marker_score_chunk"],
                expected_rows[:, [0, 1, 3]].T.astype(np.float32),
            )
        assert stream.sequential_io_throughput.total_cells == 6
        assert stream.sequential_io_throughput.total_bytes == marker_scores.nbytes
    finally:
        stream.close()
        assert source._matrix_reader is None
        marker_memmap._mmap.close()


@pytest.mark.parametrize(("step", "expected_chunks"), [(1, 16), (2, 1)])
def test_marker_window_keeps_32m_contiguous_default(tmp_path, step, expected_chunks):
    matrix = np.memmap(tmp_path / "matrix.dat", mode="w+", shape=(1000, 10000), dtype=np.float16)
    rows = np.arange(0, 1000, step)
    source = SpatialLDChunkSource(
        mode="weight_matrix",
        chunk_starts=list(range(0, len(rows), 100)),
        chunk_size=100,
        cell_end=len(rows),
        spot_indices=rows,
        spot_names_filtered=rows.astype(str),
        marker_score_adata=type("MarkerAdata", (), {"X": matrix})(),
    )
    try:
        assert source.recommended_marker_window_chunks() == expected_chunks
        assert source.recommended_marker_window_chunks(32 * 1024**2) == expected_chunks
    finally:
        matrix._mmap.close()


def test_noncontiguous_chunk_progress_counts_rows_not_source_span():
    class ReadyArray:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def block_until_ready(self):
            return self

        def __array__(self, dtype=None, copy=None):
            del copy
            return np.asarray(self.values, dtype=dtype)

    added = []
    computer = LDScoreComputePool.__new__(LDScoreComputePool)
    computer.throughput_lock = threading.Lock()
    computer.throughput = pipeline.ComponentThroughput()
    computer.stats_lock = threading.Lock()
    computer.total_cells_processed = 0
    computer.total_chunks_processed = 0
    computer.processor = type(
        "Processor",
        (),
        {"_add_chunk_result": lambda self, *args: added.append(args)},
    )()

    computer._finalize_chunk(
        (
            0,
            pd.Index(["cell0", "cell2", "cell4"]),
            0,
            5,
            ReadyArray([1, 2, 3]),
            ReadyArray([0.1, 0.2, 0.3]),
            object(),
            time.time(),
        )
    )

    assert computer.get_stats() == (3, 1)
    assert len(added) == 1


def test_stream_close_unblocks_a_full_result_queue():
    fetched = threading.Event()

    class Processor:
        total_chunks = 1

        def _fetch_ldscore_chunk(self, chunk_idx):
            fetched.set()
            return np.zeros((1, 1)), [f"cell{chunk_idx}"], chunk_idx, chunk_idx + 1

    result_queue = queue.Queue(maxsize=1)
    result_queue.put({"occupied": True})
    stream = LDScoreStream(
        Processor(),
        num_workers=1,
        output_queue=result_queue,
    )
    stream.start()
    assert fetched.wait(timeout=2)

    stream.close()

    assert not stream.workers[0].is_alive()


def test_parallel_transforms_are_published_in_chunk_order():
    class Processor:
        total_chunks = 3

    stream = LDScoreStream(Processor(), num_workers=2)

    def result(chunk_idx):
        return {
            "chunk_idx": chunk_idx,
            "success": True,
        }

    # Simulate completion in reverse order. Later chunks remain in the bounded
    # reorder buffer until the missing first chunk arrives.
    assert stream._publish_transformed_in_order(result(2))
    assert stream._publish_transformed_in_order(result(1))
    assert stream.result_queue.empty()
    assert stream._publish_transformed_in_order(result(0))

    assert [stream.get_result()["chunk_idx"] for _ in range(3)] == [0, 1, 2]


def test_prepare_ldscore_backend_follows_shared_use_gpu(monkeypatch):
    processor = type(
        "Processor",
        (),
        {
            "mode": "weight_matrix",
            "snp_gene_weight_sparse": object(),
            "defer_ldscore_to_compute": False,
        },
    )()
    computer = LDScoreComputePool.__new__(LDScoreComputePool)
    computer.processor = processor

    monkeypatch.setattr("gsMap.spatial_ldsc.execution.pipeline.jax.default_backend", lambda: "cpu")
    computer._prepare_ldscore_backend()
    assert computer.ldscore_backend is None
    assert processor.defer_ldscore_to_compute is False

    def fake_select(sparse_weight):
        assert sparse_weight is processor.snp_gene_weight_sparse
        return _FakeTorchBackend(sparse_weight)

    monkeypatch.setattr("gsMap.spatial_ldsc.execution.pipeline.jax.default_backend", lambda: "gpu")
    monkeypatch.setattr(
        "gsMap.spatial_ldsc.execution.pipeline.select_gpu_ldscore_backend", fake_select
    )

    computer._prepare_ldscore_backend()

    assert computer.use_gpu_ldscore is True
    assert computer.gpu_ldscore_backend == "torch"
    assert isinstance(computer.ldscore_backend, _FakeTorchBackend)
    assert processor.defer_ldscore_to_compute is True


def test_construct_spatial_ld_falls_back_to_cpu_after_gpu_oom():
    class OOMBackend:
        name = "torch"

        def __init__(self):
            self.closed = False

        def compute(self, _marker_score_chunk):
            raise RuntimeError("CUDA out of memory during sparse matmul")

        def close(self):
            self.closed = True

    sparse_weight = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    marker_chunk = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )

    processor = type(
        "Processor",
        (),
        {
            "_compute_ldscore_from_marker_chunk": lambda self, chunk: (
                sparse_weight @ chunk
            ).astype(np.float32),
        },
    )()
    backend = OOMBackend()
    computer = LDScoreComputePool.__new__(LDScoreComputePool)
    computer.processor = processor
    computer.ldscore_backend = backend
    computer.ldscore_backend_lock = threading.Lock()
    computer.use_gpu_ldscore = True
    computer.gpu_ldscore_backend = "torch"

    result = computer._construct_spatial_ld_from_marker_chunk(marker_chunk)

    np.testing.assert_allclose(np.asarray(result), sparse_weight @ marker_chunk)
    assert backend.closed is True
    assert computer.ldscore_backend is None
    assert computer.use_gpu_ldscore is False
    assert computer.gpu_ldscore_backend is None


def test_construct_spatial_ld_reraises_non_memory_gpu_errors():
    class BrokenBackend:
        def compute(self, _marker_score_chunk):
            raise RuntimeError("invalid CSR index")

    computer = LDScoreComputePool.__new__(LDScoreComputePool)
    computer.ldscore_backend = BrokenBackend()

    with pytest.raises(RuntimeError, match="invalid CSR index"):
        computer._construct_spatial_ld_from_marker_chunk(np.ones((2, 2), dtype=np.float32))

import threading

import numpy as np
import pytest
import scipy.sparse as sp

from gsMap.spatial_ldsc import ldscore_backend
from gsMap.spatial_ldsc.chunk_source import SpatialLDChunkSource
from gsMap.spatial_ldsc.pipeline import ParallelLDScoreComputer


class _FakeTorchBackend:
    name = "torch"

    def __init__(self, sparse_weight):
        self.sparse_weight = sparse_weight


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
        sample_start_offset=0,
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


def test_chunk_source_accumulates_gene_block_ldscore():
    sparse_weight = sp.csr_matrix(np.asarray([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    marker_scores = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )

    class Store:
        def iter_spot_marker_blocks(self, start, end, marker_gene_indices):
            selected = marker_scores[start:end, marker_gene_indices].T
            yield np.asarray([0, 1]), selected[:2]
            yield np.asarray([2]), selected[2:]

    source = SpatialLDChunkSource(
        mode="gene_block",
        chunk_starts=[1],
        chunk_size=2,
        cell_end=3,
        sample_start_offset=0,
        spot_indices=np.asarray([0, 1, 2]),
        spot_names_filtered=np.asarray(["cell0", "cell1", "cell2"]),
        marker_score_gene_indices=np.asarray([0, 1, 2]),
        snp_gene_weight_sparse=sparse_weight,
        marker_score_store=Store(),
    )

    actual = source.fetch_chunk(0, defer_ldscore_to_compute=False)
    expected = sparse_weight @ marker_scores[1:3].T
    np.testing.assert_allclose(actual[0], expected)
    assert list(actual[1]) == ["cell1", "cell2"]


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
    computer = ParallelLDScoreComputer.__new__(ParallelLDScoreComputer)
    computer.processor = processor

    monkeypatch.setattr("gsMap.spatial_ldsc.pipeline.jax.default_backend", lambda: "cpu")
    computer._prepare_ldscore_backend()
    assert computer.ldscore_backend is None
    assert processor.defer_ldscore_to_compute is False

    def fake_select(sparse_weight):
        assert sparse_weight is processor.snp_gene_weight_sparse
        return _FakeTorchBackend(sparse_weight)

    monkeypatch.setattr("gsMap.spatial_ldsc.pipeline.jax.default_backend", lambda: "gpu")
    monkeypatch.setattr("gsMap.spatial_ldsc.pipeline.select_gpu_ldscore_backend", fake_select)

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
    computer = ParallelLDScoreComputer.__new__(ParallelLDScoreComputer)
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

    computer = ParallelLDScoreComputer.__new__(ParallelLDScoreComputer)
    computer.ldscore_backend = BrokenBackend()

    with pytest.raises(RuntimeError, match="invalid CSR index"):
        computer._construct_spatial_ld_from_marker_chunk(np.ones((2, 2), dtype=np.float32))

"""Refactoring must preserve values, backend policy and resource lifetimes."""

import gc
import threading
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gsMap.spatial_ldsc import ldscore_backend, multitrait
from gsMap.spatial_ldsc.moment_cpu import CPUMomentRegression
from gsMap.spatial_ldsc.prefetch import prefetched_chunks
from gsMap.spatial_ldsc.regression_backend import RegressionBatch, SharedRegressionBackend

from .test_moments import compare, fixture_data, reference


@pytest.mark.parametrize("mode", ["moments", "jax", "direct"])
def test_uniform_backend_values_and_tail(mode):
    a, marker, prepared = fixture_data(blocks=7, cells=17, traits=3)
    config = SimpleNamespace(
        n_blocks=7,
        ldsc_trait_batch_size=2,
        ldsc_compute_workers=2,
        ldsc_regression_backend="moments" if mode == "moments" else "jax",
    )
    with SharedRegressionBackend(
        config, None if mode == "direct" else a, prepared, a.shape[0]
    ) as backend:
        for width in (17, 1):
            m = marker[:, :width]
            batch = backend.compute(marker=m, ldscore=a @ m)
            batch.wait()
            got = batch.to_host()
            if mode == "moments":
                with backend.thread_context():
                    exact = CPUMomentRegression(a, prepared, 7, threads=2, trait_batch=2).compute(
                        m
                    )
                np.testing.assert_array_equal(got, exact)
            else:
                compare(got, reference(a, m, prepared, 7))
    backend.close()  # idempotent
    with pytest.raises(RuntimeError, match="closed"):
        backend.compute(marker=marker)


def test_result_batch_holds_producer_owners():
    class Owner:
        pass

    owner = Owner()
    ref = weakref.ref(owner)
    groups = [np.ones((2, 2, 3), np.float32)]
    batch = RegressionBatch(groups, owner, validate_moments=True)
    del owner
    gc.collect()
    assert ref() is not None
    np.testing.assert_array_equal(batch.to_host(), groups[0])
    batch.wait()
    assert ref() is not None
    del batch
    gc.collect()
    assert ref() is None


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0])
def test_invalid_moment_result_is_never_published(invalid):
    result = np.ones((1, 2, 3), np.float32)
    result[0, 1, 1] = invalid
    with pytest.raises(FloatingPointError, match="invalid"):
        RegressionBatch([result], validate_moments=True).to_host()


def test_backend_construction_and_close_failures_release_ld(monkeypatch):
    events = []
    a, _, prepared = fixture_data()
    config = SimpleNamespace(n_blocks=7, ldsc_trait_batch_size=4)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        ldscore_backend,
        "select_gpu_ldscore_backend",
        lambda _: SimpleNamespace(close=lambda: events.append("ld_closed")),
    )

    def fail_prepare(*args):
        raise ValueError("design failed")

    with pytest.raises(ValueError, match="design failed"):
        SharedRegressionBackend(config, a, prepared, a.shape[0], moment_selector=fail_prepare)
    assert events == ["ld_closed"]

    def fail_close():
        events.append("moment_closed")
        raise RuntimeError("cleanup failed")

    backend = SharedRegressionBackend(
        config,
        a,
        prepared,
        a.shape[0],
        moment_selector=lambda *args: SimpleNamespace(close=fail_close),
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        backend.close()
    assert events == ["ld_closed", "moment_closed", "ld_closed"]
    backend.close()


def test_file_reader_is_serial_and_joined():
    lock = threading.Lock()
    source = SimpleNamespace(
        marker_score_adata=SimpleNamespace(X=object()), chunk_starts=range(11)
    )
    active = []

    def fetch(index, deferred):
        assert lock.acquire(blocking=False), "Concurrent access to a shared file reader"
        try:
            assert deferred
            active.append(index)
            return index
        finally:
            lock.release()

    source.fetch_chunk = fetch
    assert list(prefetched_chunks(source, SimpleNamespace(ldsc_read_workers=8))) == list(range(11))
    assert active == list(range(11))


@pytest.mark.parametrize("failure", ["read", "compute", "write", "finalize"])
def test_pipeline_cleanup_order_on_failure(tmp_path, monkeypatch, failure):
    events = []
    source = SimpleNamespace(
        cell_end=5,
        chunk_starts=list(range(5)),
        ldscore_snp_positions=np.arange(2),
        close_sequential_marker_access=lambda: events.append("source_closed"),
    )

    def chunks(*args):
        try:
            for i in range(5):
                if failure == "read" and i == 1:
                    raise OSError("read failed")
                yield (None, [str(i)], i, i + 1, np.ones((2, 1), np.float32))
        finally:
            events.append("reader_joined")

    class Batch:
        def wait(self):
            events.append("wait")

        def to_host(self):
            return np.ones((2, 2, 1), np.float32)

    class Backend:
        name = "fake"
        count = 0

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("backend_closed")

        def thread_context(self):
            return nullcontext()

        def compute(self, **kwargs):
            self.count += 1
            if failure == "compute" and self.count == 2:
                raise OSError("compute failed")
            return Batch()

    class Writer:
        closed = False

        def __init__(self, config, name, *args, **kwargs):
            self.name = name

        def check_errors(self):
            pass

        def add_chunk_result(self, *args):
            if failure == "write":
                raise OSError("write failed")

        def finalize(self):
            if failure == "finalize":
                raise OSError("finalize failed")

        def abort(self):
            self.closed = True
            events.append("abort_" + self.name)

    monkeypatch.setattr(multitrait, "_prefetched_chunks", chunks)
    monkeypatch.setattr(multitrait, "SharedRegressionBackend", Backend)
    monkeypatch.setattr(multitrait, "ResultWriter", Writer)
    with pytest.raises(OSError, match=failure + " failed"):
        multitrait.run_shared_chunks(
            source, SimpleNamespace(), [{"name": "a"}, {"name": "b"}], object(), tmp_path, 5
        )
    assert (
        events.index("reader_joined")
        < events.index("backend_closed")
        < events.index("source_closed")
    )
    assert "abort_a" in events
    assert "abort_b" in events
    if failure != "finalize":
        assert events.index("wait") < events.index("backend_closed")

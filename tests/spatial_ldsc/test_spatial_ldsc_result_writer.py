import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gsMap.spatial_ldsc.result_writer import ResultWriter


def _config():
    return SimpleNamespace(
        project_name="project",
        cell_indices_range=None,
        sample_filter=None,
    )


def _add(writer, chunk_idx, rows, start):
    rows = np.asarray(rows, dtype=np.float32)
    writer.add_chunk_result(
        chunk_idx=chunk_idx,
        betas=rows,
        ses=np.ones(rows.size, dtype=np.float32),
        spot_names=pd.Index([f"cell{start + i}" for i in range(rows.size)]),
        abs_start=start,
        abs_end=start + rows.size,
    )


def test_result_writer_streams_out_of_order_chunks_atomically(tmp_path):
    writer = ResultWriter(
        config=_config(),
        trait_name="trait",
        output_dir=tmp_path,
        n_spots=5,
        total_chunks=3,
        expected_cells=5,
        output_batch_rows=2,
    )
    expected_path = tmp_path / "project_trait.csv.gz"

    _add(writer, 1, [3.0, 4.0], 2)
    assert not expected_path.exists()
    _add(writer, 0, [1.0, 2.0], 0)
    _add(writer, 2, [5.0], 4)

    output_path = writer.finalize()
    assert output_path == expected_path
    assert expected_path.is_file()
    assert not list(tmp_path.glob("*.partial.csv.gz"))

    result = pd.read_csv(output_path)
    assert result["spot"].tolist() == [f"cell{i}" for i in range(5)]
    np.testing.assert_allclose(result["beta"], np.arange(1, 6, dtype=np.float32))
    np.testing.assert_allclose(result["se"], 1.0)


def test_result_writer_rejects_duplicate_chunk_and_aborts(tmp_path):
    writer = ResultWriter(
        config=_config(),
        trait_name="trait",
        output_dir=tmp_path,
        n_spots=2,
        total_chunks=2,
        expected_cells=2,
        output_batch_rows=10,
    )
    _add(writer, 0, [1.0], 0)

    with pytest.raises(ValueError, match="Duplicate result chunk"):
        _add(writer, 0, [1.0], 0)

    writer.abort()
    assert not list(tmp_path.iterdir())


def test_result_writer_does_not_publish_incomplete_output(tmp_path):
    writer = ResultWriter(
        config=_config(),
        trait_name="trait",
        output_dir=tmp_path,
        n_spots=2,
        total_chunks=2,
        expected_cells=2,
        output_batch_rows=1,
    )
    _add(writer, 0, [1.0], 0)

    with pytest.raises(RuntimeError, match="Missing 1/2 LDSC chunks"):
        writer.finalize()

    assert not (tmp_path / "project_trait.csv.gz").exists()
    assert not list(tmp_path.glob("*.partial.csv.gz"))


def test_result_writer_backpressures_late_chunks_without_blocking_gap(tmp_path, monkeypatch):
    writer = ResultWriter(
        config=_config(),
        trait_name="trait",
        output_dir=tmp_path,
        n_spots=4,
        total_chunks=4,
        expected_cells=4,
        output_batch_rows=10,
        max_pending_chunks=2,
    )
    _add(writer, 1, [2.0], 1)
    _add(writer, 2, [3.0], 2)

    late_finished = threading.Event()

    def add_late_chunk():
        _add(writer, 3, [4.0], 3)
        late_finished.set()

    late_thread = threading.Thread(target=add_late_chunk)
    waiting = threading.Event()
    original_wait = writer.condition.wait

    def tracked_wait(timeout=None):
        if threading.current_thread() is late_thread:
            waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(writer.condition, "wait", tracked_wait)
    try:
        late_thread.start()
        assert waiting.wait(3), "Late chunk never reached the backpressure wait"
        assert not late_finished.is_set()
        assert len(writer.pending) == writer.max_pending_chunks

        # The missing chunk bypasses the full-buffer wait and drains the gap.
        _add(writer, 0, [1.0], 0)
        late_thread.join(timeout=3)
        assert not late_thread.is_alive()
        assert late_finished.is_set()

        result = pd.read_csv(writer.finalize())
        assert result["spot"].tolist() == ["cell0", "cell1", "cell2", "cell3"]
    finally:
        writer.abort()
        late_thread.join(timeout=3)


def test_duplicate_chunk_is_rechecked_after_backpressure_wait(tmp_path, monkeypatch):
    writer = ResultWriter(_config(), "trait", tmp_path, 3, 3, 3, max_pending_chunks=1)
    _add(writer, 1, [1], 1)
    waiting = set()
    both_waiting = threading.Event()
    original_wait = writer.condition.wait
    errors = []

    def tracked_wait(timeout=None):
        waiting.add(threading.get_ident())
        if len(waiting) == 2:
            both_waiting.set()
        return original_wait(timeout)

    def produce():
        try:
            _add(writer, 2, [2], 2)
        except ValueError as error:
            errors.append(error)

    monkeypatch.setattr(writer.condition, "wait", tracked_wait)
    threads = [threading.Thread(target=produce) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        assert both_waiting.wait(3)
        _add(writer, 0, [0], 0)
        for thread in threads:
            thread.join(3)
        assert not any(thread.is_alive() for thread in threads)
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert "Duplicate result chunk 2" in str(errors[0])
        assert pd.read_csv(writer.finalize()).spot.tolist() == ["cell0", "cell1", "cell2"]
    finally:
        writer.abort()
        for thread in threads:
            thread.join(3)


def test_slow_output_is_bounded_and_does_not_block_initial_dispatch(tmp_path, monkeypatch):
    entered, release, finished = (threading.Event() for _ in range(3))
    original = ResultWriter._write_batch

    def slow_write(self, results):
        entered.set()
        assert release.wait(5)
        original(self, results)

    monkeypatch.setattr(ResultWriter, "_write_batch", slow_write)
    writer = ResultWriter(_config(), "trait", tmp_path, 5, 5, 5, output_batch_rows=1)
    producer_errors = []

    def produce():
        try:
            _add(writer, 3, [4], 3)
            _add(writer, 4, [5], 4)
            finished.set()
        except BaseException as error:  # noqa: BLE001 - assert background failures in the main test
            producer_errors.append(error)

    thread = threading.Thread(target=produce)
    try:
        _add(writer, 0, [1], 0)
        assert entered.wait(2)
        # One in-flight batch and exactly two queued batches, not unlimited RAM.
        _add(writer, 1, [2], 1)
        _add(writer, 2, [3], 2)
        assert writer._output_queue.qsize() == writer._output_queue.maxsize == 2
        thread.start()
        assert not finished.wait(0.1)
        assert not (tmp_path / "project_trait.csv.gz").exists()
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(3)
    assert not thread.is_alive()
    assert not producer_errors
    result = pd.read_csv(writer.finalize())
    assert result.spot.tolist() == [f"cell{i}" for i in range(5)]
    np.testing.assert_array_equal(result.beta, [1, 2, 3, 4, 5])
    assert writer.finalize() == writer.final_path
    assert not writer._writer_thread.is_alive()


def test_async_output_owns_inputs_and_flushes_final_tail(tmp_path):
    writer = ResultWriter(_config(), "trait", tmp_path, 2, 1, 2, output_batch_rows=100)
    betas, ses = np.array([1.0, 2.0]), np.ones(2)
    names = pd.Index(["a", "b"])
    writer.add_chunk_result(0, betas, ses, names, 0, 2)
    betas[:] = 999
    ses[:] = 999
    names.values[:] = "changed"
    result = pd.read_csv(writer.finalize())
    assert result.spot.tolist() == ["a", "b"]
    np.testing.assert_array_equal(result.beta, [1, 2])
    np.testing.assert_array_equal(result.se, [1, 1])


@pytest.mark.parametrize("failure", ["write", "close"])
def test_async_io_errors_never_publish_output(tmp_path, monkeypatch, failure):
    original = ResultWriter._write_batch

    class FailingClose:
        def __init__(self, handle):
            self.handle = handle

        def close(self):
            self.handle.close()
            raise OSError("injected close error")

    def fail(self, results):
        if failure == "write":
            raise OSError("injected write error")
        original(self, results)
        self.handle = FailingClose(self.handle)

    monkeypatch.setattr(ResultWriter, "_write_batch", fail)
    writer = ResultWriter(_config(), "trait", tmp_path, 1, 1, 1, output_batch_rows=100)
    _add(writer, 0, [1], 0)
    with pytest.raises(RuntimeError, match=f"injected {failure} error"):
        writer.finalize()
    assert not writer._writer_thread.is_alive()
    assert not list(tmp_path.iterdir())


def test_abort_unblocks_backpressured_output_producer(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked_write(self, results):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(ResultWriter, "_write_batch", blocked_write)
    writer = ResultWriter(_config(), "trait", tmp_path, 4, 4, 4, output_batch_rows=1)
    errors = []

    def produce():
        try:
            _add(writer, 3, [4], 3)
        except RuntimeError as error:
            errors.append(error)

    _add(writer, 0, [1], 0)
    assert entered.wait(2)
    _add(writer, 1, [2], 1)
    _add(writer, 2, [3], 2)
    producer = threading.Thread(target=produce)
    aborter = threading.Thread(target=writer.abort)
    producer.start()
    try:
        aborter.start()
        producer.join(2)
        assert not producer.is_alive()
        assert errors
    finally:
        release.set()
        aborter.join(2)
        producer.join(2)
    assert not aborter.is_alive()
    assert not writer._writer_thread.is_alive()
    assert not list(tmp_path.iterdir())


def test_concurrent_out_of_order_producers_with_output_backpressure(tmp_path):
    n = 120
    writer = ResultWriter(
        _config(), "trait", tmp_path, n, n, n, output_batch_rows=1, max_pending_chunks=n
    )
    errors = []

    def produce(indices):
        try:
            for i in indices:
                _add(writer, i, [i], i)
        except BaseException as error:  # noqa: BLE001 - assert background failures in the main test
            errors.append(error)

    indices = np.random.default_rng(42).permutation(n)
    threads = [
        threading.Thread(target=produce, args=(part,)) for part in np.array_split(indices, 4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    try:
        assert not any(thread.is_alive() for thread in threads)
        assert not errors
        frame = pd.read_csv(writer.finalize())
        assert frame.spot.tolist() == [f"cell{i}" for i in range(n)]
        np.testing.assert_array_equal(frame.beta, np.arange(n))
    finally:
        writer.abort()

import errno
import os

import numpy as np
import pytest

from gsMap.latent2gene import memmap_io
from gsMap.latent2gene.memmap_io import MemMapDense, ParallelMarkerScoreWriter


@pytest.mark.parametrize(
    "indices",
    [
        slice(2, 6),
        slice(6, 2, -1),
        np.array([6, 0, 3, 2]),
        np.array([-1, -2, 0, 1]),
        np.array([3, 3, 2, 2]),
        slice(0, 0),
    ],
)
def test_positional_rows_match_numpy(tmp_path, indices):
    marker = MemMapDense(tmp_path / "marker.dat", (8, 7), num_write_workers=0)
    writer = ParallelMarkerScoreWriter(marker, num_workers=1)
    expected = np.zeros((8, 7), dtype=np.float16)
    count = expected[indices].shape[0]
    # Exercise non-contiguous input and implicit dtype conversion.
    values = np.arange(count * 14, dtype=np.float32).reshape(count, 14)[:, ::2]
    expected[indices] = values
    writer.write_queue.put((0, values, indices))
    writer.close()
    marker.close()
    actual = np.fromfile(marker.data_path, dtype=np.float16).reshape(expected.shape)
    np.testing.assert_array_equal(actual, expected)
    assert MemMapDense.check_complete(marker.path)[0]


def test_disjoint_batches_and_out_of_order_offsets(tmp_path):
    marker = MemMapDense(tmp_path / "marker.dat", (19, 11), num_write_workers=0)
    writer = ParallelMarkerScoreWriter(marker, num_workers=3, queue_size=2)
    expected = np.arange(19 * 11, dtype=np.float16).reshape(19, 11)
    for start in [15, 0, 10, 5]:
        stop = min(start + 5, 19)
        writer.write_queue.put((start, expected[start:stop], slice(start, stop)))
    writer.close()
    writer.close()
    assert writer.get_completed_count() == 4
    assert writer._output_fd is None
    assert writer.write_queue.unfinished_tasks == 0
    # POSIX file writes are coherent with the existing downstream mmap.
    np.testing.assert_array_equal(marker.memmap, expected)
    marker.close()
    np.testing.assert_array_equal(np.fromfile(marker.data_path, np.float16), expected.ravel())


def test_partial_and_interrupted_writes_preserve_every_byte(tmp_path, monkeypatch):
    original = os.pwrite
    calls = 0

    def partial(fd, data, offset):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError(errno.EINTR, "interrupted")
        return original(fd, data[:3], offset)

    monkeypatch.setattr(os, "pwrite", partial)
    marker = MemMapDense(tmp_path / "marker.dat", (2, 7), num_write_workers=0)
    writer = ParallelMarkerScoreWriter(marker, num_workers=1)
    values = np.arange(7, dtype=np.float16).reshape(1, 7)
    os.lseek(writer._output_fd, 5, os.SEEK_SET)
    writer.write_queue.put((0, values, 1))
    writer.write_queue.join()
    assert os.lseek(writer._output_fd, 0, os.SEEK_CUR) == 5
    writer.close()
    assert calls >= 6
    np.testing.assert_array_equal(marker.memmap[1], values[0])
    marker.close()


@pytest.mark.parametrize("failure", ["zero-progress", "disk-full", "fsync"])
def test_write_failure_prevents_complete_metadata(tmp_path, monkeypatch, failure):
    marker = MemMapDense(tmp_path / "marker.dat", (2, 3), num_write_workers=0)
    writer = ParallelMarkerScoreWriter(marker, num_workers=1, queue_size=2)

    def fail(*args):
        if failure == "zero-progress":
            return 0
        raise OSError(errno.ENOSPC, "synthetic disk full")

    monkeypatch.setattr(os, "fsync" if failure == "fsync" else "pwrite", fail)
    for i in range(2):
        writer.write_queue.put((i, np.ones((1, 3), dtype=np.float16), slice(i, i + 1)))
    with pytest.raises(RuntimeError):
        writer.close()
    assert writer._output_fd is None
    assert all(not t.is_alive() for t in writer.workers)
    with pytest.raises(RuntimeError, match="refusing"):
        marker.mark_complete()
    writer.abort()
    marker.abort()
    assert not MemMapDense.check_complete(marker.path)[0]


@pytest.mark.parametrize(
    "indices",
    [
        np.array([9]),
        np.array([-9]),
        np.array([2**64 - 1], dtype=np.uint64),
        slice(0, 2),
        np.array([0.5]),
    ],
)
def test_invalid_indices_cannot_extend_or_corrupt_file(tmp_path, indices):
    marker = MemMapDense(tmp_path / "marker.dat", (8, 3), num_write_workers=0)
    writer = ParallelMarkerScoreWriter(marker, num_workers=1)
    writer.write_queue.put((0, np.ones((1, 3), dtype=np.float16), indices))
    with pytest.raises(RuntimeError):
        writer.close()
    assert marker.data_path.stat().st_size == 8 * 3 * 2
    marker.abort()


def test_positional_writes_target_scratch_not_final_path(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(memmap_io, "_should_use_tmp_storage", lambda *args: True)
    marker = MemMapDense(tmp_path / "marker.dat", (3, 2), num_write_workers=0, tmp_dir=scratch)
    writer = ParallelMarkerScoreWriter(marker, num_workers=1)
    expected = np.arange(6, dtype=np.float16).reshape(3, 2)
    writer.write_queue.put((0, expected, slice(0, 3)))
    writer.close()
    assert not marker.original_path.exists()
    np.testing.assert_array_equal(np.fromfile(marker.data_path, np.float16), expected.ravel())
    marker.close()
    np.testing.assert_array_equal(np.fromfile(marker.original_path, np.float16), expected.ravel())
    assert not list(scratch.iterdir())


def test_mmap_fallback_without_pwrite(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "pwrite")
    marker = MemMapDense(tmp_path / "marker.dat", (2, 3), num_write_workers=0)
    writer = ParallelMarkerScoreWriter(marker, num_workers=1)
    assert writer.io_backend == "mmap"
    writer.write_queue.put((0, np.ones((1, 3), np.float16), slice(1, 2)))
    writer.close()
    marker.close()
    np.testing.assert_array_equal(
        np.fromfile(marker.data_path, np.float16).reshape(2, 3), [[0, 0, 0], [1, 1, 1]]
    )


def test_aborted_writer_cannot_publish_incomplete_parent(tmp_path):
    marker = MemMapDense(tmp_path / "marker.dat", (2, 3), num_write_workers=0)
    writer = ParallelMarkerScoreWriter(marker, num_workers=2)
    writer.abort()
    assert writer._output_fd is None
    with pytest.raises(RuntimeError, match="refusing"):
        marker.publish()
    marker.abort()
    assert not MemMapDense.check_complete(marker.path)[0]


@pytest.mark.parametrize("positional", [False, True])
def test_readonly_marker_output_rejected_before_worker_start(tmp_path, monkeypatch, positional):
    path = tmp_path / "marker.dat"
    with MemMapDense(path, (2, 3), num_write_workers=0):
        pass
    if not positional:
        monkeypatch.delattr(os, "pwrite")
    with MemMapDense(path, (2, 3), mode="r") as marker:
        with pytest.raises(ValueError, match="writable"):
            ParallelMarkerScoreWriter(marker)


@pytest.mark.parametrize("failed_thread", [1, 2])
def test_partial_worker_startup_closes_descriptor_and_invalidates_output(
    tmp_path, monkeypatch, failed_thread
):
    marker = MemMapDense(tmp_path / "marker.dat", (2, 3), num_write_workers=0)
    original_open = os.open
    original_start = memmap_io.threading.Thread.start
    opened, started = [], []

    def track_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def fail_start(thread):
        if len(started) + 1 == failed_thread:
            raise RuntimeError("synthetic thread creation error")
        started.append(thread)
        original_start(thread)

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(memmap_io.threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread creation"):
        ParallelMarkerScoreWriter(marker, num_workers=2)
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
    assert all(not thread.is_alive() for thread in started)
    with pytest.raises(RuntimeError, match="refusing"):
        marker.mark_complete()
    marker.abort()

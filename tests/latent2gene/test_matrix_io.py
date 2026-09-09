import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import numpy as np
import pytest

from gsMap.latent2gene.memmap_io import MemMapDense
from gsMap.utils.matrix_io import MatrixRowReader, row_read_spans


@pytest.mark.parametrize("rows", [[0, 1, 2, 5, 7], [7, 1, 1, 0], [], [7]])
@pytest.mark.parametrize("view_start", [0, 2])
def test_positional_reads_preserve_order_and_memmap_view_offset(tmp_path, rows, view_start):
    matrix = np.memmap(
        tmp_path / "matrix.dat", mode="w+", shape=(10, 7), dtype=np.float16, offset=128
    )
    matrix[:] = np.arange(70).reshape(10, 7)
    matrix.flush()
    view = matrix[view_start:]
    with closing(MatrixRowReader(view)) as reader:
        output = np.empty((len(rows), 7), np.float16)
        reader.read_into(np.array(rows, dtype=np.int64), output)
        np.testing.assert_array_equal(output, view[rows])
        np.testing.assert_array_equal(reader.read_contiguous(1, 4), view[1:4])
        fd = reader.fd
    if fd is not None:
        with pytest.raises(OSError):
            os.fstat(fd)
    matrix._mmap.close()


def test_read_short_interrupted_and_truncated(tmp_path, monkeypatch):
    matrix = np.memmap(tmp_path / "matrix.dat", mode="w+", shape=(4, 3), dtype=np.float16)
    matrix[:] = np.arange(12).reshape(4, 3)
    matrix.flush()
    original, calls = os.preadv, []

    def short(fd, buffers, offset):
        calls.append(offset)
        if len(calls) == 1:
            raise InterruptedError
        return original(fd, [buffers[0][:3]], offset)

    monkeypatch.setattr(os, "preadv", short)
    with closing(MatrixRowReader(matrix)) as reader:
        np.testing.assert_array_equal(reader.read_contiguous(0, 4), matrix)
        monkeypatch.setattr(os, "preadv", lambda *args: 0)
        with pytest.raises(EOFError):
            reader.read_contiguous(0, 4)
    assert len(calls) > 8
    matrix._mmap.close()


def test_parallel_reads_and_bounded_overread(tmp_path):
    matrix = np.memmap(tmp_path / "matrix.dat", mode="w+", shape=(1000, 101), dtype=np.float16)
    matrix[:] = np.arange(101, dtype=np.float16)[None, :] + np.arange(1000)[:, None]
    matrix.flush()
    rows = np.array([0, 1, 3, 5, 40, 42, 80, 999])
    spans = list(row_read_spans(rows, 202, max_bytes=2020, max_overread=2))
    assert sum(int(rows[b - 1] - rows[a] + 1) for a, b in spans) <= 2 * len(rows)
    assert all(int(rows[b - 1] - rows[a] + 1) <= 10 for a, b in spans)
    with closing(MatrixRowReader(matrix)) as reader, ThreadPoolExecutor(4) as pool:

        def read(_):
            output = np.empty((len(rows), 101), np.float16)
            actual_bytes = reader.read_into(rows, output, max_bytes=2020)
            assert actual_bytes <= 2 * output.nbytes
            np.testing.assert_array_equal(output, matrix[rows])

        list(pool.map(read, range(20)))
    matrix._mmap.close()


@pytest.mark.parametrize("fortran", [False, True])
def test_array_fallback_and_invalid_selection(fortran):
    matrix = np.array(np.arange(12).reshape(4, 3), order="F" if fortran else "C")
    with closing(MatrixRowReader(matrix)) as reader:
        assert reader.fd is None
        output = np.empty((2, 3), matrix.dtype)
        reader.read_into(np.array([3, 0]), output)
        np.testing.assert_array_equal(output, matrix[[3, 0]])
        with pytest.raises(IndexError):
            reader.read_into(np.array([-1, 0]), output)
        with pytest.raises(ValueError):
            reader.read_into(np.array([1.0, 0.0]), output)
        with pytest.raises(ValueError):
            reader.read_into(np.array([1, 0]), output.astype(np.float16))


def test_copy_on_write_memmap_reads_private_changes(tmp_path):
    path = tmp_path / "private.dat"
    np.arange(12, dtype=np.float16).tofile(path)
    matrix = np.memmap(path, mode="c", dtype=np.float16, shape=(4, 3))
    matrix[1] = 42
    with closing(MatrixRowReader(matrix)) as reader:
        assert reader.fd is None
        np.testing.assert_array_equal(reader.read_contiguous(1, 2), [[42, 42, 42]])
    matrix._mmap.close()
    np.testing.assert_array_equal(np.fromfile(path, np.float16), np.arange(12))


@pytest.mark.parametrize("kind", ["copy", "fancy", "strided"])
def test_memmap_arrays_without_contiguous_file_backing_use_numpy(tmp_path, kind):
    matrix = np.memmap(tmp_path / "array.dat", mode="w+", shape=(6, 3), dtype=np.float16)
    matrix[:] = np.arange(18).reshape(6, 3)
    matrix.flush()
    view = {
        "copy": lambda: matrix.copy(),
        "fancy": lambda: matrix[[5, 2, 0]],
        "strided": lambda: matrix[::2],
    }[kind]()
    with closing(MatrixRowReader(view)) as reader:
        assert reader.fd is None
        np.testing.assert_array_equal(reader.read_contiguous(0, 2), view[:2])
    matrix._mmap.close()


def test_reader_context_closes_and_rejects_reuse(tmp_path):
    matrix = np.memmap(tmp_path / "array.dat", mode="w+", shape=(2, 3), dtype=np.float16)
    with MatrixRowReader(matrix) as reader:
        fd = reader.fd
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.read_contiguous(0, 1)
    with pytest.raises(RuntimeError, match="closed"):
        reader.read_into(np.array([0]), np.empty((1, 3), np.float16))
    with pytest.raises(OSError):
        os.fstat(fd)
    matrix._mmap.close()


def test_reader_fallback_without_preadv(tmp_path, monkeypatch):
    matrix = np.memmap(tmp_path / "array.dat", mode="w+", shape=(4, 3), dtype=np.float16)
    matrix[:] = np.arange(12).reshape(4, 3)
    monkeypatch.delattr(os, "preadv")
    with closing(MatrixRowReader(matrix)) as reader:
        assert reader.fd is None
        np.testing.assert_array_equal(reader.read_contiguous(1, 3), matrix[1:3])
    matrix._mmap.close()


def test_read_rejects_readonly_destination_before_syscall(tmp_path, monkeypatch):
    matrix = np.memmap(tmp_path / "array.dat", mode="w+", shape=(2, 3), dtype=np.float16)
    output = np.zeros((1, 3), np.float16)
    output.setflags(write=False)
    with closing(MatrixRowReader(matrix)) as reader:
        monkeypatch.setattr(os, "preadv", lambda *args: pytest.fail("Read attempted"))
        with pytest.raises(ValueError, match="writable"):
            reader.read_into(np.array([0]), output)
    matrix._mmap.close()


def test_scratch_publish_failure_keeps_final_metadata_incomplete(tmp_path, monkeypatch):
    from gsMap.latent2gene import memmap_io

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(memmap_io, "_should_use_tmp_storage", lambda *args: True)
    marker = MemMapDense(tmp_path / "rank.dat", (3, 2), num_write_workers=1, tmp_dir=scratch)
    marker.write_batch(np.ones((3, 2), np.float16), slice(None))
    fsync = os.fsync

    def fail_final(fd):
        if os.readlink(f"/proc/self/fd/{fd}") == str(marker.original_path):
            raise OSError("test publication error")
        fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_final)
    with pytest.raises(OSError, match="publication error"):
        marker.publish()
    assert not MemMapDense.check_complete(marker.original_path)[0]
    marker.abort()


@pytest.mark.parametrize("workers", [0, 3])
def test_rank_writer_positional_and_column_fallback(tmp_path, workers):
    path = tmp_path / "rank.dat"
    marker = MemMapDense(path, (7, 5), num_write_workers=workers)
    values = np.arange(35, dtype=np.float16).reshape(7, 5)
    marker.write_batch(values[3:6], 3)
    marker.write_batch(values[:3], slice(0, 3))
    marker.write_batch(values[-1:], -1)
    marker.write_queue.join()
    marker.write_batch(np.ones((1, 2), np.float16), slice(0, 1), slice(0, 2))
    values[0, :2] = 1
    marker.publish()
    np.testing.assert_array_equal(marker.memmap, values)
    with pytest.raises(RuntimeError):
        marker.write_batch(values, slice(None))
    marker.close()
    assert marker._write_fd is None
    np.testing.assert_array_equal(np.fromfile(path, np.float16), values.ravel())
    assert MemMapDense.check_complete(path)[0]


@pytest.mark.parametrize("failure", ["pwrite", "fsync"])
def test_rank_writer_failure_never_marks_complete(tmp_path, monkeypatch, failure):
    marker = MemMapDense(tmp_path / "rank.dat", (2, 3), num_write_workers=2)

    def fail(*args):
        raise OSError("test write error")

    monkeypatch.setattr(os, failure, fail)
    marker.write_batch(np.ones((2, 3), np.float16), slice(None))
    with pytest.raises((RuntimeError, OSError)):
        marker.publish()
    assert not MemMapDense.check_complete(marker.path)[0]
    marker.abort()
    assert marker._write_fd is None


@pytest.mark.parametrize("start", [-16, -9, 8])
@pytest.mark.parametrize("workers", [0, 2])
def test_rank_batch_start_out_of_bounds_never_wraps_twice(tmp_path, start, workers):
    marker = MemMapDense(tmp_path / "rank.dat", (8, 3), num_write_workers=workers)
    values = np.ones((1, 3), np.float16)
    if workers == 0:
        with pytest.raises(IndexError):
            marker.write_batch(values, start)
    else:
        marker.write_batch(values, start)
    with pytest.raises(RuntimeError, match="refusing"):
        marker.publish()
    marker.abort()
    np.testing.assert_array_equal(np.fromfile(marker.data_path, np.float16), np.zeros(24))

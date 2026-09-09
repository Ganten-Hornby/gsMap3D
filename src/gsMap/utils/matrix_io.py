"""Bounded positional I/O for row-major matrices on local and shared storage."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import DTypeLike

_READ_BUFFER_BYTES = 8 * 1024**2
_MAX_READ_AMPLIFICATION = 2.0
_MAX_READ_GAP_BYTES = 1024**2


def pread_into(fd: int, values: np.ndarray, offset: int) -> None:
    """Fill a contiguous array exactly, handling short reads and EINTR."""
    if not values.flags.c_contiguous or not values.flags.writeable:
        raise ValueError("Positional read target must be writable and C-contiguous")
    if not values.size:
        return
    view = memoryview(values).cast("B")
    done = 0
    while done < view.nbytes:
        try:
            count = os.preadv(fd, [view[done:]], offset + done)
        except InterruptedError:
            continue
        if count <= 0:
            raise EOFError(f"Unexpected end of matrix after {done:,}/{view.nbytes:,} bytes")
        done += count


def pwrite_rows(
    fd: int,
    shape: tuple[int, int],
    dtype: DTypeLike,
    values: np.ndarray,
    rows: int | slice | np.ndarray,
) -> None:
    """Write complete rows of a zero-offset file without changing its seek position.

    Calls from different threads must target disjoint rows. Repeated indices
    within one call preserve NumPy's last-write-wins ordering.
    """
    values = np.asarray(values, dtype=dtype)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != shape[1]:
        raise ValueError(f"Invalid matrix batch shape: {values.shape}")
    n_rows = shape[0]

    def write_block(block: np.ndarray, start: int) -> None:
        if not block.size:
            return
        data = memoryview(np.ascontiguousarray(block)).cast("B")
        offset = int(start) * shape[1] * np.dtype(dtype).itemsize
        while data:
            try:
                count = os.pwrite(fd, data, offset)
            except InterruptedError:
                continue
            if count <= 0:
                raise OSError("Matrix pwrite made no progress")
            offset += count
            data = data[count:]

    if isinstance(rows, slice):
        start, stop, step = rows.indices(n_rows)
        if len(range(start, stop, step)) != len(values):
            raise ValueError("Matrix row slice and batch lengths disagree")
        if step == 1:
            write_block(values, start)
            return
        indices = np.arange(start, stop, step, dtype=np.int64)
    else:
        indices = np.atleast_1d(np.asarray(rows))
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("Matrix row indices must be a one-dimensional integer array")
        if len(indices) != len(values):
            raise ValueError("Matrix row indices and batch lengths disagree")
        if np.any(indices >= n_rows) or (indices.dtype.kind == "i" and np.any(indices < -n_rows)):
            raise IndexError("Matrix row index out of bounds")
        indices = indices.astype(np.int64)
        indices = np.where(indices < 0, indices + n_rows, indices)
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    for start, end in zip(np.r_[0, breaks], np.r_[breaks, len(indices)], strict=True):
        if start < end:
            write_block(values[start:end], indices[start])


def row_read_spans(
    rows: np.ndarray,
    row_bytes: int,
    *,
    max_bytes: int = _READ_BUFFER_BYTES,
    max_overread: float = _MAX_READ_AMPLIFICATION,
    max_gap_bytes: int = _MAX_READ_GAP_BYTES,
) -> Iterator[tuple[int, int]]:
    """Coalesce sorted unique rows; bound buffers and read amplification.

    Bounds index the input rows, not the source matrix. A single matrix row
    wider than max_bytes is the only permitted exception to the buffer limit.
    """
    if row_bytes < 1 or max_bytes < 1 or max_overread < 1 or max_gap_bytes < 0:
        raise ValueError("Row/read sizes must be positive and read amplification at least one")
    max_rows = max(1, max_bytes // row_bytes)
    max_gap = max_gap_bytes // row_bytes
    start = 0
    for end in range(1, len(rows)):
        span = int(rows[end]) - int(rows[start]) + 1
        gap = int(rows[end]) - int(rows[end - 1]) - 1
        if span > max_rows or span > (end - start + 1) * max_overread or gap > max_gap:
            yield start, end
            start = end
    if start < len(rows):
        yield start, len(rows)


class MatrixRowReader:
    """Read rows through preadv without faulting file pages into process RSS.

    Each caller owns its returned/target buffer; a single descriptor can be
    shared by rank-cache read threads. close() must follow their termination.
    In-memory arrays and unsupported platforms retain NumPy semantics.
    """

    def __init__(self, matrix: np.ndarray, *, sequential: bool = False) -> None:
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("Matrix row reader requires a 2D matrix with at least one column")
        self.matrix = matrix
        self.fd: int | None = None
        self._closed = False
        self.offset = 0
        self.row_bytes = int(matrix.shape[1]) * matrix.dtype.itemsize
        if not (
            isinstance(matrix, np.memmap)
            and matrix.flags.c_contiguous
            and matrix.mode in ("r", "r+", "w+")
            and matrix.filename is not None
            and callable(getattr(os, "preadv", None))
        ):
            return
        if matrix._mmap is not None and matrix._mmap.closed:
            raise ValueError("Cannot read a closed matrix mapping")
        # A C-contiguous row view can have .offset == 0 even when it starts
        # part-way into the original mapping. Include its pointer displacement.
        base = matrix
        while isinstance(base.base, np.ndarray):
            base = base.base
        self.offset = int(getattr(base, "offset", 0)) + matrix.ctypes.data - base.ctypes.data
        self.fd = os.open(os.fspath(matrix.filename), os.O_RDONLY)
        advice = getattr(os, "POSIX_FADV_SEQUENTIAL" if sequential else "POSIX_FADV_RANDOM", None)
        if advice is not None and hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(self.fd, 0, 0, advice)
            except OSError:
                pass

    def read_into(
        self,
        rows: np.ndarray,
        output: np.ndarray,
        *,
        max_bytes: int = _READ_BUFFER_BYTES,
        max_overread: float = _MAX_READ_AMPLIFICATION,
    ) -> int:
        """Fill rows in caller order; return bytes requested, including overread.

        This counts logical I/O, including page-cache hits, not physical disk
        traffic. Each coalesced read has a bounded temporary buffer.
        """
        self._check_open()
        indices = np.asarray(rows)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("Read rows must be a one-dimensional integer array")
        if (
            output.shape != (len(indices), self.matrix.shape[1])
            or output.dtype != self.matrix.dtype
        ):
            raise ValueError("Read output shape/dtype does not match the requested matrix rows")
        if not output.flags.c_contiguous or not output.flags.writeable:
            raise ValueError("Read output must be writable and C-contiguous")
        if indices.size and (indices.min() < 0 or indices.max() >= self.matrix.shape[0]):
            raise IndexError("Read row index out of bounds")
        if not len(indices):
            return 0
        if self.fd is None:
            output[:] = self.matrix[indices]
            return output.nbytes
        if np.any(indices[1:] <= indices[:-1]):
            unique, inverse = np.unique(indices, return_inverse=True)
            values = np.empty((len(unique), self.matrix.shape[1]), dtype=self.matrix.dtype)
            count = self.read_into(unique, values, max_bytes=max_bytes, max_overread=max_overread)
            output[:] = values[inverse]
            return count
        read_bytes = 0
        for first, end in row_read_spans(
            indices, self.row_bytes, max_bytes=max_bytes, max_overread=max_overread
        ):
            source_start = int(indices[first])
            span_rows = int(indices[end - 1]) - source_start + 1
            contiguous = span_rows == end - first
            buffer = (
                output[first:end]
                if contiguous
                else np.empty((span_rows, self.matrix.shape[1]), dtype=self.matrix.dtype)
            )
            pread_into(self.fd, buffer, self.offset + source_start * self.row_bytes)
            read_bytes += buffer.nbytes
            if not contiguous:
                output[first:end] = buffer[indices[first:end] - source_start]
        return read_bytes

    def read_contiguous(self, start: int, end: int) -> np.ndarray:
        """Return an owned RAM buffer for a half-open source row range."""
        self._check_open()
        if not 0 <= start <= end <= self.matrix.shape[0]:
            raise IndexError("Contiguous read rows out of bounds")
        output = np.empty((end - start, self.matrix.shape[1]), dtype=self.matrix.dtype)
        if self.fd is None:
            output[:] = self.matrix[start:end]
        elif output.size:
            pread_into(self.fd, output, self.offset + start * self.row_bytes)
        return output

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Matrix row reader is closed")

    def close(self) -> None:
        self._closed = True
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)

    def __enter__(self) -> MatrixRowReader:
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj) -> None:
        self.close()

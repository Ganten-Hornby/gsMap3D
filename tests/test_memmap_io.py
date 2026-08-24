import errno
import os

import numpy as np
import pytest

from gsMap.latent2gene.memmap_io import MemMapDense


def test_memmap_dense_preallocates_backing_file(monkeypatch, tmp_path) -> None:
    calls = []

    def record_fallocate(file_descriptor: int, offset: int, byte_size: int) -> None:
        calls.append((offset, byte_size))
        os.ftruncate(file_descriptor, byte_size)

    monkeypatch.setattr(os, "posix_fallocate", record_fallocate)

    matrix = MemMapDense(
        tmp_path / "matrix",
        shape=(3, 4),
        dtype=np.float32,
        mode="w",
        num_write_workers=0,
    )
    try:
        assert calls == [(0, 3 * 4 * np.dtype(np.float32).itemsize)]
        assert matrix.data_path.stat().st_size == 3 * 4 * np.dtype(np.float32).itemsize
    finally:
        matrix.close()


def test_memmap_dense_removes_partial_file_when_preallocation_fails(
    monkeypatch, tmp_path
) -> None:
    def fail_fallocate(file_descriptor: int, offset: int, byte_size: int) -> None:
        raise OSError(errno.ENOSPC, "no space left")

    monkeypatch.setattr(os, "posix_fallocate", fail_fallocate)
    data_path = tmp_path / "matrix.dat"

    with pytest.raises(OSError, match="no space left"):
        MemMapDense(
            tmp_path / "matrix",
            shape=(3, 4),
            dtype=np.float32,
            mode="w",
            num_write_workers=0,
        )

    assert not data_path.exists()

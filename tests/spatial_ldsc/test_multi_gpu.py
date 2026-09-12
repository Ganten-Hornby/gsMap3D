"""Production device allocation, shard publication and failure propagation."""

import gzip
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gsMap.spatial_ldsc.multi_gpu import (
    merge_gzip_members,
    partition_cells,
    visible_gpu_tokens,
    wait_workers,
    worker_environment,
)
from gsMap.spatial_ldsc.result_writer import ResultWriter


@pytest.mark.parametrize(
    ("cells", "batch", "gpus"), [(1000037, 100, 8), (1037, 100, 4), (3, 100, 8), (8, 1, 8)]
)
def test_partition_has_no_gaps_or_empty_ranges(cells, batch, gpus):
    ranges = partition_cells(cells, batch, gpus)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == cells
    assert all(left < right for left, right in ranges)
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    assert all(left % batch == 0 for left, _ in ranges)


@pytest.mark.parametrize("visible", ["", "-1", "0,0", "2"])
def test_gpu_allocation_rejects_missing_or_duplicate_tokens(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="Requested"):
        visible_gpu_tokens(2)


@pytest.mark.parametrize("visible", ["3,7,1", "GPU-abc,GPU-def", "MIG-aaa,MIG-bbb"])
def test_gpu_allocation_preserves_visible_order(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert visible_gpu_tokens(2) == visible.split(",")[:2]


@pytest.mark.parametrize("order", [None, "FASTEST_FIRST", "PCI_BUS_ID"])
def test_worker_environment_preserves_cuda_ordinal_order(monkeypatch, order):
    if order is None:
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    else:
        monkeypatch.setenv("CUDA_DEVICE_ORDER", order)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    env = worker_environment("7", 4)
    assert env.get("CUDA_DEVICE_ORDER") == order
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["JAX_PLATFORMS"] == "cuda"
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_real_writers_merge_one_header_and_all_rows(tmp_path):
    cfg = SimpleNamespace(project_name="p", sample_filter=None, cell_indices_range=None)
    paths = []
    for rank in range(2):
        writer = ResultWriter(
            cfg, "trait", tmp_path / f"rank{rank}", 3, 1, 3, emit_header=rank == 0
        )
        writer.add_chunk_result(
            0, np.arange(3), np.ones(3), pd.Index([f"s{i + rank * 3}" for i in range(3)]), 0, 3
        )
        paths.append(writer.finalize())
    output = tmp_path / "merged.csv.gz"
    merge_gzip_members(paths, output)
    frame = pd.read_csv(output)
    assert frame.spot.tolist() == [f"s{i}" for i in range(6)]
    with gzip.open(output, "rt") as handle:
        assert handle.read().count("spot,beta,se,z,p,neg_log10_p\n") == 1
    with pytest.raises(FileExistsError):
        merge_gzip_members(paths, output)
    assert not list(tmp_path.glob("*.partial"))


def test_duplicate_headers_do_not_publish(tmp_path):
    part = tmp_path / "part.gz"
    with gzip.open(part, "wt") as handle:
        handle.write("spot,beta,se,z,p,neg_log10_p\ns,1,1,1,.1,1\n")
    with pytest.raises(ValueError, match="header"):
        merge_gzip_members([part, part], tmp_path / "final.gz")
    assert not (tmp_path / "final.gz").exists()


def test_worker_failure_terminates_sibling_and_reaps_children():
    processes = [
        subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"]),
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]),
    ]
    with pytest.raises(RuntimeError, match="status 7"):
        wait_workers(processes)
    assert all(process.poll() is not None for process in processes)

import json
from dataclasses import fields

import numpy as np
import pytest

from gsMap.config.latent2gene_config import LatentToGeneComputeConfig
from gsMap.latent2gene.memmap_io import set_memmap_advice
from gsMap.latent2gene.rank_calculator import (
    _automatic_rank_batch_size,
    _automatic_rank_write_interval,
)


def _write_rank_memmap(tmp_path, values: np.ndarray):
    base_path = tmp_path / "ranks"
    data_path = base_path.with_suffix(".dat")
    metadata_path = base_path.with_suffix(".meta.json")
    rank_memmap = np.memmap(data_path, dtype=values.dtype, mode="w+", shape=values.shape)
    rank_memmap[:] = values
    rank_memmap.flush()
    del rank_memmap
    metadata_path.write_text(
        json.dumps({"shape": list(values.shape), "dtype": str(values.dtype)}),
        encoding="utf-8",
    )
    return base_path


def test_set_memmap_advice_rejects_unknown_value(tmp_path):
    values = np.arange(12, dtype=np.float16).reshape(3, 4)
    base_path = _write_rank_memmap(tmp_path, values)
    rank_memmap = np.memmap(
        base_path.with_suffix(".dat"), dtype=np.float16, mode="r", shape=(3, 4)
    )

    with pytest.raises(ValueError, match="Invalid rank mmap advice"):
        set_memmap_advice(rank_memmap, "unknown")


def test_hybrid_configuration_defaults():
    defaults = {field.name: field.default for field in fields(LatentToGeneComputeConfig)}
    workload_automatic = {
        "rank_batch_size": 0,
        "mkscore_batch_size": 0,
        "rank_write_interval": 0,
    }
    stable_concurrency = {
        "find_homogeneous_batch_size": 100,
        "rank_read_workers": 120,
        "mkscore_compute_workers": 4,
        "mkscore_write_workers": 2,
        "compute_input_queue_size": 1,
        "writer_queue_size": 20,
    }
    removed_execution_fields = {
        "rank_read_max_workers",
        "rank_read_memory_budget_gib",
        "rank_mmap_advice",
        "available_cpus",
        "available_memory",
    }

    assert {name: defaults[name] for name in workload_automatic} == workload_automatic
    assert {name: defaults[name] for name in stable_concurrency} == stable_concurrency
    assert defaults.keys().isdisjoint(removed_execution_fields)


def test_rank_batch_and_write_interval_are_workload_automatic():
    rank_batch = _automatic_rank_batch_size(12_253)

    assert rank_batch == 500
    assert _automatic_rank_write_interval(rank_batch) == 10

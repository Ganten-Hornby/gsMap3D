import json
import queue
from dataclasses import fields

import numpy as np
import pytest

from gsMap.config.latent2gene_config import LatentToGeneComputeConfig
from gsMap.latent2gene.marker_scores import (
    ParallelMarkerScoreComputer,
    _automatic_marker_batch_size,
)
from gsMap.latent2gene.memmap_io import (
    ParallelRankReader,
    set_memmap_advice,
)
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


@pytest.mark.parametrize("advice", ["normal", "random", "sequential"])
def test_parallel_rank_reader_reconstructs_exact_rows(tmp_path, advice):
    values = np.arange(8 * 5, dtype=np.float16).reshape(8, 5)
    base_path = _write_rank_memmap(tmp_path, values)
    output_queue = queue.Queue(maxsize=1)
    reader = ParallelRankReader(
        str(base_path),
        num_workers=1,
        output_queue=output_queue,
        mmap_advice=advice,
    )
    neighbors = np.array([[3, 1, 3], [0, 2, 1]], dtype=np.int64)

    try:
        reader.submit_batch(7, neighbors, {"source": "test"})
        batch_id, rank_data, rank_indices, neighbor_shape, metadata = output_queue.get(timeout=5)
        reader.check_errors()
    finally:
        reader.close()

    assert reader.shared_memmap._mmap.closed
    expected_unique, expected_inverse = np.unique(neighbors.reshape(-1), return_inverse=True)
    assert batch_id == 7
    assert metadata == {"source": "test"}
    assert neighbor_shape == neighbors.shape
    assert np.array_equal(rank_data, values[expected_unique])
    assert np.array_equal(rank_indices, expected_inverse)
    assert rank_indices.dtype == np.int32
    reconstructed = rank_data[rank_indices].reshape(*neighbors.shape, values.shape[1])
    assert np.array_equal(reconstructed, values[neighbors])


def test_return_inverse_matches_legacy_dictionary_mapping():
    neighbors = np.array([[9, 2, 9, 4], [2, 5, 4, 9]], dtype=np.int64)
    flat_neighbors = neighbors.reshape(-1)
    legacy_unique = np.unique(flat_neighbors)
    legacy_map = {index: position for position, index in enumerate(legacy_unique)}
    legacy_inverse = np.asarray([legacy_map[index] for index in flat_neighbors])

    optimized_unique, optimized_inverse = np.unique(flat_neighbors, return_inverse=True)

    assert np.array_equal(optimized_unique, legacy_unique)
    assert np.array_equal(optimized_inverse, legacy_inverse)


def test_set_memmap_advice_rejects_unknown_value(tmp_path):
    values = np.arange(12, dtype=np.float16).reshape(3, 4)
    base_path = _write_rank_memmap(tmp_path, values)
    rank_memmap = np.memmap(
        base_path.with_suffix(".dat"), dtype=np.float16, mode="r", shape=(3, 4)
    )

    with pytest.raises(ValueError, match="Invalid rank mmap advice"):
        set_memmap_advice(rank_memmap, "unknown")


def test_automatic_marker_batch_for_macaque_workload():
    assert _automatic_marker_batch_size(12_253, np.dtype("float16"), 100) == 50


@pytest.mark.parametrize(
    ("neighbors_per_cell", "expected_batch_size"),
    [(20, 200), (100, 50), (500, 10)],
)
def test_automatic_marker_batch_adapts_to_workload(neighbors_per_cell, expected_batch_size):
    assert (
        _automatic_marker_batch_size(12_253, np.dtype("float16"), neighbors_per_cell)
        == expected_batch_size
    )


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


def test_marker_score_context_keeps_full_weights_in_host_memory():
    computer = object.__new__(ParallelMarkerScoreComputer)
    weights = np.arange(24, dtype=np.float32).reshape(6, 4)
    cell_indices = np.arange(6, dtype=np.int64)

    computer.set_batch_context(weights, cell_indices, batch_size=2, n_cells=6)

    assert isinstance(computer.neighbor_weights, np.ndarray)
    assert np.array_equal(computer.neighbor_weights, weights)

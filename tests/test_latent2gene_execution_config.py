import json
from dataclasses import fields
from unittest.mock import call, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

from gsMap.config.latent2gene_config import LatentToGeneComputeConfig
from gsMap.latent2gene.memmap_io import set_memmap_advice
from gsMap.latent2gene.rank_calculator import (
    _RankProgressReporter,
    _automatic_sparse_rank_batch_size,
    _build_log_rank_lookup,
    _sparse_log_rank_bits,
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


def test_sparse_rank_batch_is_workload_automatic():
    assert _automatic_sparse_rank_batch_size(27_122) == 1000


def test_rank_progress_reporter_persists_progress_and_throughput():
    reporter = _RankProgressReporter(
        100,
        percent_step=10,
        log_interval_seconds=float("inf"),
    )
    with patch("gsMap.latent2gene.rank_calculator.logger.info") as log_info:
        reporter.update(
            processed_cells=5,
            active_seconds=1.0,
            sample_name="slice_1",
            sample_processed_cells=5,
            sample_total_cells=100,
        )
        log_info.assert_not_called()

        reporter.update(
            processed_cells=10,
            active_seconds=1.0,
            sample_name="slice_1",
            sample_processed_cells=10,
            sample_total_cells=100,
        )
        reporter.update(
            processed_cells=100,
            active_seconds=8.0,
            sample_name="slice_1",
            sample_processed_cells=100,
            sample_total_cells=100,
        )

    assert log_info.call_args_list == [
        call(
            "Rank progress: 10/100 cells (10.0%); active throughput=5 cells/s; "
            "sample=slice_1 (10/100)"
        ),
        call(
            "Rank progress: 100/100 cells (100.0%); active throughput=10 cells/s; "
            "sample=slice_1 (100/100)"
        ),
    ]


def test_sparse_rank_kernel_matches_dense_average_ranks():
    dense = np.array(
        [
            [0, 1, 1, 3, 0, 2],
            [0, 0.5, 2.5, 0.5, 0, 3.5],
            [-2, 0, 4, -1, 0, 4],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    sparse = csr_matrix(dense)
    log_rank_float32, log_rank_float16_bits = _build_log_rank_lookup(dense.shape[1])

    rank_bits, sum_log_ranks, sum_fractions = _sparse_log_rank_bits(
        sparse.indptr,
        sparse.indices,
        sparse.data,
        dense.shape[1],
        log_rank_float32,
        log_rank_float16_bits,
    )

    expected = np.log(rankdata(dense, method="average", axis=1)).astype(np.float32)
    assert np.array_equal(rank_bits.view(np.float16), expected.astype(np.float16))
    assert np.allclose(sum_log_ranks, expected.sum(axis=0), rtol=1e-6, atol=1e-6)
    assert np.array_equal(sum_fractions, np.count_nonzero(dense, axis=0))

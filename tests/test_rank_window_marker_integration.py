import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gsMap.latent2gene.marker_scores import (
    MarkerScoreCalculator,
    compute_marker_scores_from_cache_jax,
)


class _ImmediateWriteQueue:
    def __init__(self, writer):
        self.writer = writer

    def put(self, item):
        _, marker_scores, cell_indices = item
        self.writer.output[cell_indices] = marker_scores
        self.writer.completed += 1

    def join(self):
        return None


class _Writer:
    def __init__(self, shape):
        self.output = np.zeros(shape, dtype=np.float16)
        self.completed = 0
        self.write_queue = _ImmediateWriteQueue(self)

    def reset_for_cell_type(self, cell_type):
        del cell_type
        self.completed = 0

    def check_errors(self):
        return None

    def get_completed_count(self):
        return self.completed


@pytest.mark.parametrize("cache_rows", [3, 5], ids=["multiple-windows", "one-window"])
@pytest.mark.parametrize("prefetch_and_retain", [False, True])
def test_unified_rank_cache_matches_full_rank_matrix(cache_rows, prefetch_and_retain):
    ranks = np.asarray(
        [
            [0.10, 0.20, 0.30],
            [0.15, 0.25, 0.35],
            [0.20, 0.30, 0.40],
            [0.25, 0.35, 0.45],
            [0.30, 0.40, 0.50],
        ],
        dtype=np.float16,
    )
    neighbors = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
    weights = np.ones(neighbors.shape, dtype=np.float16)
    output_indices = np.asarray([2, 0, 3, 1], dtype=np.int64)

    calculator = object.__new__(MarkerScoreCalculator)
    calculator.rank_values = ranks
    calculator.rank_shape = ranks.shape
    calculator.rank_dtype = ranks.dtype
    calculator.writer = _Writer(ranks.shape)
    calculator.marker_batch_size = 2
    calculator.rank_read_workers = 2
    calculator.compute_workers = 2
    calculator.compute_pending_limit = 2
    calculator.cross_slice_strategy = None
    calculator.n_slices = 1
    calculator.num_homogeneous_per_slice = 2
    calculator.global_log_gmean = jnp.zeros(ranks.shape[1], dtype=jnp.float32)
    calculator.global_expr_frac = jnp.zeros(ranks.shape[1], dtype=jnp.float32)
    calculator.no_expression_fraction = True
    calculator.cell_type_rank_cache_limit_bytes = (
        cache_rows * ranks.shape[1] * ranks.dtype.itemsize
    )
    calculator.rank_window_prefetch = prefetch_and_retain
    calculator.rank_window_adjacent_retention = prefetch_and_retain

    calculator._calculate_marker_scores_with_rank_cache(
        "synthetic",
        neighbors,
        weights,
        output_indices,
    )

    expected_in_trace_order = np.asarray(
        compute_marker_scores_from_cache_jax(
            jax.device_put(ranks),
            jax.device_put(neighbors),
            jax.device_put(weights),
            neighbors.shape[0],
            neighbors.shape[1],
            calculator.global_log_gmean,
            calculator.global_expr_frac,
            True,
        ),
        dtype=np.float16,
    )
    expected = np.zeros_like(ranks)
    expected[output_indices] = expected_in_trace_order

    assert np.array_equal(calculator.writer.output, expected)

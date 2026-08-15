import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gsMap.latent2gene.marker_scores import (
    MarkerScoreCalculator,
    compute_marker_scores_from_cache_jax,
)


class _Reader:
    def __init__(self, ranks):
        self.shared_memmap = ranks
        self.shape = ranks.shape
        self.dtype = ranks.dtype

    def reset_for_cell_type(self, cell_type):
        del cell_type


class _Computer:
    def __init__(self, genes):
        self.num_workers = 2
        self.cross_slice_strategy = None
        self.n_slices = 1
        self.num_homogeneous_per_slice = 2
        self.global_log_gmean = jnp.zeros(genes, dtype=jnp.float32)
        self.global_expr_frac = jnp.zeros(genes, dtype=jnp.float32)
        self.no_expression_fraction = True

    def reset_for_cell_type(self, cell_type):
        del cell_type


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


@pytest.mark.parametrize("prefetch_and_retain", [False, True])
def test_rank_window_path_matches_full_rank_cache(prefetch_and_retain):
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
    calculator.reader = _Reader(ranks)
    calculator.computer = _Computer(ranks.shape[1])
    calculator.writer = _Writer(ranks.shape)
    calculator.cached_marker_batch_size = 2
    calculator.effective_rank_read_workers = 2
    calculator.cell_type_rank_cache_limit_bytes = 3 * ranks.shape[1] * ranks.dtype.itemsize
    calculator.rank_window_prefetch = prefetch_and_retain
    calculator.rank_window_adjacent_retention = prefetch_and_retain

    used = calculator._calculate_marker_scores_from_rank_windows(
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
            calculator.computer.global_log_gmean,
            calculator.computer.global_expr_frac,
            True,
        ),
        dtype=np.float16,
    )
    expected = np.zeros_like(ranks)
    expected[output_indices] = expected_in_trace_order

    assert used
    assert np.array_equal(calculator.writer.output, expected)

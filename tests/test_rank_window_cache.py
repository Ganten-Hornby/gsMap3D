import numpy as np
import pytest

from gsMap.latent2gene.window_cache import (
    AdjacentRankWindowIndex,
    independent_rank_window_mapping,
    maximal_rank_windows,
    plan_rank_windows,
)


def test_plan_uses_one_right_sized_window_when_cell_type_fits():
    neighbors = np.asarray([[4, 1, 4], [2, 1, 2]], dtype=np.int32)

    windows = plan_rank_windows(neighbors, max_cache_rows=4, global_rows=8)

    assert windows == [type(windows[0])(start=0, end=2, unique_rows=3)]


def test_plan_switches_to_capacity_maximal_windows_for_large_type():
    neighbors = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)

    windows = plan_rank_windows(neighbors, max_cache_rows=3, global_rows=5)

    assert [(window.start, window.end) for window in windows] == [(0, 2), (2, 4)]


def test_plan_rejects_non_integer_neighbor_indices():
    with pytest.raises(TypeError, match="must be integers"):
        plan_rank_windows(
            np.asarray([[0.0, 1.0]], dtype=np.float32),
            max_cache_rows=2,
            global_rows=2,
        )


def test_maximal_windows_cover_trace_and_fill_until_next_row_would_overflow():
    neighbors = np.asarray(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
        ],
        dtype=np.int32,
    )
    capacity = 3

    windows = maximal_rank_windows(neighbors, capacity, global_rows=6)

    assert [(window.start, window.end) for window in windows] == [(0, 2), (2, 4), (4, 5)]
    assert windows[0].start == 0
    assert windows[-1].end == neighbors.shape[0]
    for left, window in enumerate(windows):
        observed = np.unique(neighbors[window.start : window.end]).size
        assert observed == window.unique_rows
        assert observed <= capacity
        if left + 1 < len(windows):
            with_next = np.unique(neighbors[window.start : window.end + 1]).size
            assert with_next > capacity


def test_independent_window_mapping_reconstructs_rank_rows():
    ranks = np.arange(30, dtype=np.float16).reshape(6, 5)
    neighbors = np.asarray([[5, 1, 5], [2, 1, 2]], dtype=np.int32)

    mapping = independent_rank_window_mapping(neighbors)
    cache = ranks[mapping.rows_to_read]

    assert np.array_equal(cache[mapping.local_neighbor_indices], ranks[neighbors])
    assert mapping.retained_rows == 0


def test_adjacent_mapping_retains_shared_rows_and_reconstructs_both_windows():
    ranks = np.arange(40, dtype=np.float16).reshape(8, 5)
    first_neighbors = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    second_neighbors = np.asarray([[1, 2], [2, 3]], dtype=np.int32)
    index = AdjacentRankWindowIndex(capacity_rows=4, global_rows=8)
    cache = np.zeros((4, ranks.shape[1]), dtype=ranks.dtype)

    first = index.map_window(first_neighbors)
    cache[first.update_slots] = ranks[first.rows_to_read]
    assert np.array_equal(cache[first.local_neighbor_indices], ranks[first_neighbors])

    second = index.map_window(second_neighbors)
    cache[second.update_slots] = ranks[second.rows_to_read]
    assert second.retained_rows == 2
    assert np.array_equal(second.rows_to_read, np.asarray([3]))
    assert np.array_equal(cache[second.local_neighbor_indices], ranks[second_neighbors])


@pytest.mark.parametrize("bad_value", [-1, 6])
def test_maximal_windows_reject_indices_outside_rank_matrix(bad_value):
    neighbors = np.asarray([[0, bad_value]], dtype=np.int32)

    with pytest.raises(IndexError, match="outside the rank matrix"):
        maximal_rank_windows(neighbors, max_cache_rows=3, global_rows=6)

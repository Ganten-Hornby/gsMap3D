"""Sparse or empty neighbor pools retain the fixed output contract."""

import numpy as np
import pytest

from gsMap.latent2gene.connectivity import find_spatial_neighbors_with_slices


def test_one_neighbor_with_multiple_queries_per_section():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 5], [1, 0, 5]], dtype=float)
    neighbors, _ = find_spatial_neighbors_with_slices(
        coords, np.array([0, 0, 1, 1]), k_central=1, k_adjacent=1, n_adjacent_slices=1
    )
    np.testing.assert_array_equal(neighbors, [[0, 2, -1], [1, 3, -1], [2, 0, -1], [3, 1, -1]])


@pytest.mark.parametrize("empty_pool", [False, True])
def test_unsliced_neighbors_have_fixed_shape_and_padding(empty_pool):
    coords = np.array([[0, 0], [1, 0]], dtype=float)
    neighbors, _ = find_spatial_neighbors_with_slices(
        coords, k_central=3, high_quality_cell_mask=np.array([False, not empty_pool])
    )
    assert neighbors.shape == (2, 3)
    np.testing.assert_array_equal(neighbors[:, 1:], -1)
    np.testing.assert_array_equal(neighbors[:, 0], -1 if empty_pool else 1)


def test_empty_queries_preserve_full_width():
    neighbors, _ = find_spatial_neighbors_with_slices(
        np.zeros((2, 3)),
        slice_ids=np.array([0, 1]),
        query_cell_mask=np.zeros(2, dtype=bool),
        k_central=2,
        k_adjacent=3,
        n_adjacent_slices=2,
    )
    assert neighbors.shape == (0, 14)


def test_zero_adjacent_neighbors_omits_cross_slice_slots():
    neighbors, _ = find_spatial_neighbors_with_slices(
        np.array([[0, 0], [1, 0]]),
        slice_ids=np.array([0, 1]),
        k_central=1,
        k_adjacent=0,
        n_adjacent_slices=2,
    )
    np.testing.assert_array_equal(neighbors, [[0], [1]])

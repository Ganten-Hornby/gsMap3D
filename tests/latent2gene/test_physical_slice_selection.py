"""Tests for measured-z cross-section candidate selection."""

from __future__ import annotations

import numpy as np
import pytest

from gsMap.config import CrossSliceSelectionMode
from gsMap.latent2gene.connectivity import find_spatial_neighbors_with_slices


def irregular_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Two non-tied x/y points on each of three unequally spaced sections."""
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.1, 0.0, 5.0],
            [2.1, 0.0, 5.0],
            [0.2, 0.0, 25.0],
            [2.2, 0.0, 25.0],
        ]
    )
    return coords, np.repeat(np.asarray([10, 20, 30], dtype=np.int32), 2)


def find_neighbors(
    coords: np.ndarray,
    slice_ids: np.ndarray,
    **kwargs: object,
) -> np.ndarray:
    neighbors, _ = find_spatial_neighbors_with_slices(
        coords,
        slice_ids=slice_ids,
        k_central=2,
        k_adjacent=2,
        n_adjacent_slices=1,
        **kwargs,
    )
    return neighbors


def test_physical_window_uses_calibrated_distance_and_pads_missing_planes() -> None:
    coords, slice_ids = irregular_fixture()
    neighbors = find_neighbors(
        coords,
        slice_ids,
        cross_slice_selection_mode=CrossSliceSelectionMode.PHYSICAL_WINDOW,
        cross_slice_max_distance=20.0,
        cross_slice_z_scale=4.0,
    )

    # Raw z=5 is 20 physical units away and qualifies; raw z=25 is 100 away.
    assert set(neighbors[0, 2:4]) == {2, 3}
    assert np.all(neighbors[0, 4:6] == -1)
    assert set(neighbors[2, 2:4]) == {0, 1}
    assert np.all(neighbors[2, 4:6] == -1)
    assert np.all(neighbors[4, 2:6] == -1)


def test_ordinal_mode_preserves_boundary_compensation() -> None:
    coords, _ = irregular_fixture()
    ordinal_ids = np.repeat(np.arange(3, dtype=np.int32), 2)
    implicit = find_neighbors(coords, ordinal_ids)
    explicit = find_neighbors(
        coords,
        ordinal_ids,
        cross_slice_selection_mode=CrossSliceSelectionMode.ORDINAL,
    )

    np.testing.assert_array_equal(implicit, explicit)
    # The legacy rule fills both cross-section slots at the boundary, including
    # the physically distant third section.
    assert set(implicit[0, 2:4]) == {2, 3}
    assert set(implicit[0, 4:6]) == {4, 5}


def test_physical_window_requires_a_z_coordinate() -> None:
    coords, slice_ids = irregular_fixture()
    with pytest.raises(ValueError, match="exactly three coordinate"):
        find_neighbors(
            coords[:, :2],
            slice_ids,
            cross_slice_selection_mode=CrossSliceSelectionMode.PHYSICAL_WINDOW,
            cross_slice_max_distance=20.0,
        )


def test_physical_window_always_uses_the_third_column() -> None:
    coords, slice_ids = irregular_fixture()
    neighbors = find_neighbors(
        coords,
        slice_ids,
        cross_slice_selection_mode="physical_window",
        cross_slice_max_distance=6.0,
    )
    # Every section has the same median Y, while only the first two Z positions
    # fit the window. X/Y must never be inferred as the stacking axis.
    assert set(neighbors[0, 2:4]) == {2, 3}
    assert np.all(neighbors[4, 2:6] == -1)


def test_physical_window_rejects_extra_coordinate_columns() -> None:
    coords, slice_ids = irregular_fixture()
    with pytest.raises(ValueError, match="exactly three coordinate"):
        find_neighbors(
            np.column_stack((coords, np.zeros(len(coords)))),
            slice_ids,
            cross_slice_selection_mode="physical_window",
            cross_slice_max_distance=20.0,
        )

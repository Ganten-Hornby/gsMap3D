import numpy as np
import pytest

from gsMap.latent2gene.row_ordering_jax import optimize_spatial_row_order


def test_spatial_row_order_is_a_deterministic_permutation():
    rng = np.random.default_rng(17)
    coords = rng.normal(size=(100, 3))

    first = optimize_spatial_row_order(coords)
    second = optimize_spatial_row_order(coords)

    assert first.dtype == np.int32
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(coords.shape[0]))


def test_spatial_row_order_supports_2d_constant_and_missing_axes():
    coords = np.array(
        [
            [0.0, 2.0],
            [1.0, 2.0],
            [np.nan, 2.0],
            [3.0, 2.0],
        ]
    )

    order = optimize_spatial_row_order(coords)

    assert np.array_equal(np.sort(order), np.arange(coords.shape[0]))


def test_spatial_row_order_preserves_input_order_for_coordinate_ties():
    coords = np.ones((5, 3), dtype=np.float32)

    assert np.array_equal(optimize_spatial_row_order(coords), np.arange(5, dtype=np.int32))


def test_spatial2d_orders_each_section_independently():
    coords = np.asarray([[4.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    slice_ids = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.int32)

    order = optimize_spatial_row_order(
        coords,
        slice_ids=slice_ids,
        dataset_type="spatial2D",
    )

    assert np.array_equal(slice_ids[order], np.asarray([0, 0, 0, 1, 1, 1]))
    for slice_id in (0, 1):
        section_rows = np.flatnonzero(slice_ids == slice_id)
        local_order = optimize_spatial_row_order(coords[section_rows])
        observed_rows = order[slice_ids[order] == slice_id]
        assert np.array_equal(observed_rows, section_rows[local_order])


def test_spatial3d_uses_physical_z_instead_of_slice_labels():
    coords = np.asarray(
        [
            [0.0, 0.0, 30.0],
            [2.0, 0.0, 10.0],
            [1.0, 0.0, 20.0],
            [3.0, 0.0, 0.0],
        ]
    )
    first_labels = np.asarray([0, 1, 2, 3], dtype=np.int32)
    permuted_labels = np.asarray([2, 0, 3, 1], dtype=np.int32)

    first = optimize_spatial_row_order(
        coords,
        slice_ids=first_labels,
        dataset_type="spatial3D",
        z_coordinate=2,
        z_scale=2.5,
    )
    second = optimize_spatial_row_order(
        coords,
        slice_ids=permuted_labels,
        dataset_type="spatial3D",
        z_coordinate=-1,
        z_scale=2.5,
    )

    assert np.array_equal(first, second)


def test_spatial3d_xy_only_falls_back_to_ordinal_slice_rank():
    coords = np.asarray([[0.0, 2.0], [3.0, 0.0], [1.0, 3.0], [2.0, 1.0]])
    slice_ids = np.asarray([10, 30, 10, 20], dtype=np.int32)
    _, ordinal_z = np.unique(slice_ids, return_inverse=True)

    observed = optimize_spatial_row_order(
        coords,
        slice_ids=slice_ids,
        dataset_type="spatial3D",
    )
    expected = optimize_spatial_row_order(np.column_stack((coords, ordinal_z)))

    assert np.array_equal(observed, expected)


def test_spatial3d_constant_or_incomplete_z_falls_back_to_slice_rank():
    slice_ids = np.asarray([0, 1, 0, 1], dtype=np.int32)
    xy = np.asarray([[0.0, 2.0], [3.0, 0.0], [1.0, 3.0], [2.0, 1.0]])
    _, ordinal_z = np.unique(slice_ids, return_inverse=True)
    expected = optimize_spatial_row_order(np.column_stack((xy, ordinal_z)))

    constant = optimize_spatial_row_order(
        np.column_stack((xy, np.ones(4))),
        slice_ids=slice_ids,
        dataset_type="spatial3D",
    )
    incomplete = optimize_spatial_row_order(
        np.column_stack((xy, [0.0, 1.0, np.nan, 2.0])),
        slice_ids=slice_ids,
        dataset_type="spatial3D",
    )

    assert np.array_equal(constant, expected)
    assert np.array_equal(incomplete, expected)


@pytest.mark.parametrize(
    ("coords", "slice_ids", "kwargs", "error", "match"),
    [
        (np.ones((3, 2)), None, {}, ValueError, "requires slice IDs"),
        (np.ones((3, 2)), np.zeros(2), {}, ValueError, "must have shape"),
        (
            np.ones((3, 3)),
            np.zeros(3),
            {"z_coordinate": 3},
            IndexError,
            "outside",
        ),
        (
            np.ones((3, 3)),
            np.zeros(3),
            {"z_coordinate": 1},
            ValueError,
            "distinct from x/y",
        ),
        (
            np.ones((3, 3)),
            np.zeros(3),
            {"z_scale": 0.0},
            ValueError,
            "must be positive",
        ),
    ],
)
def test_dataset_aware_spatial_order_validates_geometry(
    coords,
    slice_ids,
    kwargs,
    error,
    match,
):
    with pytest.raises(error, match=match):
        optimize_spatial_row_order(
            coords,
            slice_ids=slice_ids,
            dataset_type="spatial3D",
            **kwargs,
        )

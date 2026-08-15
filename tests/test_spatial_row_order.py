import numpy as np

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

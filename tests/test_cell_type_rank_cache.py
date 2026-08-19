import numpy as np

from gsMap.latent2gene.marker_scores import (
    _automatic_cached_marker_batch_size,
    compute_marker_scores_3d_from_cache_jax,
    compute_marker_scores_3d_hierarchical_pool_jax,
    compute_marker_scores_from_cache_jax,
    compute_marker_scores_jax,
)


def test_cached_standard_kernel_matches_direct_kernel_exactly():
    ranks = np.log1p(np.arange(8 * 6, dtype=np.float16).reshape(8, 6))
    local_neighbors = np.array([[0, 2, 1], [3, 1, 2]], dtype=np.int32)
    weights = np.array([[0.5, 0.3, 0.2], [0.25, 0.25, 0.5]], dtype=np.float16)
    global_mean = np.zeros(6, dtype=np.float32)
    global_fraction = np.zeros(6, dtype=np.float32)

    direct = compute_marker_scores_jax(
        ranks[local_neighbors.reshape(-1)],
        weights,
        2,
        3,
        global_mean,
        global_fraction,
        False,
    )
    cached = compute_marker_scores_from_cache_jax(
        ranks,
        local_neighbors,
        weights,
        2,
        3,
        global_mean,
        global_fraction,
        False,
    )

    assert np.array_equal(np.asarray(cached), np.asarray(direct))


def test_cached_hierarchical_kernel_matches_direct_kernel_exactly():
    ranks = np.log1p(np.arange(9 * 6, dtype=np.float16).reshape(9, 6))
    local_neighbors = np.array([[0, 2, 1, 4], [3, 1, 2, 5]], dtype=np.int32)
    weights = np.array([[0.5, 0.5, 0.75, 0.25], [0.25, 0.75, 0.6, 0.4]], dtype=np.float16)
    global_mean = np.zeros(6, dtype=np.float32)
    global_fraction = np.zeros(6, dtype=np.float32)

    direct = compute_marker_scores_3d_hierarchical_pool_jax(
        ranks[local_neighbors.reshape(-1)],
        weights,
        2,
        2,
        2,
        global_mean,
        global_fraction,
        False,
    )
    cached = compute_marker_scores_3d_from_cache_jax(
        ranks,
        local_neighbors,
        weights,
        2,
        2,
        2,
        global_mean,
        global_fraction,
        False,
    )

    assert np.array_equal(np.asarray(cached), np.asarray(direct))


def test_cached_batch_size_uses_a100_specific_measured_target():
    assert _automatic_cached_marker_batch_size(12_253, np.float16, 100, "NVIDIA A100") == 200
    assert _automatic_cached_marker_batch_size(12_253, np.float16, 100, "NVIDIA A40") == 50
    assert _automatic_cached_marker_batch_size(12_253, np.float16, 100, "cpu") == 50

import numpy as np

from gsMap.latent2gene.marker_scores import (
    _automatic_cached_marker_batch_size,
    _build_cell_type_rank_cache,
    compute_marker_scores_3d_from_cache_jax,
    compute_marker_scores_3d_hierarchical_pool_jax,
    compute_marker_scores_from_cache_jax,
    compute_marker_scores_jax,
)


def test_cell_type_rank_cache_reconstructs_exact_neighbor_rows():
    ranks = np.arange(7 * 5, dtype=np.float16).reshape(7, 5)
    neighbors = np.array([[4, 1, 4], [2, 1, 6]], dtype=np.int32)

    cache = _build_cell_type_rank_cache(
        ranks,
        neighbors,
        max_cache_bytes=1_000_000,
        read_workers=2,
    )

    assert cache is not None
    reconstructed = np.asarray(cache.device_ranks)[cache.local_neighbor_indices]
    assert np.array_equal(reconstructed, ranks[neighbors])
    assert cache.unique_rows == 4
    assert cache.cache_bytes == 4 * ranks.shape[1] * ranks.dtype.itemsize


def test_cell_type_rank_cache_respects_byte_cap():
    ranks = np.arange(7 * 5, dtype=np.float16).reshape(7, 5)
    neighbors = np.array([[4, 1, 4], [2, 1, 6]], dtype=np.int32)

    cache = _build_cell_type_rank_cache(
        ranks,
        neighbors,
        max_cache_bytes=39,
        read_workers=1,
    )

    assert cache is None


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

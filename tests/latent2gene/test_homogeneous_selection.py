from __future__ import annotations

import numpy as np
import pytest

from gsMap.latent2gene.connectivity import (
    _find_homogeneous_3d_memory_efficient,
    _find_homogeneous_spot_dual_embedding_batch_jit,
    find_spatial_neighbors_with_slices,
)


def _selection_fixture() -> tuple[np.ndarray, ...]:
    query_niche = np.asarray([[1.0, 0.0]], dtype=np.float32)
    query_cell = np.asarray([[1.0, 0.0]], dtype=np.float32)
    spatial_neighbors = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    all_niche = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    all_cell = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
            [0.9, np.sqrt(0.19)],
        ],
        dtype=np.float32,
    )
    return query_niche, query_cell, spatial_neighbors, all_niche, all_cell


def _select(hard_niche_filter: bool) -> tuple[np.ndarray, ...]:
    query_niche, query_cell, spatial_neighbors, all_niche, all_cell = _selection_fixture()
    result = _find_homogeneous_spot_dual_embedding_batch_jit(
        emb_niche_batch_norm=query_niche,
        emb_indv_batch_norm=query_cell,
        spatial_neighbors=spatial_neighbors,
        all_emb_niche_norm=all_niche,
        all_emb_indv_norm=all_cell,
        homogeneous_neighbors=3,
        cell_embedding_similarity_threshold=0.0,
        spatial_domain_similarity_threshold=0.6,
        hard_niche_filter=hard_niche_filter,
    )
    return tuple(np.asarray(value) for value in result)


def test_soft_priority_fills_remaining_neighbor_slots() -> None:
    indices, cell_sims, niche_sims = _select(hard_niche_filter=False)

    np.testing.assert_array_equal(indices[0, :2], [1, 2])
    assert np.count_nonzero(cell_sims > 0) == 3
    assert niche_sims[0, 2] < 0.6


def test_hard_niche_filter_reproduces_historical_zero_weight_rule() -> None:
    indices, cell_sims, niche_sims = _select(hard_niche_filter=True)

    np.testing.assert_array_equal(indices[0, :2], [1, 2])
    np.testing.assert_allclose(cell_sims[0, :2], [0.8, 0.6], atol=1e-6)
    assert np.count_nonzero(cell_sims > 0) == 2
    assert cell_sims[0, 2] == 0
    assert niche_sims[0, 2] < 0.6


def _normalized_vectors(similarities, dtype=np.float32):
    similarities = np.asarray(similarities, dtype=dtype)
    return np.column_stack((similarities, np.sqrt(1 - similarities**2))).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize(
    ("excluded_similarity", "threshold"), [(0.0, 0.0), (-0.5, 0.0), (0.25, 0.5)]
)
def test_zero_weight_niche_match_cannot_displace_positive_fallback(
    dtype, excluded_similarity, threshold
) -> None:
    # Candidate 0 passes the niche threshold but cannot contribute any weight.
    # Candidate 1 fails the niche threshold but is a usable soft-priority fallback.
    indices, weights, niches = _find_homogeneous_spot_dual_embedding_batch_jit(
        _normalized_vectors([1.0], dtype),
        _normalized_vectors([1.0], dtype),
        np.array([[0, 1]], dtype=np.int32),
        _normalized_vectors([1.0, 0.0], dtype),
        _normalized_vectors([excluded_similarity, 0.75], dtype),
        homogeneous_neighbors=1,
        cell_embedding_similarity_threshold=threshold,
        spatial_domain_similarity_threshold=0.6,
    )
    np.testing.assert_array_equal(indices, [[1]])
    np.testing.assert_allclose(weights, [[0.75]])
    np.testing.assert_array_equal(niches, [[0.0]])


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("hard_niche_filter", [False, True])
@pytest.mark.parametrize("has_eligible_neighbor", [False, True])
def test_ineligible_selected_slots_are_explicitly_missing(
    dtype, hard_niche_filter, has_eligible_neighbor
) -> None:
    # Positive similarity exactly at the configured threshold remains eligible.
    # Zero, negative, below-threshold and padding candidates must be missing.
    indices, weights, niches = _find_homogeneous_spot_dual_embedding_batch_jit(
        _normalized_vectors([1.0], dtype),
        _normalized_vectors([1.0], dtype),
        np.array([[0, 1, 2, 3, 4, -1]], dtype=np.int32),
        _normalized_vectors([1.0] * 5, dtype),
        _normalized_vectors([0.5 if has_eligible_neighbor else 0.0, 0.0, -0.5, 0.25, 0.0], dtype),
        homogeneous_neighbors=6,
        cell_embedding_similarity_threshold=0.5,
        spatial_domain_similarity_threshold=0.6,
        hard_niche_filter=hard_niche_filter,
    )
    first_index = 0 if has_eligible_neighbor else -1
    np.testing.assert_array_equal(indices, [[first_index, -1, -1, -1, -1, -1]])
    np.testing.assert_array_equal(
        weights, [[0.5 if has_eligible_neighbor else 0.0, 0, 0, 0, 0, 0]]
    )
    np.testing.assert_array_equal(niches, [[1.0 if has_eligible_neighbor else 0.0, 0, 0, 0, 0, 0]])


def test_selection_preserves_high_quality_annotation_and_section_pool() -> None:
    coords = np.array([[0, 0], [1, 0], [2, 0], [0, 0], [1, 0]], dtype=np.float32)
    cell_mask = np.array([True, True, True, False, True])
    neighbors, _ = find_spatial_neighbors_with_slices(
        coords,
        slice_ids=np.array([0, 0, 0, 1, 1]),
        query_cell_mask=cell_mask,
        high_quality_cell_mask=np.array([False, True, True, True, False]),
        k_central=3,
        n_adjacent_slices=0,
    )
    embeddings = _normalized_vectors([1.0] * 5)
    indices, weights, _ = _find_homogeneous_spot_dual_embedding_batch_jit(
        embeddings[cell_mask], embeddings[cell_mask], neighbors, embeddings, embeddings, 3
    )
    indices, weights = np.asarray(indices), np.asarray(weights)
    for row in indices[:3]:
        assert set(row[row >= 0]) == {1, 2}
    # Section 1 has an HQ bin of another annotation, but no eligible same-label bin.
    # Neither borrowing from section 0 nor a low-quality self fallback is allowed.
    np.testing.assert_array_equal(indices[-1], [-1, -1, -1])
    np.testing.assert_array_equal(weights[-1], [0, 0, 0])


def test_constrained_3d_selection_applies_eligibility_in_each_slice() -> None:
    query = _normalized_vectors([1.0], np.float16)
    indices, weights, niches = _find_homogeneous_3d_memory_efficient(
        emb_niche_masked_jax=query,
        emb_indv_masked_jax=query,
        spatial_neighbors=np.array([[0, 1, -1, 2, 3, -1, -1, -1, -1]], dtype=np.int32),
        all_emb_niche_norm_jax=_normalized_vectors([1.0, 0.0, 1.0, 0.0], np.float16),
        all_emb_indv_norm_jax=_normalized_vectors([0.0, 0.5, -0.5, 0.75], np.float16),
        num_homogeneous_per_slice=1,
        k_central=3,
        k_adjacent=3,
        n_adjacent_slices=1,
        cell_embedding_similarity_threshold=0.0,
        spatial_domain_similarity_threshold=0.6,
        hard_niche_filter=False,
        find_homogeneous_batch_size=1,
    )
    np.testing.assert_array_equal(indices, [[1, 3, -1]])
    np.testing.assert_array_equal(weights, [[0.5, 0.75, 0.0]])
    np.testing.assert_array_equal(niches, [[0.0, 0.0, 0.0]])

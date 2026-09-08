from __future__ import annotations

import numpy as np

from gsMap.latent2gene.connectivity import (
    _find_homogeneous_spot_dual_embedding_batch_jit,
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

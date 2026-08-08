from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from gsMap.latent2gene.marker_scores import MarkerScoreCalculator


def test_clear_homogeneous_outputs_removes_only_recalculated_fields() -> None:
    adata = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b", "c"]),
    )
    adata.obsm["emb"] = np.ones((3, 2), dtype=np.float32)
    adata.obsm["gsMap_homo_indices"] = np.full((3, 4), 7, dtype=np.int32)
    adata.obsm["gsMap_homo_cell_sims"] = np.ones((3, 4), dtype=np.float32)
    adata.obsm["gsMap_homo_niche_sims"] = np.ones((3, 4), dtype=np.float32)
    adata.obs["gsMap_homo_neighbor_count"] = 4
    adata.obs["gsMap_homo_cell_sims_median"] = 0.5
    adata.obs["gsMap_homo_niche_sims_median"] = 0.5

    MarkerScoreCalculator._clear_homogeneous_outputs(adata)

    assert set(adata.obsm) == {"emb"}
    assert not any(column.startswith("gsMap_homo_") for column in adata.obs)


def test_unprocessed_rows_receive_explicit_invalid_neighbor_indices() -> None:
    adata = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b", "c"]),
    )
    calculator = object.__new__(MarkerScoreCalculator)
    neighbor_indices = np.asarray([[0, 2, -1], [2, 0, 1]], dtype=np.int32)
    cell_sims = np.asarray([[1.0, 0.7, 0.0], [1.0, 0.6, 0.2]], dtype=np.float32)
    niche_sims = np.asarray([[1.0, 0.8, 0.0], [1.0, 0.7, 0.4]], dtype=np.float32)

    calculator._save_homogeneous_data_to_adata(
        adata=adata,
        neighbor_indices=neighbor_indices,
        cell_sims=cell_sims,
        niche_sims=niche_sims,
        cell_indices_sorted=np.asarray([0, 2]),
        has_real_niche_embedding=True,
    )

    np.testing.assert_array_equal(adata.obsm["gsMap_homo_indices"][1], [-1, -1, -1])
    np.testing.assert_array_equal(adata.obsm["gsMap_homo_cell_sims"][1], [0, 0, 0])
    np.testing.assert_array_equal(adata.obsm["gsMap_homo_niche_sims"][1], [0, 0, 0])
    assert adata.obs.loc["b", "gsMap_homo_neighbor_count"] == 0
    assert np.isnan(adata.obs.loc["b", "gsMap_homo_cell_sims_median"])
    assert np.isnan(adata.obs.loc["b", "gsMap_homo_niche_sims_median"])

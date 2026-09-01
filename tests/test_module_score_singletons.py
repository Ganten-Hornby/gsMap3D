from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from gsMap.find_latent import st_process


def test_calculate_module_score_keeps_singleton_annotation(monkeypatch):
    adata = ad.AnnData(
        X=np.array(
            [
                [4.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
                [5.0, 1.0, 0.0],
                [0.0, 4.0, 1.0],
            ]
        ),
        obs=pd.DataFrame(
            {"cell_type": pd.Categorical(["common", "common", "common", "singleton"])},
            index=["spot0", "spot1", "spot2", "spot3"],
        ),
        var=pd.DataFrame(index=["gene0", "gene1", "gene2"]),
    )
    requested_groups: list[str] = []

    def fake_rank_genes_groups(adata, *, groups, **kwargs):
        requested_groups.extend(groups)
        adata.uns["rank_genes_groups"] = {}

    def fake_score_genes(adata, *, score_name, **kwargs):
        adata.obs[score_name] = 1.0

    monkeypatch.setattr(st_process.sc.tl, "rank_genes_groups", fake_rank_genes_groups)
    monkeypatch.setattr(st_process.sc.tl, "score_genes", fake_score_genes)
    monkeypatch.setattr(st_process, "filter_significant_degs", lambda *args: ["gene0"])

    result = st_process.calculate_module_score(adata, "cell_type")

    assert requested_groups == ["common"]
    assert "singleton" in result.obs["cell_type"].cat.categories
    np.testing.assert_array_equal(result.obs["singleton_module_score"].to_numpy(), 0.0)

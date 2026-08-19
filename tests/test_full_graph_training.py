from collections import OrderedDict
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from gsMap.find_latent.gnn.gcn import GCN
from gsMap.find_latent.st_process import (
    TrainingData,
    create_subsampled_adata,
    propagate_spatial_features,
)


def _full_section_fixture() -> anndata.AnnData:
    coordinates = []
    sections = []
    for section_index in range(20):
        xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
        coordinates.append(
            np.column_stack((xy, np.full(xy.shape[0], section_index * 0.25)))
        )
        sections.extend([f"section-{section_index:02d}"] * xy.shape[0])

    n_obs = len(sections)
    expression = np.column_stack(
        (
            np.arange(n_obs, dtype=np.float32),
            np.square(np.arange(n_obs, dtype=np.float32)) / n_obs,
        )
    )
    obs_names = [f"spot-{index:03d}" for index in range(n_obs)]
    obs = pd.DataFrame(
        {
            "section": sections,
            "spatial_domain": np.where(np.arange(n_obs) % 2, "region-b", "region-a"),
        },
        index=obs_names,
    )
    adata = anndata.AnnData(
        X=sp.csr_matrix(expression),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    adata.obsm["spatial"] = np.concatenate(coordinates)
    return adata


def _params(graph_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        K=2,
        annotation="spatial_domain",
        spatial_key="spatial",
        graph_mode=graph_mode,
        graph_section_key="section",
        graph_z_scale=1.0,
        graph_radius=None,
        graph_sigma=None,
        n_neighbors=2,
        inference_chunk_size=6,
        data_layer="X",
        do_sampling=True,
    )


@pytest.mark.parametrize("graph_mode", ["sectionwise_2d", "section_aware_3d"])
def test_sampled_training_rows_use_full_density_graph_context(monkeypatch, graph_mode):
    full_adata = _full_section_fixture()
    full_expression = full_adata.X.toarray()
    monkeypatch.setattr(
        "gsMap.find_latent.st_process.sc.read_h5ad",
        lambda _path: full_adata.copy(),
    )

    dense_row_counts = []
    original_toarray = sp.csr_matrix.toarray

    def record_toarray(matrix, *args, **kwargs):
        dense_row_counts.append(matrix.shape[0])
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", record_toarray)
    params = _params(graph_mode)
    sources = OrderedDict([("sample", "full.h5ad")])
    np.random.seed(2024)
    sampled = create_subsampled_adata(sources, [12], params)

    training = TrainingData(params)
    training.prepare(sampled, ["g0", "g1"], source_h5ad_dict=sources)

    expected, _ = propagate_spatial_features(
        expression=torch.from_numpy(full_expression),
        coords=np.asarray(full_adata.obsm["spatial"]),
        stable_ids=full_adata.obs_names.to_numpy(),
        params=params,
        gcov=GCN(params.K),
        section_labels=full_adata.obs["section"].to_numpy(),
    )
    source_rows = sampled.obs["_gsmap_source_obs_index"].to_numpy(dtype=np.int64)
    torch.testing.assert_close(training.expression_gcn_merge, expected[source_rows])

    sampled_only, _ = propagate_spatial_features(
        expression=torch.from_numpy(sampled.X.toarray()),
        coords=np.asarray(sampled.obsm["spatial"]),
        stable_ids=sampled.obs_names.to_numpy(),
        params=params,
        gcov=GCN(params.K),
        section_labels=sampled.obs["section"].to_numpy(),
    )
    assert not torch.allclose(training.expression_gcn_merge, sampled_only)
    assert max(dense_row_counts) < full_adata.n_obs

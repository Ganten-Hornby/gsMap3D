from types import SimpleNamespace

import anndata
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder

from gsMap.find_latent.gnn.gcn import GCN
from gsMap.find_latent.st_process import (
    InferenceData,
    iter_section_inference_chunks,
    propagate_spatial_features,
)


def _section_fixture(n_sections=12):
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.25, 0.8]])
    coords = []
    sections = []
    stable_ids = []
    for section_index in range(n_sections):
        section = f"section-{section_index:02d}"
        section_xy = xy + np.asarray([0.03 * section_index, 0.0])
        section_z = np.full((xy.shape[0], 1), 0.2 * section_index)
        coords.append(np.column_stack((section_xy, section_z)))
        sections.extend([section] * xy.shape[0])
        stable_ids.extend(f"{section}-spot-{spot}" for spot in range(xy.shape[0]))
    return np.concatenate(coords), np.asarray(sections), np.asarray(stable_ids)


def _params(graph_mode, propagation_depth):
    return SimpleNamespace(
        K=propagation_depth,
        graph_mode=graph_mode,
        graph_z_scale=1.0,
        graph_radius=None,
        graph_sigma=None,
        n_neighbors=2,
    )


def test_section_inference_chunks_cover_each_focal_once_without_full_context():
    coords, sections, _ = _section_fixture()

    chunks = list(
        iter_section_inference_chunks(
            coords,
            sections,
            graph_mode="section_aware_3d",
            target_focal_spots=6,
            propagation_depth=2,
        )
    )

    focal_indices = np.concatenate([chunk.focal_global_indices for chunk in chunks])
    np.testing.assert_array_equal(np.sort(focal_indices), np.arange(coords.shape[0]))
    assert np.unique(focal_indices).size == coords.shape[0]
    assert max(chunk.context_global_indices.size for chunk in chunks) < coords.shape[0]


@pytest.mark.parametrize("graph_mode", ["sectionwise_2d", "section_aware_3d"])
def test_chunked_section_propagation_matches_full_graph(graph_mode):
    coords, sections, stable_ids = _section_fixture()
    generator = torch.Generator().manual_seed(2026)
    expression = torch.rand((coords.shape[0], 5), generator=generator)
    params = _params(graph_mode, propagation_depth=2)

    expected, _ = propagate_spatial_features(
        expression=expression,
        coords=coords,
        stable_ids=stable_ids,
        params=params,
        gcov=GCN(params.K),
        section_labels=sections,
    )
    observed = torch.empty_like(expected)

    for chunk in iter_section_inference_chunks(
        coords,
        sections,
        graph_mode=graph_mode,
        target_focal_spots=6,
        propagation_depth=params.K,
    ):
        context = chunk.context_global_indices
        propagated, _ = propagate_spatial_features(
            expression=expression[context],
            coords=coords[context],
            stable_ids=stable_ids[context],
            params=params,
            gcov=GCN(params.K),
            section_labels=sections[context],
        )
        observed[chunk.focal_global_indices] = propagated[chunk.focal_local_indices]

    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)


def test_3d_chunk_halo_preserves_fallback_edges_and_normalized_degrees():
    xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    z_values = [-0.5, 0.0, 1.5, 2.0, 3.0]
    coords = np.concatenate([np.column_stack((xy, np.full(3, z_value))) for z_value in z_values])
    sections = np.repeat([f"section-{index}" for index in range(len(z_values))], 3)
    stable_ids = np.asarray(
        [f"section-{section}-spot-{spot}" for section in range(len(z_values)) for spot in range(3)]
    )
    expression = torch.arange(coords.shape[0] * 2, dtype=torch.float32).reshape(-1, 2) / 10
    params = _params("section_aware_3d", propagation_depth=1)

    expected, _ = propagate_spatial_features(
        expression,
        coords,
        stable_ids,
        params,
        GCN(params.K),
        sections,
    )
    first_chunk = next(
        iter_section_inference_chunks(
            coords,
            sections,
            graph_mode="section_aware_3d",
            target_focal_spots=3,
            propagation_depth=params.K,
        )
    )
    context = first_chunk.context_global_indices
    observed, _ = propagate_spatial_features(
        expression[context],
        coords[context],
        stable_ids[context],
        params,
        GCN(params.K),
        sections[context],
    )

    torch.testing.assert_close(
        observed[first_chunk.focal_local_indices],
        expected[first_chunk.focal_global_indices],
        rtol=1e-6,
        atol=1e-7,
    )


class _IdentityEmbeddingModel(torch.nn.Module):
    def encode(self, x_list, batch):
        del batch
        return [x_list[0][:, :2], x_list[1][:, :2]]


def test_section_inference_never_densifies_the_full_expression_matrix(monkeypatch, tmp_path):
    coords, sections, stable_ids = _section_fixture()
    expression = sp.csr_matrix(
        np.arange(coords.shape[0] * 5, dtype=np.float32).reshape(coords.shape[0], 5)
    )
    adata = anndata.AnnData(expression)
    adata.obs_names = stable_ids
    adata.var_names = [f"gene-{index}" for index in range(5)]
    adata.obsm["spatial"] = coords
    sample_h5ad_dict = {}
    for section in np.unique(sections):
        section_adata = adata[sections == section].copy()
        section_path = tmp_path / f"{section}.h5ad"
        section_adata.write_h5ad(section_path)
        sample_h5ad_dict[section] = section_path

    dense_row_counts = []
    original_toarray = sp.csr_matrix.toarray

    def record_toarray(matrix, *args, **kwargs):
        dense_row_counts.append(matrix.shape[0])
        return original_toarray(matrix, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", record_toarray)
    params = SimpleNamespace(
        K=2,
        graph_mode="section_aware_3d",
        graph_z_scale=1.0,
        graph_radius=None,
        graph_sigma=None,
        n_neighbors=2,
        inference_chunk_size=6,
        embedding_size=2,
        spatial_key="spatial",
        data_layer="X",
        latent_dir=tmp_path,
        latent_representation_cell="emb_cell",
        latent_representation_niche="emb_niche",
    )

    observed = InferenceData(
        adata.var_names.tolist(),
        batch_size=len(sample_h5ad_dict),
        model=_IdentityEmbeddingModel(),
        label_name=None,
        params=params,
        batch_label_encoder=LabelEncoder().fit(list(sample_h5ad_dict)),
    )
    observed = dict(observed.infer_embedding_sections(sample_h5ad_dict))

    assert dense_row_counts
    assert max(dense_row_counts) < adata.n_obs
    assert sum(item.n_obs for item in observed.values()) == adata.n_obs
    assert all(item.obsm["emb_cell"].shape == (item.n_obs, 2) for item in observed.values())
    assert all(item.obsm["emb_niche"].shape == (item.n_obs, 2) for item in observed.values())
    assert all(np.isfinite(item.obsm["emb_cell"]).all() for item in observed.values())
    assert all(np.isfinite(item.obsm["emb_niche"]).all() for item in observed.values())

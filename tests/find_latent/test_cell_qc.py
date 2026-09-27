from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder

from gsMap.find_latent.gnn.gcn import GCN
from gsMap.find_latent.section_data import SectionDataSource, section_source
from gsMap.find_latent.st_process import (
    InferenceData,
    TrainingData,
    create_subsampled_adata,
    load_ordered_section_graph_data,
    propagate_spatial_features,
)


def _write(path, encoding="csr", layer="X", z=0):
    x = np.asarray(
        [
            [0, 0, 0, 100],
            [4, 1, 0, 0],
            [3, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 2, 3, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    matrix = {"csr": sp.csr_matrix, "csc": sp.csc_matrix, "dense": np.asarray}[encoding](x)
    data = ad.AnnData(matrix, obs=pd.DataFrame(index=[f"cell{i}" for i in range(6)]))
    data.var_names = ["g0", "g1", "g2", "MT-CO1"]
    data.obsm["spatial3d"] = np.column_stack([np.arange(6), np.zeros(6), np.full(6, z)])
    if layer != "X":
        data.layers[layer] = data.X.copy()
        data.X = sp.csr_matrix(data.shape)
    data.write_h5ad(path)
    return data


def _params(tmp_path, **changes):
    settings = dict(
        dataset_type="spatial3D",
        data_layer="X",
        cell_qc_min_genes=2,
        cell_qc_min_counts=None,
        cell_qc_chunk_size=2,
        spatial_key="spatial3d",
        latent_dir=tmp_path / "out",
        annotation=None,
        do_sampling=True,
        K=2,
        graph_mode="section_aware_3d",
        n_neighbors=1,
        graph_z_scale=1,
        graph_radius=None,
        graph_sigma=None,
        inference_chunk_size=3,
        embedding_size=3,
        latent_representation_cell="emb_cell",
        latent_representation_niche="emb_niche",
    )
    return SimpleNamespace(**(settings | changes))


@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
@pytest.mark.parametrize("layer", ["X", "counts"])
def test_qc_rows_and_reads_are_consistent(tmp_path, encoding, layer):
    path = tmp_path / "slice.h5ad"
    original = _write(path, encoding, layer)
    source = SectionDataSource(path, _params(tmp_path, data_layer=layer))
    first = source.load()
    np.testing.assert_array_equal(source.keep_indices, [1, 3, 4])
    np.testing.assert_array_equal(first.obs["_gsmap_source_obs_index"], [1, 3, 4])
    np.testing.assert_array_equal(first.obsm["spatial3d"], original.obsm["spatial3d"][[1, 3, 4]])
    second = source.load(genes=["g2", "g0"])
    np.testing.assert_array_equal(second.X.toarray(), first[:, ["g2", "g0"]].X.toarray())
    assert ad.read_h5ad(path).shape == (6, 4)
    assert source.summary()["n_zero_expression"] == 1  # MT-only cell is empty after exclusion.
    assert source.summary()["n_low_genes"] == 3
    with np.load(source.cache_path, allow_pickle=False) as audit:
        np.testing.assert_array_equal(audit["n_genes"], [0, 2, 1, 3, 2, 1])
        assert audit["reasons"][0] == 3  # zero-expression AND too few genes


def test_cached_qc_skips_scan_and_invalidates_for_new_thresholds(tmp_path, monkeypatch):
    path = tmp_path / "slice.h5ad"
    _write(path)
    params = _params(tmp_path)
    first = SectionDataSource(path, params)
    first.load()
    original_hash = first.summary()["keep_sha256"]
    cached = SectionDataSource(path, params)
    monkeypatch.setattr(cached, "_scan", lambda *a, **k: pytest.fail("QC was recomputed"))
    assert cached.load().n_obs == 3
    changed = SectionDataSource(path, _params(tmp_path, cell_qc_min_genes=1))
    assert changed.cache_path != first.cache_path
    assert changed.load().n_obs == 5
    assert changed.summary()["keep_sha256"] != original_hash
    changed_counts = SectionDataSource(path, _params(tmp_path, cell_qc_min_counts=4))
    assert changed_counts.load().n_obs == 2
    np.testing.assert_array_equal(changed_counts.keep_indices, [1, 4])


def test_chunk_size_does_not_change_qc_or_population(tmp_path):
    path = tmp_path / "slice.h5ad"
    _write(path)
    a = SectionDataSource(path, _params(tmp_path, latent_dir=None, cell_qc_chunk_size=1))
    b = SectionDataSource(path, _params(tmp_path, latent_dir=None, cell_qc_chunk_size=100))
    np.testing.assert_array_equal(a.load().X.toarray(), b.load().X.toarray())
    assert a.summary()["keep_sha256"] == b.summary()["keep_sha256"]


def test_sampling_reads_only_chosen_expression_rows(tmp_path, monkeypatch):
    path = tmp_path / "slice.h5ad"
    _write(path)
    params = _params(tmp_path)
    source = section_source(path, params)
    source.summary()  # Preceding HVG pass has already established the population.
    calls = []
    read_blocks = source._blocks

    def record(matrix, rows, columns):
        calls.append(rows.copy())
        yield from read_blocks(matrix, rows, columns)

    monkeypatch.setattr(source, "_blocks", record)
    sampled = create_subsampled_adata({"slice": path}, [2], params)
    assert len(calls) == 1
    assert len(calls[0]) == 2
    np.testing.assert_array_equal(calls[0], sampled.obs["_gsmap_source_obs_index"])


def test_output_preserves_counts_layer_for_independent_latent2gene(tmp_path):
    path = tmp_path / "slice.h5ad"
    data = _write(path)
    data.layers["counts"] = data.X.copy()
    data.X.data = np.log1p(data.X.data)
    data.uns["example_provenance"] = "source"
    data.write_h5ad(path)
    result = SectionDataSource(path, _params(tmp_path)).load(all_obsm=True, original_rows=[4, 1])
    np.testing.assert_array_equal(
        result.layers["counts"].toarray(), data.layers["counts"][[4, 1], :3].toarray()
    )
    np.testing.assert_array_equal(result.obs["_gsmap_source_obs_index"], [4, 1])
    assert result.uns["example_provenance"] == "source"


def test_modified_source_invalidates_cache_and_active_population(tmp_path):
    path = tmp_path / "slice.h5ad"
    data = _write(path)
    first = SectionDataSource(path, _params(tmp_path))
    first.load()
    data.X[2, 1] = 1
    data.write_h5ad(path)
    for access in (first.load, first.observations, first.summary):
        with pytest.raises(ValueError, match="Source changed"):
            access()
    fresh = SectionDataSource(path, _params(tmp_path))
    assert fresh.cache_path != first.cache_path
    assert fresh.load().n_obs == 4


def test_counts_threshold_rejects_continuous_expression(tmp_path):
    path = tmp_path / "slice.h5ad"
    data = _write(path)
    data.X.data = np.log1p(data.X.data)
    data.write_h5ad(path)
    with pytest.raises(ValueError, match="raw-count data_layer"):
        SectionDataSource(path, _params(tmp_path, cell_qc_min_counts=1)).load()


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
def test_counts_threshold_must_be_finite_and_nonnegative(tmp_path, threshold):
    path = tmp_path / "slice.h5ad"
    _write(path)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SectionDataSource(path, _params(tmp_path, cell_qc_min_counts=threshold))


def test_empty_section_and_unseen_original_rows_fail(tmp_path):
    path = tmp_path / "slice.h5ad"
    _write(path)
    source = SectionDataSource(path, _params(tmp_path))
    source.load()
    np.testing.assert_array_equal(source.filtered_positions([4, 1]), [2, 0])
    with pytest.raises(RuntimeError, match="absent from the QC population"):
        source.filtered_positions([2])
    with pytest.raises(RuntimeError, match="absent from the QC population"):
        source.filtered_positions([99])
    empty = SectionDataSource(path, _params(tmp_path, cell_qc_min_genes=10))
    for access in (empty.load, empty.load, empty.observations, empty.summary):
        with pytest.raises(ValueError, match="Refusing to silently remove"):
            access()


class _Identity(torch.nn.Module):
    def encode(self, x_list, batch):
        return [x_list[0], x_list[1][:, :3]]


@pytest.mark.parametrize("mode", ["section_aware_3d", "sectionwise_2d", "symmetric_knn"])
def test_training_graph_and_inference_share_filtered_population(tmp_path, mode):
    paths = {"slice0": tmp_path / "s0.h5ad", "slice1": tmp_path / "s1.h5ad"}
    for i, path in enumerate(paths.values()):
        _write(path, z=i * 0.2)
    params = _params(tmp_path, graph_mode=mode)
    np.random.seed(0)
    sampled = create_subsampled_adata(paths, [2, 2], params)
    assert set(sampled.obs["_gsmap_source_obs_index"]) <= {1, 3, 4}
    assert sampled.n_obs == 4
    training = TrainingData(params)
    training.prepare(sampled, ["g0", "g1", "g2"], source_h5ad_dict=paths)
    inference = InferenceData(
        ["g0", "g1", "g2"],
        2,
        _Identity(),
        None,
        params,
        LabelEncoder().fit(list(paths)),
    )
    if mode == "section_aware_3d":
        graph_data = load_ordered_section_graph_data(paths, ["g0", "g1", "g2"], params)
        assert graph_data.expression.shape == (6, 3)
        np.testing.assert_array_equal(graph_data.offsets, [0, 3, 6])
        expected, _ = propagate_spatial_features(
            torch.from_numpy(graph_data.expression.toarray()),
            graph_data.coords,
            graph_data.stable_ids,
            params,
            GCN(params.K),
            section_labels=graph_data.section_labels,
            section_order=np.arange(2),
        )
        rows = [
            list(paths).index(name) * 3
            + section_source(paths[name], params).filtered_positions([row])[0]
            for name, row in zip(
                sampled.obs.sample_name, sampled.obs["_gsmap_source_obs_index"], strict=True
            )
        ]
        torch.testing.assert_close(training.expression_gcn_merge, expected[rows])
        outputs = dict(inference.infer_embedding_sections(paths))
    else:
        outputs = {
            name: inference.infer_embedding_single(name, path) for name, path in paths.items()
        }
    for name, output in outputs.items():
        np.testing.assert_array_equal(output.obs["_gsmap_source_obs_index"], [1, 3, 4])
        assert output.obsm["emb_cell"].shape == (3, 3)
        # Every sampled graph feature must agree with inference on the same QC graph.
        batch = sampled.obs.sample_name == name
        local_rows = section_source(paths[name], params).filtered_positions(
            sampled.obs.loc[batch, "_gsmap_source_obs_index"]
        )
        np.testing.assert_allclose(
            training.expression_gcn_merge[batch.to_numpy(), :3],
            output.obsm["emb_niche"][local_rows],
            rtol=1e-6,
        )

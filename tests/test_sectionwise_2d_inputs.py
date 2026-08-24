from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder

from gsMap.config import FindLatentRepresentationsConfig
from gsMap.find_latent.find_latent_representation import (
    write_latent_representation_outputs,
)
from gsMap.find_latent.gnn.gcn import GCN, sectionwise_2d_slice_labels
from gsMap.find_latent.st_process import (
    InferenceData,
    create_subsampled_adata,
    propagate_spatial_features,
)


def _write_slice(path, z_values):
    n_obs = len(z_values)
    adata = anndata.AnnData(
        X=sp.csr_matrix(np.ones((n_obs, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=[f"{path.stem}-spot-{index}" for index in range(n_obs)]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    adata.obsm["spatial"] = np.column_stack(
        (np.arange(n_obs, dtype=float), np.zeros(n_obs), np.asarray(z_values))
    )
    adata.write_h5ad(path)
    return adata


def test_sectionwise_2d_uses_one_slice_identity_per_h5ad_even_with_multiple_z_values():
    coords = np.asarray([[0.0, 0.0, -1.0], [1.0, 0.0, 4.0]])

    np.testing.assert_array_equal(sectionwise_2d_slice_labels(coords), [0, 0])


def test_sectionwise_2d_preserves_file_order_and_warns_on_z_count_mismatch(tmp_path, monkeypatch):
    second = tmp_path / "slice-02.h5ad"
    first = tmp_path / "slice-01.h5ad"
    _write_slice(second, [2.0, 2.5])
    _write_slice(first, [1.0, 1.0])
    warnings = []
    monkeypatch.setattr(
        "gsMap.config.find_latent_config.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )

    config = FindLatentRepresentationsConfig(
        workdir=tmp_path,
        project_name="ordered-slices",
        h5ad_path=[second, first],
        graph_mode="sectionwise_2d",
    )

    assert list(config.sample_h5ad_dict) == ["slice-02", "slice-01"]
    warning_text = "\n".join(warnings)
    assert "uses the H5AD list order as physical slice order" in warning_text
    assert "3 unique z values across 2 H5AD slice files" in warning_text


def test_section_aware_3d_uses_file_list_as_slice_identity(tmp_path, monkeypatch):
    first = tmp_path / "slice-01.h5ad"
    second = tmp_path / "slice-02.h5ad"
    _write_slice(first, [0.0, 0.1])
    _write_slice(second, [1.0, 1.0])
    warnings = []
    monkeypatch.setattr(
        "gsMap.config.find_latent_config.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )

    config = FindLatentRepresentationsConfig(
        workdir=tmp_path,
        project_name="ordered-3d-slices",
        h5ad_path=[first, second],
        graph_mode="section_aware_3d",
    )

    assert list(config.sample_h5ad_dict) == ["slice-01", "slice-02"]
    warning_text = "\n".join(warnings)
    assert "uses the H5AD list order as physical slice order" in warning_text
    assert "3 unique z values across 2 H5AD slice files" in warning_text
    assert not hasattr(config, "graph_section_key")


def test_sectionwise_2d_preserves_files_with_duplicate_stems(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "slice.h5ad"
    second = second_dir / "slice.h5ad"
    _write_slice(first, [1.0])
    _write_slice(second, [2.0])

    config = FindLatentRepresentationsConfig(
        workdir=tmp_path,
        project_name="duplicate-stems",
        h5ad_path=[first, second],
        graph_mode="sectionwise_2d",
    )

    assert list(config.sample_h5ad_dict) == ["slice", "slice__2"]
    assert list(config.sample_h5ad_dict.values()) == [first, second]


def test_sectionwise_2d_training_slice_ids_follow_input_file_order(tmp_path):
    first = tmp_path / "slice-01.h5ad"
    second = tmp_path / "slice-02.h5ad"
    first_adata = _write_slice(first, [305.0, 305.0])
    second_adata = _write_slice(second, [310.0, 310.0])
    first_adata.obs["slice_id"] = pd.Categorical([305.0, 305.0])
    second_adata.obs["slice_id"] = pd.Categorical([310.0, 310.0])
    first_adata.write_h5ad(first)
    second_adata.write_h5ad(second)
    params = SimpleNamespace(
        annotation=None,
        data_layer="X",
        do_sampling=False,
        graph_mode="sectionwise_2d",
    )

    observed = create_subsampled_adata(
        {"slice-01": first, "slice-02": second}, [2, 2], params
    )

    np.testing.assert_array_equal(observed.obs["slice_id"], [0, 0, 1, 1])
    assert observed.obs["slice_id"].dtype == np.dtype(np.int32)
    observed.write_h5ad(tmp_path / "training_adata.h5ad")


def test_sectionwise_2d_warns_when_xyz_has_no_z_values(tmp_path, monkeypatch):
    empty_slice = tmp_path / "empty.h5ad"
    _write_slice(empty_slice, [])
    warnings = []
    monkeypatch.setattr(
        "gsMap.config.find_latent_config.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )

    FindLatentRepresentationsConfig(
        workdir=tmp_path,
        project_name="empty-slice",
        h5ad_path=[empty_slice],
        graph_mode="sectionwise_2d",
    )

    assert "0 unique z values across 1 H5AD slice files" in "\n".join(warnings)


def test_sectionwise_2d_writes_one_output_for_an_xyz_h5ad(tmp_path):
    adata = _write_slice(tmp_path / "source.h5ad", [1.0, 2.0])
    config = SimpleNamespace(
        graph_mode="sectionwise_2d",
        latent_dir=tmp_path / "latent",
        spatial_key="spatial",
    )
    config.latent_dir.mkdir()

    outputs = write_latent_representation_outputs(
        adata,
        sample_name="source",
        config=config,
    )

    assert list(outputs) == ["source"]
    assert outputs["source"].name == "source_add_latent.h5ad"
    assert anndata.read_h5ad(outputs["source"]).n_obs == adata.n_obs


class _IdentityEmbeddingModel(torch.nn.Module):
    def encode(self, x_list, batch):
        del batch
        return [x_list[0][:, :2], x_list[1][:, :2]]


def test_section_aware_3d_inference_uses_cross_file_graph_context(tmp_path):
    first = tmp_path / "slice-01.h5ad"
    second = tmp_path / "slice-02.h5ad"
    first_adata = _write_slice(first, [0.0, 0.0])
    second_adata = _write_slice(second, [0.2, 0.2])
    first_adata.X = sp.csr_matrix(np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32))
    second_adata.X = sp.csr_matrix(np.asarray([[9.0, 0.0], [11.0, 0.0]], dtype=np.float32))
    first_adata.write_h5ad(first)
    second_adata.write_h5ad(second)
    sample_h5ad_dict = {"slice-01": first, "slice-02": second}
    params = SimpleNamespace(
        K=1,
        graph_mode="section_aware_3d",
        graph_z_scale=1.0,
        graph_radius=None,
        graph_sigma=None,
        n_neighbors=1,
        inference_chunk_size=2,
        embedding_size=2,
        spatial_key="spatial",
        data_layer="X",
        latent_dir=tmp_path,
        latent_representation_cell="emb_cell",
        latent_representation_niche="emb_niche",
    )
    inference = InferenceData(
        ["g0", "g1"],
        batch_size=2,
        model=_IdentityEmbeddingModel(),
        label_name=None,
        params=params,
        batch_label_encoder=LabelEncoder().fit(list(sample_h5ad_dict)),
    )

    observed = dict(inference.infer_embedding_sections(sample_h5ad_dict))

    combined_expression = torch.from_numpy(
        np.vstack((first_adata.X.toarray(), second_adata.X.toarray()))
    )
    combined_coords = np.vstack(
        (first_adata.obsm["spatial"], second_adata.obsm["spatial"])
    )
    expected, _ = propagate_spatial_features(
        combined_expression,
        combined_coords,
        np.arange(4),
        params,
        GCN(1),
        section_labels=np.repeat([0, 1], 2),
        section_order=np.asarray([0, 1]),
    )
    np.testing.assert_allclose(observed["slice-01"].obsm["emb_niche"], expected[:2])
    np.testing.assert_allclose(observed["slice-02"].obsm["emb_niche"], expected[2:])
    assert list(observed) == ["slice-01", "slice-02"]

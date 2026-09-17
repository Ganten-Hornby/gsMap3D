import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch
import yaml

from gsMap.config import FindLatentRepresentationsConfig
from gsMap.find_latent.checkpoint import read_checkpoint
from gsMap.find_latent.find_latent_representation import run_find_latent_representation


def test_train_save_reload_pipeline_uses_checkpoint_hvgs(tmp_path, monkeypatch):
    """Exercise real HVG selection, training, H5AD/metadata output and checkpoint reuse."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(42)
    paths = []
    for section in range(2):
        x = rng.poisson(0.4, size=(80, 400)).astype(np.float32)
        x[0] = 0
        data = ad.AnnData(sp.csr_matrix(x))
        data.var_names = [f"g{i:04}" for i in range(400)]
        data.obs_names = [f"s{section}_c{i}" for i in range(80)]
        data.obsm["spatial"] = rng.random((80, 2))
        path = tmp_path / f"slice{section}.h5ad"
        data.write_h5ad(path)
        paths.append(path)
    config = FindLatentRepresentationsConfig(
        workdir=tmp_path,
        project_name="smoke",
        h5ad_path=paths,
        graph_mode="sectionwise_2d",
        K=1,
        feat_cell=100,
        hidden_size=32,
        embedding_size=8,
        itermax=1,
        patience=1,
        do_sampling=False,
        two_stage=False,
        high_quality_cell_qc=False,
        batch_size=48,
    )
    metadata = run_find_latent_representation(config)
    payload = read_checkpoint(config.model_path)
    schema = payload["feature_schema"]
    assert schema == metadata["feature_schema"]
    assert len(schema["hvg_order"]) == 100
    assert schema["cell_qc"]["slice0"]["n_zero_expression"] == 1
    assert schema["hvg_selection"]["sections"]["slice0"]["n_zero_cells_excluded_from_fit"] == 0
    training = ad.read_h5ad(config.latent_dir / "training_adata.h5ad")
    assert training.uns["hvg_order"].tolist() == schema["hvg_order"]
    assert training.var.highly_variable.sum() == 100
    assert training.n_obs == 158  # The same QC population feeds training and inference.
    saved_metadata = yaml.safe_load(config.find_latent_metadata_path.read_text())
    assert saved_metadata["feature_schema"]["hvg_order"] == schema["hvg_order"]
    for output in metadata["outputs"]["latent_files"].values():
        assert ad.read_h5ad(output).n_obs == 79
    original_checkpoint = config.model_path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Existing checkpoint must reuse HVGs and skip training")

    monkeypatch.setattr("gsMap.find_latent.st_process._section_hvg", forbidden)
    monkeypatch.setattr("gsMap.find_latent.gnn.train_step.ModelTrain.train", forbidden)
    repeated = run_find_latent_representation(config)
    assert repeated["feature_schema"] == schema
    assert config.model_path.read_bytes() == original_checkpoint

    # A legacy checkpoint is rejected before selection/training and is not replaced.
    torch.save(payload["state_dict"], config.model_path)
    legacy_bytes = config.model_path.read_bytes()
    with pytest.raises(ValueError, match="legacy state_dict"):
        run_find_latent_representation(config)
    assert config.model_path.read_bytes() == legacy_bytes

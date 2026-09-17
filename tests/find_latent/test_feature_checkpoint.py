from copy import deepcopy
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
import torch

from gsMap.find_latent.checkpoint import (
    feature_hash,
    load_checkpoint_state,
    make_feature_schema,
    read_checkpoint,
    save_checkpoint,
    validate_input_contract,
)
from gsMap.find_latent.gnn.train_step import ModelTrain
from tests.find_latent.test_embedding_inference import _make_variational_model


def _schema(tmp_path):
    path = tmp_path / "input.h5ad"
    data = ad.AnnData(np.ones((2, 4), dtype=np.float32))
    data.var_names = ["a", "b", "c", "d"]
    data.write_h5ad(path)
    config = SimpleNamespace(
        sample_h5ad_dict={"first": path, "second": path},
        data_layer="X",
        feat_cell=4,
        cell_qc_min_genes=1,
        to_dict_with_paths_as_strings=lambda: {"random_seed": 42},
    )
    return make_feature_schema(["d", "b", "a", "c"], config, True, {}), config


def test_checkpoint_roundtrip_preserves_order_and_exact_mean_inference(tmp_path):
    schema, config = _schema(tmp_path)
    model = _make_variational_model().eval()
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, schema)
    payload = read_checkpoint(path)
    assert payload["feature_schema"]["hvg_order"] == ["d", "b", "a", "c"]
    validate_input_contract(payload["feature_schema"], config)
    restored = _make_variational_model().eval()
    assert load_checkpoint_state(restored, path, schema)
    inputs = [torch.rand(10, 4), torch.rand(10, 8)]
    with torch.no_grad():
        for actual, expected in zip(
            model.encode(inputs, sample=False), restored.encode(inputs, sample=False), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_same_width_different_gene_order_is_rejected(tmp_path):
    schema, _ = _schema(tmp_path)
    path = tmp_path / "model.pt"
    model = _make_variational_model()
    save_checkpoint(model, path, schema)
    changed = deepcopy(schema)
    changed["hvg_order"].reverse()
    changed["feature_sha256"] = feature_hash(changed["hvg_order"])
    with pytest.raises(ValueError, match="feature mismatch"):
        load_checkpoint_state(model, path, changed)


def test_legacy_and_corrupt_feature_schema_are_rejected(tmp_path):
    schema, _ = _schema(tmp_path)
    path = tmp_path / "model.pt"
    model = _make_variational_model()
    torch.save(model.state_dict(), path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="legacy state_dict"):
        read_checkpoint(path)
    assert path.read_bytes() == before
    schema["hvg_order"].reverse()
    with pytest.raises(ValueError, match="feature hash"):
        save_checkpoint(model, path, schema)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["section_order", "layer", "source_stats"])
def test_changed_input_contract_is_rejected(tmp_path, change):
    schema, config = _schema(tmp_path)
    if change == "section_order":
        config.sample_h5ad_dict = dict(reversed(list(config.sample_h5ad_dict.items())))
    elif change == "layer":
        config.data_layer = "counts"
    else:
        next(iter(config.sample_h5ad_dict.values())).write_bytes(b"changed input")
    with pytest.raises(ValueError, match="input sections/order"):
        validate_input_contract(schema, config)


def test_training_saves_feature_bundle(tmp_path, monkeypatch):
    schema, _ = _schema(tmp_path)
    model = _make_variational_model()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "model.pt"
    trainer = ModelTrain(
        model, torch.optim.Adam(model.parameters()), "nb", "reconstruction", 1e-3, path, schema
    )
    monkeypatch.setattr(trainer, "_mini_batch", lambda *a, **kw: torch.tensor(1.0))
    trainer.train(1, patience=1)
    assert read_checkpoint(path)["feature_schema"] == schema


def test_interrupted_checkpoint_save_preserves_previous_model(tmp_path, monkeypatch):
    schema, _ = _schema(tmp_path)
    path = tmp_path / "model.pt"
    model = _make_variational_model()
    save_checkpoint(model, path, schema)
    before = path.read_bytes()

    def interrupted(*args, **kwargs):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="interrupted write"):
        save_checkpoint(model, path, schema)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".model.pt.*"))

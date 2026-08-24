from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from gsMap.find_latent.st_process import InferenceData, TrainingData


class _IdentityEmbeddingModel(torch.nn.Module):
    def encode(self, x_list, batch):
        del batch
        return x_list


def _params(tmp_path):
    return SimpleNamespace(
        K=1,
        annotation=None,
        spatial_key="spatial",
        graph_mode="no_graph",
        graph_z_scale=1.0,
        graph_radius=None,
        graph_sigma=None,
        n_neighbors=1,
        inference_chunk_size=8,
        data_layer="X",
        latent_dir=tmp_path,
        embedding_size=2,
    )


def test_training_batch_encoder_is_reused_for_inference(tmp_path):
    batch_labels = np.asarray([f"S{index}" for index in range(12)])
    adata = anndata.AnnData(
        X=sp.csr_matrix(np.ones((batch_labels.size, 2), dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "batch_id": batch_labels,
                "label": "unknown",
            },
            index=[f"spot-{index}" for index in range(batch_labels.size)],
        ),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    adata.obsm["spatial"] = np.column_stack(
        (np.arange(batch_labels.size), np.zeros(batch_labels.size))
    )

    training = TrainingData(_params(tmp_path))
    training.prepare(adata, ["g0", "g1"])

    expected_codes = training.batch_label_encoder.transform(batch_labels)
    np.testing.assert_array_equal(training.batch_merge.numpy(), expected_codes)
    assert training.batch_merge.dtype == torch.long
    assert training.batch_size == batch_labels.size
    assert expected_codes[2] != 2  # S10 and S11 sort before S2.

    inference = InferenceData(
        hvg=["g0", "g1"],
        batch_size=training.batch_size,
        model=_IdentityEmbeddingModel(),
        label_name=None,
        params=_params(tmp_path),
        batch_label_encoder=training.batch_label_encoder,
    )
    assert inference._encode_batch_label("S2") == int(expected_codes[2])
    with pytest.raises(RuntimeError, match="was not present during training"):
        inference._encode_batch_label("unseen-sample")

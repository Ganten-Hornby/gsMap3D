from types import SimpleNamespace

import anndata
import numpy as np
import scipy.sparse as sp

from gsMap.find_latent.find_latent_representation import _resolve_expression_model
from gsMap.find_latent.st_process import find_common_hvg, setup_data_layer


def test_float32_integer_counts_in_x_enable_nb_variational_model():
    adata = anndata.AnnData(
        X=sp.csr_matrix(np.asarray([[0.0, 2.0], [3.0, 1.0]], dtype=np.float32))
    )

    is_count_data, actual_layer = setup_data_layer(adata, "X", verbose=False)
    model_config = _resolve_expression_model(is_count_data, "nb", use_tf=True)

    assert actual_layer == "X"
    assert is_count_data
    assert model_config == ("nb", True, True)


def test_continuous_x_uses_gaussian_nonvariational_model():
    adata = anndata.AnnData(
        X=sp.csr_matrix(np.asarray([[0.0, 0.25], [1.5, 2.75]], dtype=np.float32))
    )

    is_count_data, actual_layer = setup_data_layer(adata, "X", verbose=False)
    model_config = _resolve_expression_model(is_count_data, "nb", use_tf=True)

    assert actual_layer == "X"
    assert not is_count_data
    assert model_config == ("gaussian", False, False)


def test_find_common_hvg_accepts_gene_named_var_index(tmp_path, monkeypatch):
    genes = [f"gene_{i}" for i in range(400)]
    sample_paths = {}
    for sample_index in range(2):
        adata = anndata.AnnData(
            X=sp.csr_matrix(np.ones((4, len(genes)), dtype=np.float32))
        )
        adata.var_names = genes
        adata.var.index.name = "gene"
        sample_path = tmp_path / f"sample_{sample_index}.h5ad"
        adata.write_h5ad(sample_path)
        sample_paths[f"sample_{sample_index}"] = sample_path

    def fake_highly_variable_genes(adata, n_top_genes, subset):
        adata.var["residual_variances"] = np.arange(adata.n_vars, dtype=float)
        adata.var["highly_variable"] = np.arange(adata.n_vars) < n_top_genes

    monkeypatch.setattr(
        "gsMap.find_latent.st_process.sc.experimental.pp.highly_variable_genes",
        fake_highly_variable_genes,
    )
    params = SimpleNamespace(
        data_layer="X",
        feat_cell=300,
        n_cell_training=6,
        species=None,
        homolog_file=None,
    )

    hvg, n_cell_used, gene_homolog_dict, is_count_data = find_common_hvg(
        sample_paths,
        params,
    )

    assert len(hvg) == 300
    assert n_cell_used == [3, 3]
    assert len(gene_homolog_dict) == 400
    assert is_count_data

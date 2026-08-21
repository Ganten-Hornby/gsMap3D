import anndata
import numpy as np
import scipy.sparse as sp

from gsMap.find_latent.find_latent_representation import _resolve_expression_model
from gsMap.find_latent.st_process import setup_data_layer


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

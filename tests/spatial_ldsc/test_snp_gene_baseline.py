import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gsMap.spatial_ldsc.io import construct_baseline_ld_from_snp_gene_weights


def _weight_adata(matrix, var_names):
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=["rs1", "rs2"]),
        var=pd.DataFrame(index=var_names),
    )
    return adata


@pytest.mark.parametrize("use_sparse", [False, True], ids=["dense", "sparse"])
def test_construct_baseline_uses_terminal_unmapped_column(use_sparse):
    weights = np.asarray(
        [
            [1.0, 2.0, 10.0],
            [0.5, 4.0, 3.0],
        ],
        dtype=np.float32,
    )
    matrix = sp.csr_matrix(weights) if use_sparse else weights
    adata = _weight_adata(matrix, ["GENE_A", "GENE_B", "Unmapped"])

    observed = construct_baseline_ld_from_snp_gene_weights(adata)

    expected = pd.DataFrame(
        {
            "base": np.asarray([13.0, 7.5], dtype=np.float32),
            "all_gene": np.asarray([3.0, 4.5], dtype=np.float32),
        },
        index=pd.Index(["rs1", "rs2"], name="SNP"),
    )
    pd.testing.assert_frame_equal(observed, expected)


def test_construct_baseline_rejects_missing_unmapped_column():
    adata = _weight_adata(
        np.ones((2, 3), dtype=np.float32),
        ["GENE_A", "GENE_B", "GENE_C"],
    )

    with pytest.raises(ValueError, match="exactly one column named 'Unmapped'; found 0"):
        construct_baseline_ld_from_snp_gene_weights(adata)


def test_construct_baseline_rejects_nonterminal_unmapped_column():
    adata = _weight_adata(
        np.ones((2, 3), dtype=np.float32),
        ["GENE_A", "Unmapped", "GENE_B"],
    )

    with pytest.raises(ValueError, match="'Unmapped' must be last"):
        construct_baseline_ld_from_snp_gene_weights(adata)


def test_construct_baseline_requires_a_mapped_gene_column():
    adata = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["rs1", "rs2"]),
        var=pd.DataFrame(index=["Unmapped"]),
    )

    with pytest.raises(ValueError, match="at least one mapped-gene column"):
        construct_baseline_ld_from_snp_gene_weights(adata)

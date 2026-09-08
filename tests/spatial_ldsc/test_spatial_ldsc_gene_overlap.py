from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gsMap.spatial_ldsc.ldscore_quick_mode import (
    SpatialLDSCEngine,
    _align_marker_score_genes,
    _canonicalize_sparse_weight_matrix,
)


def _gene_names(count):
    return [f"GENE_{index:03d}" for index in range(count)]


def test_gene_alignment_accepts_more_than_200_exact_matches():
    common = _gene_names(201)
    marker_genes = [*common, "MARKER_ONLY"]
    weight_genes = [*reversed(common), "WEIGHT_ONLY", "Unmapped"]

    marker_mask, observed_common, weight_indices = _align_marker_score_genes(
        marker_genes,
        weight_genes,
    )

    np.testing.assert_array_equal(marker_mask, [True] * 201 + [False])
    assert observed_common.tolist() == common
    assert [weight_genes[index] for index in weight_indices] == common


def test_gene_alignment_rejects_exactly_200_matches_with_diagnostics():
    common = _gene_names(200)
    marker_genes = [*common, "marker_only_feature"]
    weight_genes = [*common, "WEIGHT_ONLY_FEATURE", "Unmapped"]

    with pytest.raises(ValueError) as error:
        _align_marker_score_genes(marker_genes, weight_genes)

    message = str(error.value)
    assert "found 200 exactly matching genes" in message
    assert "more than 200 are required" in message
    assert "exact and case-sensitive" in message
    assert "marker_only_feature" in message
    assert "WEIGHT_ONLY_FEATURE" in message


def test_gene_alignment_excludes_unmapped_from_gene_overlap():
    common = _gene_names(200)
    marker_genes = [*common, "Unmapped"]
    weight_genes = [*common, "Unmapped"]

    with pytest.raises(ValueError, match="found 200 exactly matching genes"):
        _align_marker_score_genes(marker_genes, weight_genes)


def test_sparse_weight_is_canonical_after_gene_column_reordering():
    original = sp.csr_matrix(np.array([[1, 2, 3], [4, 0, 6]], dtype=np.float32))
    reordered = original[:, [2, 0, 1]].tocsr()
    assert not reordered.has_canonical_format

    observed = _canonicalize_sparse_weight_matrix(reordered)

    assert sp.isspmatrix_csr(observed)
    assert observed.has_sorted_indices
    assert observed.has_canonical_format
    np.testing.assert_array_equal(observed.toarray(), original.toarray()[:, [2, 0, 1]])


def test_spatial_ldsc_engine_enforces_gene_overlap_threshold(tmp_path):
    common = _gene_names(200)
    marker_scores = ad.AnnData(
        X=np.ones((2, len(common)), dtype=np.float32),
        var=pd.DataFrame(index=common),
    )
    snp_gene_weights = ad.AnnData(
        X=np.ones((3, len(common) + 1), dtype=np.float32),
        var=pd.DataFrame(index=[*common, "Unmapped"]),
    )

    with pytest.raises(ValueError, match="found 200 exactly matching genes"):
        SpatialLDSCEngine(
            config=SimpleNamespace(),
            output_dir=tmp_path,
            marker_score_adata=marker_scores,
            snp_gene_weight_adata=snp_gene_weights,
            baseline_ld=pd.DataFrame(),
            w_ld=pd.DataFrame(),
        )

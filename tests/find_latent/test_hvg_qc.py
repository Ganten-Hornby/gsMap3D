from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from gsMap.find_latent.st_process import _section_hvg, find_common_hvg


def _adata(sparse=True):
    counts = np.random.default_rng(42).poisson(0.3, (80, 400)).astype(np.float32)
    counts = np.column_stack([counts, np.zeros((80, 100), dtype=np.float32)])
    result = ad.AnnData(sp.csr_matrix(counts) if sparse else counts)
    result.var_names = [f"g{i:04}" for i in range(result.n_vars)]
    return result


def _fit(adata):
    return _section_hvg(adata, is_count_data=True, n_top_genes=100, sample_name="test")


@pytest.mark.parametrize("sparse", [True, False])
def test_zero_cell_injection_does_not_change_hvg_or_scores(sparse):
    original = _adata(sparse)
    empty = ad.AnnData(
        sp.csr_matrix((1, original.n_vars)) if sparse else np.zeros((1, original.n_vars)),
        var=original.var.copy(),
    )
    dirty = ad.concat([original, empty], index_unique="-")
    before = original.X.copy()
    clean_result, _ = _fit(original)
    result, audit = _fit(dirty)
    np.testing.assert_allclose(result.residual_variances, clean_result.residual_variances)
    assert result.highly_variable.equals(clean_result.highly_variable)
    assert not result.loc[original.var_names[400:], "highly_variable"].any()
    assert np.isfinite(result.residual_variances).all()
    assert audit["n_zero_cells_excluded_from_fit"] == 1
    assert dirty.n_obs == 81
    np.testing.assert_array_equal(
        original.X.toarray() if sparse else original.X,
        before.toarray() if sparse else before,
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 0.5])
def test_invalid_count_values_fail(value):
    data = _adata(False)
    data.X[0, 0] = value
    with pytest.raises(ValueError, match="HVG input test"):
        _fit(data)


def test_all_zero_section_fails():
    data = ad.AnnData(sp.csr_matrix((10, 400)))
    with pytest.raises(ValueError, match="nonzero cells"):
        _fit(data)


def test_nonfinite_score_in_unselected_gene_fails(monkeypatch):
    def broken(adata, **kwargs):
        adata.var["highly_variable"] = False
        adata.var["residual_variances"] = 1.0
        adata.var.iloc[-1, adata.var.columns.get_loc("residual_variances")] = np.nan

    monkeypatch.setattr(
        "gsMap.find_latent.st_process.sc.experimental.pp.highly_variable_genes", broken
    )
    with pytest.raises(ValueError, match="non-finite residual_variances"):
        _fit(_adata())


def _params():
    return SimpleNamespace(data_layer="X", feat_cell=100, n_cell_training=80, species=None)


def test_multisection_ties_are_stable_and_section_specific_genes_survive(tmp_path, monkeypatch):
    def ties(adata, n_top_genes, subset, **kwargs):
        adata.var["residual_variances"] = 1.0
        adata.var["highly_variable"] = np.arange(adata.n_vars) < n_top_genes
        if "g0000" in adata.var_names:
            adata.var.loc["g0000", "residual_variances"] = 3.0

    monkeypatch.setattr(
        "gsMap.find_latent.st_process.sc.experimental.pp.highly_variable_genes", ties
    )
    first = _adata(False)
    second = first.copy()
    second.X[:, 0] = 0  # Present by identity in both, expressed only in the first.
    paths = {"first": tmp_path / "first.h5ad", "second": tmp_path / "second.h5ad"}
    first.write_h5ad(paths["first"])
    second.write_h5ad(paths["second"])
    selected, n_cells, homologs, _ = find_common_hvg(paths, _params())
    assert "g0000" in homologs
    assert "g0000" in selected
    assert not set(first.var_names[400:]) & set(selected)
    first[:, ::-1].write_h5ad(paths["first"])
    second[:, ::-1].write_h5ad(paths["second"])
    repeated = find_common_hvg(paths, _params())
    assert selected == repeated[0]
    assert n_cells == repeated[1] == [40, 40]


def test_failed_section_is_not_silently_skipped(tmp_path):
    paths = {"good": tmp_path / "good.h5ad", "bad": tmp_path / "bad.h5ad"}
    data = _adata()
    data.write_h5ad(paths["good"])
    data.X = sp.csr_matrix(data.shape)
    data.write_h5ad(paths["bad"])
    with pytest.raises(ValueError, match="Cell QC bad"):
        find_common_hvg(paths, _params())


def test_saved_hvg_is_reused_without_reselection(tmp_path, monkeypatch):
    path = tmp_path / "section.h5ad"
    data = _adata()
    data.write_h5ad(path)
    genes = data.var_names[:100][::-1].tolist()

    def forbidden(*args, **kwargs):
        raise AssertionError("Checkpoint inference must not reselect HVGs")

    monkeypatch.setattr(
        "gsMap.find_latent.st_process.sc.experimental.pp.highly_variable_genes", forbidden
    )
    assert find_common_hvg({"section": path}, _params(), saved_hvg=genes)[0] == genes
    genes[-1] = "missing"
    with pytest.raises(ValueError, match="missing from input"):
        find_common_hvg({"section": path}, _params(), saved_hvg=genes)


def test_too_few_expressed_common_genes_fails(tmp_path):
    data = _adata(False)
    data.X[:, 90:] = 0
    path = tmp_path / "section.h5ad"
    data.write_h5ad(path)
    with pytest.raises(ValueError, match="90 expressed common genes"):
        find_common_hvg({"section": path}, _params())


def test_continuous_log_expression_retains_seurat_branch():
    data = _adata(False)
    data.X = np.log1p(data.X)
    result, audit = _section_hvg(
        data, is_count_data=False, n_top_genes=100, sample_name="continuous"
    )
    assert "dispersions_norm" in result
    assert np.isfinite(result.dispersions_norm).all()
    assert not (result.highly_variable & ~result.expressed).any()
    assert audit["n_fit_genes"] == 400

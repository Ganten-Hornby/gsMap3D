"""
Tests for spatial LDSC with:
  - h5ad marker-score format (toy fixture)
  - feather marker-score format (toy fixture)
  - cell_indices_range option
  - sample_filter option
  - additional_baseline_h5ad_path_list option
"""

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gsMap.cli import app

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _read_gene_names(snp_gene_weight_path, n_genes=100):
    """Read up to n_genes gene names from the SNP-gene weight matrix."""
    w = anndata.read_h5ad(snp_gene_weight_path, backed="r")
    names = list(w.var_names[: min(n_genes, w.n_vars)])
    w.file.close()
    return names


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_h5ad_plain(tmp_path_factory, snp_gene_weight_path):
    """Small h5ad marker-score file (50 spots, genes from real weight matrix)."""
    tmp_dir = tmp_path_factory.mktemp("ldsc_plain")
    rng = np.random.default_rng(20)
    gene_names = _read_gene_names(snp_gene_weight_path)

    n_spots = 50
    adata = anndata.AnnData(X=rng.random((n_spots, len(gene_names))).astype(np.float32))
    adata.var_names = gene_names
    adata.obs_names = [f"spot_{i}" for i in range(n_spots)]

    path = tmp_dir / "marker_scores.h5ad"
    adata.write_h5ad(path)
    return path


@pytest.fixture(scope="session")
def synthetic_h5ad_with_sample(tmp_path_factory, snp_gene_weight_path):
    """h5ad marker scores with a 'sample' obs column (contiguous blocks).

    Spots 0-24 → 'sampleA', spots 25-49 → 'sampleB'.
    Contiguity is required by the sample_filter code path.
    """
    tmp_dir = tmp_path_factory.mktemp("ldsc_sample")
    rng = np.random.default_rng(21)
    gene_names = _read_gene_names(snp_gene_weight_path)

    n_spots = 50
    adata = anndata.AnnData(X=rng.random((n_spots, len(gene_names))).astype(np.float32))
    adata.var_names = gene_names
    adata.obs_names = [f"spot_{i}" for i in range(n_spots)]
    adata.obs["sample"] = ["sampleA"] * 25 + ["sampleB"] * 25

    path = tmp_dir / "marker_scores_sample.h5ad"
    adata.write_h5ad(path)
    return path


@pytest.fixture(scope="session")
def synthetic_marker_scores_feather(tmp_path_factory, snp_gene_weight_path):
    """Feather marker-score file.

    Layout expected by FeatherAnnData(index_col='HUMAN_GENE_SYM', transpose=True):
      - rows    = genes
      - columns = [HUMAN_GENE_SYM, spot_0, spot_1, ...]
    """
    import pyarrow.feather as feather

    tmp_dir = tmp_path_factory.mktemp("ldsc_feather")
    rng = np.random.default_rng(22)
    gene_names = _read_gene_names(snp_gene_weight_path)

    n_spots = 50
    spot_names = [f"spot_{i}" for i in range(n_spots)]
    data = rng.random((len(gene_names), n_spots)).astype(np.float32)

    df = pd.DataFrame(data, columns=spot_names)
    df.insert(0, "HUMAN_GENE_SYM", gene_names)

    path = tmp_dir / "marker_scores.feather"
    feather.write_feather(df, str(path))
    return path


@pytest.fixture(scope="session")
def synthetic_additional_baseline(tmp_path_factory, snp_gene_weight_path):
    """Additional baseline h5ad: obs = SNP names from weight matrix, vars = 3 annotations."""
    tmp_dir = tmp_path_factory.mktemp("ldsc_baseline")
    rng = np.random.default_rng(23)

    w = anndata.read_h5ad(snp_gene_weight_path, backed="r")
    snp_names = list(w.obs_names)
    w.file.close()

    X = sp.random(len(snp_names), 3, density=0.1, random_state=rng).tocsr().astype(np.float32)

    adata = anndata.AnnData(X=X)
    adata.obs_names = snp_names
    adata.var_names = ["annot_0", "annot_1", "annot_2"]

    path = tmp_dir / "additional_baseline.h5ad"
    adata.write_h5ad(path)
    return path


# ---------------------------------------------------------------------------
# Shared CLI arg builder
# ---------------------------------------------------------------------------


def _base_ldsc_args(work_dir, project_name):
    """Minimal quick-mode args to run only the spatial_ldsc step."""
    return [
        "quick-mode",
        "--workdir",
        str(work_dir),
        "--project-name",
        project_name,
        "--annotation",
        "annotation",
        "--start-step",
        "spatial_ldsc",
        "--stop-step",
        "spatial_ldsc",
        "--no-gpu",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.real_data
def test_spatial_ldsc_h5ad_format(
    cli_runner,
    work_dir,
    synthetic_h5ad_plain,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
):
    """spatial-ldsc with h5ad marker-score format (toy synthetic data)."""
    project_name = "test_ldsc_h5ad"
    args = _base_ldsc_args(work_dir, project_name) + [
        "--marker-score-format",
        "h5ad",
        "--marker-score-h5ad-path",
        str(synthetic_h5ad_plain),
        "--snp-gene-weight-adata-path",
        str(snp_gene_weight_path),
        "--w-ld-dir",
        str(w_ld_dir),
        "--trait-name",
        "IQ",
        "--sumstats-file",
        str(sumstats_file),
    ]
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 0, f"h5ad spatial-ldsc failed:\n{result.output}"

    ldsc_dir = work_dir / project_name / "spatial_ldsc"
    assert ldsc_dir.is_dir()
    assert len(list(ldsc_dir.glob("*_IQ.csv.gz"))) > 0


@pytest.mark.real_data
def test_spatial_ldsc_feather_format(
    cli_runner,
    work_dir,
    synthetic_marker_scores_feather,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
):
    """spatial-ldsc with feather marker-score format.

    Exercises FeatherAnnData and LazyFeatherX code paths.
    """
    project_name = "test_ldsc_feather"
    args = _base_ldsc_args(work_dir, project_name) + [
        "--marker-score-format",
        "feather",
        "--marker-score-feather-path",
        str(synthetic_marker_scores_feather),
        "--snp-gene-weight-adata-path",
        str(snp_gene_weight_path),
        "--w-ld-dir",
        str(w_ld_dir),
        "--trait-name",
        "IQ",
        "--sumstats-file",
        str(sumstats_file),
    ]
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 0, f"feather spatial-ldsc failed:\n{result.output}"

    ldsc_dir = work_dir / project_name / "spatial_ldsc"
    assert ldsc_dir.is_dir()
    assert len(list(ldsc_dir.glob("*_IQ.csv.gz"))) > 0


@pytest.mark.real_data
def test_spatial_ldsc_cell_indices_range(
    cli_runner,
    work_dir,
    synthetic_h5ad_plain,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
):
    """spatial-ldsc with cell_indices_range=(0, 25) processes only the first 25 spots."""
    project_name = "test_ldsc_cell_range"
    args = _base_ldsc_args(work_dir, project_name) + [
        "--marker-score-format",
        "h5ad",
        "--marker-score-h5ad-path",
        str(synthetic_h5ad_plain),
        "--snp-gene-weight-adata-path",
        str(snp_gene_weight_path),
        "--w-ld-dir",
        str(w_ld_dir),
        "--trait-name",
        "IQ",
        "--sumstats-file",
        str(sumstats_file),
        "--cell-indices-range",
        "0",
        "25",
    ]
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 0, f"cell_indices_range spatial-ldsc failed:\n{result.output}"

    ldsc_dir = work_dir / project_name / "spatial_ldsc"
    assert ldsc_dir.is_dir()
    # Output filename contains '_cells_' when a range is specified
    result_files = list(ldsc_dir.glob("*_cells_*.csv.gz"))
    assert len(result_files) > 0, "Expected cell-range output file not found"

    df = pd.read_csv(result_files[0], compression="gzip")
    assert len(df) == 25, f"Expected 25 spots in range output, got {len(df)}"


@pytest.mark.real_data
def test_spatial_ldsc_sample_filter(
    cli_runner,
    work_dir,
    synthetic_h5ad_with_sample,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
):
    """spatial-ldsc with sample_filter='sampleA' processes only the 25 contiguous sampleA spots."""
    project_name = "test_ldsc_sample"
    args = _base_ldsc_args(work_dir, project_name) + [
        "--marker-score-format",
        "h5ad",
        "--marker-score-h5ad-path",
        str(synthetic_h5ad_with_sample),
        "--snp-gene-weight-adata-path",
        str(snp_gene_weight_path),
        "--w-ld-dir",
        str(w_ld_dir),
        "--trait-name",
        "IQ",
        "--sumstats-file",
        str(sumstats_file),
        "--sample-filter",
        "sampleA",
    ]
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 0, f"sample_filter spatial-ldsc failed:\n{result.output}"

    ldsc_dir = work_dir / project_name / "spatial_ldsc"
    assert ldsc_dir.is_dir()
    # Filename encodes sample name
    result_files = list(ldsc_dir.glob("*_IQ_sampleA*.csv.gz"))
    assert len(result_files) > 0, "Expected sampleA output file not found"

    df = pd.read_csv(result_files[0], compression="gzip")
    assert len(df) == 25, f"Expected 25 sampleA spots, got {len(df)}"


@pytest.mark.real_data
def test_spatial_ldsc_additional_baseline(
    cli_runner,
    work_dir,
    synthetic_h5ad_plain,
    synthetic_additional_baseline,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
):
    """spatial-ldsc with additional_baseline_h5ad_path_list adds extra annotation columns.

    Exercises the additional-baseline loading path in load_common_resources.
    """
    project_name = "test_ldsc_add_baseline"
    args = _base_ldsc_args(work_dir, project_name) + [
        "--marker-score-format",
        "h5ad",
        "--marker-score-h5ad-path",
        str(synthetic_h5ad_plain),
        "--snp-gene-weight-adata-path",
        str(snp_gene_weight_path),
        "--w-ld-dir",
        str(w_ld_dir),
        "--additional-baseline-h5ad-path-list",
        str(synthetic_additional_baseline),
        "--trait-name",
        "IQ",
        "--sumstats-file",
        str(sumstats_file),
    ]
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 0, f"additional_baseline spatial-ldsc failed:\n{result.output}"

    ldsc_dir = work_dir / project_name / "spatial_ldsc"
    assert ldsc_dir.is_dir()
    assert len(list(ldsc_dir.glob("*_IQ.csv.gz"))) > 0

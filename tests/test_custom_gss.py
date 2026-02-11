import anndata
import numpy as np
import pytest

from gsMap.cli import app


@pytest.fixture(scope="session")
def synthetic_marker_scores_h5ad(tmp_path_factory, snp_gene_weight_path):
    """Create a small h5ad with random marker scores.

    Gene names are read from the real SNP-gene weight matrix so the
    spatial-ldsc step can find overlapping genes.
    """
    tmp_dir = tmp_path_factory.mktemp("synthetic_gss")
    rng = np.random.default_rng(42)

    # Read gene names from the weight matrix
    weight_adata = anndata.read_h5ad(snp_gene_weight_path, backed="r")
    all_genes = list(weight_adata.var_names)
    weight_adata.file.close()

    # Use a subset of real gene names
    n_genes = min(100, len(all_genes))
    gene_names = all_genes[:n_genes]
    n_spots = 50

    adata = anndata.AnnData(
        X=rng.random((n_spots, n_genes)).astype(np.float32),
    )
    adata.var_names = gene_names
    adata.obs_names = [f"spot_{i}" for i in range(n_spots)]

    h5ad_path = tmp_dir / "synthetic_marker_scores.h5ad"
    adata.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.mark.real_data
def test_custom_gss_spatial_ldsc(
    cli_runner,
    work_dir,
    synthetic_marker_scores_h5ad,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
    h5ad_file_2d,
    homolog_file,
):
    """Test spatial-ldsc with a custom h5ad marker-score input via quick-mode."""
    project_name = "test_custom_gss"
    result = cli_runner.invoke(
        app,
        [
            "quick-mode",
            "--workdir",
            str(work_dir),
            "--project-name",
            project_name,
            "--dataset-type",
            "spatial2D",
            "--h5ad-path",
            str(h5ad_file_2d),
            "--homolog-file",
            str(homolog_file),
            "--annotation",
            "annotation",
            "--data-layer",
            "count",
            "--spatial-key",
            "spatial",
            "--marker-score-format",
            "h5ad",
            "--marker-score-h5ad-path",
            str(synthetic_marker_scores_h5ad),
            "--snp-gene-weight-adata-path",
            str(snp_gene_weight_path),
            "--w-ld-dir",
            str(w_ld_dir),
            "--trait-name",
            "IQ",
            "--sumstats-file",
            str(sumstats_file),
            "--start-step",
            "spatial_ldsc",
            "--stop-step",
            "spatial_ldsc",
            "--no-gpu",
        ],
    )
    assert result.exit_code == 0, f"spatial-ldsc failed:\n{result.output}"

    ldsc_dir = work_dir / project_name / "spatial_ldsc"
    assert ldsc_dir.is_dir(), "spatial_ldsc directory not created"
    result_files = list(ldsc_dir.glob("*_IQ.csv.gz"))
    assert len(result_files) > 0, "No LDSC result files produced"

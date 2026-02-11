import pytest

from gsMap.cli import app

PROJECT_NAME = "test_step_by_step"


@pytest.fixture(scope="session")
def step_project_dir(work_dir):
    return work_dir / PROJECT_NAME


# ---------------------------------------------------------------------------
# Run the full pipeline once via quick-mode, then verify each step's outputs.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def full_pipeline_result(
    cli_runner,
    work_dir,
    h5ad_file_2d,
    homolog_file,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
    step_project_dir,
):
    """Run the full quick-mode pipeline once and return the project directory."""
    result = cli_runner.invoke(
        app,
        [
            "quick-mode",
            "--workdir",
            str(work_dir),
            "--project-name",
            PROJECT_NAME,
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
            "--w-ld-dir",
            str(w_ld_dir),
            "--snp-gene-weight-adata-path",
            str(snp_gene_weight_path),
            "--trait-name",
            "IQ",
            "--sumstats-file",
            str(sumstats_file),
            "--no-gpu",
        ],
    )
    assert result.exit_code == 0, f"quick-mode pipeline failed:\n{result.output}"
    return step_project_dir


# ---------------------------------------------------------------------------
# Test functions — each verifies the outputs from its corresponding step
# ---------------------------------------------------------------------------


@pytest.mark.real_data
def test_find_latent(full_pipeline_result):
    """Verify find-latent produced the expected outputs."""
    latent_dir = full_pipeline_result / "find_latent_representations"
    assert latent_dir.is_dir()
    h5ad_files = list(latent_dir.glob("*_add_latent.h5ad")) + list(
        latent_dir.glob("*_latent_adata.h5ad")
    )
    assert len(h5ad_files) > 0, "No latent h5ad files found"


@pytest.mark.real_data
def test_latent_to_gene(full_pipeline_result):
    """Verify latent-to-gene produced the expected outputs."""
    l2g_dir = full_pipeline_result / "latent_to_gene"
    assert l2g_dir.is_dir()
    assert (l2g_dir / "marker_scores.dat").exists() or any(l2g_dir.glob("*.parquet"))


@pytest.mark.real_data
def test_spatial_ldsc(full_pipeline_result):
    """Verify spatial-ldsc produced the expected outputs."""
    ldsc_dir = full_pipeline_result / "spatial_ldsc"
    assert ldsc_dir.is_dir()
    result_files = list(ldsc_dir.glob("*_IQ.csv.gz"))
    assert len(result_files) > 0, "No LDSC result files found"


@pytest.mark.real_data
def test_cauchy_combination(full_pipeline_result):
    """Verify cauchy-combination produced the expected outputs."""
    cauchy_dir = full_pipeline_result / "cauchy_combination"
    assert cauchy_dir.is_dir()
    result_files = list(cauchy_dir.glob("*.cauchy.csv")) + list(
        cauchy_dir.glob("*.sample_cauchy.csv")
    )
    assert len(result_files) > 0, "No Cauchy result files found"

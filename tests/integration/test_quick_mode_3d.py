import pytest

from gsMap.cli import app


@pytest.mark.real_data
def test_quick_mode_3d_pipeline(
    cli_runner,
    work_dir,
    h5ad_file_3d_slice1,
    h5ad_file_3d_slice2,
    homolog_file,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
):
    """Test the full 3D quick-mode pipeline end-to-end."""
    project_name = "test_3d"
    result = cli_runner.invoke(
        app,
        [
            "quick-mode",
            "--workdir",
            str(work_dir),
            "--project-name",
            project_name,
            "--dataset-type",
            "spatial3D",
            "--h5ad-path",
            str(h5ad_file_3d_slice1),
            "--h5ad-path",
            str(h5ad_file_3d_slice2),
            "--homolog-file",
            str(homolog_file),
            "--annotation",
            "annotation",
            "--data-layer",
            "count",
            "--spatial-key",
            "spatial",
            "--n-adjacent-slices",
            "1",
            "--cross-slice-marker-score-strategy",
            "hierarchical_pool",
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
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"

    project_dir = work_dir / project_name
    assert (project_dir / "find_latent_representations").is_dir()
    assert (project_dir / "latent_to_gene").is_dir()
    assert (project_dir / "spatial_ldsc").is_dir()
    assert (project_dir / "cauchy_combination").is_dir()

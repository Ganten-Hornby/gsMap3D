import pytest

from gsMap.cli import app


@pytest.mark.real_data
def test_ldscore_weight_matrix_gencode(
    cli_runner,
    work_dir,
    ld_ref_panel_prefix,
    hapmap3_snps_dir,
    gencode_bed,
):
    """Test the ldscore-weight-matrix command with gencode BED mapping."""
    output_dir = work_dir / "ldscore_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = cli_runner.invoke(
        app,
        [
            "ldscore-weight-matrix",
            "--bfile-root",
            f"{ld_ref_panel_prefix}.{{chr}}",
            "--hm3-snp-dir",
            str(hapmap3_snps_dir),
            "--mapping-file",
            str(gencode_bed),
            "--mapping-type",
            "bed",
            "--output-dir",
            str(output_dir),
            "--output-filename",
            "test_weights",
            "--feature-window-size",
            "50000",
            "--strategy",
            "tss",
            "--chromosomes",
            "22",
        ],
    )
    assert result.exit_code == 0, f"ldscore-weight-matrix failed:\n{result.output}"

    # Verify output h5ad exists
    h5ad_files = list(output_dir.glob("*.h5ad"))
    assert len(h5ad_files) > 0, f"No output h5ad files found in {output_dir}"

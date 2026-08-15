"""
Tests for the direct ldscore input mode in spatial_ldsc.

In this mode the user supplies a pre-computed SNP × Cell LD score h5ad
(obs=SNPs, var=cells) together with a baseline h5ad list that provides
all regression baseline annotations.  The on-the-fly matrix multiplication
(snp_gene_weight_sparse @ mk_score_chunk) is bypassed entirely.
"""

import anndata
import numpy as np
import pandas as pd
import pytest

from gsMap.cli import app
from gsMap.spatial_ldsc.chunk_source import SpatialLDChunkSource

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _read_snp_names(snp_gene_weight_path, n_snps=5000):
    """Read the first *n_snps* SNP obs_names from the real weight matrix."""
    w = anndata.read_h5ad(snp_gene_weight_path, backed="r")
    names = list(w.obs_names[: min(n_snps, w.n_obs)])
    w.file.close()
    return names


@pytest.fixture(scope="session")
def synthetic_ldscore_h5ad(tmp_path_factory, snp_gene_weight_path):
    """Pre-computed SNP × Cell LD score h5ad for direct-mode testing.

    Structure expected by the direct ldscore path:
      obs_names  = SNP IDs  (sampled from the real weight matrix)
      var_names  = cell IDs (synthetic)
      X          = random float32 LD scores  shape (n_snps, n_cells)
    """
    tmp_dir = tmp_path_factory.mktemp("direct_ldscore")
    rng = np.random.default_rng(42)

    snp_names = _read_snp_names(snp_gene_weight_path, n_snps=5000)
    n_cells = 30
    cell_names = [f"cell_{i}" for i in range(n_cells)]

    adata = anndata.AnnData(X=rng.random((len(snp_names), n_cells)).astype(np.float32))
    adata.obs_names = snp_names
    adata.var_names = cell_names

    path = tmp_dir / "ldscore.h5ad"
    adata.write_h5ad(path)
    return path


@pytest.fixture(scope="session")
def synthetic_baseline_h5ad(tmp_path_factory, snp_gene_weight_path):
    """Baseline annotation h5ad for direct-mode testing.

    Structure expected by load_common_resources_direct_ldscore:
      obs_names  = SNP IDs  (same set as synthetic_ldscore_h5ad)
      var_names  = annotation column names
      X          = positive float32 values  shape (n_snps, n_annots)
    """
    tmp_dir = tmp_path_factory.mktemp("direct_baseline")
    rng = np.random.default_rng(43)

    snp_names = _read_snp_names(snp_gene_weight_path, n_snps=5000)

    adata = anndata.AnnData(X=(rng.random((len(snp_names), 2)).astype(np.float32) + 0.1))
    adata.obs_names = snp_names
    adata.var_names = ["base", "all_gene"]

    path = tmp_dir / "baseline.h5ad"
    adata.write_h5ad(path)
    return path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _direct_ldsc_args(
    work_dir, project_name, ldscore_h5ad, baseline_h5ad, w_ld_dir, sumstats_file, extra=None
):
    """Build CLI args for direct ldscore mode, running only the spatial_ldsc step."""
    args = [
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
        "--ldscore-h5ad-path",
        str(ldscore_h5ad),
        "--additional-baseline-h5ad-path-list",
        str(baseline_h5ad),
        "--w-ld-dir",
        str(w_ld_dir),
        "--trait-name",
        "IQ",
        "--sumstats-file",
        str(sumstats_file),
    ]
    if extra:
        args.extend(extra)
    return args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_direct_chunk_uses_np_take_and_preserves_snp_order(monkeypatch):
    matrix = np.arange(30, dtype=np.float32).reshape(6, 5)
    positions = np.array([4, 1, 3, 1], dtype=np.int64)
    source = SpatialLDChunkSource(
        mode="direct_ldscore",
        chunk_starts=[0],
        chunk_size=2,
        cell_end=2,
        sample_start_offset=0,
        spot_indices=np.arange(2),
        spot_names_filtered=np.array(["cell_0", "cell_1"]),
        ldscore_adata=type("AdataStub", (), {"X": matrix})(),
        ldscore_snp_positions=positions,
    )
    original_take = np.take
    calls = []

    def tracked_take(array, indices, axis=None):
        calls.append((array.shape, np.asarray(indices).copy(), axis))
        return original_take(array, indices, axis=axis)

    monkeypatch.setattr("gsMap.spatial_ldsc.chunk_source.np.take", tracked_take)
    observed = source._fetch_direct_ldscore_chunk(1, 3)

    expected = matrix[:, 1:3][positions, :]
    np.testing.assert_array_equal(observed, expected)
    assert len(calls) == 1
    assert calls[0][0] == (6, 2)
    np.testing.assert_array_equal(calls[0][1], positions)
    assert calls[0][2] == 0


@pytest.mark.real_data
def test_spatial_ldsc_direct_ldscore_basic(
    cli_runner,
    work_dir,
    synthetic_ldscore_h5ad,
    synthetic_baseline_h5ad,
    sumstats_file,
    w_ld_dir,
):
    """Direct ldscore mode: all 30 synthetic cells are processed."""
    project_name = "test_direct_ldscore_basic"
    args = _direct_ldsc_args(
        work_dir,
        project_name,
        synthetic_ldscore_h5ad,
        synthetic_baseline_h5ad,
        w_ld_dir,
        sumstats_file,
    )
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 0, f"direct ldscore basic failed:\n{result.output}"

    ldsc_dir = work_dir / project_name / "spatial_ldsc"
    assert ldsc_dir.is_dir(), "spatial_ldsc output directory not created"

    result_files = list(ldsc_dir.glob("*_IQ.csv.gz"))
    assert len(result_files) > 0, "Expected IQ output file not found"

    df = pd.read_csv(result_files[0], compression="gzip")
    assert len(df) == 30, f"Expected 30 cells in output, got {len(df)}"
    assert {"p", "beta", "se", "neg_log10_p"}.issubset(df.columns), (
        f"Missing expected columns; got: {list(df.columns)}"
    )


@pytest.mark.real_data
def test_spatial_ldsc_direct_ldscore_cell_indices_range(
    cli_runner,
    work_dir,
    synthetic_ldscore_h5ad,
    synthetic_baseline_h5ad,
    sumstats_file,
    w_ld_dir,
):
    """Direct ldscore mode: cell_indices_range=(0, 15) yields exactly 15 output rows."""
    project_name = "test_direct_ldscore_range"
    args = _direct_ldsc_args(
        work_dir,
        project_name,
        synthetic_ldscore_h5ad,
        synthetic_baseline_h5ad,
        w_ld_dir,
        sumstats_file,
        extra=["--cell-indices-range", "0", "15"],
    )
    result = cli_runner.invoke(app, args)
    assert result.exit_code == 0, f"direct ldscore cell-range failed:\n{result.output}"

    ldsc_dir = work_dir / project_name / "spatial_ldsc"
    result_files = list(ldsc_dir.glob("*_cells_*.csv.gz"))
    assert len(result_files) > 0, "Expected cell-range output file not found"

    df = pd.read_csv(result_files[0], compression="gzip")
    assert len(df) == 15, f"Expected 15 cells in range output, got {len(df)}"

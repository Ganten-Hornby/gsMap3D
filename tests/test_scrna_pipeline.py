"""
End-to-end test for scRNA-seq dataset type through the quick-mode pipeline.

For scRNA-seq:
  - No spatial coordinates required
  - Cell embeddings (e.g. PCA / SCVI) go in obsm and are pointed to by
    --latent-representation-cell
  - Niche embedding is not needed (a dummy all-ones matrix is created internally)
  - Connectivity is built via KNN on the cell embedding space, not on spatial coords
  - The pipeline runs: latent2gene → spatial_ldsc → cauchy
    (find_latent is skipped since embeddings are pre-supplied)
"""

import anndata
import numpy as np
import pytest

from gsMap.cli import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _read_gene_names(snp_gene_weight_path, n_genes=200):
    """Read gene names from the real SNP-gene weight matrix."""
    w = anndata.read_h5ad(snp_gene_weight_path, backed="r")
    names = list(w.var_names[: min(n_genes, w.n_vars)])
    w.file.close()
    return names


@pytest.fixture(scope="session")
def synthetic_scrna_h5ad(tmp_path_factory, snp_gene_weight_path):
    """Synthetic scRNA-seq h5ad with PCA cell embeddings.

    Layout required for latent2gene (scRNA path):
      - X               : raw counts  (n_cells × n_genes)
      - obs['cell_type']: annotation column
      - obsm['X_pca']   : cell embeddings, shape (n_cells, 30)
                          mimics SCVI / PCA output from find_latent
      - var_names       : subset of real gene names so spatial_ldsc finds overlaps
      - layers['count'] : copy of X (used as data_layer)

    No spatial coordinates are included – scRNA connectivity is KNN-only.
    """
    tmp_dir = tmp_path_factory.mktemp("scrna_pipeline")
    rng = np.random.default_rng(99)

    gene_names = _read_gene_names(snp_gene_weight_path)
    n_cells = 120
    n_genes = len(gene_names)

    # Simulate three cell-type clusters with distinct embedding centroids
    n_per_type = n_cells // 3
    cell_types = (
        ["TypeA"] * n_per_type + ["TypeB"] * n_per_type + ["TypeC"] * (n_cells - 2 * n_per_type)
    )

    # Cluster-specific centroids in 30-d PCA space
    centroids = rng.standard_normal((3, 30)).astype(np.float32) * 3.0
    embeddings = np.vstack(
        [
            centroids[i] + rng.standard_normal((n, 30)).astype(np.float32) * 0.5
            for i, n in enumerate([n_per_type, n_per_type, n_cells - 2 * n_per_type])
        ]
    )

    counts = rng.poisson(5.0, (n_cells, n_genes)).astype(np.float32)

    adata = anndata.AnnData(X=counts)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = gene_names
    adata.obs["cell_type"] = cell_types
    adata.obsm["X_pca"] = embeddings
    adata.layers["count"] = counts.copy()

    path = tmp_dir / "synthetic_scrna.h5ad"
    adata.write_h5ad(path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.real_data
def test_scrna_quick_mode_full_pipeline(
    cli_runner,
    work_dir,
    synthetic_scrna_h5ad,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
):
    """Full quick-mode pipeline for scRNA-seq: latent2gene → spatial_ldsc → cauchy → report.

    Key differences from spatial2D:
      - dataset_type=scRNA: no spatial key, KNN connectivity in embedding space
      - latent_representation_cell=X_pca: pre-computed embeddings (SCVI/PCA style)
      - find_latent is skipped (--start-step latent2gene)
      - latent_representation_niche is auto-set to None for scRNA (no niche needed)
    """
    project_name = "test_scrna_pipeline"
    result = cli_runner.invoke(
        app,
        [
            "quick-mode",
            "--workdir",
            str(work_dir),
            "--project-name",
            project_name,
            # scRNA dataset – no spatial coordinates
            "--dataset-type",
            "scRNA",
            "--h5ad-path",
            str(synthetic_scrna_h5ad),
            "--annotation",
            "cell_type",
            "--data-layer",
            "count",
            # Point to pre-computed PCA / SCVI-style cell embedding
            "--latent-representation-cell",
            "X_pca",
            # Skip find_latent; embeddings are already in the h5ad
            "--start-step",
            "latent2gene",
            # Spatial LDSC resources
            "--snp-gene-weight-adata-path",
            str(snp_gene_weight_path),
            "--w-ld-dir",
            str(w_ld_dir),
            "--trait-name",
            "IQ",
            "--sumstats-file",
            str(sumstats_file),
            "--no-gpu",
        ],
    )
    assert result.exit_code == 0, f"scRNA quick-mode failed:\n{result.output}"

    project_dir = work_dir / project_name

    # latent2gene output: marker scores (memmap format)
    l2g_dir = project_dir / "latent_to_gene"
    assert l2g_dir.is_dir(), "latent_to_gene directory not created"
    assert (l2g_dir / "marker_scores.dat").exists(), "marker_scores.dat not found"

    # spatial_ldsc output
    ldsc_dir = project_dir / "spatial_ldsc"
    assert ldsc_dir.is_dir(), "spatial_ldsc directory not created"
    assert len(list(ldsc_dir.glob("*_IQ.csv.gz"))) > 0, "No spatial_ldsc result files produced"

    # cauchy output
    cauchy_dir = project_dir / "cauchy_combination"
    assert cauchy_dir.is_dir(), "cauchy_combination directory not created"
    cauchy_files = list(cauchy_dir.glob("*.cauchy.csv")) + list(
        cauchy_dir.glob("*.sample_cauchy.csv")
    )
    assert len(cauchy_files) > 0, "No Cauchy result files produced"

    # report output
    report_dir = project_dir / "gsmap_web_report"
    assert report_dir.is_dir(), "gsmap_web_report directory not created"
    assert (report_dir / "index.html").exists(), "index.html not found in report"

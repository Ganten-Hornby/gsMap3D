from pathlib import Path

import anndata
import numpy as np
import pytest

from gsMap.cli import app
from gsMap.config import LatentToGeneConfig


@pytest.fixture(scope="session")
def synthetic_h5ad_with_embeddings(tmp_path_factory):
    """Create a small synthetic h5ad with custom PCA embeddings."""
    tmp_dir = tmp_path_factory.mktemp("synthetic_embed")
    rng = np.random.default_rng(42)

    n_spots = 50
    n_genes = 100
    gene_names = [f"GENE{i}" for i in range(n_genes)]

    adata = anndata.AnnData(
        X=rng.poisson(5, (n_spots, n_genes)).astype(np.float32),
    )
    adata.var_names = gene_names
    adata.obs["annotation"] = rng.choice(["TypeA", "TypeB"], n_spots)
    adata.obsm["spatial"] = rng.random((n_spots, 2)).astype(np.float32) * 1000
    adata.obsm["X_pca"] = rng.standard_normal((n_spots, 10)).astype(np.float32)
    adata.layers["count"] = adata.X.copy()

    h5ad_path = tmp_dir / "synthetic_embeddings.h5ad"
    adata.write_h5ad(h5ad_path)
    return h5ad_path


@pytest.mark.real_data
def test_custom_embedding_cli(
    cli_runner,
    work_dir,
    synthetic_h5ad_with_embeddings,
    sumstats_file,
    w_ld_dir,
    snp_gene_weight_path,
):
    """Test quick-mode with --start-step latent2gene and a custom embedding key."""
    project_name = "test_custom_embed"
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
            str(synthetic_h5ad_with_embeddings),
            "--annotation",
            "annotation",
            "--data-layer",
            "count",
            "--spatial-key",
            "spatial",
            "--latent-representation-cell",
            "X_pca",
            "--start-step",
            "latent2gene",
            "--stop-step",
            "latent2gene",
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

    l2g_dir = work_dir / project_name / "latent_to_gene"
    assert l2g_dir.is_dir(), "latent_to_gene directory not created"


def test_custom_embedding_python_api(synthetic_h5ad_with_embeddings, tmp_path):
    """Verify LatentToGeneConfig accepts a custom embedding key."""
    config = LatentToGeneConfig(
        workdir=tmp_path,
        project_name="api_test",
        dataset_type="spatial2D",
        h5ad_path=[Path(synthetic_h5ad_with_embeddings)],
        annotation="annotation",
        data_layer="count",
        spatial_key="spatial",
        latent_representation_cell="X_pca",
        use_gpu=False,
    )
    assert config.latent_representation_cell == "X_pca"

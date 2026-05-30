import anndata as ad
import numpy as np
import pandas as pd

from gsMap.config import TraitModuleConfig
from gsMap.trait_modules import (
    _condition_residual_on_profile,
    _module_from_best_seed,
    discover_trait_modules,
)


def _write_trait_module_inputs(
    tmp_path,
    project_name: str,
    trait_name: str,
    marker_scores: np.ndarray,
    p_values: np.ndarray,
):
    project_dir = tmp_path / project_name
    ldsc_dir = project_dir / "spatial_ldsc"
    ldsc_dir.mkdir(parents=True)

    cells = [f"cell_{i}" for i in range(marker_scores.shape[0])]
    genes = [f"gene_{i}" for i in range(marker_scores.shape[1])]

    marker_adata = ad.AnnData(X=marker_scores.astype(np.float32))
    marker_adata.obs_names = cells
    marker_adata.var_names = genes
    marker_path = tmp_path / f"{project_name}_marker_scores.h5ad"
    marker_adata.write_h5ad(marker_path)

    ldsc = pd.DataFrame(
        {
            "spot": cells,
            "beta": np.ones(marker_scores.shape[0]),
            "se": np.full(marker_scores.shape[0], 0.1),
            "z": np.full(marker_scores.shape[0], 10.0),
            "p": p_values,
        }
    )
    ldsc_path = ldsc_dir / f"{project_name}_{trait_name}.csv.gz"
    ldsc.to_csv(ldsc_path, index=False, compression="gzip")
    return cells, genes, marker_path


def test_discover_trait_modules_recovers_two_marker_programs(tmp_path):
    workdir = tmp_path
    project_name = "module_test"

    n_cells = 60
    n_genes = 40

    rng = np.random.default_rng(42)
    marker_scores = rng.normal(0.05, 0.02, size=(n_cells, n_genes)).astype(np.float32)
    marker_scores[:20, :6] += 2.0
    marker_scores[20:40, 10:16] += 1.8
    marker_scores = np.maximum(marker_scores, 0.0)

    p_values = np.full(n_cells, 0.9)
    p_values[:20] = 1e-8
    p_values[20:40] = 1e-6
    cells, genes, marker_path = _write_trait_module_inputs(
        tmp_path, project_name, "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=workdir,
        project_name=project_name,
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.15,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=4,
        max_marker_genes=6,
    )

    cell_df, summary_df, genes_df = discover_trait_modules(config)

    assert summary_df["n_cells"].tolist() == [20, 20]
    assert set(cell_df.loc[cell_df["trait_module"] > 0, "spot"]) == set(cells[:40])

    module_1_genes = set(genes_df.loc[genes_df["module"] == 1, "gene"])
    module_2_genes = set(genes_df.loc[genes_df["module"] == 2, "gene"])

    assert module_1_genes <= set(genes[:6])
    assert module_2_genes <= set(genes[10:16])


def test_discover_trait_modules_uses_global_background_for_pure_signal(tmp_path):
    n_cells = 100
    n_genes = 30
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:20, :5] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:20] = 1e-8
    cells, genes, marker_path = _write_trait_module_inputs(
        tmp_path, "pure_module_test", "BMI", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="pure_module_test",
        trait_name="BMI",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.2,
        min_module_cells=5,
        module_similarity_threshold=0.8,
        max_modules=3,
        max_marker_genes=5,
    )

    cell_df, summary_df, genes_df = discover_trait_modules(config)

    assert summary_df["n_cells"].tolist() == [20]
    assert set(cell_df.loc[cell_df["trait_module"] == 1, "spot"]) == set(cells[:20])
    assert set(genes_df["gene"]) == set(genes[:5])


def test_condition_residual_on_profile_removes_claimed_program_only():
    residual = np.array(
        [
            [2.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    profile = np.array([2.0, 2.0, 0.0, 0.0], dtype=np.float32)

    conditioned = _condition_residual_on_profile(residual, profile)

    np.testing.assert_allclose(conditioned[:2], 0.0, atol=1e-6)
    np.testing.assert_allclose(conditioned[2], residual[2], atol=1e-6)


def test_module_similarity_uses_marker_gene_subspace():
    residual = np.zeros((2, 100), dtype=np.float32)
    residual[0, :5] = 2.0
    residual[1, :5] = 2.0
    residual[1, 5:] = 5.0

    module_idx, _, _ = _module_from_best_seed(
        residual=residual,
        remaining_mask=np.array([True, True]),
        weights=np.array([100.0, 1.0], dtype=np.float32),
        top_n=5,
        similarity_threshold=0.9,
    )

    assert module_idx.tolist() == [0, 1]

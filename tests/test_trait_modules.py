import anndata as ad
import numpy as np
import pandas as pd

from gsMap.config import TraitModuleConfig
from gsMap.trait_modules import discover_trait_modules


def test_discover_trait_modules_recovers_two_marker_programs(tmp_path):
    workdir = tmp_path
    project_name = "module_test"
    project_dir = workdir / project_name
    ldsc_dir = project_dir / "spatial_ldsc"
    ldsc_dir.mkdir(parents=True)

    n_cells = 60
    n_genes = 40
    cells = [f"cell_{i}" for i in range(n_cells)]
    genes = [f"gene_{i}" for i in range(n_genes)]

    rng = np.random.default_rng(42)
    marker_scores = rng.normal(0.05, 0.02, size=(n_cells, n_genes)).astype(np.float32)
    marker_scores[:20, :6] += 2.0
    marker_scores[20:40, 10:16] += 1.8
    marker_scores = np.maximum(marker_scores, 0.0)

    marker_adata = ad.AnnData(X=marker_scores)
    marker_adata.obs_names = cells
    marker_adata.var_names = genes
    marker_path = tmp_path / "marker_scores.h5ad"
    marker_adata.write_h5ad(marker_path)

    p_values = np.full(n_cells, 0.9)
    p_values[:20] = 1e-8
    p_values[20:40] = 1e-6
    ldsc = pd.DataFrame(
        {
            "spot": cells,
            "beta": np.ones(n_cells),
            "se": np.full(n_cells, 0.1),
            "z": np.full(n_cells, 10.0),
            "p": p_values,
        }
    )
    ldsc_path = ldsc_dir / f"{project_name}_IQ.csv.gz"
    ldsc.to_csv(ldsc_path, index=False, compression="gzip")

    config = TraitModuleConfig(
        workdir=workdir,
        project_name=project_name,
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        min_module_cells=5,
        module_similarity_threshold=0.2,
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

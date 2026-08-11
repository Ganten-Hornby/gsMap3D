import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).parents[1] / "analysis" / "revision_graph_3d" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_mouse_e115_supervised_trait_inputs import prepare_inputs  # noqa: E402


def test_prepare_inputs_maps_genes_and_preserves_all_observations(tmp_path) -> None:
    mouse_genes = [f"Mouse{i}" for i in range(301)]
    human_genes = [f"HUMAN{i}" for i in range(301)]
    source = tmp_path / "source.h5ad"
    homolog = tmp_path / "homolog.tsv"
    output_dir = tmp_path / "sections"
    manifest = tmp_path / "sections.yaml"
    audit = tmp_path / "audit.json"

    adata = ad.AnnData(
        X=np.ones((4, len(mouse_genes)), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "slices": ["later", "earlier", "later", "earlier"],
                "heart_regions": ["LV", "RV", "LV", "RV"],
                "ctype_user": ["c", "c", "c", "c"],
            },
            index=["s0", "s1", "s2", "s3"],
        ),
        var=pd.DataFrame(index=mouse_genes),
        obsm={
            "aligned_spatial_3D_mm": np.array(
                [[0, 0, 0.1], [0, 0, 0.0], [1, 0, 0.1], [1, 0, 0.0]],
                dtype=np.float32,
            ),
            "emb_cell": np.ones((4, 2), dtype=np.float32),
            "emb_niche": np.ones((4, 2), dtype=np.float32),
        },
    )
    adata.write_h5ad(source)
    pd.DataFrame(
        {"MOUSE_GENE_SYM": mouse_genes, "HUMAN_GENE_SYM": human_genes}
    ).to_csv(homolog, sep="\t", index=False)

    prepare_inputs(
        argparse.Namespace(
            source=source,
            homolog_file=homolog,
            output_dir=output_dir,
            manifest=manifest,
            audit=audit,
            section_key="slices",
            z_key="aligned_spatial_3D_mm",
        )
    )

    with manifest.open() as handle:
        section_paths = yaml.safe_load(handle)
    first, second = [ad.read_h5ad(path) for path in section_paths.values()]
    assert first.obs["slices"].unique().tolist() == ["earlier"]
    assert second.obs["slices"].unique().tolist() == ["later"]
    assert first.var_names.tolist() == human_genes
    assert first.var["MOUSE_GENE_SYM"].tolist() == mouse_genes

    audit_data = json.loads(audit.read_text())
    assert audit_data["source_n_obs"] == 4
    assert audit_data["output_n_obs"] == 4
    assert audit_data["mapped_source_genes"] == 301
    assert audit_data["no_downsampling"] is True

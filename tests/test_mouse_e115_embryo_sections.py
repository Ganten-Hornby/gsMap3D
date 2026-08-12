import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parents[1] / "analysis" / "revision_graph_3d" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_mouse_e115_embryo_input import main, set_slice_id_from_z_axis  # noqa: E402

from gsMap.find_latent.find_latent_representation import (  # noqa: E402
    write_latent_representation_outputs,
)


def test_z_axis_replaces_duplicate_source_slice_labels() -> None:
    adata = ad.AnnData(
        X=np.ones((5, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"slice_id": ["S0", "S01", "S1", "S02", "S2"]},
            index=[f"spot-{index}" for index in range(5)],
        ),
        obsm={
            "z_correction": np.array(
                [[0, 0, 305], [1, 0, 305], [0, 0, 310], [1, 0, 310], [0, 0, 315]],
                dtype=np.float64,
            )
        },
    )

    unique_z, _ = set_slice_id_from_z_axis(adata)

    np.testing.assert_array_equal(unique_z, [305, 310, 315])
    assert adata.obs["slice_id"].astype(float).tolist() == [305, 305, 310, 310, 315]
    assert adata.obs["source_slice_id"].tolist() == ["S0", "S01", "S1", "S02", "S2"]


def test_section_graph_outputs_one_h5ad_per_unique_z_in_physical_order(tmp_path) -> None:
    adata = ad.AnnData(
        X=np.ones((4, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"slice_id": [315.0, 305.0, 310.0, 305.0]},
            index=[f"spot-{index}" for index in range(4)],
        ),
        obsm={
            "z_correction": np.array(
                [[0, 0, 315], [0, 0, 305], [0, 0, 310], [1, 0, 305]], dtype=np.float64
            )
        },
    )
    config = SimpleNamespace(
        graph_mode="sectionwise_2d",
        graph_section_key="slice_id",
        spatial_key="z_correction",
        latent_dir=tmp_path,
    )

    outputs = write_latent_representation_outputs(
        adata,
        sample_name="embryo",
        config=config,
    )

    assert list(outputs) == [
        "embryo__section_000_305.0",
        "embryo__section_001_310.0",
        "embryo__section_002_315.0",
    ]
    assert [ad.read_h5ad(path).n_obs for path in outputs.values()] == [2, 1, 1]


def test_preparation_allows_source_slice_label_to_span_multiple_z_values(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.h5ad"
    output = tmp_path / "prepared.h5ad"
    audit = tmp_path / "audit.json"
    adata = ad.AnnData(
        X=np.ones((4, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "slice_id": ["S11", "S11", "S12", "S12"],
                "mapped_celltype": ["a", "a", "b", "b"],
            },
            index=[f"spot-{index}" for index in range(4)],
        ),
        obsm={
            "z_correction": np.array(
                [[0, 0, 305], [1, 0, 310], [0, 0, 310], [1, 0, 315]], dtype=np.float64
            )
        },
    )
    adata.write_h5ad(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_mouse_e115_embryo_input.py",
            "--source",
            str(source),
            "--output",
            str(output),
            "--audit",
            str(audit),
        ],
    )

    main()

    prepared = ad.read_h5ad(output)
    assert prepared.obs["slice_id"].astype(float).tolist() == [305, 310, 310, 315]
    mappings = {
        item["slice_id"]: item
        for item in json.loads(audit.read_text())["slice_to_physical_section"]
    }
    assert mappings["S11"]["z_axis_slice_ids"] == [305.0, 310.0]
    assert mappings["S11"]["n_z_axis_slices"] == 2

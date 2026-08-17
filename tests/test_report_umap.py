import json

import anndata as ad
import numpy as np
import pandas as pd

from gsMap.report.report_data import _export_umap_js, _render_scanpy_umap_panels


def test_scanpy_umap_panels_record_full_count_and_dpi(tmp_path):
    n_obs = 20
    adata = ad.AnnData(
        np.zeros((n_obs, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "Trait": np.linspace(0, 1, n_obs),
                "Annotation": pd.Categorical(["a", "b"] * (n_obs // 2)),
            }
        ),
    )
    coordinates = np.column_stack(
        [np.linspace(0, 1, n_obs), np.linspace(1, 0, n_obs)]
    ).astype(np.float32)

    panels, audit = _render_scanpy_umap_panels(
        adata,
        {"cell": coordinates},
        ["Trait"],
        ["Annotation"],
        tmp_path,
    )

    assert panels["cell"]["trait"]["Trait"] == "umap_scanpy/umap_cell_trait_Trait.png"
    assert audit["observations"] == n_obs
    assert len(audit["images"]) == 2
    for image in audit["images"].values():
        assert image["observations"] == n_obs
        assert image["stored_dpi_x"] >= 319.9
        assert image["stored_dpi_y"] >= 319.9
        assert image["pixel_width"] > 0
        assert image["pixel_height"] > 0


def test_export_umap_js_uses_compact_static_manifest(tmp_path):
    data_dir = tmp_path / "data"
    js_dir = tmp_path / "js"
    data_dir.mkdir()
    js_dir.mkdir()
    pd.DataFrame(
        {
            "spot": ["a", "b"],
            "sample_name": ["s", "s"],
            "umap_cell_x": [0.0, 1.0],
            "umap_cell_y": [1.0, 0.0],
        }
    ).to_csv(data_dir / "umap_data.csv", index=False)
    static_panels = {
        "cell": {
            "trait": {"Trait": "umap_scanpy/umap_cell_trait_Trait.png"},
            "annotation": {
                "Annotation": "umap_scanpy/umap_cell_annotation_Annotation.png"
            },
        }
    }

    _export_umap_js(
        data_dir,
        js_dir,
        {
            "umap_info": {
                "render_mode": "scanpy_static",
                "n_points": 2,
                "has_niche": False,
                "static_panels": static_panels,
            }
        },
    )

    payload = (js_dir / "umap_data.js").read_text(encoding="utf-8")
    exported = json.loads(payload.removeprefix("window.GSMAP_UMAP = ").removesuffix(";"))
    assert exported["n_points"] == 2
    assert exported["spot"] == []
    assert exported["static_panels"] == static_panels

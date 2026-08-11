from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection


def load_comparison_module():
    script = (
        Path(__file__).parents[1]
        / "analysis"
        / "revision_graph_3d"
        / "scripts"
        / "compare_mouse_e95_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location("embedding_comparison_analysis", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_umap_panel_is_plotted_through_scanpy(monkeypatch):
    comparison = load_comparison_module()
    coordinates = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    labels = np.asarray(["A", "B"], dtype=object)
    colors = {"A": "#112233", "B": "#445566"}
    recorded = {}

    def fake_scanpy_umap(adata, **kwargs):
        recorded["coordinates"] = adata.obsm["X_umap"].copy()
        recorded["color"] = kwargs["color"]
        recorded["palette"] = kwargs["palette"]
        kwargs["ax"].add_collection(PathCollection([]))

    monkeypatch.setattr(comparison.sc.pl, "umap", fake_scanpy_umap)
    figure, axis = plt.subplots()
    comparison.plot_scanpy_umap_panel(
        axis,
        coordinates,
        labels,
        colors,
        "Scanpy UMAP",
        ((0.0, 2.0), (1.0, 3.0)),
    )

    np.testing.assert_array_equal(recorded["coordinates"], coordinates)
    assert recorded["color"] == "transf_anno_E95"
    assert recorded["palette"] == ["#112233", "#445566"]
    assert all(collection.get_rasterized() for collection in axis.collections)
    plt.close(figure)


def test_latent_metric_names_follow_configured_neighbor_count():
    comparison = load_comparison_module()

    names = comparison.LatentMetricNames.for_k(7)

    assert names.k == 7
    assert names.recall == "kNN recall@7"
    assert names.jaccard == "kNN Jaccard@7"
    assert names.label_purity == "Label kNN purity@7"
    assert names.different_section == "Different-section latent kNN fraction@7"
    assert names.adjacent_section == "Adjacent-section latent kNN fraction@7"

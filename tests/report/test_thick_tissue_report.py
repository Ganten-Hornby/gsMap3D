"""Synthetic volume geometry through report caches, JS payloads and SpatialVista."""

import json

import anndata as ad
import numpy as np
import pandas as pd

from gsMap.config import QuickModeConfig
from gsMap.config.quick_mode_config import (
    _report_visualization_config_matches,
    get_report_missing_files,
)
from gsMap.report.report import _render_report_html
from gsMap.report.report_data import ReportDataManager
from gsMap.report.spatialvista import export_spatialvista


def test_volume_report_and_resume(tmp_path):
    xyz = np.random.default_rng(42).random((20, 3))
    data = ad.AnnData(np.ones((20, 3), dtype=np.float32))
    data.obsm["spatial"] = xyz
    data.obs["celltype"] = ["A", "B"] * 10
    data.obs["sample_name"] = "volume"
    path = tmp_path / "volume.h5ad"
    data.write_h5ad(path)
    gwas = tmp_path / "IQ.sumstats.gz"
    gwas.touch()
    config = QuickModeConfig(
        workdir=tmp_path,
        project_name="report",
        h5ad_path=[path],
        dataset_type="spatial3D",
        annotation="celltype",
        sumstats_config_dict={"IQ": gwas},
        start_step="report",
    )
    data.write_h5ad(config.concatenated_latent_adata_path)
    associations = pd.DataFrame(
        {
            "sample_name": "volume",
            "celltype": data.obs.celltype,
            "IQ": np.linspace(0, 5, 20),
        },
        index=data.obs_names,
    )
    associations.index.name = "spot"
    associations.to_parquet(config.ldsc_combined_parquet_path)

    def generate():
        manager = ReportDataManager(config)
        manager.prepare_base_metadata()
        manager.prepare_spot_metadata()
        manager.render_static_plots()
        manager.prepare_3d_visualization()
        manager.finalize_report()
        return manager

    first = generate()
    assert "sx" not in first.metadata
    assert "sy" not in first.metadata
    np.testing.assert_allclose(first.metadata[["3d_x", "3d_y", "3d_z"]], xyz)
    web = config.web_report_dir
    meta = json.loads((web / "report_meta.json").read_text())
    assert meta["spatial_layout"] == "volume"
    assert meta["has_3d_widget"]
    assert not (web / "js_data/spatial_2d").exists()
    assert not (web / "gene_diagnostic_plots").exists()
    assert (web / "js_data/sample_index.js").read_text() == "window.GSMAP_SAMPLE_INDEX={};"
    assert "showUMAP" in _render_report_html("Volume smoke")
    export = export_spatialvista(config.project_dir, metadata=first.metadata)

    exported = ad.read_h5ad(export)
    np.testing.assert_allclose(exported.obsm["spatial"], xyz)
    assert len(exported) == 20
    before = (config.report_data_dir / "spot_metadata.csv").stat().st_mtime_ns
    second = generate()
    assert (config.report_data_dir / "spot_metadata.csv").stat().st_mtime_ns == before
    pd.testing.assert_frame_equal(first.metadata, second.metadata, check_dtype=False)
    assert _report_visualization_config_matches(config)
    missing = sum(get_report_missing_files(config), [])
    assert not any("spatial_2d" in str(p) or "gene_diagnostic_plots" in str(p) for p in missing)
    # Choosing a different coordinate array invalidates the report cache.
    config.spatial_key = "other_xyz"
    assert not _report_visualization_config_matches(config)

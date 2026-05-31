import json
import os
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

import gsMap.report.report_data as report_data
from gsMap.config.latent2gene_config import DatasetType
from gsMap.config.quick_mode_config import get_report_missing_files
from gsMap.report.report_data import (
    _export_per_sample_spatial_js,
    _export_trait_module_js,
    _merge_trait_module_metadata,
    _prepare_3d_visualization,
    _trait_module_outputs_newer_than,
)


def _report_config(tmp_path):
    project_name = "module_report"
    return SimpleNamespace(
        project_name=project_name,
        project_dir=tmp_path / project_name,
        trait_name_list=["IQ"],
    )


def test_merge_trait_module_metadata_adds_trait_specific_columns(tmp_path):
    config = _report_config(tmp_path)
    module_dir = config.project_dir / "trait_modules"
    module_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "spot": ["cell_1", "cell_3"],
            "trait_module": [2, 1],
            "module_specificity": [0.9234, 0.5123],
            "second_best_module_specificity": [0.1234, 0.1122],
            "module_specificity_margin": [0.8, 0.4],
            "module_specificity_ratio": [7.482982, 4.565954],
            "is_module_cell": [True, True],
            "is_unique_module_cell": [True, True],
            "module_status": ["assigned", "rejected"],
            "module_reason": ["accepted", "specificity_null_pvalue"],
        }
    ).to_csv(module_dir / "module_report_IQ.cell_modules.csv", index=False)
    pd.DataFrame(
        {
            "spot": ["cell_1", "cell_2", "cell_3"],
            "best_trait_module": [2, 2, 1],
            "best_module_specificity": [0.9234, 0.2211, 0.5123],
            "second_best_trait_module": [1, 1, 2],
            "second_best_module_specificity": [0.1234, 0.1111, 0.1122],
            "module_specificity_margin": [0.8, 0.11, 0.4],
            "module_specificity_ratio": [7.482982, 1.990099, 4.565954],
            "module_1_specificity": [0.1234, 0.1111, 0.5123],
            "module_2_specificity": [0.9234, 0.2211, 0.1122],
        }
    ).to_csv(module_dir / "module_report_IQ.module_specificity.csv", index=False)

    metadata = pd.DataFrame(
        {
            "spot": ["cell_1", "cell_2", "cell_3"],
            "sample_name": ["sampleA", "sampleA", "sampleB"],
            "IQ": [8.1, 0.2, 6.4],
        }
    )

    merged = _merge_trait_module_metadata(metadata, config)

    assert merged["IQ_trait_module"].tolist() == [2, -1, 1]
    assert merged["IQ_module_specificity"].iloc[[0, 2]].tolist() == [0.923, 0.512]
    assert np.isnan(merged["IQ_module_specificity"].iloc[1])
    assert merged["IQ_assigned_module_specificity_margin"].iloc[[0, 2]].tolist() == [0.8, 0.4]
    assert merged["IQ_assigned_module_specificity_ratio"].iloc[[0, 2]].tolist() == [
        7.483,
        4.566,
    ]
    assert merged["IQ_best_trait_module"].tolist() == [2, 2, 1]
    assert merged["IQ_best_module_specificity"].tolist() == [0.923, 0.221, 0.512]
    assert merged["IQ_second_best_trait_module"].tolist() == [1, 1, 2]
    assert merged["IQ_second_best_module_specificity"].tolist() == [0.123, 0.111, 0.112]
    assert merged["IQ_module_specificity_margin"].tolist() == [0.8, 0.11, 0.4]
    assert merged["IQ_module_specificity_ratio"].tolist() == [7.483, 1.99, 4.566]
    assert merged["IQ_module_1_specificity"].tolist() == [0.123, 0.111, 0.512]
    assert merged["IQ_module_2_specificity"].tolist() == [0.923, 0.221, 0.112]
    assert merged["IQ_is_module_cell"].tolist() == [True, False, True]
    assert merged["IQ_is_unique_module_cell"].tolist() == [True, False, True]
    assert merged["IQ_module_status"].tolist() == ["assigned", "unresolved", "rejected"]
    assert merged["IQ_module_reason"].tolist() == [
        "accepted",
        "not_selected",
        "specificity_null_pvalue",
    ]


def test_export_per_sample_spatial_js_includes_trait_module_columns(tmp_path):
    data_dir = tmp_path / "report_data"
    js_dir = tmp_path / "js_data"
    data_dir.mkdir()
    js_dir.mkdir()

    pd.DataFrame(
        {
            "spot": ["cell_1", "cell_2"],
            "sample_name": ["sampleA", "sampleA"],
            "sx": [1.0, 2.0],
            "sy": [3.0, 4.0],
            "annotation": ["A", "B"],
            "IQ": [8.1, 0.2],
            "IQ_trait_module": [2, -1],
            "IQ_module_specificity": [0.9234, None],
            "IQ_best_trait_module": [2, 2],
            "IQ_best_module_specificity": [0.9234, 0.2211],
            "IQ_second_best_trait_module": [1, 1],
            "IQ_second_best_module_specificity": [0.1234, 0.1111],
            "IQ_module_specificity_margin": [0.8, 0.11],
            "IQ_module_specificity_ratio": [7.482982, 1.990099],
            "IQ_module_1_specificity": [0.1234, 0.1111],
            "IQ_module_2_specificity": [0.9234, 0.2211],
            "IQ_is_module_cell": ["True", "False"],
            "IQ_is_unique_module_cell": ["True", "False"],
            "IQ_module_status": ["assigned", "rejected"],
            "IQ_module_reason": ["accepted", "module_profile_similarity"],
        }
    ).to_csv(data_dir / "spot_metadata.csv", index=False)

    _export_per_sample_spatial_js(
        data_dir,
        js_dir,
        {
            "samples": ["sampleA"],
            "traits": ["IQ"],
            "annotations": ["annotation"],
            "downsampling_n_spots_2d": 10,
        },
    )

    js_text = (js_dir / "sample_sampleA_spatial.js").read_text()
    payload = js_text.split("=", 1)[1].rstrip(";")
    data = json.loads(payload)

    assert data["IQ_trait_module"] == [2, -1]
    assert data["IQ_module_specificity"] == [0.923, None]
    assert data["IQ_best_trait_module"] == [2, 2]
    assert data["IQ_best_module_specificity"] == [0.923, 0.221]
    assert data["IQ_second_best_trait_module"] == [1, 1]
    assert data["IQ_second_best_module_specificity"] == [0.123, 0.111]
    assert data["IQ_module_specificity_margin"] == [0.8, 0.11]
    assert data["IQ_module_specificity_ratio"] == [7.483, 1.99]
    assert data["IQ_module_1_specificity"] == [0.123, 0.111]
    assert data["IQ_module_2_specificity"] == [0.923, 0.221]
    assert data["IQ_is_module_cell"] == [True, False]
    assert data["IQ_is_unique_module_cell"] == [True, False]
    assert data["IQ_module_status"] == ["assigned", "rejected"]
    assert data["IQ_module_reason"] == ["accepted", "module_profile_similarity"]


def test_export_trait_module_js_includes_summary_and_trace(tmp_path):
    config = _report_config(tmp_path)
    module_dir = config.project_dir / "trait_modules"
    module_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trait": ["IQ"],
            "module": [1],
            "n_cells": [12],
            "module_signal_fraction": [0.62],
            "cumulative_signal_fraction": [0.62],
            "residual_signal_fraction_after_module": [0.18],
            "module_evidence_score": [0.87],
            "module_evidence_status": ["strong"],
            "mean_specificity": [0.91],
            "conditional_ldsc_neg_log10_p": [4.2],
            "conditional_ldsc_p": [6.3e-5],
            "stability_profile_similarity": [0.96],
            "stability_assignment_fraction": [0.94],
            "stability_null_pvalue": [0.02],
            "specificity_enrichment": [2.3],
            "specificity_null_pvalue": [0.01],
            "max_profile_similarity": [0.0],
            "most_similar_module": [-1],
        }
    ).to_csv(module_dir / "module_report_IQ.module_summary.csv", index=False)
    pd.DataFrame(
        {
            "trait": ["IQ", "IQ"],
            "iteration": [1, 2],
            "decision": ["accepted", "stopped"],
            "reason": ["accepted", "min_module_cells"],
            "candidate_signal_fraction": [0.62, None],
            "accepted_signal_fraction": [0.62, 0.62],
            "residual_signal_fraction_before": [1.0, 0.18],
            "residual_signal_fraction_after": [0.18, 0.18],
            "module_evidence_score": [0.87, None],
            "module_evidence_status": ["strong", "not_evaluated"],
            "candidate_n_cells": [12, 1],
            "remaining_cells_before": [20, 8],
            "remaining_cells_after": [8, 8],
            "conditional_ldsc_neg_log10_p": [4.2, None],
            "conditional_ldsc_p": [6.3e-5, None],
            "stability_profile_similarity": [0.96, None],
            "stability_assignment_fraction": [0.94, None],
            "stability_null_pvalue": [0.02, None],
        }
    ).to_csv(module_dir / "module_report_IQ.module_trace.csv", index=False)
    pd.DataFrame(
        {
            "trait": ["IQ"],
            "n_seed_cells": [20],
            "n_modules": [1],
            "n_module_cells": [12],
            "module_cell_fraction": [0.6],
            "n_assigned_seed_cells": [12],
            "assigned_seed_cell_fraction": [0.6],
            "n_rejected_seed_cells": [3],
            "rejected_seed_cell_fraction": [0.15],
            "n_unresolved_seed_cells": [5],
            "unresolved_seed_cell_fraction": [0.25],
            "top_module_signal_fraction": [0.62],
            "accepted_signal_fraction": [0.62],
            "residual_signal_fraction": [0.18],
            "resolution_status": ["single_module_dominant"],
        }
    ).to_csv(module_dir / "module_report_IQ.run_summary.csv", index=False)
    pd.DataFrame(
        {
            "trait": ["IQ", "IQ"],
            "module": [1, 1],
            "rank": [1, 2],
            "gene": ["GENE1", "GENE2"],
            "score": [3.2, 2.8],
        }
    ).to_csv(module_dir / "module_report_IQ.module_genes.csv", index=False)
    pd.DataFrame(
        {
            "trait": ["IQ"],
            "module_a": [1],
            "module_b": [2],
            "profile_similarity": [0.12],
            "max_allowed_profile_similarity": [0.9],
            "profile_distinct": [True],
            "marker_jaccard": [0.1],
            "marker_gene_universe_size": [100],
            "marker_overlap_null_pvalue": [0.12],
            "shared_marker_count": [1],
            "module_a_marker_count": [5],
            "module_b_marker_count": [6],
            "shared_top_genes": ["GENE2"],
            "module_a_only_top_genes": ["GENE1;GENE3"],
            "module_b_only_top_genes": ["GENE4;GENE5"],
            "module_a_n_cells": [12],
            "module_b_n_cells": [8],
            "module_a_signal_fraction": [0.62],
            "module_b_signal_fraction": [0.22],
            "module_a_evidence_status": ["strong"],
            "module_b_evidence_status": ["moderate"],
        }
    ).to_csv(module_dir / "module_report_IQ.module_pairs.csv", index=False)

    js_dir = tmp_path / "js_data"
    js_dir.mkdir()

    _export_trait_module_js(config, js_dir)

    js_text = (js_dir / "trait_modules.js").read_text()
    payload = js_text.split("=", 1)[1].rstrip(";")
    data = json.loads(payload)

    assert data["IQ"]["summary"][0]["module"] == 1
    assert data["IQ"]["summary"][0]["specificity_enrichment"] == 2.3
    assert data["IQ"]["summary"][0]["module_signal_fraction"] == 0.62
    assert data["IQ"]["summary"][0]["residual_signal_fraction_after_module"] == 0.18
    assert data["IQ"]["summary"][0]["module_evidence_score"] == 0.87
    assert data["IQ"]["summary"][0]["module_evidence_status"] == "strong"
    assert data["IQ"]["summary"][0]["conditional_ldsc_neg_log10_p"] == 4.2
    assert data["IQ"]["summary"][0]["stability_profile_similarity"] == 0.96
    assert data["IQ"]["run_summary"]["resolution_status"] == "single_module_dominant"
    assert data["IQ"]["run_summary"]["accepted_signal_fraction"] == 0.62
    assert data["IQ"]["run_summary"]["n_assigned_seed_cells"] == 12
    assert data["IQ"]["run_summary"]["rejected_seed_cell_fraction"] == 0.15
    assert data["IQ"]["run_summary"]["n_unresolved_seed_cells"] == 5
    assert data["IQ"]["trace"][0]["decision"] == "accepted"
    assert data["IQ"]["trace"][1]["reason"] == "min_module_cells"
    assert data["IQ"]["trace"][0]["candidate_signal_fraction"] == 0.62
    assert data["IQ"]["trace"][0]["residual_signal_fraction_after"] == 0.18
    assert data["IQ"]["trace"][0]["module_evidence_score"] == 0.87
    assert data["IQ"]["trace"][0]["module_evidence_status"] == "strong"
    assert data["IQ"]["trace"][0]["conditional_ldsc_p"] == 6.3e-05
    assert data["IQ"]["trace"][0]["stability_null_pvalue"] == 0.02
    assert data["IQ"]["genes"][0]["gene"] == "GENE1"
    assert data["IQ"]["genes"][1]["rank"] == 2
    assert data["IQ"]["pairs"][0]["module_a"] == 1
    assert data["IQ"]["pairs"][0]["module_b"] == 2
    assert data["IQ"]["pairs"][0]["profile_distinct"] is True
    assert data["IQ"]["pairs"][0]["marker_overlap_null_pvalue"] == 0.12
    assert data["IQ"]["pairs"][0]["shared_top_genes"] == "GENE2"


def test_trait_module_outputs_newer_than_detects_metadata_staleness(tmp_path):
    config = _report_config(tmp_path)
    module_dir = config.project_dir / "trait_modules"
    module_dir.mkdir(parents=True)
    metadata_path = config.project_dir / "report_data" / "spot_metadata.csv"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text("spot\ncell_1\n")

    specificity_path = module_dir / "module_report_IQ.module_specificity.csv"
    pd.DataFrame(
        {
            "spot": ["cell_1"],
            "best_trait_module": [1],
            "best_module_specificity": [0.9],
        }
    ).to_csv(specificity_path, index=False)
    os.utime(metadata_path, (1000, 1000))
    os.utime(specificity_path, (2000, 2000))

    assert _trait_module_outputs_newer_than(config, [metadata_path], metadata_only=True)


def test_prepare_3d_visualization_writes_trait_module_specificity_views(tmp_path, monkeypatch):
    config = SimpleNamespace(annotation_list=["annotation"], downsampling_n_spots_3d=100)
    metadata = pd.DataFrame(
        {
            "spot": ["cell_1", "cell_2", "cell_3"],
            "sample_name": ["sampleA", "sampleA", "sampleB"],
            "3d_x": [0.0, 1.0, 2.0],
            "3d_y": [0.0, 1.0, 2.0],
            "3d_z": [0.0, 0.0, 1.0],
            "IQ": [3.2, 0.4, 1.1],
            "IQ_best_trait_module": [1, -1, 2],
            "IQ_best_module_specificity": [0.9, 0.0, 0.8],
            "IQ_second_best_trait_module": [2, -1, 1],
            "IQ_second_best_module_specificity": [0.1, 0.0, 0.2],
            "IQ_module_specificity_margin": [0.8, 0.0, 0.6],
            "IQ_module_specificity_ratio": [9.0, None, 4.0],
            "IQ_module_1_specificity": [0.9, 0.0, 0.2],
            "IQ_module_2_specificity": [0.1, 0.0, 0.8],
            "annotation": ["A", "B", "A"],
        }
    )
    calls = []

    class FakePlotter:
        pass

    def fake_three_d_plot(**kwargs):
        calls.append(kwargs)
        return FakePlotter()

    def fake_three_d_plot_save(plotter, filename):
        Path(filename).with_suffix(".html").write_text("ok")

    monkeypatch.setattr(report_data, "three_d_plot", fake_three_d_plot)
    monkeypatch.setattr(report_data, "three_d_plot_save", fake_three_d_plot_save)

    result = _prepare_3d_visualization(
        config,
        metadata,
        ["IQ"],
        tmp_path / "report_data",
        tmp_path / "web_report",
    )

    web_3d = tmp_path / "web_report" / "spatial_3d"
    assert result == "spatial_3d/spatial_3d_anno_annotation.html"
    assert (web_3d / "spatial_3d_trait_IQ.html").exists()
    assert (web_3d / "spatial_3d_trait_module_IQ_best_module_specificity.html").exists()
    assert (web_3d / "spatial_3d_trait_module_IQ_module_specificity_margin.html").exists()
    assert (web_3d / "spatial_3d_trait_module_IQ_module_1_specificity.html").exists()
    assert (web_3d / "spatial_3d_trait_module_IQ_module_2_specificity.html").exists()
    assert not (web_3d / "spatial_3d_trait_module_IQ_second_best_trait_module.html").exists()
    assert not (web_3d / "spatial_3d_trait_module_IQ_module_specificity_ratio.html").exists()
    continuous_keys = [call["keys"][0] for call in calls if call["keys"][0] != "annotation"]
    assert continuous_keys == [
        "IQ",
        "IQ_best_module_specificity",
        "IQ_second_best_module_specificity",
        "IQ_module_specificity_margin",
        "IQ_module_1_specificity",
        "IQ_module_2_specificity",
    ]
    saved = ad.read_h5ad(tmp_path / "report_data" / "spatial_3d" / "spatial_3d.h5ad")
    assert "IQ_module_1_specificity" in saved.obs.columns


def test_report_missing_files_requires_trait_module_3d_views(tmp_path):
    project_dir = tmp_path / "project"
    report_data_dir = project_dir / "report_data"
    web_report_dir = project_dir / "gsmap_web_report"
    module_dir = project_dir / "trait_modules"
    js_dir = web_report_dir / "js_data"
    spatial_3d_dir = web_report_dir / "spatial_3d"
    data_3d_dir = report_data_dir / "spatial_3d"
    for path in [
        report_data_dir / "gss_stats",
        report_data_dir / "manhattan_data",
        js_dir / "gss_stats",
        module_dir,
        spatial_3d_dir,
        data_3d_dir,
    ]:
        path.mkdir(parents=True)

    for path in [
        web_report_dir / "index.html",
        web_report_dir / "report_meta.json",
        js_dir / "sample_index.js",
        js_dir / "report_meta.js",
        js_dir / "cauchy_results.js",
        report_data_dir / "spot_metadata.csv",
        report_data_dir / "gene_list.csv",
        report_data_dir / "cauchy_results.csv",
        report_data_dir / "gss_stats" / "gene_trait_correlation_IQ.csv",
        report_data_dir / "manhattan_data" / "IQ_manhattan.csv",
        js_dir / "gss_stats" / "gene_trait_correlation_IQ.js",
        js_dir / "manhattan_IQ.js",
        js_dir / "umap_data.js",
        report_data_dir / "umap_data.csv",
        data_3d_dir / "spatial_3d.h5ad",
        spatial_3d_dir / "spatial_3d_trait_IQ.html",
        spatial_3d_dir / "spatial_3d_anno_annotation.html",
        web_report_dir / "gene_diagnostic_plots",
    ]:
        if path.suffix:
            path.write_text("ok")
        else:
            path.mkdir(exist_ok=True)

    pd.DataFrame(
        {
            "spot": ["cell_1"],
            "best_trait_module": [1],
            "best_module_specificity": [0.9],
            "second_best_trait_module": [-1],
            "second_best_module_specificity": [0.0],
            "module_specificity_margin": [0.9],
            "module_specificity_ratio": [None],
            "module_1_specificity": [0.9],
        }
    ).to_csv(module_dir / "module_report_IQ.module_specificity.csv", index=False)

    config = SimpleNamespace(
        dataset_type=DatasetType.SPATIAL_3D,
        web_report_dir=web_report_dir,
        report_data_dir=report_data_dir,
        project_dir=project_dir,
        project_name="module_report",
        trait_name_list=["IQ"],
        annotation_list=["annotation"],
        sample_h5ad_dict=None,
        generate_multi_sample_plots=False,
        concatenated_latent_adata_path=None,
    )

    _, missing_web_files = get_report_missing_files(config)

    assert (
        spatial_3d_dir / "spatial_3d_trait_module_IQ_best_module_specificity.html"
    ) in missing_web_files
    assert (
        spatial_3d_dir / "spatial_3d_trait_module_IQ_second_best_module_specificity.html"
    ) in missing_web_files
    assert (
        spatial_3d_dir / "spatial_3d_trait_module_IQ_module_specificity_margin.html"
    ) in missing_web_files
    assert (
        spatial_3d_dir / "spatial_3d_trait_module_IQ_second_best_trait_module.html"
    ) not in missing_web_files
    assert (
        spatial_3d_dir / "spatial_3d_trait_module_IQ_module_specificity_ratio.html"
    ) not in missing_web_files
    assert (
        spatial_3d_dir / "spatial_3d_trait_module_IQ_module_1_specificity.html"
    ) in missing_web_files


def test_report_missing_files_marks_trait_module_outputs_as_stale_inputs(tmp_path):
    project_dir = tmp_path / "project"
    report_data_dir = project_dir / "report_data"
    web_report_dir = project_dir / "gsmap_web_report"
    module_dir = project_dir / "trait_modules"
    js_dir = web_report_dir / "js_data"
    spatial_3d_dir = web_report_dir / "spatial_3d"
    data_3d_dir = report_data_dir / "spatial_3d"
    concat_adata_path = project_dir / "latent_to_gene" / "concatenated_latent_adata.h5ad"
    for path in [
        report_data_dir / "gss_stats",
        report_data_dir / "manhattan_data",
        js_dir / "gss_stats",
        module_dir,
        spatial_3d_dir,
        data_3d_dir,
        concat_adata_path.parent,
    ]:
        path.mkdir(parents=True)

    existing_files = [
        web_report_dir / "index.html",
        web_report_dir / "report_meta.json",
        js_dir / "sample_index.js",
        js_dir / "report_meta.js",
        js_dir / "cauchy_results.js",
        js_dir / "trait_modules.js",
        js_dir / "sample_sampleA_spatial.js",
        report_data_dir / "spot_metadata.csv",
        report_data_dir / "gene_list.csv",
        report_data_dir / "cauchy_results.csv",
        report_data_dir / "gss_stats" / "gene_trait_correlation_IQ.csv",
        report_data_dir / "manhattan_data" / "IQ_manhattan.csv",
        js_dir / "gss_stats" / "gene_trait_correlation_IQ.js",
        js_dir / "manhattan_IQ.js",
        js_dir / "umap_data.js",
        report_data_dir / "umap_data.csv",
        data_3d_dir / "spatial_3d.h5ad",
        spatial_3d_dir / "spatial_3d_trait_IQ.html",
        spatial_3d_dir / "spatial_3d_anno_annotation.html",
        spatial_3d_dir / "spatial_3d_trait_module_IQ_best_module_specificity.html",
        spatial_3d_dir / "spatial_3d_trait_module_IQ_module_1_specificity.html",
        concat_adata_path,
    ]
    for path in existing_files:
        path.write_text("ok")
        os.utime(path, (1000, 1000))
    (web_report_dir / "gene_diagnostic_plots").mkdir()

    specificity_path = module_dir / "module_report_IQ.module_specificity.csv"
    pd.DataFrame(
        {
            "spot": ["cell_1"],
            "best_trait_module": [1],
            "best_module_specificity": [0.9],
            "second_best_trait_module": [-1],
            "second_best_module_specificity": [0.0],
            "module_specificity_margin": [0.9],
            "module_specificity_ratio": [None],
            "module_1_specificity": [0.9],
        }
    ).to_csv(specificity_path, index=False)
    os.utime(specificity_path, (2000, 2000))

    config = SimpleNamespace(
        dataset_type=DatasetType.SPATIAL_3D,
        web_report_dir=web_report_dir,
        report_data_dir=report_data_dir,
        project_dir=project_dir,
        project_name="module_report",
        trait_name_list=["IQ"],
        annotation_list=["annotation"],
        sample_h5ad_dict={"sampleA": tmp_path / "sampleA.h5ad"},
        generate_multi_sample_plots=False,
        concatenated_latent_adata_path=concat_adata_path,
    )

    missing_data_files, missing_web_files = get_report_missing_files(config)

    assert report_data_dir / "spot_metadata.csv" in missing_data_files
    assert report_data_dir / "umap_data.csv" in missing_data_files
    assert data_3d_dir / "spatial_3d.h5ad" in missing_data_files
    assert js_dir / "umap_data.js" in missing_web_files
    assert js_dir / "sample_sampleA_spatial.js" in missing_web_files
    assert js_dir / "trait_modules.js" in missing_web_files
    assert (
        spatial_3d_dir / "spatial_3d_trait_module_IQ_best_module_specificity.html"
    ) in missing_web_files

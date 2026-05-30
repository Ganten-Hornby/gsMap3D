import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gsMap.report.report_data import (
    _export_per_sample_spatial_js,
    _export_trait_module_js,
    _merge_trait_module_metadata,
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
            "is_module_cell": [True, True],
        }
    ).to_csv(module_dir / "module_report_IQ.cell_modules.csv", index=False)

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
    assert merged["IQ_is_module_cell"].tolist() == [True, False, True]


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
            "IQ_is_module_cell": ["True", "False"],
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
    assert data["IQ_is_module_cell"] == [True, False]


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
            "trait": ["IQ", "IQ"],
            "module": [1, 1],
            "rank": [1, 2],
            "gene": ["GENE1", "GENE2"],
            "score": [3.2, 2.8],
        }
    ).to_csv(module_dir / "module_report_IQ.module_genes.csv", index=False)

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

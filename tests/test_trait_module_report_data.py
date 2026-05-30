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
            "module_status": ["assigned", "rejected"],
            "module_reason": ["accepted", "specificity_null_pvalue"],
        }
    ).to_csv(module_dir / "module_report_IQ.cell_modules.csv", index=False)
    pd.DataFrame(
        {
            "spot": ["cell_1", "cell_2", "cell_3"],
            "best_trait_module": [2, 2, 1],
            "best_module_specificity": [0.9234, 0.2211, 0.5123],
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
    assert merged["IQ_best_trait_module"].tolist() == [2, 2, 1]
    assert merged["IQ_best_module_specificity"].tolist() == [0.923, 0.221, 0.512]
    assert merged["IQ_module_1_specificity"].tolist() == [0.123, 0.111, 0.512]
    assert merged["IQ_module_2_specificity"].tolist() == [0.923, 0.221, 0.112]
    assert merged["IQ_is_module_cell"].tolist() == [True, False, True]
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
            "IQ_module_1_specificity": [0.1234, 0.1111],
            "IQ_module_2_specificity": [0.9234, 0.2211],
            "IQ_is_module_cell": ["True", "False"],
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
    assert data["IQ_module_1_specificity"] == [0.123, 0.111]
    assert data["IQ_module_2_specificity"] == [0.923, 0.221]
    assert data["IQ_is_module_cell"] == [True, False]
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

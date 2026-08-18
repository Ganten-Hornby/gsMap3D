import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

from gsMap.config.report_config import ReportConfig
from gsMap.report.diagnosis import load_gwas_data
from gsMap.report.report_data import (
    _available_cpu_count,
    _load_chrom_tick_positions,
    _ManhattanWeightContext,
    _materialize_plot_alias,
    _per_sample_spatial_outputs_current,
    _prepare_manhattan_for_trait,
    _render_single_sample_gene_plot_task,
    _report_worker_count,
    _sample_spatial_export_signature,
    _write_text_if_changed,
)
from gsMap.report.spatial_payload import SPATIAL_PAYLOAD_VERSION


def test_report_3d_display_default_balances_density_and_interaction():
    field = ReportConfig.__dataclass_fields__["downsampling_n_spots_3d"]

    assert field.default == 100_000


def test_available_cpu_count_prefers_slurm_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64")

    assert _available_cpu_count() == 7


def test_report_worker_count_uses_existing_cpu_configuration(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.setattr("gsMap.report.report_data._available_cpu_count", lambda: 64)
    config = SimpleNamespace(ldsc_read_workers=15, ldsc_compute_workers=10)

    assert _report_worker_count(config, ceiling=20) == 15
    assert _report_worker_count(config, ceiling=4) == 4


def test_load_gwas_data_can_select_columns_and_calculate_p(tmp_path):
    source = tmp_path / "trait.sumstats.gz"
    pd.DataFrame(
        {
            "SNP": ["rs1", "rs2"],
            "A1": ["A", "C"],
            "A2": ["G", "T"],
            "Z": [0.0, 2.0],
            "N": [100, 100],
        }
    ).to_csv(source, sep="\t", index=False, compression="gzip")

    result = load_gwas_data(source, usecols=["SNP", "Z"])

    assert result.columns.tolist() == ["SNP", "Z", "P"]
    assert result["SNP"].tolist() == ["rs1", "rs2"]
    assert result.loc[0, "P"] == 1.0
    assert 0 < result.loc[1, "P"] < 0.1


def test_prepare_manhattan_preserves_schema_and_weight_order(tmp_path):
    source = tmp_path / "trait.sumstats.gz"
    pd.DataFrame(
        {
            "A1": ["A", "C", "G", "T"],
            "A2": ["G", "T", "A", "C"],
            "Z": [1.0, 4.0, 3.0, 10.0],
            "N": [100, 100, 100, 100],
            "SNP": ["rs3", "rs2", "rs1", "not_in_weights"],
        }
    ).to_csv(source, sep="\t", index=False, compression="gzip")

    context = _ManhattanWeightContext(
        snp_index=pd.Index(["rs1", "rs2", "rs3"]),
        chromosome=np.array([1, 1, 2]),
        base_pair=np.array([10, 20, 5]),
        matrix=sparse.csr_matrix([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]),
        target_gene_indices=np.array([0, 1]),
        target_genes=np.array(["GENE1", "GENE2"]),
    )
    config = SimpleNamespace(
        sumstats_config_dict={"trait": source},
        top_corr_genes=1,
    )
    trait_pcc = pd.DataFrame({"gene": ["GENE2", "GENE1"], "PCC": [0.9, 0.1]})

    ticks = _prepare_manhattan_for_trait(config, "trait", context, trait_pcc, tmp_path)
    result = pd.read_csv(tmp_path / "trait_manhattan.csv")

    assert result.columns.tolist() == [
        "SNP",
        "A1",
        "A2",
        "Z",
        "N",
        "P",
        "CHR",
        "BP",
        "GENE",
        "is_top_pcc",
        "BP_cum",
        "CHR_INDEX",
    ]
    assert result["SNP"].tolist() == ["rs1", "rs2", "rs3"]
    assert result["GENE"].fillna("None").tolist() == ["GENE1", "GENE2", "None"]
    assert result["is_top_pcc"].tolist() == [False, True, False]
    assert set(ticks) == {1, 2}


def test_load_chrom_tick_positions_from_cached_csv(tmp_path):
    path = tmp_path / "cached.csv"
    pd.DataFrame({"CHR": [1, 1, 2, 2], "BP_cum": [10.0, 20.0, 40.0, 60.0]}).to_csv(
        path, index=False
    )

    assert _load_chrom_tick_positions(path) == {1: 15.0, 2: 50.0}


def test_materialize_plot_alias_replaces_target(tmp_path):
    source = tmp_path / "source.png"
    target = tmp_path / "target.png"
    source.write_bytes(b"new-image")
    target.write_bytes(b"old-image")

    _materialize_plot_alias(source, target)

    assert target.read_bytes() == b"new-image"


def test_diagnostic_worker_publishes_complete_png_atomically(tmp_path):
    output = tmp_path / "plot.png"
    _render_single_sample_gene_plot_task(
        {
            "gene": "GENE1",
            "sample_name": "sample",
            "plot_type": "gss",
            "coords": np.array([[0.0, 0.0], [1.0, 1.0]]),
            "values": np.array([0.0, 1.0]),
            "output_path": output,
            "dpi": 20,
        }
    )

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(tmp_path.glob(".*.tmp.png"))


def test_text_output_and_per_sample_cache_are_content_and_dependency_aware(tmp_path):
    data_dir = tmp_path / "report_data"
    js_dir = tmp_path / "web" / "js_data"
    data_dir.mkdir()
    js_dir.mkdir(parents=True)
    metadata = data_dir / "spot_metadata.csv"
    metadata.write_text("spot,sample_name\na,sample 1\n")

    sample_index = js_dir / "sample_index.js"
    sample_data = js_dir / "sample_sample_1_spatial.js"
    assert _write_text_if_changed(sample_index, "index")
    assert _write_text_if_changed(sample_data, "sample")
    first_mtime = sample_index.stat().st_mtime_ns
    assert not _write_text_if_changed(sample_index, "index")
    assert sample_index.stat().st_mtime_ns == first_mtime

    meta = {"samples": ["sample 1"]}
    manifest = {
        "version": SPATIAL_PAYLOAD_VERSION,
        "signature": _sample_spatial_export_signature(data_dir, meta),
        "outputs": [sample_index.name, sample_data.name],
    }
    (js_dir / "sample_spatial_manifest.json").write_text(json.dumps(manifest))
    assert _per_sample_spatial_outputs_current(data_dir, js_dir, meta)
    assert not _per_sample_spatial_outputs_current(
        data_dir, js_dir, {"samples": ["different sample"]}
    )

    replacement = data_dir / "replacement.csv"
    replacement.write_text("spot,sample_name\nb,sample 1\n")
    os.replace(replacement, metadata)
    assert not _per_sample_spatial_outputs_current(data_dir, js_dir, meta)

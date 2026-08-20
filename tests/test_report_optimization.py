import json
import os
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from gsMap.config.latent2gene_config import DatasetType
from gsMap.config.quick_mode_config import (
    _report_visualization_config_matches,
    get_report_missing_files,
)
from gsMap.config.report_config import ReportConfig
from gsMap.latent2gene.marker_layout import SequentialMarkerRowWriter
from gsMap.report.diagnosis import load_gwas_data
from gsMap.report.report_data import (
    _available_cpu_count,
    _calculate_pcc_for_traits_fast,
    _calculate_umap_from_embeddings,
    _cauchy_payloads_current,
    _export_annotation_schemas_js,
    _export_cauchy_js,
    _load_chrom_tick_positions,
    _load_gss_pcc_sample,
    _ManhattanWeightContext,
    _materialize_plot_alias,
    _per_sample_spatial_outputs_current,
    _prepare_manhattan_for_trait,
    _read_marker_score_rows,
    _render_single_sample_gene_plot_task,
    _report_worker_count,
    _sample_spatial_export_signature,
    _write_text_if_changed,
)
from gsMap.report.spatial_payload import SPATIAL_PAYLOAD_VERSION


def test_report_3d_display_default_balances_density_and_interaction():
    field = ReportConfig.__dataclass_fields__["downsampling_n_spots_3d"]

    assert field.default == 100_000


def test_report_umap_and_2d_defaults_prioritize_visual_density():
    umap_field = ReportConfig.__dataclass_fields__["downsampling_n_spots_umap"]
    spatial_2d_field = ReportConfig.__dataclass_fields__["downsampling_n_spots_2d"]

    assert umap_field.default == 50_000
    assert spatial_2d_field.default == 100_000


def test_report_completion_detects_changed_visualization_limits(tmp_path):
    config = SimpleNamespace(
        web_report_dir=tmp_path,
        downsampling_n_spots_umap=50_000,
        downsampling_n_spots_2d=100_000,
        downsampling_n_spots_3d=100_000,
    )
    (tmp_path / "report_meta.json").write_text(
        json.dumps(
            {
                "downsampling_n_spots_umap": 50_000,
                "downsampling_n_spots_2d": 100_000,
                "downsampling_n_spots_3d": 100_000,
            }
        )
    )

    assert _report_visualization_config_matches(config)

    config.downsampling_n_spots_2d = 50_000
    assert not _report_visualization_config_matches(config)


def test_report_completion_requires_lazy_cauchy_index_and_manifest(tmp_path):
    web_report_dir = tmp_path / "web"
    js_data_dir = web_report_dir / "js_data"
    report_data_dir = tmp_path / "data"
    js_data_dir.mkdir(parents=True)
    report_data_dir.mkdir()
    for path in (
        web_report_dir / "index.html",
        web_report_dir / "report_meta.json",
        js_data_dir / "sample_index.js",
        js_data_dir / "report_meta.js",
        js_data_dir / "cauchy_results.js",
        report_data_dir / "spot_metadata.csv",
        report_data_dir / "gene_list.csv",
        report_data_dir / "cauchy_results.csv",
    ):
        path.touch()

    config = SimpleNamespace(
        dataset_type=DatasetType.SCRNA_SEQ,
        web_report_dir=web_report_dir,
        report_data_dir=report_data_dir,
        trait_name_list=[],
        annotation_list=[],
        generate_multi_sample_plots=False,
        concatenated_latent_adata_path=None,
    )
    _, missing_web = get_report_missing_files(config)
    assert js_data_dir / "cauchy_index.js" in missing_web
    assert js_data_dir / "cauchy_manifest.json" in missing_web
    assert js_data_dir / "cauchy_results.js" not in missing_web

    (js_data_dir / "cauchy_index.js").touch()
    (js_data_dir / "cauchy_manifest.json").touch()
    missing_data, missing_web = get_report_missing_files(config)
    assert missing_data == []
    assert missing_web == []


def test_cauchy_export_is_file_scheme_compatible_and_sharded(tmp_path):
    data_dir = tmp_path / "report_data"
    js_data_dir = tmp_path / "js_data"
    data_dir.mkdir()
    js_data_dir.mkdir()
    rows = []
    for result_type, sample_names in (
        ("aggregated", ["All Samples"]),
        ("sample", ["sample_a", "sample_b"]),
    ):
        for trait in ("trait/a", "trait_b"):
            for annotation_name in ("cell type", "region"):
                for annotation_index, annotation in enumerate(("A", "B")):
                    for sample_name in sample_names:
                        value = annotation_index + (0.5 if sample_name == "sample_b" else 0.0)
                        rows.append(
                            {
                                "trait": trait,
                                "annotation_name": annotation_name,
                                "annotation": annotation,
                                "p_cauchy": 10 ** -(value + 1),
                                "p_median": 10 ** -(value + 0.5),
                                "top_95_quantile": value + 2,
                                "sig_spots": 3,
                                "total_spots": 10,
                                "type": result_type,
                                "sample_name": sample_name,
                                "mlog10_p_cauchy": value + 1,
                                "mlog10_p_median": value + 0.5,
                            }
                        )
    pd.DataFrame(rows).to_csv(data_dir / "cauchy_results.csv", index=False)
    (js_data_dir / "cauchy_results.js").write_text("legacy monolith")

    _export_cauchy_js(data_dir, js_data_dir)

    index_text = (js_data_dir / "cauchy_index.js").read_text(encoding="utf-8")
    index = json.loads(index_text.split("=", 1)[1].rstrip(";"))
    assert index["version"] == 2
    assert set(index["table"]) == {"all", "sample"}
    entry = index["table"]["sample"]["trait/a"]["cell type"]
    assert entry["n_rows"] == 4
    assert entry["file"].startswith("js_data/cauchy/")
    shard_text = (tmp_path / entry["file"]).read_text(encoding="utf-8")
    assert "window.GSMAP_CAUCHY_SHARDS" in shard_text
    assert "fetch(" not in shard_text
    assert not (js_data_dir / "cauchy_results.js").exists()
    assert _cauchy_payloads_current(data_dir / "cauchy_results.csv", js_data_dir)

    heatmap_entry = index["heatmap"]["sample"]["cell type"]
    assert heatmap_entry["aggregation"] == "median_across_samples"
    assert heatmap_entry["n_rows"] == 4


def test_annotation_schema_export_is_small_canonical_palette(tmp_path):
    metadata = pd.DataFrame(
        {
            "region": ["cortex", "thalamus", "cortex", None],
            "unused": [1, 2, 3, 4],
        }
    )

    _export_annotation_schemas_js(tmp_path, metadata, ["region", "missing"])

    content = (tmp_path / "annotation_schemas.js").read_text(encoding="utf-8")
    schemas = json.loads(content.split("=", 1)[1].rstrip(";"))
    assert set(schemas) == {"region"}
    assert schemas["region"]["categories"] == ["NaN", "cortex", "thalamus"]
    assert len(schemas["region"]["colors"]) == 3
    assert schemas["region"]["colors"][0] == "#d3d3d3"


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


def test_nonfinite_precomputed_umap_is_recomputed_from_embedding(monkeypatch):
    adata = ad.AnnData(X=np.zeros((4, 1), dtype=np.float32))
    adata.obsm["emb_cell"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    adata.obsm["X_umap"] = np.array(
        [[0.0, 1.0], [np.nan, np.nan], [2.0, 3.0], [4.0, 5.0]],
        dtype=np.float32,
    )
    calls = []

    def fake_neighbors(target, *, use_rep):
        calls.append(("neighbors", use_rep))

    expected = np.array(
        [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]],
        dtype=np.float32,
    )

    def fake_umap(target):
        calls.append(("umap", None))
        target.obsm["X_umap"] = expected.copy()

    monkeypatch.setattr("gsMap.report.report_data.sc.pp.neighbors", fake_neighbors)
    monkeypatch.setattr("gsMap.report.report_data.sc.tl.umap", fake_umap)

    result = _calculate_umap_from_embeddings(adata, "emb_cell")

    assert calls == [("neighbors", "emb_cell"), ("umap", None)]
    np.testing.assert_array_equal(result, expected)


def test_finite_precomputed_umap_is_reused(monkeypatch):
    adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
    adata.obsm["emb_cell"] = np.ones((3, 2), dtype=np.float32)
    expected = np.arange(6, dtype=np.float32).reshape(3, 2)
    adata.obsm["X_umap"] = expected.copy()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("finite precomputed UMAP should be reused")

    monkeypatch.setattr("gsMap.report.report_data.sc.pp.neighbors", fail_if_called)
    monkeypatch.setattr("gsMap.report.report_data.sc.tl.umap", fail_if_called)

    result = _calculate_umap_from_embeddings(adata, "emb_cell")

    np.testing.assert_array_equal(result, expected)


def test_report_marker_reader_preserves_requested_row_and_gene_order(tmp_path):
    values = np.arange(40, dtype=np.float16).reshape(8, 5)
    data_path = tmp_path / "marker_scores.dat"
    values.tofile(data_path)

    result = _read_marker_score_rows(
        data_path=data_path,
        shape=values.shape,
        dtype=values.dtype,
        marker_rows=np.array([6, 1, 2, 7]),
        gene_positions=np.array([4, 0, 2]),
        n_workers=3,
    )

    np.testing.assert_array_equal(result, values[[6, 1, 2, 7]][:, [4, 0, 2]])


def test_batched_pcc_matches_independent_trait_correlations():
    rng = np.random.default_rng(17)
    spots = np.array([f"spot_{index}" for index in range(80)])
    marker_scores = rng.normal(size=(80, 9)).astype(np.float32)
    centered = marker_scores - marker_scores.mean(axis=0)
    ssq = np.sum(centered**2, axis=0)
    traits = ["trait_a", "trait_b", "trait_c"]
    trait_values = rng.normal(size=(80, len(traits))).astype(np.float32)
    ldsc_results = pd.DataFrame(trait_values, index=spots, columns=traits)

    result = _calculate_pcc_for_traits_fast(
        traits,
        ldsc_results,
        spots,
        centered,
        ssq,
    )
    expected = np.column_stack(
        [
            np.array(
                [
                    np.corrcoef(marker_scores[:, gene], trait_values[:, trait])[0, 1]
                    for gene in range(marker_scores.shape[1])
                ]
            )
            for trait in range(len(traits))
        ]
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_pcc_sample_reader_honors_sequential_marker_row_layout(tmp_path):
    latent_dir = tmp_path / "latent_to_gene"
    latent_dir.mkdir()
    source_path = latent_dir / "concatenated_latent_adata.h5ad"
    obs_names = ["spot_a", "spot_b", "spot_c", "spot_d"]
    var_names = ["g1", "g2", "g3"]
    source = ad.AnnData(
        X=np.zeros((4, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "sample_name": ["sample"] * 4,
                "cell_type": ["A", "B", "C", "D"],
            },
            index=obs_names,
        ),
        var=pd.DataFrame(index=var_names),
    )
    source.write_h5ad(source_path)

    marker_path = latent_dir / "marker_scores.dat"
    marker_values = np.array(
        [[20, 21, 22], [0, 1, 2], [30, 31, 32], [10, 11, 12]],
        dtype=np.float16,
    )
    marker_values.tofile(marker_path)
    layout_writer = SequentialMarkerRowWriter(marker_path, n_obs=4)
    layout_writer.append(np.array([2, 0, 3, 1]))
    row_layout = layout_writer.finalize()
    marker_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "shape": [4, 3],
                "dtype": "float16",
                "complete": True,
                "row_layout": row_layout,
            }
        )
    )
    mean_frac_path = latent_dir / "mean_frac.parquet"
    pd.DataFrame({"frac": [0.5, 0.5]}, index=["g2", "g1"]).to_parquet(mean_frac_path)
    ldsc_results = pd.DataFrame(
        {"sample_name": ["sample"] * 4, "trait": np.arange(4)},
        index=obs_names,
    )
    config = SimpleNamespace(
        marker_scores_memmap_path=marker_path,
        concatenated_latent_adata_path=source_path,
        mean_frac_path=mean_frac_path,
        downsampling_n_spots_pcc=4,
        annotation_list=["cell_type"],
        memmap_tmp_dir=None,
        ldsc_read_workers=2,
        ldsc_compute_workers=2,
    )

    sample = _load_gss_pcc_sample(config, ldsc_results, tmp_path / "report_data")

    assert sample.analysis_spots.tolist() == obs_names
    assert sample.gene_names == ["g2", "g1"]
    assert sample.annotations.tolist() == ["A", "B", "C", "D"]
    np.testing.assert_array_equal(
        sample.marker_scores,
        marker_values[[1, 3, 0, 2]][:, [1, 0]],
    )


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

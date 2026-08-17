import json
import os
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import gsMap.pipeline.quick_mode as quick_mode_module
from gsMap.config.cauchy_config import check_cauchy_done
from gsMap.config.spatial_ldsc_config import check_spatial_ldsc_done
from gsMap.latent2gene.gene_blocks import (
    GENE_BLOCK_FORMAT_VERSION,
    MAX_GENE_BLOCK_BYTES,
    GeneBlockMarkerStore,
    accumulate_annotated_ldscore_blocks,
    calculate_gene_block_marker_scores,
    compute_marker_score_gene_block,
    ensure_rank_row_min,
    gene_names_sha256,
)
from gsMap.latent2gene.marker_scores import (
    compute_marker_scores_3d_hierarchical_pool_jax,
    compute_marker_scores_jax,
)
from gsMap.latent2gene.provenance import artifact_signature, atomic_json_dump
from gsMap.latent2gene.rank_calculator import (
    rank_outputs_match_inputs,
    write_rank_provenance,
)
from gsMap.report.report_data import (
    ReportDataManager,
    _GeneBlockReportSource,
    _load_gss_and_calculate_stats_base,
)
from gsMap.spatial_ldsc.result_writer import (
    RESULT_METADATA_FORMAT_VERSION,
    ResultWriter,
    result_metadata_path,
)
from gsMap.spatial_ldsc.spatial_ldsc_jax import _raise_for_trait_failures


def _rank_fixture():
    ranks = np.asarray(
        [
            [1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            [0.5, 0.5, 0.5, 1.5, 2.0, 2.5, 3.0],
            [0.2, 0.7, 0.7, 0.7, 1.7, 2.7, 3.7],
            [0.4, 0.4, 1.4, 1.4, 2.4, 3.4, 4.4],
            [0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1],
            [0.3, 0.3, 0.8, 1.3, 1.8, 2.3, 2.8],
        ],
        dtype=np.float16,
    )
    neighbors = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    weights = np.asarray([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]], dtype=np.float32)
    global_log_gmean = np.asarray([0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6], dtype=np.float32)
    global_expr_frac = np.asarray([0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4], dtype=np.float32)
    return ranks, neighbors, weights, global_log_gmean, global_expr_frac


def test_gene_blocks_match_full_standard_marker_scores():
    ranks, neighbors, weights, global_log_gmean, global_expr_frac = _rank_fixture()
    flat_neighbors = neighbors.ravel()
    full = np.asarray(
        compute_marker_scores_jax(
            ranks[flat_neighbors],
            weights,
            len(neighbors),
            neighbors.shape[1],
            global_log_gmean,
            global_expr_frac,
        )
    )

    row_min = ranks.min(axis=1)
    blocks = []
    for start, end in ((0, 2), (2, 5), (5, 7)):
        blocks.append(
            compute_marker_score_gene_block(
                rank_rows=ranks[:, start:end],
                rank_inverse=flat_neighbors,
                rank_row_min=row_min,
                neighbor_weights=weights,
                global_log_gmean=global_log_gmean[start:end],
                global_expr_frac=global_expr_frac[start:end],
                n_slices=1,
                homogeneous_neighbors=neighbors.shape[1],
                hierarchical_pool=False,
            )
        )

    np.testing.assert_allclose(np.concatenate(blocks, axis=1), full, rtol=2e-3, atol=2e-3)


def test_gene_blocks_match_full_hierarchical_marker_scores():
    ranks, neighbors, weights, global_log_gmean, global_expr_frac = _rank_fixture()
    flat_neighbors = neighbors.ravel()
    full = np.asarray(
        compute_marker_scores_3d_hierarchical_pool_jax(
            ranks[flat_neighbors],
            weights,
            len(neighbors),
            3,
            1,
            global_log_gmean,
            global_expr_frac,
        )
    )

    row_min = ranks.min(axis=1)
    blocks = []
    for start, end in ((0, 3), (3, 4), (4, 7)):
        blocks.append(
            compute_marker_score_gene_block(
                rank_rows=ranks[:, start:end],
                rank_inverse=flat_neighbors,
                rank_row_min=row_min,
                neighbor_weights=weights,
                global_log_gmean=global_log_gmean[start:end],
                global_expr_frac=global_expr_frac[start:end],
                n_slices=3,
                homogeneous_neighbors=1,
                hierarchical_pool=True,
            )
        )

    np.testing.assert_allclose(np.concatenate(blocks, axis=1), full, rtol=2e-3, atol=2e-3)


def test_annotated_ldscore_is_additive_across_gene_blocks():
    marker_scores = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [0.5, 0.0, 1.0],
            [2.0, 1.5, 0.5],
            [0.0, 3.0, 4.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    snp_gene_weight = sp.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0, 2.0, 0.0, 1.0],
                [0.0, 3.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.5, 0.0, 4.0],
            ],
            dtype=np.float32,
        )
    )

    blocks = [
        (np.asarray([0, 1]), marker_scores[0:2]),
        (np.asarray([2]), marker_scores[2:3]),
        (np.asarray([3, 4]), marker_scores[3:5]),
    ]
    actual = accumulate_annotated_ldscore_blocks(snp_gene_weight, blocks)
    expected = snp_gene_weight @ marker_scores

    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1e-6, atol=1e-6)


def test_gene_block_store_reads_only_common_columns_for_spot_chunk(tmp_path):
    marker_scores = np.arange(30, dtype=np.float16).reshape(5, 6)
    blocks = []
    for start, end in ((0, 2), (2, 5), (5, 6)):
        path = tmp_path / f"block_{start}_{end}.dat"
        block = np.memmap(path, mode="w+", dtype=np.float16, shape=(5, end - start))
        block[:] = marker_scores[:, start:end]
        block.flush()
        del block
        blocks.append({"start": start, "end": end, "path": path.name, "complete": True})

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": GENE_BLOCK_FORMAT_VERSION,
                "shape": [5, 6],
                "dtype": "float16",
                "gene_block_size": 3,
                "complete": True,
                "gene_names_sha256": gene_names_sha256(
                    [f"gene-{index}" for index in range(6)]
                ),
                "source_fingerprint": "0" * 64,
                "blocks": blocks,
            }
        )
    )
    common_gene_indices = np.asarray([0, 2, 3, 5])
    weight = sp.csr_matrix(np.arange(8, dtype=np.float32).reshape(2, 4))

    store = GeneBlockMarkerStore(manifest_path)
    actual = accumulate_annotated_ldscore_blocks(
        weight,
        store.iter_spot_marker_blocks(1, 4, common_gene_indices),
    )
    expected = weight @ marker_scores[1:4, common_gene_indices].T.astype(np.float32)

    np.testing.assert_allclose(actual, expected)
    store.close()


def test_gene_block_store_reads_arbitrary_spot_and_gene_subset(tmp_path):
    marker_scores = np.arange(42, dtype=np.float16).reshape(6, 7)
    blocks = []
    for start, end in ((0, 3), (3, 5), (5, 7)):
        path = tmp_path / f"block_{start}_{end}.dat"
        block = np.memmap(path, mode="w+", dtype=np.float16, shape=(6, end - start))
        block[:] = marker_scores[:, start:end]
        block.flush()
        del block
        blocks.append({"start": start, "end": end, "path": path.name, "complete": True})

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": GENE_BLOCK_FORMAT_VERSION,
                "shape": [6, 7],
                "dtype": "float16",
                "gene_block_size": 3,
                "complete": True,
                "gene_names_sha256": gene_names_sha256(
                    [f"gene-{index}" for index in range(7)]
                ),
                "source_fingerprint": "0" * 64,
                "blocks": blocks,
            }
        )
    )

    store = GeneBlockMarkerStore(manifest_path)
    spot_indices = np.asarray([5, 1, 3])
    gene_indices = np.asarray([6, 0, 4, 2])

    actual = store.read_subset(spot_indices, gene_indices)
    expected = marker_scores[np.ix_(spot_indices, gene_indices)].astype(np.float32)

    np.testing.assert_array_equal(actual, expected)


def test_report_reads_gene_block_marker_scores_without_full_memmap(tmp_path):
    obs_names = pd.Index(["spot-a", "spot-b", "spot-c", "spot-d"])
    gene_names = pd.Index(["gene-a", "gene-b", "gene-c"])
    marker_scores = np.asarray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float16
    )
    block_dir = tmp_path / "marker_score_blocks"
    block_dir.mkdir()
    blocks = []
    for start, end in ((0, 2), (2, 3)):
        path = block_dir / f"block_{start}_{end}.dat"
        block = np.memmap(path, mode="w+", dtype=np.float16, shape=(4, end - start))
        block[:] = marker_scores[:, start:end]
        block.flush()
        del block
        blocks.append({"start": start, "end": end, "path": path.name, "complete": True})
    manifest_path = block_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": GENE_BLOCK_FORMAT_VERSION,
                "shape": [4, 3],
                "dtype": "float16",
                "gene_block_size": 2,
                "complete": True,
                "gene_names_sha256": gene_names_sha256(gene_names),
                "source_fingerprint": "0" * 64,
                "blocks": blocks,
            }
        )
    )
    latent_path = tmp_path / "concatenated_latent_adata.h5ad"
    ad.AnnData(
        X=np.zeros((4, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {"sample_name": ["s1", "s1", "s2", "s2"], "region": ["x", "x", "y", "y"]},
            index=obs_names,
        ),
        var=pd.DataFrame(index=gene_names),
    ).write_h5ad(latent_path)
    mean_frac_path = tmp_path / "mean_frac.parquet"
    pd.DataFrame({"frac": [0.5, 0.5, 0.5]}, index=gene_names).to_parquet(mean_frac_path)
    ldsc_results = pd.DataFrame(
        {"sample_name": ["s1", "s1", "s2", "s2"], "trait": [1.0, 2.0, 3.0, 4.0]},
        index=obs_names,
    )
    config = SimpleNamespace(
        marker_score_format="gene_block",
        marker_score_blocks_manifest_path=manifest_path,
        concatenated_latent_adata_path=latent_path,
        downsampling_n_spots_pcc=None,
        mean_frac_path=mean_frac_path,
        annotation_list=["region"],
        report_data_dir=tmp_path / "report_data",
        web_report_dir=tmp_path / "gsmap_web_report",
        sample_h5ad_dict={},
        force_report_re_run=False,
    )
    config.get_gene_diagnostic_info_save_path = (
        lambda trait: config.report_data_dir
        / "gss_stats"
        / f"gene_trait_correlation_{trait}.csv"
    )

    common_spots, source, gene_stats, analysis_spots = _load_gss_and_calculate_stats_base(
        config,
        ldsc_results,
        pd.DataFrame(index=obs_names),
        tmp_path,
    )

    assert isinstance(source, _GeneBlockReportSource)
    np.testing.assert_array_equal(common_spots, obs_names.sort_values())
    np.testing.assert_array_equal(analysis_spots, obs_names.sort_values())
    np.testing.assert_array_equal(
        source.read_subset(["spot-d", "spot-b"], ["gene-c", "gene-a"]),
        marker_scores[np.ix_([3, 1], [2, 0])].astype(np.float32),
    )
    assert set(gene_stats["gene"]) == set(gene_names)
    source.close()

    manager = ReportDataManager(config)
    manager.ldsc_results = ldsc_results
    manager.traits = ["trait"]
    manager.coords = pd.DataFrame(index=obs_names)
    manager.prepare_gss_stats()
    assert config.get_gene_diagnostic_info_save_path("trait").is_file()
    manager.close()


def test_gene_block_store_rejects_incomplete_or_noncontiguous_blocks(tmp_path):
    block_path = tmp_path / "block.dat"
    block_path.write_bytes(b"\0" * (5 * 2 * np.dtype(np.float16).itemsize))
    manifest_path = tmp_path / "manifest.json"
    base_manifest = {
        "format_version": GENE_BLOCK_FORMAT_VERSION,
        "shape": [5, 4],
        "dtype": "float16",
        "gene_block_size": 2,
        "complete": True,
        "gene_names_sha256": gene_names_sha256(["a", "b", "c", "d"]),
        "source_fingerprint": "0" * 64,
    }

    manifest_path.write_text(
        json.dumps(
            {
                **base_manifest,
                "blocks": [
                    {"start": 0, "end": 2, "path": block_path.name, "complete": True}
                ],
            }
        )
    )
    assert not GeneBlockMarkerStore.check_complete(manifest_path)

    manifest_path.write_text(
        json.dumps(
            {
                **base_manifest,
                "blocks": [
                    {"start": 0, "end": 2, "path": block_path.name, "complete": True},
                    {"start": 3, "end": 4, "path": "missing.dat", "complete": False},
                ],
            }
        )
    )
    assert not GeneBlockMarkerStore.check_complete(manifest_path)


def test_gene_block_store_validates_gene_order(tmp_path):
    marker_scores = np.arange(12, dtype=np.float16).reshape(3, 4)
    block_path = tmp_path / "block.dat"
    block_path.write_bytes(marker_scores.tobytes())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": GENE_BLOCK_FORMAT_VERSION,
                "shape": [3, 4],
                "dtype": "float16",
                "gene_block_size": 4,
                "complete": True,
                "gene_names_sha256": gene_names_sha256(["a", "b", "c", "d"]),
                "source_fingerprint": "0" * 64,
                "blocks": [
                    {"start": 0, "end": 4, "path": block_path.name, "complete": True}
                ],
            }
        )
    )

    store = GeneBlockMarkerStore(manifest_path)
    store.validate_gene_names(["a", "b", "c", "d"])
    with np.testing.assert_raises_regex(ValueError, "gene order"):
        store.validate_gene_names(["b", "a", "c", "d"])


def test_rank_row_min_is_recomputed_when_rank_source_changes(tmp_path):
    ranks_path = tmp_path / "ranks.dat"
    ranks = np.memmap(ranks_path, mode="w+", dtype=np.float16, shape=(2, 3))
    ranks[:] = [[1, 2, 3], [4, 5, 6]]
    ranks.flush()
    del ranks
    ranks_path.with_suffix(".meta.json").write_text(
        json.dumps({"shape": [2, 3], "dtype": "float16", "complete": True})
    )
    row_min_path = tmp_path / "rank_row_min.npy"

    ensure_rank_row_min(ranks_path, row_min_path)
    np.testing.assert_array_equal(np.load(row_min_path), [1, 4])

    tampered = np.load(row_min_path, mmap_mode="r+")
    tampered[:] = [9, 9]
    tampered.flush()
    del tampered
    ensure_rank_row_min(ranks_path, row_min_path)
    np.testing.assert_array_equal(np.load(row_min_path), [1, 4])

    ranks = np.memmap(ranks_path, mode="r+", dtype=np.float16, shape=(2, 3))
    ranks[:] = [[7, 8, 9], [0, 5, 6]]
    ranks.flush()
    del ranks

    ensure_rank_row_min(ranks_path, row_min_path)
    np.testing.assert_array_equal(np.load(row_min_path), [7, 0])


def test_rank_outputs_are_bound_to_current_h5ad_inputs(tmp_path):
    input_path = tmp_path / "input.h5ad"
    input_path.write_bytes(b"input-v1")
    rank_path = tmp_path / "ranks.dat"
    rank_path.write_bytes(b"rank-data")
    rank_path.with_suffix(".meta.json").write_text(
        json.dumps({"shape": [1, 1], "dtype": "float16", "complete": True})
    )
    mean_frac_path = tmp_path / "mean.parquet"
    mean_frac_path.write_bytes(b"mean-v1")
    concatenated_path = tmp_path / "concatenated.h5ad"
    concatenated_path.write_bytes(b"metadata-v1")
    samples = {"sample": input_path}
    provenance_kwargs = {
        "sample_h5ad_dict": samples,
        "rank_memmap_path": rank_path,
        "mean_frac_path": mean_frac_path,
        "concatenated_adata_path": concatenated_path,
        "annotation_key": "ManualAnnotation",
        "data_layer": "X",
        "min_cells_per_type": 21,
    }

    write_rank_provenance(**provenance_kwargs)
    assert rank_outputs_match_inputs(**provenance_kwargs)

    input_path.write_bytes(b"input-v2-is-different")
    assert not rank_outputs_match_inputs(**provenance_kwargs)

    write_rank_provenance(**provenance_kwargs)
    assert rank_outputs_match_inputs(**provenance_kwargs)
    mean_frac_path.write_bytes(b"mean-v2-is-different")
    assert not rank_outputs_match_inputs(**provenance_kwargs)


def test_gene_block_size_cannot_create_a_giant_single_mmap(tmp_path):
    rank_path = tmp_path / "ranks.dat"
    n_spots = MAX_GENE_BLOCK_BYTES // np.dtype(np.float16).itemsize + 1
    rank_path.with_suffix(".meta.json").write_text(
        json.dumps({"shape": [n_spots, 1], "dtype": "float16", "complete": True})
    )

    with pytest.raises(ValueError, match="no rank or marker mmap exceeds 32 GiB"):
        calculate_gene_block_marker_scores(
            rank_memmap_path=rank_path,
            rank_row_min_path=tmp_path / "row_min.npy",
            mean_frac_path=tmp_path / "missing.parquet",
            neighbor_adata_path=tmp_path / "missing.h5ad",
            output_dir=tmp_path / "blocks",
            manifest_path=tmp_path / "blocks" / "manifest.json",
            gene_block_size=1,
            spot_batch_size=1,
            n_slices=1,
            homogeneous_neighbors=1,
            hierarchical_pool=False,
            no_expression_fraction=False,
            tmp_dir=tmp_path / "local",
        )


def test_late_quick_mode_restart_rejects_stale_predecessors(monkeypatch):
    config = SimpleNamespace(
        marker_score_format="gene_block",
        start_step="cauchy",
        sumstats_config_dict={"Height": "height.sumstats"},
    )
    monkeypatch.setattr(quick_mode_module, "check_latent2gene_done", lambda *a, **k: True)
    monkeypatch.setattr(quick_mode_module, "check_spatial_ldsc_done", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="Restart from spatial_ldsc"):
        quick_mode_module._validate_late_start_prerequisites(config, 3)

    config.start_step = "report"
    monkeypatch.setattr(quick_mode_module, "check_spatial_ldsc_done", lambda *a, **k: True)
    monkeypatch.setattr(quick_mode_module, "check_cauchy_done", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="Restart from cauchy"):
        quick_mode_module._validate_late_start_prerequisites(config, 4)


def test_downstream_outputs_older_than_gene_blocks_are_not_complete(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "trait.csv.gz"
    manifest_path.write_text("{}")
    result_path.write_text("result")
    os.utime(result_path, ns=(1_000, 1_000))
    os.utime(manifest_path, ns=(2_000, 2_000))
    config = SimpleNamespace(
        marker_score_format="gene_block",
        marker_score_blocks_manifest_path=manifest_path,
        rank_memmap_path=tmp_path / "rank.dat",
        mean_frac_path=tmp_path / "mean.parquet",
        concatenated_latent_adata_path=tmp_path / "metadata.h5ad",
        get_ldsc_result_file=lambda _trait: result_path,
    )
    monkeypatch.setattr(
        "gsMap.latent2gene.gene_blocks.gene_block_store_matches_inputs",
        lambda *args, **kwargs: True,
    )

    assert not check_spatial_ldsc_done(config, "trait")
    os.utime(result_path, ns=(3_000, 3_000))
    atomic_json_dump(
        result_metadata_path(result_path),
        {
            "format_version": RESULT_METADATA_FORMAT_VERSION,
            "complete": True,
            "row_count": 1,
            "total_chunks": 1,
            "processed_chunks": 1,
            "output_signature": artifact_signature(result_path),
        },
    )
    assert check_spatial_ldsc_done(config, "trait")
    result_path.write_text("tampered")
    assert not check_spatial_ldsc_done(config, "trait")


def test_result_writer_refuses_missing_chunks_and_writes_completion_sidecar(tmp_path):
    config = SimpleNamespace(
        project_name="project",
        cell_indices_range=None,
        sample_filter=None,
    )
    writer = ResultWriter(config, "Height", tmp_path, n_spots=2)
    writer.add_chunk_result(
        0,
        np.asarray([0.1], dtype=np.float32),
        np.asarray([0.2], dtype=np.float32),
        pd.Index(["spot-0"]),
        0,
        1,
    )
    with pytest.raises(RuntimeError, match="missing chunks"):
        writer.validate_merge_and_save(total_chunks=2)

    writer.add_chunk_result(
        1,
        np.asarray([0.2], dtype=np.float32),
        np.asarray([0.3], dtype=np.float32),
        pd.Index(["spot-1"]),
        1,
        2,
    )
    result = writer.validate_merge_and_save(total_chunks=2)
    output_path = tmp_path / "project_Height.csv.gz"
    assert len(result) == 2
    assert output_path.exists()
    with open(result_metadata_path(output_path)) as handle:
        completion = json.load(handle)
    assert completion["complete"] is True
    assert completion["processed_chunks"] == completion["total_chunks"] == 2
    assert completion["output_signature"] == artifact_signature(output_path)


def test_result_writer_never_leaves_partial_final_csv(tmp_path, monkeypatch):
    config = SimpleNamespace(
        project_name="project",
        cell_indices_range=None,
        sample_filter=None,
    )
    writer = ResultWriter(config, "IQ", tmp_path, n_spots=1)
    writer.add_chunk_result(
        0,
        np.asarray([0.1], dtype=np.float32),
        np.asarray([0.2], dtype=np.float32),
        pd.Index(["spot-0"]),
        0,
        1,
    )

    def fail_mid_write(_self, path, **_kwargs):
        path.write_bytes(b"partial")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_mid_write)
    with pytest.raises(OSError, match="simulated interrupted write"):
        writer.validate_merge_and_save(total_chunks=1)

    assert not (tmp_path / "project_IQ.csv.gz").exists()
    assert not (tmp_path / ".project_IQ.csv.gz.partial").exists()


def test_trait_failures_propagate_after_all_traits_are_attempted():
    first_error = ValueError("bad trait")
    with pytest.raises(RuntimeError, match="Height, IQ") as error:
        _raise_for_trait_failures(
            [("Height", first_error), ("IQ", RuntimeError("also bad"))]
        )
    assert error.value.__cause__ is first_error


def test_cauchy_outputs_older_than_ldsc_are_not_complete(tmp_path):
    ldsc_path = tmp_path / "trait.csv.gz"
    annotation_path = tmp_path / "annotation.csv"
    sample_path = tmp_path / "sample.csv"
    for path in (ldsc_path, annotation_path, sample_path):
        path.write_text("result")
    os.utime(annotation_path, ns=(1_000, 1_000))
    os.utime(sample_path, ns=(1_000, 1_000))
    os.utime(ldsc_path, ns=(2_000, 2_000))

    def get_cauchy_result_file(_trait, *, annotation, all_samples):
        assert annotation == "region"
        return annotation_path if all_samples else sample_path

    config = SimpleNamespace(
        annotation_list=["region"],
        get_ldsc_result_file=lambda _trait: ldsc_path,
        get_cauchy_result_file=get_cauchy_result_file,
    )
    assert not check_cauchy_done(config, "trait")

    os.utime(annotation_path, ns=(3_000, 3_000))
    os.utime(sample_path, ns=(3_000, 3_000))
    assert check_cauchy_done(config, "trait")


def test_gene_block_calculator_materializes_full_equivalent_scores(tmp_path):
    ranks, _, _, global_log_gmean, global_expr_frac = _rank_fixture()
    n_spots, n_genes = ranks.shape
    neighbors = np.asarray(
        [[i, (i + 1) % n_spots, (i + 2) % n_spots] for i in range(n_spots)],
        dtype=np.int32,
    )
    weights = np.tile(np.asarray([0.5, 0.3, 0.2], dtype=np.float32), (n_spots, 1))

    rank_path = tmp_path / "ranks.dat"
    rank_memmap = np.memmap(rank_path, mode="w+", dtype=np.float16, shape=ranks.shape)
    rank_memmap[:] = ranks
    rank_memmap.flush()
    del rank_memmap
    rank_path.with_suffix(".meta.json").write_text(
        json.dumps({"shape": list(ranks.shape), "dtype": "float16", "complete": True})
    )

    mean_frac_path = tmp_path / "mean_frac.parquet"
    pd.DataFrame({"G_Mean": global_log_gmean, "frac": global_expr_frac}).to_parquet(mean_frac_path)
    neighbor_path = tmp_path / "neighbors.h5ad"
    neighbor_adata = ad.AnnData(np.zeros((n_spots, n_genes), dtype=np.float32))
    neighbor_adata.obsm["gsMap_homo_indices"] = neighbors
    neighbor_adata.obsm["gsMap_homo_cell_sims"] = weights
    neighbor_adata.write_h5ad(neighbor_path)

    manifest_path = calculate_gene_block_marker_scores(
        rank_memmap_path=rank_path,
        rank_row_min_path=tmp_path / "rank_row_min.npy",
        mean_frac_path=mean_frac_path,
        neighbor_adata_path=neighbor_path,
        output_dir=tmp_path / "blocks",
        manifest_path=tmp_path / "blocks" / "manifest.json",
        gene_block_size=3,
        spot_batch_size=2,
        n_slices=1,
        homogeneous_neighbors=3,
        hierarchical_pool=False,
        no_expression_fraction=False,
        tmp_dir=tmp_path / "local",
    )

    expected = np.asarray(
        compute_marker_scores_jax(
            ranks[neighbors.ravel()],
            weights,
            n_spots,
            3,
            global_log_gmean,
            global_expr_frac,
        )
    )
    store = GeneBlockMarkerStore(manifest_path)
    actual = np.zeros_like(expected)
    for block in store.blocks:
        source = store._open_block(block)
        actual[:, block["start"] : block["end"]] = source

    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    first_fingerprint = store.source_fingerprint
    store.close()

    changed_log_gmean = global_log_gmean + 0.5
    pd.DataFrame(
        {"G_Mean": changed_log_gmean, "frac": global_expr_frac}
    ).to_parquet(mean_frac_path)
    calculate_gene_block_marker_scores(
        rank_memmap_path=rank_path,
        rank_row_min_path=tmp_path / "rank_row_min.npy",
        mean_frac_path=mean_frac_path,
        neighbor_adata_path=neighbor_path,
        output_dir=tmp_path / "blocks",
        manifest_path=tmp_path / "blocks" / "manifest.json",
        gene_block_size=3,
        spot_batch_size=2,
        n_slices=1,
        homogeneous_neighbors=3,
        hierarchical_pool=False,
        no_expression_fraction=False,
        tmp_dir=tmp_path / "local",
    )
    changed_expected = np.asarray(
        compute_marker_scores_jax(
            ranks[neighbors.ravel()],
            weights,
            n_spots,
            3,
            changed_log_gmean,
            global_expr_frac,
        )
    )
    changed_store = GeneBlockMarkerStore(manifest_path)
    changed_actual = np.zeros_like(changed_expected)
    for block in changed_store.blocks:
        source = changed_store._open_block(block)
        changed_actual[:, block["start"] : block["end"]] = source

    assert changed_store.source_fingerprint != first_fingerprint
    np.testing.assert_allclose(changed_actual, changed_expected, rtol=2e-3, atol=2e-3)
    changed_store.close()

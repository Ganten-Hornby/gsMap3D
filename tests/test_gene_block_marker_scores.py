import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from gsMap.latent2gene.gene_blocks import (
    GENE_BLOCK_FORMAT_VERSION,
    GeneBlockMarkerStore,
    accumulate_annotated_ldscore_blocks,
    gene_block_store_is_read_compatible,
    gene_block_store_matches_inputs,
    gene_names_sha256,
)
from gsMap.report.report_data import _GeneBlockReportSource, _load_gss_pcc_sample


def _write_store(tmp_path, marker_scores, gene_names):
    block_dir = tmp_path / "marker_score_blocks"
    block_dir.mkdir()
    blocks = []
    boundaries = ((0, 2), (2, marker_scores.shape[1]))
    for start, end in boundaries:
        path = block_dir / f"block_{start}_{end}.dat"
        block = np.memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(marker_scores.shape[0], end - start),
        )
        block[:] = marker_scores[:, start:end]
        block.flush()
        del block
        blocks.append({"start": start, "end": end, "path": path.name, "complete": True})

    manifest_path = block_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": GENE_BLOCK_FORMAT_VERSION,
                "shape": list(marker_scores.shape),
                "dtype": "float16",
                "gene_block_size": 2,
                "complete": True,
                "gene_names_sha256": gene_names_sha256(gene_names),
                "source_fingerprint": "0" * 64,
                "blocks": blocks,
            }
        )
    )
    return manifest_path


def test_existing_gene_block_store_reads_arbitrary_subset(tmp_path):
    marker_scores = np.arange(24, dtype=np.float16).reshape(6, 4)
    manifest_path = _write_store(
        tmp_path,
        marker_scores,
        [f"gene-{index}" for index in range(marker_scores.shape[1])],
    )
    store = GeneBlockMarkerStore(manifest_path)

    spots = np.asarray([5, 1, 3])
    genes = np.asarray([3, 0, 2])
    actual = store.read_subset(spots, genes)

    np.testing.assert_array_equal(
        actual,
        marker_scores[np.ix_(spots, genes)].astype(np.float32),
    )


def test_existing_gene_blocks_accumulate_same_ldscore_as_dense_matrix(tmp_path):
    marker_scores = np.arange(24, dtype=np.float16).reshape(6, 4)
    manifest_path = _write_store(
        tmp_path,
        marker_scores,
        [f"gene-{index}" for index in range(marker_scores.shape[1])],
    )
    store = GeneBlockMarkerStore(manifest_path)
    weight = np.asarray([[1, 0, 2, 0], [0, 3, 0, 4]], dtype=np.float32)

    actual = accumulate_annotated_ldscore_blocks(
        weight,
        store.iter_spot_marker_blocks(1, 5, np.arange(4)),
    )
    expected = weight @ marker_scores[1:5].T.astype(np.float32)

    np.testing.assert_allclose(actual, expected)


def test_report_reads_existing_gene_blocks(tmp_path):
    obs_names = pd.Index(["spot-a", "spot-b", "spot-c", "spot-d"])
    gene_names = pd.Index(["gene-a", "gene-b", "gene-c"])
    marker_scores = np.arange(1, 13, dtype=np.float16).reshape(4, 3)
    manifest_path = _write_store(tmp_path, marker_scores, gene_names)

    latent_path = tmp_path / "concatenated_latent_adata.h5ad"
    ad.AnnData(
        X=np.zeros(marker_scores.shape, dtype=np.float32),
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
        marker_scores_memmap_path=tmp_path / "missing-marker-score.dat",
        concatenated_latent_adata_path=latent_path,
        downsampling_n_spots_pcc=None,
        mean_frac_path=mean_frac_path,
        annotation_list=["region"],
        memmap_tmp_dir=None,
    )

    source = _GeneBlockReportSource(config)
    np.testing.assert_array_equal(
        source.read_subset(["spot-d", "spot-b"], ["gene-c", "gene-a"]),
        marker_scores[np.ix_([3, 1], [2, 0])].astype(np.float32),
    )
    source.close()

    sample = _load_gss_pcc_sample(config, ldsc_results, tmp_path)
    np.testing.assert_array_equal(sample.marker_scores, marker_scores.astype(np.float32))
    assert sample.gene_names == gene_names.tolist()


def test_legacy_gene_block_store_remains_readable_after_metadata_rewrite(tmp_path):
    gene_names = pd.Index(["gene-a", "gene-b", "gene-c"])
    marker_scores = np.arange(12, dtype=np.float16).reshape(4, 3)
    manifest_path = _write_store(tmp_path, marker_scores, gene_names)

    metadata_path = tmp_path / "metadata.h5ad"
    ad.AnnData(
        X=np.zeros(marker_scores.shape, dtype=np.float32),
        obs=pd.DataFrame(index=["spot-a", "spot-b", "spot-c", "spot-d"]),
        var=pd.DataFrame(index=gene_names),
    ).write_h5ad(metadata_path)
    rank_path = tmp_path / "ranks.dat"
    np.zeros(marker_scores.shape, dtype=np.float16).tofile(rank_path)
    (tmp_path / "ranks.meta.json").write_text(
        json.dumps({"shape": list(marker_scores.shape), "dtype": "float16"})
    )
    mean_frac_path = tmp_path / "mean_frac.parquet"
    pd.DataFrame({"frac": np.ones(len(gene_names))}, index=gene_names).to_parquet(
        mean_frac_path
    )

    assert not gene_block_store_matches_inputs(
        manifest_path,
        rank_memmap_path=rank_path,
        mean_frac_path=mean_frac_path,
        marker_metadata_path=metadata_path,
    )
    assert gene_block_store_is_read_compatible(
        manifest_path,
        rank_memmap_path=rank_path,
        mean_frac_path=mean_frac_path,
        marker_metadata_path=metadata_path,
    )


def test_legacy_gene_block_store_rejects_spot_axis_change(tmp_path):
    gene_names = pd.Index(["gene-a", "gene-b", "gene-c"])
    manifest_path = _write_store(tmp_path, np.zeros((4, 3), dtype=np.float16), gene_names)
    metadata_path = tmp_path / "metadata.h5ad"
    ad.AnnData(
        X=np.zeros((3, 3), dtype=np.float32),
        obs=pd.DataFrame(index=["spot-a", "spot-b", "spot-c"]),
        var=pd.DataFrame(index=gene_names),
    ).write_h5ad(metadata_path)
    rank_path = tmp_path / "ranks.dat"
    np.zeros((3, 3), dtype=np.float16).tofile(rank_path)
    (tmp_path / "ranks.meta.json").write_text(
        json.dumps({"shape": [3, 3], "dtype": "float16"})
    )
    mean_frac_path = tmp_path / "mean_frac.parquet"
    pd.DataFrame({"frac": np.ones(len(gene_names))}, index=gene_names).to_parquet(
        mean_frac_path
    )

    assert not gene_block_store_is_read_compatible(
        manifest_path,
        rank_memmap_path=rank_path,
        mean_frac_path=mean_frac_path,
        marker_metadata_path=metadata_path,
    )

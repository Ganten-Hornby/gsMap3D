from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from gsMap.latent2gene.marker_layout import (
    MARKER_SCORE_LAYOUT_VERSION,
    SequentialMarkerRowWriter,
    marker_obs_indices_path,
    open_marker_obs_indices,
)
from gsMap.latent2gene.marker_scores import MarkerScoreCalculator
from gsMap.latent2gene.memmap_io import (
    MemMapDense,
    ParallelMarkerScoreWriter,
    _automatic_lustre_stripe_count,
    _should_use_tmp_storage,
)
from gsMap.spatial_ldsc.io import _shared_marker_cache_base, load_marker_scores_memmap_format


def test_sequential_marker_layout_appends_skipped_rows_and_is_atomic(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    writer = SequentialMarkerRowWriter(marker_path, n_obs=5)

    assert writer.append(np.asarray([2, 0], dtype=np.int32)) == slice(0, 2)
    assert writer.append(np.asarray([3], dtype=np.int64)) == slice(2, 3)
    metadata = writer.finalize()

    mapping = open_marker_obs_indices(marker_path, {"row_layout": metadata}, expected_obs=5)
    try:
        np.testing.assert_array_equal(mapping, [2, 0, 3, 1, 4])
    finally:
        mapping._mmap.close()

    assert metadata["scored_rows"] == 3
    assert metadata["unscored_rows"] == 2
    assert not list(tmp_path.glob("*.incomplete.*"))


def test_shared_marker_read_cache_is_reused(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    marker_path = source_dir / "marker_scores.dat"
    marker_values = np.arange(12, dtype=np.float16).reshape(4, 3)
    marker_values.tofile(marker_path)
    metadata = {"shape": [4, 3], "dtype": "float16", "complete": True}

    monkeypatch.setattr("gsMap.spatial_ldsc.io._paths_share_storage_device", lambda *_: False)
    monkeypatch.setattr(
        "gsMap.spatial_ldsc.io.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )

    first = _shared_marker_cache_base(marker_path, cache_dir, metadata)
    second = _shared_marker_cache_base(marker_path, cache_dir, metadata)

    assert first == second
    np.testing.assert_array_equal(
        np.fromfile(first.with_suffix(".dat"), dtype=np.float16).reshape(4, 3),
        marker_values,
    )
    assert first.with_suffix(".meta.json").is_file()
    assert len(list((cache_dir / "gsmap_read_cache").glob("*.dat"))) == 1


def test_shared_marker_read_cache_can_be_reused_without_populating(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    marker_path = source_dir / "marker_scores.dat"
    np.arange(12, dtype=np.float16).tofile(marker_path)
    metadata = {"shape": [4, 3], "dtype": "float16", "complete": True}
    monkeypatch.setattr("gsMap.spatial_ldsc.io._paths_share_storage_device", lambda *_: False)

    resolved = _shared_marker_cache_base(
        marker_path,
        cache_dir,
        metadata,
        populate=False,
    )

    assert resolved == marker_path
    assert not list((cache_dir / "gsmap_read_cache").glob("*.dat"))


@pytest.mark.parametrize(
    "rows",
    [
        np.asarray([1, 1], dtype=np.int64),
        np.asarray([0, 1, 2, 0], dtype=np.int64),
    ],
)
def test_sequential_marker_layout_rejects_duplicate_rows(tmp_path, rows):
    writer = SequentialMarkerRowWriter(tmp_path / "marker_scores.dat", n_obs=4)
    writer.append(rows)
    try:
        with pytest.raises(ValueError, match="duplicate"):
            writer.finalize()
    finally:
        writer.abort()


def test_sequential_marker_layout_rejects_invalid_indices(tmp_path):
    writer = SequentialMarkerRowWriter(tmp_path / "marker_scores.dat", n_obs=3)
    try:
        with pytest.raises(IndexError, match="outside"):
            writer.append(np.asarray([0, 3], dtype=np.int64))
        with pytest.raises(TypeError, match="integer"):
            writer.append(np.asarray([0.0, 1.0]))
    finally:
        writer.abort()


def _write_marker_memmap(path, values, row_order=None):
    marker = MemMapDense(
        path,
        shape=values.shape,
        dtype=values.dtype,
        mode="w",
        num_write_workers=0,
    )
    marker.memmap[:] = values
    metadata_updates = None
    if row_order is not None:
        layout_writer = SequentialMarkerRowWriter(path, n_obs=values.shape[0])
        layout_writer.append(np.asarray(row_order, dtype=np.int64))
        row_layout = layout_writer.finalize()
        metadata_updates = {
            "marker_score_layout_version": MARKER_SCORE_LAYOUT_VERSION,
            "row_layout": row_layout,
        }
    marker.close(metadata_updates=metadata_updates)


def _source_adata(path):
    obs_names = ["spot_a", "spot_b", "spot_c", "spot_d"]
    obs = pd.DataFrame(
        {
            "annotation": ["A", "B", "C", "D"],
            "sample_name": ["s1", "s2", "s1", "s2"],
        },
        index=obs_names,
    )
    source = ad.AnnData(
        X=np.zeros((4, 3), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    source.obsm["spatial"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    source.obsm["frame"] = pd.DataFrame(
        np.arange(8, dtype=np.float32).reshape(4, 2),
        index=obs_names,
        columns=["x", "y"],
    )
    source.write_h5ad(path)
    return source


def test_loader_aligns_marker_rows_with_original_annotations_and_coordinates(tmp_path):
    source_path = tmp_path / "concatenated_latent_adata.h5ad"
    source = _source_adata(source_path)
    marker_path = tmp_path / "marker_scores.dat"
    marker_values = np.asarray(
        [
            [20, 21, 22],
            [0, 1, 2],
            [30, 31, 32],
            [10, 11, 12],
        ],
        dtype=np.float16,
    )
    _write_marker_memmap(marker_path, marker_values, row_order=[2, 0, 3])

    config = SimpleNamespace(
        marker_scores_memmap_path=marker_path,
        concatenated_latent_adata_path=source_path,
        memmap_tmp_dir=None,
        spatial_key="spatial",
    )
    loaded = load_marker_scores_memmap_format(config)
    try:
        assert loaded.obs_names.tolist() == ["spot_c", "spot_a", "spot_d", "spot_b"]
        assert loaded.obs["annotation"].tolist() == ["C", "A", "D", "B"]
        np.testing.assert_array_equal(loaded.X, marker_values)
        np.testing.assert_array_equal(
            loaded.obsm["spatial"],
            source.obsm["spatial"][[2, 0, 3, 1]],
        )
        assert "frame" not in loaded.obsm
        assert loaded.uns["marker_score_row_layout"]["unscored_rows"] == 1
    finally:
        loaded.uns["memmap_manager"].close()

    # The mapping is applied only by the wrapper; the rank-indexed source stays
    # in its original order for safe latent-to-gene reruns.
    unchanged = ad.read_h5ad(source_path)
    assert unchanged.obs_names.tolist() == source.obs_names.tolist()


def test_loader_preserves_legacy_identity_layout(tmp_path):
    source_path = tmp_path / "concatenated_latent_adata.h5ad"
    source = _source_adata(source_path)
    marker_path = tmp_path / "legacy_marker_scores.dat"
    marker_values = np.arange(12, dtype=np.float16).reshape(4, 3)
    _write_marker_memmap(marker_path, marker_values)

    config = SimpleNamespace(
        marker_scores_memmap_path=marker_path,
        concatenated_latent_adata_path=source_path,
        memmap_tmp_dir=None,
        spatial_key="spatial",
    )
    loaded = load_marker_scores_memmap_format(config)
    try:
        assert loaded.obs_names.tolist() == source.obs_names.tolist()
        np.testing.assert_array_equal(loaded.X, marker_values)
        assert "marker_score_row_layout" not in loaded.uns
    finally:
        loaded.uns["memmap_manager"].close()


def test_complete_marker_memmap_requires_its_row_mapping(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    _write_marker_memmap(
        marker_path,
        np.zeros((3, 2), dtype=np.float16),
        row_order=[2, 0, 1],
    )
    assert MemMapDense.check_complete(marker_path)[0]

    marker_obs_indices_path(marker_path).unlink()

    assert not MemMapDense.check_complete(marker_path)[0]


@pytest.mark.parametrize(
    ("size_gib", "expected"),
    [(1, 1), (8, 4), (128, 8), (1024, 16)],
)
def test_automatic_lustre_stripe_count_scales_with_file_size(size_gib, expected):
    assert _automatic_lustre_stripe_count(size_gib * 1024**3) == expected


def test_optional_tmp_storage_falls_back_when_capacity_is_insufficient(
    tmp_path,
    monkeypatch,
):
    usage = type("Usage", (), {"free": 8 * 1024**3})()
    monkeypatch.setattr(
        "gsMap.latent2gene.memmap_io._same_filesystem",
        lambda first, second: False,
    )
    monkeypatch.setattr(
        "gsMap.latent2gene.memmap_io.shutil.disk_usage",
        lambda path: usage,
    )

    assert not _should_use_tmp_storage(
        tmp_path / "shared" / "marker_scores.dat",
        tmp_path / "scratch",
        file_bytes=10 * 1024**3,
    )


def _force_distinct_scratch_with_capacity(monkeypatch):
    monkeypatch.setattr(
        "gsMap.latent2gene.memmap_io._same_filesystem",
        lambda first, second: False,
    )
    usage = type("Usage", (), {"free": 100 * 1024**3})()
    monkeypatch.setattr(
        "gsMap.latent2gene.memmap_io.shutil.disk_usage",
        lambda path: usage,
    )


def test_rank_publish_retains_scratch_and_syncs_project_once(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    rank_path = project_dir / "ranks.dat"
    values = np.arange(12, dtype=np.float16).reshape(4, 3)
    _force_distinct_scratch_with_capacity(monkeypatch)

    sync_calls = []
    original_sync = MemMapDense._sync_tmp_to_original

    def tracked_sync(memmap):
        sync_calls.append(memmap.data_path)
        original_sync(memmap)

    monkeypatch.setattr(MemMapDense, "_sync_tmp_to_original", tracked_sync)

    rank = MemMapDense(
        rank_path,
        shape=values.shape,
        dtype=values.dtype,
        mode="w",
        num_write_workers=0,
        tmp_dir=scratch_dir,
    )
    assert rank.using_tmp
    assert rank.data_path.parent.parent == scratch_dir
    assert not rank_path.is_file()
    assert not rank_path.with_suffix(".meta.json").is_file()

    rank.memmap[:] = values
    rank.publish()

    assert sync_calls == [rank.data_path]
    assert MemMapDense.check_complete(rank_path)[0]
    assert rank.data_path.is_file()
    np.testing.assert_array_equal(rank.memmap, values)

    completed = np.memmap(rank_path, dtype=np.float16, mode="r", shape=values.shape)
    try:
        np.testing.assert_array_equal(completed, values)
    finally:
        completed._mmap.close()

    scratch_subdir = rank.tmp_subdir
    rank.close()
    rank.close()
    assert sync_calls == [rank.data_path]
    assert not scratch_subdir.exists()


def test_marker_input_reuses_live_rank_scratch_mapping(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    rank_path = project_dir / "ranks.dat"
    adata_path = project_dir / "concatenated_latent_adata.h5ad"
    _force_distinct_scratch_with_capacity(monkeypatch)

    rank = MemMapDense(
        rank_path,
        shape=(3, 2),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
        tmp_dir=scratch_dir,
    )
    rank.memmap[:] = np.arange(6, dtype=np.float16).reshape(3, 2)
    rank.publish()
    scratch_entries_before = set(scratch_dir.iterdir())

    ad.AnnData(
        X=np.zeros((3, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b", "c"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(adata_path)

    calculator = object.__new__(MarkerScoreCalculator)
    calculator.config = SimpleNamespace(
        high_quality_neighbor_filter=False,
        memmap_tmp_dir=scratch_dir,
    )
    calculator.load_global_stats = lambda path: (
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )

    loaded = calculator._load_input_data(adata_path, rank, "unused.parquet")
    assert loaded[1] is rank
    assert loaded[1].data_path == rank.data_path
    assert set(scratch_dir.iterdir()) == scratch_entries_before
    np.testing.assert_array_equal(loaded[-1], np.ones(3, dtype=bool))

    rank.close()


def test_invalid_marker_input_closes_live_rank_scratch_mapping(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    rank_path = project_dir / "ranks.dat"
    adata_path = project_dir / "concatenated_latent_adata.h5ad"
    _force_distinct_scratch_with_capacity(monkeypatch)

    rank = MemMapDense(
        rank_path,
        shape=(3, 2),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
        tmp_dir=scratch_dir,
    )
    rank.publish()
    scratch_subdir = rank.tmp_subdir

    ad.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(adata_path)

    calculator = object.__new__(MarkerScoreCalculator)
    calculator.config = SimpleNamespace(
        high_quality_neighbor_filter=False,
        memmap_tmp_dir=scratch_dir,
    )
    calculator.load_global_stats = lambda path: (
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Cell count mismatch"):
        calculator._load_input_data(adata_path, rank, "unused.parquet")

    assert rank._closed
    assert not scratch_subdir.exists()


def test_staged_marker_output_publishes_data_metadata_and_mapping_together(
    tmp_path,
    monkeypatch,
):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    marker_path = project_dir / "marker_scores.dat"
    values = np.arange(12, dtype=np.float16).reshape(4, 3)

    # Exercise the cross-filesystem branch using two test directories on the
    # same CI filesystem.
    _force_distinct_scratch_with_capacity(monkeypatch)

    marker = MemMapDense(
        marker_path,
        shape=values.shape,
        dtype=values.dtype,
        mode="w",
        num_write_workers=0,
        tmp_dir=scratch_dir,
    )
    assert marker.using_tmp
    marker.memmap[:] = values

    layout_writer = SequentialMarkerRowWriter(marker_path, n_obs=4)
    layout_writer.append(np.asarray([2, 0, 3, 1], dtype=np.int64))
    row_layout = layout_writer.finalize()
    marker.close(
        metadata_updates={
            "marker_score_layout_version": MARKER_SCORE_LAYOUT_VERSION,
            "row_layout": row_layout,
        }
    )

    assert MemMapDense.check_complete(marker_path)[0]
    completed = np.memmap(marker_path, dtype=np.float16, mode="r", shape=values.shape)
    try:
        np.testing.assert_array_equal(completed, values)
    finally:
        completed._mmap.close()
    assert marker_obs_indices_path(marker_path).is_file()
    assert not list(scratch_dir.glob("memmap_*"))


def test_check_complete_accepts_metadata_path(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    _write_marker_memmap(marker_path, np.zeros((2, 3), dtype=np.float16))

    metadata_path = marker_path.with_suffix(".meta.json")
    assert MemMapDense.check_complete(metadata_path)[0]


def test_memmap_open_rejects_metadata_dtype_mismatch(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    _write_marker_memmap(marker_path, np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Dtype mismatch"):
        MemMapDense(marker_path, shape=(2, 3), dtype=np.float16, mode="r")


def test_zero_worker_memmap_write_batch_is_synchronous(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    marker = MemMapDense(
        marker_path,
        shape=(3, 2),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
    )
    values = np.arange(6, dtype=np.float16).reshape(3, 2)

    marker.write_batch(values, slice(0, 3))
    marker.close()

    completed = np.memmap(marker_path, dtype=np.float16, mode="r", shape=(3, 2))
    try:
        np.testing.assert_array_equal(completed, values)
    finally:
        completed._mmap.close()


def test_parallel_marker_writer_rejects_zero_workers(tmp_path):
    marker = MemMapDense(
        tmp_path / "marker_scores.dat",
        shape=(1, 1),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
    )
    try:
        with pytest.raises(ValueError, match="at least one worker"):
            ParallelMarkerScoreWriter(marker, num_workers=0)
    finally:
        marker.abort()


def test_annotation_free_marker_mode_does_not_create_invalid_obs_column(tmp_path):
    calculator = object.__new__(MarkerScoreCalculator)
    calculator.config = SimpleNamespace(annotation=None)
    adata = ad.AnnData(
        X=np.zeros((2, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["g1"]),
    )

    assert calculator._get_cell_types(adata) == ["all"]
    assert None not in adata.obs.columns
    adata.write_h5ad(tmp_path / "annotation_free.h5ad")


def test_parallel_marker_writer_surfaces_write_failure_without_deadlock(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    marker = MemMapDense(
        marker_path,
        shape=(2, 2),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
    )
    writer = ParallelMarkerScoreWriter(marker, num_workers=1, queue_size=2)

    class BrokenArray:
        def __setitem__(self, key, value):
            del key, value
            raise OSError("synthetic marker write failure")

    writer.shared_memmap = BrokenArray()
    writer.write_queue.put((0, np.ones((1, 2), dtype=np.float16), np.asarray([0], dtype=np.int64)))
    try:
        with pytest.raises(RuntimeError, match="synthetic marker write failure"):
            writer.close()
    finally:
        writer.abort()
        marker.abort()

    assert not MemMapDense.check_complete(marker_path)[0]


def test_memmap_abort_removes_scratch_without_publishing(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    marker_path = project_dir / "marker_scores.dat"
    _force_distinct_scratch_with_capacity(monkeypatch)

    marker = MemMapDense(
        marker_path,
        shape=(2, 3),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
        tmp_dir=scratch_dir,
    )
    scratch_subdir = marker.tmp_subdir
    marker.memmap[:] = 1
    marker.abort()

    assert not scratch_subdir.exists()
    assert not marker_path.exists()
    assert not marker_path.with_suffix(".meta.json").exists()


def test_memmap_constructor_failure_removes_staged_scratch(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    rank_path = project_dir / "ranks.dat"
    _write_marker_memmap(rank_path, np.zeros((3, 2), dtype=np.float16))
    _force_distinct_scratch_with_capacity(monkeypatch)

    with pytest.raises(ValueError, match="Dtype mismatch"):
        MemMapDense(
            rank_path,
            shape=(3, 2),
            dtype=np.float32,
            mode="r",
            num_write_workers=0,
            tmp_dir=scratch_dir,
        )

    assert not list(scratch_dir.glob("memmap_*"))

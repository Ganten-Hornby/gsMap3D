from __future__ import annotations

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
from gsMap.latent2gene.memmap_io import (
    MemMapDense,
    _automatic_lustre_stripe_count,
    _should_use_tmp_storage,
)
from gsMap.spatial_ldsc.io import load_marker_scores_memmap_format


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
    monkeypatch.setattr(
        "gsMap.latent2gene.memmap_io._same_filesystem",
        lambda first, second: False,
    )
    usage = type("Usage", (), {"free": 100 * 1024**3})()
    monkeypatch.setattr(
        "gsMap.latent2gene.memmap_io.shutil.disk_usage",
        lambda path: usage,
    )

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

"""Coordinate preflight accepts AnnData's DataFrame encoding, without reading X."""

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

from gsMap.config.spatial import PROBE_ROWS, probe_coordinates


def write_frame(path, values, columns):
    data = ad.AnnData(np.zeros((len(values), 1), dtype=np.float32))
    data.obsm["spatial"] = pd.DataFrame(values, index=data.obs_names, columns=columns)
    data.write_h5ad(path)
    return path


@pytest.mark.parametrize("width", [2, 3])
def test_dataframe_preserves_declared_order(tmp_path, width):
    values = np.arange(20 * width, dtype=float).reshape(20, width)
    path = write_frame(tmp_path / "frame.h5ad", values, ["z", "a", "m"][:width])
    result = probe_coordinates(path, "spatial", "spatial3D" if width == 3 else "spatial2D")
    assert result["shape"] == values.shape
    np.testing.assert_allclose(result["median"], np.median(values, axis=0))


def test_dataframe_probe_bounded_expression_free(tmp_path, monkeypatch):
    path = write_frame(
        tmp_path / "frame.h5ad", np.arange(PROBE_ROWS * 9).reshape(-1, 3), ["x", "y", "z"]
    )
    original = h5py.Dataset.__getitem__

    def read(node, key):
        assert node.name != "/X"
        value = original(node, key)
        if node.name.startswith("/obsm/spatial/"):
            assert len(value) <= PROBE_ROWS
        return value

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    probe_coordinates(path, "spatial", "spatial3D")


def test_dataframe_unused_z_ignored_in_2d(tmp_path):
    values = np.ones((8, 3))
    values[:, 2] = np.nan
    path = write_frame(tmp_path / "frame.h5ad", values, ["x", "y", "z"])
    probe_coordinates(path, "spatial", "spatial2D")
    with pytest.raises(ValueError, match="NaN"):
        probe_coordinates(path, "spatial", "spatial3D")


def test_dataframe_strings_rejected(tmp_path):
    path = write_frame(tmp_path / "frame.h5ad", [["one", "two"]] * 8, ["x", "y"])
    with pytest.raises(ValueError, match="numeric"):
        probe_coordinates(path, "spatial", "spatial2D")


def test_dataframe_mismatched_lengths_rejected(tmp_path):
    path = write_frame(tmp_path / "frame.h5ad", np.ones((8, 2)), ["x", "y"])
    with h5py.File(path, "r+") as f:
        del f["obsm/spatial/y"]
        f["obsm/spatial/y"] = np.ones(7)
    with pytest.raises(ValueError, match="lengths differ"):
        probe_coordinates(path, "spatial", "spatial2D")

"""Selective AnnData metadata and embedding reads without expression/neighbor arrays."""

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem, sparse_dataset


def _read_dataframe_columns(group, columns=None):
    """Preserve AnnData's index and categorical/nullable column encodings."""
    if columns is None:
        return read_elem(group)
    index_key = group.attrs["_index"]
    index_values = read_elem(group[index_key])
    index = pd.Index(index_values, name=None if index_key == "_index" else index_key)
    return pd.DataFrame(
        {key: read_elem(group[key]) for key in dict.fromkeys(columns) if key in group},
        index=index,
    )


def read_h5ad_metadata(
    path: Path,
    *,
    obs_columns=None,
    obsm_keys=(),
    include_var=True,
    include_uns=False,
    include_varm=False,
) -> ad.AnnData:
    """Return an in-memory, X-free AnnData with only requested metadata fields.

    Unlike read_h5ad(backed="r"), this never reads unrelated obsm arrays, layers,
    raw, or pairwise matrices. The HDF5 handle is closed before returning.
    """
    with h5py.File(path, "r") as handle:
        obs = _read_dataframe_columns(handle["obs"], obs_columns)
        var = read_elem(handle["var"]) if include_var else pd.DataFrame(index=[])
        obsm = {
            key: read_elem(handle["obsm"][key])
            for key in dict.fromkeys(obsm_keys)
            if key is not None and key in handle.get("obsm", {})
        }
        uns = read_elem(handle["uns"]) if include_uns and "uns" in handle else {}
        varm = read_elem(handle["varm"]) if include_varm and "varm" in handle else {}
    return ad.AnnData(obs=obs, var=var, obsm=obsm, uns=uns, varm=varm)


def read_obsm_rows(path: Path, keys, rows: np.ndarray) -> dict:
    """Read embedding rows in caller order using bounded sequential HDF5 windows."""
    rows = np.asarray(rows)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Embedding row indices must be a one-dimensional integer array")
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    result = {}
    with h5py.File(path, "r") as handle:
        for key in dict.fromkeys(keys):
            if key is None or key not in handle.get("obsm", {}):
                continue
            node = handle["obsm"][key]
            if isinstance(node, h5py.Dataset):
                if node.ndim < 2:
                    raise ValueError(f"Embedding {key!r} must have at least two dimensions")
                if len(rows) and (unique_rows[0] < 0 or unique_rows[-1] >= node.shape[0]):
                    raise IndexError(f"Row indices are outside embedding {key!r}")
                values = np.empty((len(unique_rows), *node.shape[1:]), dtype=node.dtype)
                row_bytes = int(np.prod(node.shape[1:])) * node.dtype.itemsize
                window_rows = max(1, min(65536, (8 << 20) // max(1, row_bytes)))
                start = 0
                while start < len(unique_rows):
                    first = int(unique_rows[start])
                    boundary = (first // window_rows + 1) * window_rows
                    stop = int(np.searchsorted(unique_rows, boundary))
                    last = int(unique_rows[stop - 1]) + 1
                    block = node[first:last]
                    values[start:stop] = block[unique_rows[start:stop] - first]
                    start = stop
                result[key] = values[inverse]
            elif node.attrs.get("encoding-type") in ("csr_matrix", "csc_matrix"):
                matrix = sparse_dataset(node)
                if len(rows) and (unique_rows[0] < 0 or unique_rows[-1] >= matrix.shape[0]):
                    raise IndexError(f"Row indices are outside embedding {key!r}")
                result[key] = matrix[rows].copy()
            else:
                matrix = read_elem(node)
                if len(rows) and (unique_rows[0] < 0 or unique_rows[-1] >= len(matrix)):
                    raise IndexError(f"Row indices are outside embedding {key!r}")
                result[key] = (
                    matrix.iloc[rows].copy()
                    if isinstance(matrix, pd.DataFrame)
                    else matrix[rows].copy()
                )
    return result

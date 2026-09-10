"""Load only expression components needed for gene diagnostic figures."""

import logging
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from anndata.io import read_elem, sparse_dataset
from scipy import sparse

from gsMap.find_latent.st_process import _looks_like_count_matrix

logger = logging.getLogger(__name__)


def read_diagnostic_expression(path: Path, spots, genes, sample_name: str, data_layer: str):
    """Preserve expression semantics and caller row order without full AnnData.

    Already log-transformed X is used exactly as in the report's original
    loader. Counts are normalized using ALL genes in each row before selecting
    diagnostic columns. CSR and dense inputs use bounded row windows; CSC
    inputs are read once to avoid rescanning every column for each row window.
    """
    started = time.perf_counter()
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        names = pd.Index(read_elem(obs[obs.attrs["_index"]]).astype(str))
        suffix = f"|{sample_name}"
        if len(names) and not str(names[0]).endswith(suffix):
            names = names + suffix
        var = handle["var"]
        gene_index = pd.Index(read_elem(var[var.attrs["_index"]]).astype(str))
        if not names.is_unique or not gene_index.is_unique:
            raise ValueError(f"Diagnostic input requires unique spot/gene identifiers: {path}")
        rows = names.get_indexer(spots)
        if np.any(rows < 0):
            raise KeyError(f"Diagnostic spots are missing from {path}")
        available = [gene for gene in genes if gene in gene_index]
        columns = gene_index.get_indexer(available)

        x_present = "X" in handle and handle["X"].attrs.get("encoding-type") != "null"
        already_logged = "log1p" in handle.get("uns", {}) and x_present
        if already_logged:
            node = handle["X"]
        elif data_layer in handle.get("layers", {}):
            node = handle["layers"][data_layer]
        elif data_layer == "X" and x_present:
            node = handle["X"]
        else:
            raise ValueError(f"Data layer {data_layer!r} is missing from {path}")
        encoding = node.attrs.get("encoding-type")
        is_sparse = encoding in ("csr_matrix", "csc_matrix")
        matrix = sparse_dataset(node) if is_sparse else node
        if not is_sparse and not isinstance(node, h5py.Dataset):
            raise ValueError(f"Unsupported expression encoding {encoding!r} in {path}")
        # The existing count detector inspects the first 100 stored values.
        probe = (
            node["data"][:100]
            if is_sparse
            else node[: max(1, (100 + matrix.shape[1] - 1) // max(1, matrix.shape[1]))].ravel()[
                :100
            ]
        )
        dtype = node["data"].dtype if is_sparse else node.dtype
        is_count = not already_logged and (
            data_layer in ("count", "counts", "raw_counts", "impute_count")
            or np.issubdtype(dtype, np.integer)
            or _looks_like_count_matrix(probe)
        )
        result = np.empty((len(rows), len(columns)), dtype=np.float32)
        if not len(rows) or not len(columns):
            return available, result
        if encoding == "csc_matrix":
            # Only the chosen expression matrix is materialized, never other
            # layers/raw/embeddings. CSR provides efficient repeated row access.
            matrix = matrix.to_memory().tocsr()
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        window_rows = (
            16_384
            if is_sparse
            else max(1, min(16_384, (64 << 20) // max(1, matrix.shape[1] * dtype.itemsize)))
        )
        start = 0
        while start < len(rows):
            first = int(sorted_rows[start])
            stop = int(np.searchsorted(sorted_rows, first + window_rows))
            last = int(sorted_rows[stop - 1]) + 1
            block = matrix[first:last]
            if is_count:
                wrapper = ad.AnnData(X=block)
                sc.pp.normalize_total(wrapper, target_sum=1e4)
                selected = wrapper.X[:, columns].copy()
                if sparse.issparse(selected):
                    selected.data = np.log1p(selected.data)
                else:
                    np.log1p(selected, out=selected)
            else:
                selected = block[:, columns]
            if sparse.issparse(selected):
                selected = selected.toarray()
            result[order[start:stop]] = selected[sorted_rows[start:stop] - first]
            start = stop
    logger.info(
        f"Diagnostic selective expression: {len(rows):,} spots, {len(available):,} genes; "
        f"{time.perf_counter() - started:.2f}s; counts normalized={is_count}"
    )
    return available, result

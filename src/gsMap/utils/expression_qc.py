"""Bounded expression statistics independent of any gsMap pipeline stage."""

import h5py
import numpy as np
import scipy.sparse as sp
from anndata.io import read_elem, sparse_dataset


def canonical_expression_block(matrix):
    """Count distinct expressed genes, not sparse storage entries."""
    block = sp.csr_matrix(matrix, copy=True)
    # Check before summing duplicates so a negative/invalid entry cannot disappear.
    if not np.isfinite(block.data).all() or (block.data < 0).any():
        raise ValueError("Expression QC requires finite, nonnegative expression values")
    block.sum_duplicates()
    block.eliminate_zeros()
    if not np.isfinite(block.data).all():
        raise ValueError("Expression QC overflow while combining duplicate entries")
    return block


def expression_row_metrics(matrix, *, chunk_rows=8192):
    """Return positive-gene counts and row sums with bounded CSR/dense reads."""
    if chunk_rows < 1:
        raise ValueError("QC chunk_rows must be positive")
    n_genes = np.empty(matrix.shape[0], dtype=np.int32)
    totals = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], chunk_rows):
        stop = min(start + chunk_rows, matrix.shape[0])
        block = canonical_expression_block(matrix[start:stop])
        n_genes[start:stop] = np.diff(block.indptr)
        totals[start:stop] = np.asarray(block.sum(axis=1, dtype=np.float64)).ravel()
    return n_genes, totals


def h5_expression_row_metrics(node, *, chunk_rows=8192):
    """Read the caller-selected X/layer, without requiring find_latent metadata."""
    if isinstance(node, h5py.Dataset):
        matrix = node
    elif node.attrs.get("encoding-type") == "csr_matrix":
        matrix = sparse_dataset(node)
    else:
        # Convert CSC once instead of repeatedly scanning all columns per row block.
        matrix = read_elem(node).tocsr()
    return expression_row_metrics(matrix, chunk_rows=chunk_rows)

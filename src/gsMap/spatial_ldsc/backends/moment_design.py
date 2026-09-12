"""Trait-specific gene projections for exact LDSC weight initialization.

Only the initializer is projected into gene space. Regression weights still
depend on each cell's LD scores, and each trait retains its own SNP blocks.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp


def supports_moments(sparse, prepared):
    """The fused fast path has spatial + two annotations + unit intercept."""
    return bool(
        sparse is not None
        and sp.isspmatrix_csr(sparse)
        and sparse.dtype == np.float32
        and prepared
        and all(
            d["baseline"].ndim == 2
            and d["baseline"].shape[1] == 3
            and np.all(d["baseline"][:, -1] == 1)
            for d in prepared
        )
    )


def prepare_moment_design(sparse, prepared, n_blocks):
    """Prepare contiguous host arrays; fail before unchecked native indexing."""
    from gsMap.spatial_ldsc.backends.cpu_sparse import _validate_csr

    if not supports_moments(sparse, prepared):
        raise ValueError("Moment backend requires float32 CSR and two baseline annotations")
    if not isinstance(n_blocks, int) or n_blocks < 2:
        raise ValueError("Moment regression needs at least two jackknife blocks")
    _validate_csr(sparse)
    if not np.isfinite(sparse.data).all():
        raise ValueError("Non-finite SNP-gene weights")
    count = len(prepared)
    lengths = np.asarray([len(d["indices"]) for d in prepared], dtype=np.int32)
    if np.any(lengths < n_blocks) or np.any(lengths % n_blocks):
        raise ValueError("Trait SNPs must preserve original jackknife block truncation")
    pad = int(lengths.max())
    positions = np.zeros((count, pad), dtype=np.int32)
    features = np.zeros((count, 6, pad), dtype=np.float32)
    projections, offsets, numerators, nbars = [], [], [], []
    for t, d in enumerate(prepared):
        idx = np.asarray(d["indices"])
        size = int(lengths[t])
        if (
            idx.ndim != 1
            or idx.dtype.kind not in "iu"
            or idx.min() < 0
            or idx.max() >= sparse.shape[0]
            or (not np.all(idx[1:] > idx[:-1]) and len(np.unique(idx)) != size)
        ):
            raise ValueError("Invalid trait SNP mapping")
        if d["baseline"].shape != (size, 3):
            raise ValueError("Invalid baseline shape")
        for key in ("N", "chisq", "baseline_sum", "w_ld"):
            if np.asarray(d[key]).shape != (size,) or not np.isfinite(d[key]).all():
                raise ValueError(f"Invalid trait array: {key}")
        nbar = float(d["Nbar"])
        if not np.isfinite(nbar) or nbar <= 0 or np.any(d["N"] <= 0):
            raise ValueError("Sample sizes and Nbar must be positive and finite")
        if not np.isfinite(d["baseline"]).all() or np.any(d["chisq"] < 0):
            raise ValueError("Invalid baseline or chi-squared response")
        n = np.asarray(d["N"], dtype=np.float32)
        total = np.asarray(d["baseline_sum"], dtype=np.float32)
        y = np.asarray(d["chisq"], dtype=np.float32)
        positions[t, :size] = idx
        features[t, :, :size] = np.stack(
            [n, d["baseline"][:, 0], d["baseline"][:, 1], y, total, np.maximum(d["w_ld"], 1)]
        )
        projections.append(np.asarray(sparse[idx].T @ n.astype(np.float64)).ravel() / size)
        offsets.append(np.mean(n.astype(np.float64) * total))
        numerators.append(10000 * (np.mean(y, dtype=np.float64) - 1))
        nbars.append(nbar)  # never recompute the pre-truncation Nbar
    return dict(
        positions=positions,
        features=features,
        lengths=lengths,
        nbar=np.asarray(nbars, dtype=np.float32),
        q=np.asarray(projections, dtype=np.float32),
        offset=np.asarray(offsets, dtype=np.float32)[:, None],
        numerator=np.asarray(numerators, dtype=np.float32)[:, None],
    )


def gene_hsq(design, marker):
    return np.ascontiguousarray(
        np.clip(design["numerator"] / (design["q"] @ marker + design["offset"]), 0, 1),
        dtype=np.float32,
    )

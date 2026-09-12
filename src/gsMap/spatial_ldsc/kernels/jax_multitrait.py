"""Shared-mean regression for multi-trait spatial LDSC.

Each trait keeps its own retained SNPs, N, Nbar and jackknife blocks. Only
the algebraic reduction order changes; no shared regression coefficients or
shared statistical weights are assumed.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from gsMap.spatial_ldsc.kernels.jax_regression import (
    _packed_block_equations,
    process_spatial_ldsc_batch,
)


def prepare_mean_design(n_union, indices, sample_sizes, baseline_sums, responses):
    design = np.zeros((len(indices), n_union), dtype=np.float32)
    counts, baseline_n, mean_y = [], [], []
    for i, (idx, n, bs, y) in enumerate(
        zip(indices, sample_sizes, baseline_sums, responses, strict=True)
    ):
        idx = np.asarray(idx)
        # Preserve general unsorted mappings, but avoid expensive unique/sort for
        # the production mapping, whose strict order proves uniqueness directly.
        if idx.ndim != 1 or not len(idx):
            raise ValueError("Invalid trait SNP mapping")
        unique = bool(np.all(idx[1:] > idx[:-1])) or len(idx) == len(np.unique(idx))
        if not unique or idx.min() < 0 or idx.max() >= n_union:
            raise ValueError("Invalid trait SNP mapping")
        if not (len(idx) == len(n) == len(bs) == len(y)):
            raise ValueError("Unequal trait input lengths")
        design[i, idx] = n
        counts.append(len(idx))
        baseline_n.append(np.mean(np.asarray(bs, dtype=np.float32) * n))
        mean_y.append(np.mean(np.asarray(y, dtype=np.float32)))
    return tuple(
        map(
            jnp.asarray,
            (
                design,
                np.asarray(counts, dtype=np.float32),
                np.asarray(baseline_n, dtype=np.float32),
                np.asarray(mean_y, dtype=np.float32),
            ),
        )
    )


@jax.jit
def shared_hsq(union_ld, design, counts, baseline_n, mean_y):
    # Unselected SNPs have zero N only in this auxiliary matrix product.
    # The subsequent regression still gathers exactly the original SNP rows.
    weighted_mean = jnp.matmul(design, union_ld, precision=jax.lax.Precision.HIGHEST)
    weighted_mean = weighted_mean / counts[:, None] + baseline_n[:, None]
    return jnp.clip(10000.0 * (mean_y[:, None] - 1.0) / weighted_mean, 0.0, 1.0)


def _from_hsq(n_blocks, ld, bs, y, n, base, wl, nbar, hsq):
    if not 1 <= base.shape[1] <= 4:
        # Keep all custom baseline columns using the existing general kernel.
        return process_spatial_ldsc_batch(n_blocks, ld, bs, y, n, base, wl, nbar)
    n_snps, cells = ld.shape
    if n_snps % n_blocks:
        raise ValueError("SNPs must retain original block truncation")
    width = n_snps // n_blocks
    total = ld + bs[:, None]
    c = hsq[None, :] * n[:, None] / 10000.0
    w = 1.0 / (2 * jnp.square(1 + c * jnp.maximum(total, 1.0)) * jnp.maximum(wl[:, None], 1.0))
    spatial = (ld * n[:, None] / nbar).reshape(n_blocks, width, cells)
    xtx, xty = _packed_block_equations(
        spatial,
        base.reshape(n_blocks, width, base.shape[1]),
        y.reshape(n_blocks, width),
        w.reshape(n_blocks, width, cells),
    )
    full_x, full_y = jnp.sum(xtx, axis=0), jnp.sum(xty, axis=0)
    est = jnp.linalg.solve(full_x, full_y[..., None]).squeeze(-1)
    delete = jnp.linalg.solve(full_x - xtx, (full_y - xty)[..., None]).squeeze(-1)
    pseudo = n_blocks * est[:, 0] - (n_blocks - 1) * delete[:, :, 0]
    beta = jnp.mean(pseudo, axis=0)
    se = jnp.sqrt(jnp.sum(jnp.square(pseudo - beta), axis=0) / (n_blocks * (n_blocks - 1)))
    return beta / nbar, se / nbar


@partial(jax.jit, static_argnums=(0,))
def mean_regression(n_blocks, union_ld, traits, hsq):
    return jnp.stack(
        [
            jnp.stack(_from_hsq(n_blocks, union_ld[idx], *params, hsq[i]))
            for i, (idx, params) in enumerate(traits)
        ]
    )

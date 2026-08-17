"""Single production regression kernel for spatial LDSC."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

jax.config.update("jax_enable_x64", False)


@partial(jit, static_argnums=(0,))
def process_spatial_ldsc_batch(
    n_blocks: int,
    spatial_ld: jnp.ndarray,
    baseline_ld_sum: jnp.ndarray,
    chisq: jnp.ndarray,
    N: jnp.ndarray,
    baseline_ann: jnp.ndarray,
    w_ld: jnp.ndarray,
    Nbar: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Regress one SNP-by-cell LD-score batch with bounded intermediates.

    The kernel accumulates weighted normal equations by jackknife block rather
    than materializing a SNP-by-cell-by-feature design tensor. It is used on
    both CPU and GPU so both backends have identical numerical semantics.
    """
    n_snps, n_spots = spatial_ld.shape
    n_baseline = baseline_ann.shape[1]

    x_tot = spatial_ld + baseline_ld_sum.reshape(-1, 1)
    N_expanded = N.reshape(-1, 1)
    mean_chisq = jnp.mean(chisq)
    mean_x_tot_N = jnp.mean(x_tot * N_expanded, axis=0)
    hsq = 10000.0 * (mean_chisq - 1.0) / mean_x_tot_N
    hsq = jnp.clip(hsq, 0.0, 1.0)

    ld_clip = jnp.maximum(x_tot, 1.0)
    w_ld_clip = jnp.maximum(w_ld.reshape(-1, 1), 1.0)
    c = (hsq.reshape(1, -1) * N_expanded) / 10000.0
    weights = jnp.sqrt(1.0 / (2 * jnp.square(1.0 + c * ld_clip) * w_ld_clip))
    weights_scaled = weights / jnp.sum(weights, axis=0, keepdims=True)
    weights_sq = jnp.square(weights_scaled)

    block_size = n_snps // n_blocks
    n_snps_used = block_size * n_blocks
    spatial_ann = spatial_ld * N_expanded / Nbar
    spatial_blocks = spatial_ann[:n_snps_used].reshape(n_blocks, block_size, n_spots)
    baseline_blocks = baseline_ann[:n_snps_used].reshape(n_blocks, block_size, n_baseline)
    chisq_blocks = chisq[:n_snps_used].reshape(n_blocks, block_size)
    weights_sq_blocks = weights_sq[:n_snps_used].reshape(n_blocks, block_size, n_spots)

    spatial_xty = jnp.einsum("nbs,nb,nbs->ns", spatial_blocks, chisq_blocks, weights_sq_blocks)
    baseline_xty = jnp.einsum("nbk,nb,nbs->nsk", baseline_blocks, chisq_blocks, weights_sq_blocks)
    xty_blocks = jnp.concatenate([spatial_xty[..., None], baseline_xty], axis=-1)

    spatial_spatial = jnp.einsum(
        "nbs,nbs,nbs->ns", spatial_blocks, spatial_blocks, weights_sq_blocks
    )
    spatial_baseline = jnp.einsum(
        "nbs,nbk,nbs->nsk", spatial_blocks, baseline_blocks, weights_sq_blocks
    )
    baseline_baseline = jnp.einsum(
        "nbk,nbl,nbs->nskl", baseline_blocks, baseline_blocks, weights_sq_blocks
    )
    top_row = jnp.concatenate(
        [spatial_spatial[..., None, None], spatial_baseline[..., None, :]], axis=-1
    )
    bottom_rows = jnp.concatenate([spatial_baseline[..., :, None], baseline_baseline], axis=-1)
    xtx_blocks = jnp.concatenate([top_row, bottom_rows], axis=-2)

    xty_total = jnp.sum(xty_blocks, axis=0)
    xtx_total = jnp.sum(xtx_blocks, axis=0)
    est = jnp.linalg.solve(xtx_total, xty_total[..., None]).squeeze(-1)

    xty_del = xty_total - xty_blocks
    xtx_del = xtx_total - xtx_blocks
    delete_ests = jnp.linalg.solve(xtx_del, xty_del[..., None]).squeeze(-1)

    spatial_pseudovalues = n_blocks * est[:, 0] - (n_blocks - 1) * delete_ests[:, :, 0]
    spatial_jknife_est = jnp.mean(spatial_pseudovalues, axis=0)
    spatial_centered = spatial_pseudovalues - spatial_jknife_est
    spatial_jknife_se = jnp.sqrt(
        jnp.sum(jnp.square(spatial_centered), axis=0) / (n_blocks * (n_blocks - 1))
    )
    return spatial_jknife_est / Nbar, spatial_jknife_se / Nbar

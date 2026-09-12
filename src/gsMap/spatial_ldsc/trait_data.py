"""Trait-specific data preparation for spatial LDSC."""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("gsMap.spatial_ldsc.trait_data")
PARAMETER_KEYS = ("baseline_sum", "chisq", "N", "baseline", "w_ld", "Nbar")


def prepare_shared_trait_arrays(data: dict, *, name: str, filename: str, n_ref: int) -> dict:
    """Normalize an already block-truncated trait without recomputing Nbar.

    Baseline annotations are scaled by N/Nbar and the unit intercept is last.
    Positions remain in the trait's original order until SNP-union indexing.
    """
    positions = np.asarray(data["snp_positions"], dtype=np.int64)
    if np.any(positions < 0) or np.any(positions >= n_ref):
        raise ValueError("Trait SNPs are missing from the LD reference")
    n = np.asarray(data["N"], dtype=np.float32)
    nbar = np.float32(data["Nbar"])
    baseline = data["baseline_ld"].values.astype(np.float32) * n[:, None] / nbar
    baseline = np.concatenate([baseline, np.ones((len(n), 1), dtype=np.float32)], axis=1)
    return dict(
        name=name,
        filename=filename,
        positions=positions,
        baseline_sum=np.asarray(data["baseline_ld_sum"], dtype=np.float32),
        chisq=np.asarray(data["chisq"], dtype=np.float32),
        N=n,
        Nbar=nbar,
        baseline=baseline,
        w_ld=np.asarray(data["w_ld"], dtype=np.float32),
    )


def prepare_snp_data_for_blocks(data: dict, n_blocks: int) -> dict:
    """Prepare SNP-related data arrays for equal-sized jackknife blocks."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2 for jackknife standard errors")
    if "chisq" in data:
        n_snps = len(data["chisq"])
    elif "N" in data:
        n_snps = len(data["N"])
    else:
        raise ValueError("Cannot determine number of SNPs from data")
    if n_snps < n_blocks:
        raise ValueError(
            f"Spatial LDSC requires at least one SNP per jackknife block: "
            f"n_snps={n_snps}, n_blocks={n_blocks}"
        )

    block_size = n_snps // n_blocks
    n_snps_used = block_size * n_blocks
    n_dropped = n_snps - n_snps_used

    if n_dropped > 0:
        logger.info(
            f"Truncating SNP data: dropping {n_dropped} SNPs "
            f"({n_dropped / n_snps * 100:.3f}%) for {n_blocks} blocks of size {block_size}"
        )

    truncated = {}
    snp_keys = ["baseline_ld_sum", "w_ld", "chisq", "N", "snp_positions"]

    for key, value in data.items():
        if key in snp_keys and isinstance(value, np.ndarray | jnp.ndarray):
            truncated[key] = value[:n_snps_used]
        elif key == "baseline_ld":
            truncated[key] = value.iloc[:n_snps_used]
        else:
            truncated[key] = value

    truncated["block_size"] = block_size
    truncated["n_blocks"] = n_blocks
    truncated["n_snps_used"] = n_snps_used
    truncated["n_snps_original"] = n_snps

    return truncated

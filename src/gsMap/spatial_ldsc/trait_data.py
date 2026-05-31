"""Trait-specific data preparation for spatial LDSC."""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("gsMap.spatial_ldsc.trait_data")


def prepare_snp_data_for_blocks(data: dict, n_blocks: int) -> dict:
    """Prepare SNP-related data arrays for equal-sized jackknife blocks."""
    if "chisq" in data:
        n_snps = len(data["chisq"])
    elif "N" in data:
        n_snps = len(data["N"])
    else:
        raise ValueError("Cannot determine number of SNPs from data")

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

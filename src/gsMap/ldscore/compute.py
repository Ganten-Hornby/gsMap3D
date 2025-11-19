"""
Simplified LD score computation without masking or padding.

Direct computation of unbiased L2 statistics from genotype matrices.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple

from .constants import LDSC_BIAS_CORRECTION_DF


@jax.jit
def compute_unbiased_l2_batch(
    X_hm3: jnp.ndarray,
    X_ref_block: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute unbiased LD scores (L2) for HM3 SNPs against reference block.

    The unbiased L2 estimator is:
        L2 = r^2 - (1 - r^2) / (N - 2)

    where r^2 is the squared correlation and N is the number of individuals.

    Parameters
    ----------
    X_hm3 : jnp.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : jnp.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)

    Returns
    -------
    jnp.ndarray
        Unbiased LD scores, shape (n_hm3_snps, n_ref_snps)

    Notes
    -----
    All genotypes are assumed to be pre-standardized (mean=0, std=1).
    The computation directly calculates r^2 and applies bias correction.
    """
    n_individuals = X_hm3.shape[0]

    # Compute correlation matrix: r = (1/N) * X_hm3^T @ X_ref_block
    # Shape: (n_hm3_snps, n_ref_snps)
    r = jnp.dot(X_hm3.T, X_ref_block) / n_individuals

    # Compute r^2
    r_squared = r ** 2

    # Apply bias correction: L2 = r^2 - (1 - r^2) / (N - 2)
    bias_correction = (1.0 - r_squared) / (n_individuals - LDSC_BIAS_CORRECTION_DF)
    l2_unbiased = r_squared - bias_correction

    return l2_unbiased


@jax.jit
def compute_ld_scores(
    X_hm3: jnp.ndarray,
    X_ref_block: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute LD scores by summing unbiased L2 over reference SNPs.

    Parameters
    ----------
    X_hm3 : jnp.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : jnp.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)

    Returns
    -------
    jnp.ndarray
        LD scores for each HM3 SNP, shape (n_hm3_snps,)

    Notes
    -----
    The LD score for each HM3 SNP is the sum of its unbiased L2 with all
    reference SNPs in the block.
    """
    # Compute unbiased L2 matrix
    l2_unbiased = compute_unbiased_l2_batch(X_hm3, X_ref_block)

    # Sum over reference SNPs (axis=1)
    ld_scores = jnp.sum(l2_unbiased, axis=1)

    return ld_scores


def compute_batch_weights_segment_sum(
    X_hm3: jnp.ndarray,
    X_ref_block: jnp.ndarray,
    block_links: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute LD score weight matrix using segment sum for efficient feature aggregation.

    This is more efficient than creating dense feature masks when features are sparse.
    Uses segment_sum to aggregate L2 scores by feature.

    Parameters
    ----------
    X_hm3 : jnp.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : jnp.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)
    block_links : jnp.ndarray
        Feature indices for each reference SNP, shape (n_ref_snps,)
        Values are feature indices in range [0, F] where F is the "unmapped" bin

    Returns
    -------
    weights : jnp.ndarray
        Weight matrix, shape (n_hm3_snps, n_unique_features)
    unique_features : jnp.ndarray
        Global feature indices, shape (n_unique_features,)

    Notes
    -----
    For each HM3 SNP and feature:
        W[i, j] = sum_{k: block_links[k] == feature_j} L2(hm3_snp_i, ref_snp_k)

    This uses JAX's segment_sum for efficient aggregation.
    """
    # Compute unbiased L2 matrix: (n_hm3_snps, n_ref_snps)
    l2_unbiased = compute_unbiased_l2_batch(X_hm3, X_ref_block)

    # Get unique features in this block and their indices
    unique_features = jnp.unique(block_links)
    n_unique = len(unique_features)

    # Map block_links to local indices [0, n_unique)
    # Since unique_features is sorted, we can use searchsorted
    local_indices = jnp.searchsorted(unique_features, block_links)
    
    # Use segment_sum on the transposed L2 matrix
    # l2_unbiased.T shape: (n_ref_snps, n_hm3_snps)
    # local_indices shape: (n_ref_snps,)
    # We sum rows of l2_unbiased.T (which are columns of l2_unbiased) based on local_indices
    # Result shape: (n_unique, n_hm3_snps)
    weights_T = jax.ops.segment_sum(l2_unbiased.T, local_indices, num_segments=n_unique)
    
    # Transpose back to get (n_hm3_snps, n_unique)
    weights = weights_T.T

    return weights, unique_features


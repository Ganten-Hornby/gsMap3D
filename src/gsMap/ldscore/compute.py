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


def compute_batch_ld_scores_numpy(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
) -> np.ndarray:
    """
    Numpy wrapper for computing LD scores.

    Parameters
    ----------
    X_hm3 : np.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : np.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)

    Returns
    -------
    np.ndarray
        LD scores for each HM3 SNP, shape (n_hm3_snps,)
    """
    # Convert to JAX arrays
    X_hm3_jax = jnp.array(X_hm3)
    X_ref_block_jax = jnp.array(X_ref_block)

    # Compute LD scores
    ld_scores_jax = compute_ld_scores(X_hm3_jax, X_ref_block_jax)

    # Convert back to numpy
    return np.array(ld_scores_jax)


@partial(jax.jit, static_argnames=["n_features"])
def compute_batch_weights(
    X_hm3: jnp.ndarray,
    X_ref_block: jnp.ndarray,
    feature_masks: jnp.ndarray,
    n_features: int,
) -> jnp.ndarray:
    """
    Compute LD score weight matrix for multiple omics features.

    For each HM3 SNP and each feature, compute the sum of unbiased L2
    with reference SNPs that overlap that feature.

    Parameters
    ----------
    X_hm3 : jnp.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : jnp.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)
    feature_masks : jnp.ndarray
        Binary mask indicating which reference SNPs overlap each feature,
        shape (n_ref_snps, n_features)
    n_features : int
        Number of omics features

    Returns
    -------
    jnp.ndarray
        Weight matrix, shape (n_hm3_snps, n_features)

    Notes
    -----
    For each HM3 SNP i and feature j:
        W[i, j] = sum_{k: feature_masks[k, j] == 1} L2(hm3_snp_i, ref_snp_k)
    """
    # Compute unbiased L2 matrix: (n_hm3_snps, n_ref_snps)
    l2_unbiased = compute_unbiased_l2_batch(X_hm3, X_ref_block)

    # Multiply by feature masks and sum over reference SNPs
    # l2_unbiased: (n_hm3_snps, n_ref_snps)
    # feature_masks: (n_ref_snps, n_features)
    # Result: (n_hm3_snps, n_features)
    weights = jnp.dot(l2_unbiased, feature_masks)

    return weights


def compute_batch_weights_numpy(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
    feature_masks: np.ndarray,
) -> np.ndarray:
    """
    Numpy wrapper for computing weight matrix.

    Parameters
    ----------
    X_hm3 : np.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : np.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)
    feature_masks : np.ndarray
        Binary mask, shape (n_ref_snps, n_features)

    Returns
    -------
    np.ndarray
        Weight matrix, shape (n_hm3_snps, n_features)
    """
    n_features = feature_masks.shape[1]

    # Convert to JAX arrays
    X_hm3_jax = jnp.array(X_hm3)
    X_ref_block_jax = jnp.array(X_ref_block)
    feature_masks_jax = jnp.array(feature_masks)

    # Compute weights
    weights_jax = compute_batch_weights(
        X_hm3_jax, X_ref_block_jax, feature_masks_jax, n_features
    )

    # Convert back to numpy
    return np.array(weights_jax)


def compute_correlation_matrix(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
) -> np.ndarray:
    """
    Compute correlation matrix between HM3 SNPs and reference block.

    Parameters
    ----------
    X_hm3 : np.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : np.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)

    Returns
    -------
    np.ndarray
        Correlation matrix, shape (n_hm3_snps, n_ref_snps)
    """
    n_individuals = X_hm3.shape[0]
    return np.dot(X_hm3.T, X_ref_block) / n_individuals


def compute_r_squared_matrix(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
) -> np.ndarray:
    """
    Compute r^2 matrix between HM3 SNPs and reference block.

    Parameters
    ----------
    X_hm3 : np.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : np.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)

    Returns
    -------
    np.ndarray
        r^2 matrix, shape (n_hm3_snps, n_ref_snps)
    """
    r = compute_correlation_matrix(X_hm3, X_ref_block)
    return r ** 2


@jax.jit
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

    # Create mapping from block_links to dense indices [0, n_unique)
    # This allows us to use segment_sum efficiently
    # We'll process each HM3 SNP separately and stack results
    def process_hm3_snp(l2_row):
        """Process one HM3 SNP's L2 scores."""
        # l2_row: (n_ref_snps,)
        # For each unique feature, sum L2 scores of ref SNPs with that feature
        weights_row = jnp.zeros(n_unique, dtype=l2_row.dtype)

        # Use segment_sum: for each unique feature, sum corresponding L2 values
        for i, feature_idx in enumerate(unique_features):
            mask = (block_links == feature_idx)
            weights_row = weights_row.at[i].set(jnp.sum(l2_row * mask))

        return weights_row

    # Vectorize over HM3 SNPs using vmap
    weights = jax.vmap(process_hm3_snp)(l2_unbiased)
    # Shape: (n_hm3_snps, n_unique_features)

    return weights, unique_features


def compute_batch_weights_segment_sum_numpy(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
    block_links: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numpy wrapper for segment sum weight computation.

    Parameters
    ----------
    X_hm3 : np.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : np.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)
    block_links : np.ndarray
        Feature indices for reference SNPs, shape (n_ref_snps,)

    Returns
    -------
    weights : np.ndarray
        Weight matrix, shape (n_hm3_snps, n_unique_features)
    unique_features : np.ndarray
        Global feature indices, shape (n_unique_features,)
    """
    # Convert to JAX arrays
    X_hm3_jax = jnp.array(X_hm3)
    X_ref_block_jax = jnp.array(X_ref_block)
    block_links_jax = jnp.array(block_links, dtype=jnp.int32)

    # Compute weights
    weights_jax, unique_features_jax = compute_batch_weights_segment_sum(
        X_hm3_jax, X_ref_block_jax, block_links_jax
    )

    # Convert back to numpy
    return np.array(weights_jax), np.array(unique_features_jax)

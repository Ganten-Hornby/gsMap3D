"""
Core JAX-accelerated computation functions for LD score weight matrix calculation.

This module provides JIT-compiled functions for efficiently computing LD scores
using batched matrix operations on GPU/TPU devices.
"""

from functools import partial

import jax
import jax.numpy as jnp


# Unbiased LD score estimator bias correction constant
# For sample size N, the bias correction is (1 - r^2) / (N - 2)
LDSC_BIAS_CORRECTION_DF = 2.0


@partial(jax.jit, static_argnames=["block_len", "n_features"])
def compute_batch_weights(
    ref_block: jnp.ndarray,
    hm3_batch: jnp.ndarray,
    block_mapping: jnp.ndarray,
    n_features: int,
) -> jnp.ndarray:
    """
    Compute LD score weight matrix using JAX JIT compilation.

    This function calculates the unbiased LD score (L2) between reference SNPs
    and HapMap3 SNPs, then aggregates by feature using segment_sum.

    Parameters
    ----------
    ref_block : jnp.ndarray
        Reference genotypes (standardized), shape (N_individuals, block_len)
    hm3_batch : jnp.ndarray
        HapMap3 genotypes (standardized), shape (N_individuals, batch_size)
    block_mapping : jnp.ndarray
        Feature index for each reference SNP, shape (block_len,)
        Values in [0, n_features). Value n_features indicates unmapped SNPs.
    n_features : int
        Total number of distinct features (F)

    Returns
    -------
    jnp.ndarray
        Weight matrix of shape (batch_size, n_features)

    Notes
    -----
    The unbiased L2 estimator is computed as:
        L2 = r^2 - (1 - r^2) / (N - 2)

    where r is the Pearson correlation coefficient and N is the sample size.
    This corrects for the upward bias in squared correlation estimates.

    References
    ----------
    .. [1] Bulik-Sullivan, B. K. et al. (2015). LD Score regression distinguishes
           confounding from polygenicity in genome-wide association studies.
           Nature Genetics, 47(3), 291-295.
    """
    N = ref_block.shape[0]

    # Step 1: Compute correlation matrix
    # For standardized genotypes (mean=0, var=1), the correlation is simply:
    # r = (X_ref.T @ X_hm3) / N
    # Shape: (block_len, batch_size)
    cov = jnp.dot(ref_block.T, hm3_batch) / N

    # Step 2: Apply unbiased L2 estimator
    # L2_unbiased = r^2 - (1 - r^2) / (N - 2)
    # This corrects for upward bias in squared correlation estimates
    l2_scores = jnp.square(cov)
    l2_unbiased = l2_scores - (1.0 - l2_scores) / (N - LDSC_BIAS_CORRECTION_DF)

    # Step 3: Aggregate LD scores by feature using segment_sum
    # For each feature, sum the LD scores of all SNPs mapped to that feature
    # block_mapping: (block_len,) contains feature indices [0, n_features)
    # Unmapped SNPs have index n_features (garbage bin)
    # Output shape: (n_features + 1, batch_size)
    aggregated = jax.ops.segment_sum(
        l2_unbiased, block_mapping, num_segments=n_features + 1
    )

    # Step 4: Remove garbage bin and transpose to output shape
    # Return shape: (batch_size, n_features)
    return aggregated[:n_features, :].T


def prepare_padding(matrix: jnp.ndarray, target_width: int) -> jnp.ndarray:
    """
    Pad or truncate matrix width to match quantized block length for JIT.

    This function ensures consistent matrix dimensions for JAX JIT compilation
    by padding short matrices or truncating long ones to a target width.

    Parameters
    ----------
    matrix : jnp.ndarray
        Input matrix of shape (n_rows, current_width)
    target_width : int
        Desired width for the output matrix

    Returns
    -------
    jnp.ndarray
        Matrix of shape (n_rows, target_width)

    Notes
    -----
    - If current_width < target_width: Pads with zeros on the right
    - If current_width > target_width: Truncates to first target_width columns
    - If current_width == target_width: Returns as-is

    This is used for window quantization in the dynamic programming approach
    to minimize JAX recompilation overhead when processing variable-sized
    LD windows.
    """
    current_width = matrix.shape[1]

    if current_width < target_width:
        # Pad with zeros on the right
        pad_width = target_width - current_width
        return jnp.pad(matrix, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)
    elif current_width > target_width:
        # Truncate to target width
        return matrix[:, :target_width]
    else:
        # Already correct size
        return matrix
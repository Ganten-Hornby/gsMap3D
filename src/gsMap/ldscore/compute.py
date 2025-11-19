"""
Core JAX-accelerated computation functions for LD score weight matrix calculation.

This module provides JIT-compiled functions for efficiently computing LD scores
using batched matrix operations on GPU/TPU devices.
"""

from functools import partial

import jax
import jax.numpy as jnp
from .constants import LDSC_BIAS_CORRECTION_DF


@partial(jax.jit, static_argnames=["n_features"])
def compute_batch_weights(
    ref_block: jnp.ndarray,
    hm3_batch: jnp.ndarray,
    block_mapping: jnp.ndarray,
    relative_starts: jnp.ndarray,
    relative_ends: jnp.ndarray,
    n_features: int,
) -> jnp.ndarray:
    """
    Compute LD score weight matrix using JAX JIT compilation with sliding window masking.

    Parameters
    ----------
    ref_block : jnp.ndarray
        Reference genotypes (standardized) for the Super Block.
        Shape: (N_individuals, block_len)
    hm3_batch : jnp.ndarray
        HapMap3 genotypes (standardized) for the current batch.
        Shape: (N_individuals, batch_size)
    block_mapping : jnp.ndarray
        Feature index for each reference SNP in the Super Block.
        Shape: (block_len,)
    relative_starts : jnp.ndarray
        Start index of the valid window for each HM3 SNP, relative to the Super Block start.
        Shape: (batch_size,)
    relative_ends : jnp.ndarray
        End index (exclusive) of the valid window for each HM3 SNP.
        Shape: (batch_size,)
    n_features : int
        Total number of distinct features (F)

    Returns
    -------
    jnp.ndarray
        Weight matrix of shape (batch_size, n_features)
    """
    N = ref_block.shape[0]
    block_len = ref_block.shape[1]
    batch_size = hm3_batch.shape[1]

    # 1. Compute Correlation Matrix (Super Block vs Batch)
    # Shape: (block_len, batch_size)
    # cov[j, i] = correlation between Ref SNP j and HM3 SNP i
    cov = jnp.dot(ref_block.T, hm3_batch) / N

    # 2. Apply Unbiased L2 Estimator
    # L2 = r^2 - (1 - r^2) / (N - 2)
    l2_scores = jnp.square(cov)
    l2_unbiased = l2_scores - (1.0 - l2_scores) / (N - LDSC_BIAS_CORRECTION_DF)

    # 3. Apply Sliding Window Mask
    # We need to zero out L2 scores for reference SNPs that are outside
    # the specific window of the HM3 SNP.

    # Create grid of indices for the block: (block_len, 1)
    idx_grid = jnp.arange(block_len)[:, None]

    # Broadcast comparison against batch limits: (1, batch_size)
    # Mask shape: (block_len, batch_size)
    # Valid if: start <= index < end
    mask = (idx_grid >= relative_starts[None, :]) & (idx_grid < relative_ends[None, :])

    # Apply mask (zeros out invalid regions)
    masked_l2 = jnp.where(mask, l2_unbiased, 0.0)

    # 4. Aggregate by Feature
    # block_mapping maps rows (Ref SNPs) to features.
    # We sum down the rows for each column (HM3 SNP).
    # Shape: (n_features + 1, batch_size)
    aggregated = jax.ops.segment_sum(
        masked_l2, block_mapping, num_segments=n_features + 1
    )

    # 5. Transpose and remove garbage bin
    return aggregated[:n_features, :].T


def prepare_padding(matrix: jnp.ndarray, target_width: int, constant_values=0) -> jnp.ndarray:
    """
    Pad or truncate matrix width to match quantized block length.
    """
    current_width = matrix.shape[1]

    if current_width < target_width:
        pad_width = target_width - current_width
        # Pad on the right (axis 1)
        return jnp.pad(matrix, ((0, 0), (0, pad_width)), mode="constant", constant_values=constant_values)
    elif current_width > target_width:
        return matrix[:, :target_width]
    else:
        return matrix

def prepare_vector_padding(vector: jnp.ndarray, target_width: int, fill_value=0) -> jnp.ndarray:
    """
    Pad 1D vector to match quantized block length.
    """
    current_width = vector.shape[0]
    if current_width < target_width:
        pad_width = target_width - current_width
        return jnp.pad(vector, (0, pad_width), mode="constant", constant_values=fill_value)
    elif current_width > target_width:
        return vector[:target_width]
    else:
        return vector
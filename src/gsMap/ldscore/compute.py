import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['block_len', 'n_features'])
def compute_batch_weights(
        ref_block: jnp.ndarray,  # Shape (N, block_len)
        hm3_batch: jnp.ndarray,  # Shape (N, batch_hm3)
        block_mapping: jnp.ndarray,  # Shape (block_len,) - feature index for each Ref SNP
        n_features: int  # Total distinct features F
):
    """
    JAX JIT-compiled core calculation.

    Args:
        ref_block: Reference Genotypes (Normalized)
        hm3_batch: HM3 Genotypes (Normalized)
        block_mapping: Vector mapping columns of ref_block to feature indices [0, F]
                       Indices equal to F are considered 'blank' and ignored/summed to trash bin.

    Returns:
        weight_matrix: Shape (batch_hm3, n_features)
    """
    N = ref_block.shape[0]

    # 1. Compute Correlation Matrix (Unbiased L2)
    # R = (X_ref.T @ X_hm3) / N
    # Shape: (block_len, batch_hm3)
    # Note: Standard LDSC formula is R^2 - bias.
    # We do (A.T @ B) which is (block_len, N) @ (N, batch) -> (block_len, batch)

    cov = jnp.dot(ref_block.T, hm3_batch) / N

    # Unbiased L2 estimator: r^2 - (1-r^2)/(N-2)
    l2_scores = jnp.square(cov)
    l2_unbiased = l2_scores - (1.0 - l2_scores) / (N - 2.0)

    # 2. Segment Sum (Map SNPs to Features)
    # We need to sum rows of l2_unbiased (dim 0) based on block_mapping.
    # block_mapping has shape (block_len,).
    # Result should be (F+1, batch_hm3). F+1 to catch the unmapped/blank SNPs.

    # jax.ops.segment_sum sums the first dimension.
    # l2_unbiased shape is (block_len, batch_hm3).
    aggregated = jax.ops.segment_sum(
        l2_unbiased,
        block_mapping,
        num_segments=n_features + 1
    )

    # Return only the valid features [0, F), excluding the F-th bin (garbage bin)
    # Result shape: (n_features, batch_hm3) -> Transpose to (batch_hm3, n_features)
    return aggregated[:n_features, :].T


def prepare_padding(matrix, target_width):
    """Pad matrix width to match quantized block length for JIT."""
    current_width = matrix.shape[1]
    if current_width < target_width:
        pad_width = target_width - current_width
        return jnp.pad(matrix, ((0, 0), (0, pad_width)))
    return matrix[:, :target_width]
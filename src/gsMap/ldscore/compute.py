"""
Simplified LD score computation without masking or padding.

Direct computation of unbiased L2 statistics from genotype matrices using NumPy and Scipy.
"""

import numpy as np
import scipy.sparse
from typing import Tuple, List

from .constants import LDSC_BIAS_CORRECTION_DF


def compute_unbiased_l2_batch(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
) -> np.ndarray:
    """
    Compute unbiased LD scores (L2) for HM3 SNPs against reference block.

    The unbiased L2 estimator is:
        L2 = r^2 - (1 - r^2) / (N - 2)

    where r^2 is the squared correlation and N is the number of individuals.

    Parameters
    ----------
    X_hm3 : np.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : np.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)

    Returns
    -------
    np.ndarray
        Unbiased LD scores, shape (n_hm3_snps, n_ref_snps)
    """
    n_individuals = X_hm3.shape[0]

    # Compute correlation matrix: r = (1/N) * X_hm3^T @ X_ref_block
    # shape: (n_hm3_snps, n_ref_snps)
    r = np.dot(X_hm3.T, X_ref_block) / n_individuals

    # Compute r^2
    r_squared = r ** 2

    # Apply bias correction
    bias_correction = (1.0 - r_squared) / (n_individuals - LDSC_BIAS_CORRECTION_DF)
    l2_unbiased = r_squared - bias_correction

    return l2_unbiased


def compute_ld_scores(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
) -> np.ndarray:
    """
    Compute LD scores by summing unbiased L2 over reference SNPs.

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
    # Compute unbiased L2 matrix
    l2_unbiased = compute_unbiased_l2_batch(X_hm3, X_ref_block)

    # Sum over reference SNPs (axis=1)
    ld_scores = np.sum(l2_unbiased, axis=1)

    return ld_scores


def compute_batch_weights_segment_sum(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
    block_links: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LD score weight matrix using efficient sparse matrix multiplication.

    Aggregates L2 scores by feature using a sparse transform matrix.

    Parameters
    ----------
    X_hm3 : np.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : np.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)
    block_links : np.ndarray
        Feature indices for each reference SNP, shape (n_ref_snps,)
        Values are global feature indices.

    Returns
    -------
    weights : np.ndarray
        Weight matrix, shape (n_hm3_snps, n_unique_features_in_batch)
    unique_features : np.ndarray
        Global feature indices found in this batch, shape (n_unique_features_in_batch,)
    """
    # 1. Compute unbiased L2 matrix: (n_hm3_snps, n_ref_snps)
    l2_unbiased = compute_unbiased_l2_batch(X_hm3, X_ref_block)

    n_ref_snps = X_ref_block.shape[1]

    # 2. Identify unique features in this block
    unique_features = np.unique(block_links)
    n_unique = len(unique_features)

    # 3. Create a sparse transform matrix T: (n_ref_snps, n_unique)
    # We map global feature indices (block_links) to local 0..n_unique indices
    feature_map = {val: i for i, val in enumerate(unique_features)}

    # Map each ref SNP to a local column index
    col_indices = np.array([feature_map[x] for x in block_links], dtype=np.int32)
    row_indices = np.arange(n_ref_snps, dtype=np.int32)
    data = np.ones(n_ref_snps, dtype=np.float32)

    # Transform Matrix T
    # T[i, j] = 1 if ref_snp i belongs to feature j (local index)
    T = scipy.sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_ref_snps, n_unique)
    )

    # 4. Compute Weights: W = L2 @ T
    # (n_hm3, n_ref) @ (n_ref, n_unique) -> (n_hm3, n_unique)
    # Using dense matrix multiplication since L2 is dense
    weights = l2_unbiased @ T

    # If weights resulted in a sparse matrix (depends on implementation), convert to dense array
    if scipy.sparse.issparse(weights):
        weights = weights.toarray()

    return weights, unique_features
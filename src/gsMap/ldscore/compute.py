"""
Simplified LD score computation without masking or padding.

Direct computation of unbiased L2 statistics from genotype matrices using NumPy and Scipy.
"""


import numpy as np
import scipy.sparse

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


def compute_batch_weights_sparse(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
    block_mapping_matrix: scipy.sparse.csr_matrix | np.ndarray,
) -> np.ndarray:
    """
    Compute LD score weight matrix using matrix multiplication.

    Works for both sparse and dense mapping matrices.
    Weights = L2_Unbiased @ Mapping_Matrix

    Parameters
    ----------
    X_hm3 : np.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : np.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)
    block_mapping_matrix : Union[scipy.sparse.csr_matrix, np.ndarray]
        Mapping matrix for the reference block, shape (n_ref_snps, n_features).
        Can be a sparse CSR matrix (from creating_snp_feature_map) or a dense array (from annotations).

    Returns
    -------
    weights : np.ndarray
        Weight matrix, shape (n_hm3_snps, n_features)
    """
    # 1. Compute unbiased L2 matrix: (n_hm3_snps, n_ref_snps)
    l2_unbiased = compute_unbiased_l2_batch(X_hm3, X_ref_block)

    # 2. Compute Weights: W = L2 @ M
    # (n_hm3, n_ref) @ (n_ref, n_features) -> (n_hm3, n_features)
    # numpy dot handles dense @ dense
    # scipy.sparse handles dense @ sparse
    if scipy.sparse.issparse(block_mapping_matrix):
        weights = l2_unbiased @ block_mapping_matrix
    else:
        weights = np.dot(l2_unbiased, block_mapping_matrix)

    # Ensure output is a dense numpy array
    if scipy.sparse.issparse(weights):
        weights = weights.toarray()
    elif isinstance(weights, np.matrix):
        weights = np.asarray(weights)

    return weights

"""
Utility functions for LD score calculation with Numba acceleration.

This module provides high-performance functions for preprocessing genomic data,
including window quantization and LD block calculation.
"""

import numpy as np
import numba


@numba.njit
def dynamic_programming_quantization(lengths: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Quantize variable block lengths into fixed bins to minimize JAX JIT recompilation.

    This function reduces the number of unique LD window sizes by rounding up to
    discrete bin values, minimizing the JAX JIT recompilation overhead that occurs
    when processing variable-sized windows.

    Parameters
    ----------
    lengths : np.ndarray
        Array of block lengths (LD window sizes) to quantize
    n_bins : int
        Number of discrete bins to use for quantization

    Returns
    -------
    np.ndarray
        Quantized lengths, where each value is rounded up to the nearest bin

    Notes
    -----
    This is a greedy quantile-based implementation. For stricter optimal binning
    that minimizes total padding waste, a dynamic programming algorithm could be
    used instead.

    The quantization strategy uses percentiles to distribute bin boundaries,
    which provides reasonable performance in practice.

    Examples
    --------
    >>> lengths = np.array([100, 150, 200, 250, 300])
    >>> quantized = dynamic_programming_quantization(lengths, n_bins=3)
    >>> print(quantized)  # Values rounded up to 3 discrete bins
    """
    max_val = lengths.max()
    if n_bins >= len(np.unique(lengths)):
        return lengths

    # Create bins (simple quantile based for demonstration)
    # Ideally, this is where the DP logic goes to find optimal cut points
    # that minimize padding waste.
    targets = np.percentile(lengths, np.linspace(0, 100, n_bins + 1)[1:])
    targets = np.unique(np.ceil(targets).astype(np.int32))

    quantized = np.zeros_like(lengths)
    for l_idx in range(len(lengths)):
        val = lengths[l_idx]
        # Find smallest target >= val
        found = False
        for t in targets:
            if t >= val:
                quantized[l_idx] = t
                found = True
                break
        if not found:
            quantized[l_idx] = max_val

    return quantized


@numba.njit
def get_block_limits(
    coords_ref: np.ndarray, coords_hm3: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate LD window boundaries for each HapMap3 SNP in the reference panel.

    This function efficiently computes the range of reference SNPs that fall
    within a genomic window around each HM3 SNP using a two-pointer algorithm.

    Parameters
    ----------
    coords_ref : np.ndarray
        Genomic coordinates (base pairs) of reference SNPs, shape (M_ref,)
        Must be sorted in ascending order
    coords_hm3 : np.ndarray
        Genomic coordinates (base pairs) of HapMap3 SNPs, shape (M_hm3,)
        Must be sorted in ascending order
    window_size : int
        LD window size in base pairs (symmetric around each HM3 SNP)

    Returns
    -------
    left_limits : np.ndarray
        Left boundary indices in coords_ref for each HM3 SNP, shape (M_hm3,)
    right_limits : np.ndarray
        Right boundary indices (exclusive) in coords_ref for each HM3 SNP, shape (M_hm3,)

    Notes
    -----
    For each HM3 SNP at position `pos`, the window is [pos - window_size, pos + window_size].
    Reference SNPs within this window have indices in range [left_limits[i], right_limits[i]).

    The two-pointer algorithm ensures O(M_ref + M_hm3) time complexity, which is
    optimal for this problem.

    Examples
    --------
    >>> coords_ref = np.array([100, 200, 300, 400, 500])
    >>> coords_hm3 = np.array([250])
    >>> left, right = get_block_limits(coords_ref, coords_hm3, window_size=150)
    >>> print(left, right)  # SNPs within [100, 400] for HM3 SNP at 250
    [0] [4]
    """
    m_hm3 = len(coords_hm3)
    m_ref = len(coords_ref)

    left_limits = np.zeros(m_hm3, dtype=np.int32)
    right_limits = np.zeros(m_hm3, dtype=np.int32)

    # Two-pointer approach for efficiency
    l_ptr = 0
    r_ptr = 0

    for i in range(m_hm3):
        pos = coords_hm3[i]
        min_bp = pos - window_size
        max_bp = pos + window_size

        # Find left bound
        while l_ptr < m_ref and coords_ref[l_ptr] < min_bp:
            l_ptr += 1
        left_limits[i] = l_ptr

        # Find right bound
        # Reset r_ptr to l_ptr ensures we don't look backwards,
        # but we must advance to find the end
        if r_ptr < l_ptr:
            r_ptr = l_ptr
        while r_ptr < m_ref and coords_ref[r_ptr] <= max_bp:
            r_ptr += 1
        right_limits[i] = r_ptr

    return left_limits, right_limits
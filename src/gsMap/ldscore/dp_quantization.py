"""
Dynamic programming based batch width quantization.

This module implements a divide-and-conquer DP algorithm to optimally partition
SNP batches into Q groups with quantized widths, minimizing total padding cost.
"""

import numpy as np
from typing import Tuple


def compute_prefix(A: np.ndarray) -> np.ndarray:
    """
    Compute prefix sum array for cost calculation.

    Parameters
    ----------
    A : np.ndarray
        Array of batch widths (sorted)

    Returns
    -------
    np.ndarray
        Prefix sum array with P[0] = 0, P[i] = sum(A[0:i])
    """
    n = len(A)
    P = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        P[i] = P[i - 1] + A[i - 1]
    return P


def cost_factory(A: np.ndarray):
    """
    Create a cost function for a given array of widths.

    The cost of assigning batches [s, t] to the same quantized width is:
        length * max_width - sum_of_widths
    where max_width = A[t-1] (since A is sorted).

    Parameters
    ----------
    A : np.ndarray
        Sorted array of batch widths

    Returns
    -------
    callable
        Cost function cost(s, t) for 1-based inclusive indexing
    """
    P = compute_prefix(A)

    def cost(s: int, t: int) -> int:
        """
        Cost of assigning batches [s, t] to the same quantized width.

        Parameters
        ----------
        s : int
            Start index (1-based, inclusive)
        t : int
            End index (1-based, inclusive)

        Returns
        -------
        int
            Cost (total padding needed)
        """
        length = t - s + 1
        at = A[t - 1]  # Max width in this range (A is sorted)
        return length * at - (P[t] - P[s - 1])

    return cost


def dp_divide_conquer(A: np.ndarray, Q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    DP with divide & conquer optimization for batch quantization.

    Finds optimal partitioning of n batches into Q groups, where each group
    is assigned the maximum width in that group. Minimizes total padding cost.

    Parameters
    ----------
    A : np.ndarray
        Array of batch widths (will be sorted internally)
    Q : int
        Number of quantization groups

    Returns
    -------
    dp_table : np.ndarray
        DP table where dp_table[t] is the minimum cost for first t batches
    assignments : np.ndarray
        Quantized width assignments for each batch

    Notes
    -----
    Uses divide-and-conquer optimization exploiting monotone matrix
    search property, reducing complexity from O(n^2 * Q) to O(n * Q * log n).
    """
    # Sort A and keep track of original indices
    n = len(A)
    sort_idx = np.argsort(A)
    A_sorted = A[sort_idx].copy()

    cost = cost_factory(A_sorted)
    INF = 10**18

    # DP tables
    prev = np.full(n + 1, INF, dtype=np.int64)
    curr = np.full(n + 1, INF, dtype=np.int64)

    # Track split points for reconstruction
    splits = np.zeros((Q + 1, n + 1), dtype=np.int32)

    # Base case: k=1 (all batches in one group)
    for t in range(1, n + 1):
        prev[t] = cost(1, t)
        splits[1, t] = 1

    def compute_layer(k: int, t_lo: int, t_hi: int, opt_lo: int, opt_hi: int):
        """
        Recursively compute DP layer using divide & conquer.

        Parameters
        ----------
        k : int
            Current group count
        t_lo : int
            Lower bound of range to compute
        t_hi : int
            Upper bound of range to compute
        opt_lo : int
            Lower bound of optimal split search range
        opt_hi : int
            Upper bound of optimal split search range
        """
        if t_lo > t_hi:
            return

        mid = (t_lo + t_hi) // 2
        best_val = INF
        best_pos = -1

        # Search best split point s in [opt_lo, opt_hi]
        start = max(k, opt_lo)
        end = min(mid, opt_hi)

        for s in range(start, end + 1):
            val = prev[s - 1] + cost(s, mid)
            if val < best_val:
                best_val = val
                best_pos = s

        curr[mid] = best_val
        splits[k, mid] = best_pos

        if t_lo == t_hi:
            return

        # Recurse on left and right halves
        compute_layer(k, t_lo, mid - 1, opt_lo, best_pos)
        compute_layer(k, mid + 1, t_hi, best_pos, opt_hi)

    # Fill DP table for k = 2 to Q
    for k in range(2, Q + 1):
        compute_layer(k, k, n, k, n)
        prev, curr = curr, prev  # Swap for next layer

    # Reconstruct quantized assignments
    assignments_sorted = np.zeros(n, dtype=np.int64)

    # Backtrack to find group boundaries
    boundaries = []
    pos = n
    for k in range(Q, 0, -1):
        start_pos = splits[k, pos]
        boundaries.append((start_pos - 1, pos - 1))  # Convert to 0-based
        pos = start_pos - 1

    boundaries.reverse()

    # Assign quantized widths (max width in each group)
    for start, end in boundaries:
        quantized_width = A_sorted[end]  # Max in sorted array
        for i in range(start, end + 1):
            assignments_sorted[i] = quantized_width

    # Reorder assignments to match original order
    assignments = np.zeros(n, dtype=np.int64)
    for i, orig_idx in enumerate(sort_idx):
        assignments[orig_idx] = assignments_sorted[i]

    return prev, assignments


def quantize_batch_widths(widths: np.ndarray, n_groups: int) -> np.ndarray:
    """
    Quantize batch widths into n_groups distinct values.

    Parameters
    ----------
    widths : np.ndarray
        Array of batch widths to quantize
    n_groups : int
        Number of distinct quantized values

    Returns
    -------
    np.ndarray
        Quantized widths (same length as input)

    Examples
    --------
    >>> widths = np.array([100, 150, 200, 250, 300])
    >>> quantized = quantize_batch_widths(widths, n_groups=3)
    >>> # Result: [200, 200, 200, 300, 300] (example)
    """
    if n_groups >= len(widths):
        # No quantization needed
        return widths.copy()

    _, assignments = dp_divide_conquer(widths, n_groups)
    return assignments

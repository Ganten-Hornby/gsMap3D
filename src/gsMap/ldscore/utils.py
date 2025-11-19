import numpy as np
import numba


@numba.njit
def dynamic_programming_quantization(lengths: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Quantizes variable block lengths into a fixed set of 'n_bins' upper bounds
    to minimize JAX JIT recompilation.

    This is a simplified greedy implementation. Replace with full DP algo
    if strict optimal binning is required.
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
def get_block_limits(coords_ref: np.ndarray, coords_hm3: np.ndarray, window_size: int):
    """
    Calculates left and right indices in the Reference array for each HM3 SNP
    based on physical position (BP) and window size.
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
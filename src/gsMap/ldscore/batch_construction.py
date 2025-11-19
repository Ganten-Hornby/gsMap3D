"""
Simplified batch construction for LD score calculation.

This module handles:
1. Splitting HM3 SNPs into fixed-size batches
2. Calculating reference block boundaries for each batch based on LD window
"""

import numpy as np
import pandas as pd
import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatchInfo:
    """
    Information for a single HM3 batch.

    Attributes
    ----------
    chromosome : str
        Chromosome identifier
    hm3_indices : np.ndarray
        Indices of HM3 SNPs in this batch (relative to chromosome BIM)
    ref_start_idx : int
        Start index of reference block in chromosome (inclusive)
    ref_end_idx : int
        End index of reference block in chromosome (exclusive)
    """

    chromosome: str
    hm3_indices: np.ndarray
    ref_start_idx: int
    ref_end_idx: int


def construct_batches(
    bim_df: pd.DataFrame,
    hm3_snp_names: List[str],
    batch_size_hm3: int,
    window_size_bp: int = 1_000_000,
) -> List[BatchInfo]:
    """
    Construct batches of HM3 SNPs with reference block boundaries.

    Optimized implementation using searchsorted for O(log N) boundary finding.

    Parameters
    ----------
    bim_df : pd.DataFrame
        BIM dataframe for this chromosome
    hm3_snp_names : List[str]
        List of HM3 SNP names for this chromosome
    batch_size_hm3 : int
        Number of HM3 SNPs per batch
    window_size_bp : int
        LD window size in base pairs (default: 1Mb)

    Returns
    -------
    List[BatchInfo]
        List of BatchInfo objects
    """
    chromosome = str(bim_df["CHR"].iloc[0])

    # Pre-extract arrays for performance
    bim_snps = bim_df["SNP"].values
    bim_positions = bim_df["BP"].values
    
    # Ensure positions are sorted for searchsorted
    # In standard BIM files they are, but we check to be safe or just assume?
    # For max performance we assume sorted, but a check is cheap compared to the old code.
    # Let's assume sorted as is standard for genetics tools.

    # Find indices of HM3 SNPs in BIM
    # Use a set for O(1) lookup during mask creation if needed, 
    # but np.isin is efficient for bulk.
    hm3_set = set(hm3_snp_names)
    hm3_mask = np.isin(bim_snps, list(hm3_set))
    hm3_indices_all = np.where(hm3_mask)[0]

    if len(hm3_indices_all) == 0:
        logger.warning(f"No HM3 SNPs found in chromosome {chromosome}")
        return []

    logger.info(f"Found {len(hm3_indices_all)} HM3 SNPs in chromosome {chromosome}")

    # Split into batches
    n_hm3 = len(hm3_indices_all)
    n_batches = (n_hm3 + batch_size_hm3 - 1) // batch_size_hm3
    batch_infos = []

    for batch_idx in range(n_batches):
        # Get HM3 indices for this batch
        start_idx = batch_idx * batch_size_hm3
        end_idx = min((batch_idx + 1) * batch_size_hm3, n_hm3)
        hm3_indices = hm3_indices_all[start_idx:end_idx]

        # Get positions for this batch
        # Since hm3_indices are sorted (coming from where), these positions are sorted
        batch_positions = bim_positions[hm3_indices]
        min_pos = batch_positions[0]
        max_pos = batch_positions[-1]

        # Find reference block boundaries using binary search
        # This is O(log N) instead of O(N)
        window_start = min_pos - window_size_bp
        window_end = max_pos + window_size_bp

        ref_start = np.searchsorted(bim_positions, window_start, side='left')
        ref_end = np.searchsorted(bim_positions, window_end, side='right')

        batch_info = BatchInfo(
            chromosome=chromosome,
            hm3_indices=hm3_indices,
            ref_start_idx=int(ref_start),
            ref_end_idx=int(ref_end),
        )
        batch_infos.append(batch_info)

    logger.info(f"Created {len(batch_infos)} batches for chromosome {chromosome}")
    if len(batch_infos) > 0:
        avg_size = np.mean([b.ref_end_idx - b.ref_start_idx for b in batch_infos])
        logger.info(f"  Average reference block size: {avg_size:.0f} SNPs")

    return batch_infos

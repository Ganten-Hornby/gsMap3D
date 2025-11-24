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
    
    # Find indices of HM3 SNPs in BIM
    hm3_set = set(hm3_snp_names)
    # np.isin on strings is very slow, use pandas isin instead
    hm3_mask = bim_df["SNP"].isin(hm3_set).values
    hm3_indices_all = np.where(hm3_mask)[0]

    n_hm3 = len(hm3_indices_all)
    if n_hm3 == 0:
        logger.warning(f"No HM3 SNPs found in chromosome {chromosome}")
        return []
    if n_hm3 < len(hm3_snp_names):
        logger.warning(f"{len(hm3_snp_names) - n_hm3} HM3 SNPs not found in chromosome {chromosome} reference plink panel")

    logger.info(f"Found {n_hm3} HM3 SNPs in chromosome {chromosome}")

    # Calculate batch boundaries
    # We want batches of size batch_size_hm3
    # Start indices: 0, batch_size, 2*batch_size, ...
    batch_starts = np.arange(0, n_hm3, batch_size_hm3)
    # End indices: batch_size, 2*batch_size, ..., n_hm3
    batch_ends = np.minimum(batch_starts + batch_size_hm3, n_hm3)
    
    n_batches = len(batch_starts)
    
    # Get indices in BIM for start and end of each batch
    # hm3_indices_all contains the BIM indices of HM3 SNPs
    # We need the BIM index of the first and last HM3 SNP in each batch
    
    # First HM3 SNP in each batch
    batch_start_bim_indices = hm3_indices_all[batch_starts]
    # Last HM3 SNP in each batch (indices are exclusive in slice, so -1 for element access)
    batch_end_bim_indices = hm3_indices_all[batch_ends - 1]
    
    # Get positions
    min_pos = bim_positions[batch_start_bim_indices]
    max_pos = bim_positions[batch_end_bim_indices]
    
    # Calculate window boundaries
    window_starts = min_pos - window_size_bp
    window_ends = max_pos + window_size_bp
    
    # Vectorized searchsorted
    # Find insertion points for all window starts and ends at once
    ref_starts = np.searchsorted(bim_positions, window_starts, side='left')
    ref_ends = np.searchsorted(bim_positions, window_ends, side='right')
    
    # Construct BatchInfo objects
    batch_infos = []
    for i in range(n_batches):
        # Slice the HM3 indices for this batch
        # This is the only part that might still be a bit slow if n_batches is huge,
        # but it's much faster than the searchsorted loop.
        b_hm3_indices = hm3_indices_all[batch_starts[i]:batch_ends[i]]
        
        batch_infos.append(BatchInfo(
            chromosome=chromosome,
            hm3_indices=b_hm3_indices,
            ref_start_idx=int(ref_starts[i]),
            ref_end_idx=int(ref_ends[i]),
        ))

    logger.info(f"Created {len(batch_infos)} batches for chromosome {chromosome}")
    if len(batch_infos) > 0:
        avg_size = np.mean(ref_ends - ref_starts)
        logger.info(f"  Average reference block size: {avg_size:.0f} SNPs")

    return batch_infos

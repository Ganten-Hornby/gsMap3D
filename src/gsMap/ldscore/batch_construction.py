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


def find_ld_window_boundaries(
    bim_df: pd.DataFrame,
    hm3_indices: np.ndarray,
    window_size_bp: int = 1_000_000,
) -> tuple[int, int]:
    """
    Find reference block boundaries for a batch of HM3 SNPs.

    Parameters
    ----------
    bim_df : pd.DataFrame
        BIM dataframe with columns: CHR, SNP, CM, BP, A1, A2
    hm3_indices : np.ndarray
        Indices of HM3 SNPs in this batch
    window_size_bp : int
        LD window size in base pairs (default: 1Mb)

    Returns
    -------
    ref_start : int
        Start index of reference block (inclusive)
    ref_end : int
        End index of reference block (exclusive)
    """
    # Get min and max positions of HM3 SNPs in this batch
    hm3_positions = bim_df.iloc[hm3_indices]["BP"].values
    min_pos = hm3_positions.min()
    max_pos = hm3_positions.max()

    # Expand window on both sides
    window_start = min_pos - window_size_bp
    window_end = max_pos + window_size_bp

    # Find all SNPs within window
    all_positions = bim_df["BP"].values
    in_window = (all_positions >= window_start) & (all_positions <= window_end)

    # Get start and end indices
    indices_in_window = np.where(in_window)[0]

    if len(indices_in_window) == 0:
        # Fallback: use HM3 indices only
        ref_start = hm3_indices.min()
        ref_end = hm3_indices.max() + 1
    else:
        ref_start = indices_in_window[0]
        ref_end = indices_in_window[-1] + 1

    return int(ref_start), int(ref_end)


def construct_batches(
    bim_df: pd.DataFrame,
    hm3_snp_names: List[str],
    batch_size_hm3: int,
    window_size_bp: int = 1_000_000,
) -> List[BatchInfo]:
    """
    Construct batches of HM3 SNPs with reference block boundaries.

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

    # Find indices of HM3 SNPs in BIM
    hm3_set = set(hm3_snp_names)
    bim_snps = bim_df["SNP"].values
    hm3_mask = np.isin(bim_snps, list(hm3_set))
    hm3_indices_all = np.where(hm3_mask)[0]

    if len(hm3_indices_all) == 0:
        logger.warning(f"No HM3 SNPs found in chromosome {chromosome}")
        return []

    logger.info(f"Found {len(hm3_indices_all)} HM3 SNPs in chromosome {chromosome}")

    # Split into batches
    n_batches = (len(hm3_indices_all) + batch_size_hm3 - 1) // batch_size_hm3
    batch_infos = []

    for batch_idx in range(n_batches):
        # Get HM3 indices for this batch
        start_idx = batch_idx * batch_size_hm3
        end_idx = min((batch_idx + 1) * batch_size_hm3, len(hm3_indices_all))
        hm3_indices = hm3_indices_all[start_idx:end_idx]

        # Find reference block boundaries
        ref_start, ref_end = find_ld_window_boundaries(
            bim_df, hm3_indices, window_size_bp
        )

        batch_info = BatchInfo(
            chromosome=chromosome,
            hm3_indices=hm3_indices,
            ref_start_idx=ref_start,
            ref_end_idx=ref_end,
        )
        batch_infos.append(batch_info)

    logger.info(f"Created {len(batch_infos)} batches for chromosome {chromosome}")
    logger.info(f"  Average reference block size: {np.mean([b.ref_end_idx - b.ref_start_idx for b in batch_infos]):.0f} SNPs")

    return batch_infos

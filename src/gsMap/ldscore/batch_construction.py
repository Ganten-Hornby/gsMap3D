"""
Simplified batch construction for LD score calculation.

This module handles:
1. Splitting HM3 SNPs into fixed-size batches
2. Calculating reference block boundaries for each batch based on LD window
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    hm3_snp_names: list[str],
    batch_size_hm3: int,
    ld_wind: float = 1.0,
    ld_unit: str = "CM",
) -> list[BatchInfo]:
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
    ld_wind : float
        LD window size (default: 1.0)
    ld_unit : str
        Unit for LD window: 'SNP', 'KB', 'CM' (default: 'CM')

    Returns
    -------
    List[BatchInfo]
        List of BatchInfo objects
    """
    chromosome = str(bim_df["CHR"].iloc[0])

    # Pre-extract arrays for performance
    _snp_values = bim_df["SNP"].values

    # helper to get coordinates based on unit
    if ld_unit == "SNP":
        coords = np.arange(len(bim_df))
        max_dist = int(ld_wind)
    elif ld_unit == "KB":
        coords = bim_df["BP"].values
        max_dist = ld_wind * 1000
    elif ld_unit == "CM":
        coords = bim_df["CM"].values
        max_dist = ld_wind
        # Fallback if CM is all zero
        if np.all(coords == 0):
            logger.warning(
                f"All CM values are 0 for chromosome {chromosome}. Fallback to 1MB window (BP)."
            )
            coords = bim_df["BP"].values
            max_dist = 1_000_000
    else:
        raise ValueError(f"Invalid ld_unit: {ld_unit}. Must be 'SNP', 'KB', or 'CM'.")

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
        logger.warning(
            f"{len(hm3_snp_names) - n_hm3} HM3 SNPs not found in chromosome {chromosome} reference plink panel"
        )

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

    # Get coordinates for these batch boundaries
    min_coords = coords[batch_start_bim_indices]
    max_coords = coords[batch_end_bim_indices]

    # Calculate window boundaries
    window_starts = min_coords - max_dist
    window_ends = max_coords + max_dist

    if ld_unit == "SNP":
        # For SNP unit, coordinates are indices, so we just clip
        ref_starts = np.maximum(window_starts, 0).astype(int)
        ref_ends = np.minimum(window_ends, len(coords)).astype(int)
    else:
        # Vectorized searchsorted
        # Find insertion points for all window starts and ends at once
        ref_starts = np.searchsorted(coords, window_starts, side="left")
        ref_ends = np.searchsorted(coords, window_ends, side="right")

    # Construct BatchInfo objects
    batch_infos = []
    for i in range(n_batches):
        # Slice the HM3 indices for this batch
        b_hm3_indices = hm3_indices_all[batch_starts[i] : batch_ends[i]]

        batch_infos.append(
            BatchInfo(
                chromosome=chromosome,
                hm3_indices=b_hm3_indices,
                ref_start_idx=int(ref_starts[i]),
                ref_end_idx=int(ref_ends[i]),
            )
        )

    logger.info(f"Created {len(batch_infos)} batches for chromosome {chromosome}")
    if len(batch_infos) > 0:
        avg_size = np.mean(ref_ends - ref_starts)
        logger.info(f"  Average reference block size: {avg_size:.0f} SNPs")

    return batch_infos

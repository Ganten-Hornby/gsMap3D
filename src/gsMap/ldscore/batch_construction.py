"""
Batch construction with DP-based quantization for LD score calculation.

This module handles:
1. Constructing HM3 SNP batches per chromosome
2. Calculating reference block boundaries (sb_start_global, sb_end_global)
3. DP-based width quantization
4. Padding batches to match quantized widths
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List
from dataclasses import dataclass

from .dp_quantization import quantize_batch_widths

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
    sb_start_global : int
        Start index of reference block in chromosome (inclusive)
    sb_end_global : int
        End index of reference block in chromosome (exclusive)
    current_width : int
        Original width of reference block
    quantized_width : int
        Quantized width after DP optimization
    """

    chromosome: str
    hm3_indices: np.ndarray
    sb_start_global: int
    sb_end_global: int
    current_width: int
    quantized_width: int


def find_ld_window_boundaries(
    bim_df: pd.DataFrame,
    hm3_indices: np.ndarray,
    window_size_bp: int = 1_000_000,
) -> Tuple[int, int]:
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
    sb_start : int
        Start index of reference block (inclusive)
    sb_end : int
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
        sb_start = hm3_indices.min()
        sb_end = hm3_indices.max() + 1
    else:
        sb_start = indices_in_window[0]
        sb_end = indices_in_window[-1] + 1

    return int(sb_start), int(sb_end)


def construct_chromosome_batches(
    bim_df: pd.DataFrame,
    hm3_snp_names: List[str],
    batch_size_hm3: int,
    window_size_bp: int = 1_000_000,
) -> List[Tuple[np.ndarray, int, int]]:
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
        LD window size in base pairs

    Returns
    -------
    List[Tuple[np.ndarray, int, int]]
        List of (hm3_indices, sb_start_global, sb_end_global) tuples
    """
    # Find indices of HM3 SNPs in BIM
    hm3_set = set(hm3_snp_names)
    bim_snps = bim_df["SNP"].values
    hm3_mask = np.isin(bim_snps, list(hm3_set))
    hm3_indices_in_bim = np.where(hm3_mask)[0]

    if len(hm3_indices_in_bim) == 0:
        logger.warning(f"No HM3 SNPs found in chromosome {bim_df['CHR'].iloc[0]}")
        return []

    logger.info(
        f"Found {len(hm3_indices_in_bim)} HM3 SNPs in chromosome {bim_df['CHR'].iloc[0]}"
    )

    # Split into batches
    n_batches = int(np.ceil(len(hm3_indices_in_bim) / batch_size_hm3))
    batches = []

    for i in range(n_batches):
        start_idx = i * batch_size_hm3
        end_idx = min((i + 1) * batch_size_hm3, len(hm3_indices_in_bim))

        batch_hm3_indices = hm3_indices_in_bim[start_idx:end_idx]

        # Find reference block boundaries
        sb_start, sb_end = find_ld_window_boundaries(
            bim_df, batch_hm3_indices, window_size_bp
        )

        batches.append((batch_hm3_indices, sb_start, sb_end))

    logger.info(f"Created {n_batches} batches for chromosome {bim_df['CHR'].iloc[0]}")

    return batches


def apply_quantization_and_padding(
    batches: List[Tuple[np.ndarray, int, int]],
    n_groups: int,
    chromosome: str,
) -> List[BatchInfo]:
    """
    Apply DP-based quantization and padding to batches.

    Parameters
    ----------
    batches : List[Tuple[np.ndarray, int, int]]
        List of (hm3_indices, sb_start, sb_end) tuples
    n_groups : int
        Number of quantization groups (Q)
    chromosome : str
        Chromosome identifier

    Returns
    -------
    List[BatchInfo]
        List of BatchInfo objects with quantized and padded boundaries
    """
    if len(batches) == 0:
        return []

    # Extract current widths
    widths = np.array([sb_end - sb_start for _, sb_start, sb_end in batches])

    # Apply DP quantization
    quantized_widths = quantize_batch_widths(widths, n_groups)

    logger.info(
        f"Chromosome {chromosome}: Quantized {len(widths)} batches into {n_groups} groups"
    )
    logger.info(f"  Original widths: min={widths.min()}, max={widths.max()}, mean={widths.mean():.1f}")
    logger.info(
        f"  Quantized widths: unique={len(np.unique(quantized_widths))}, values={np.unique(quantized_widths)}"
    )

    # Create BatchInfo objects with padding
    batch_infos = []

    for i, (hm3_indices, sb_start, sb_end) in enumerate(batches):
        current_width = sb_end - sb_start
        quantized_width = int(quantized_widths[i])

        # Calculate padding needed
        padding_needed = quantized_width - current_width

        if padding_needed < 0:
            logger.error(
                f"Quantized width ({quantized_width}) < current width ({current_width})"
            )
            padding_needed = 0

        # Distribute padding equally on both sides
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left

        # Apply padding
        sb_start_padded = max(0, sb_start - pad_left)
        sb_end_padded = sb_end + pad_right

        batch_info = BatchInfo(
            chromosome=chromosome,
            hm3_indices=hm3_indices,
            sb_start_global=sb_start_padded,
            sb_end_global=sb_end_padded,
            current_width=current_width,
            quantized_width=quantized_width,
        )

        batch_infos.append(batch_info)

    return batch_infos


def create_batch_matrix(batch_infos: List[BatchInfo]) -> np.ndarray:
    """
    Create W matrix representation of batches.

    Parameters
    ----------
    batch_infos : List[BatchInfo]
        List of BatchInfo objects

    Returns
    -------
    np.ndarray
        Matrix of shape (n_batches, 4) with columns:
        [sb_start_global, sb_end_global, current_width, quantized_width]
    """
    n_batches = len(batch_infos)
    W = np.zeros((n_batches, 4), dtype=np.int64)

    for i, batch_info in enumerate(batch_infos):
        W[i, 0] = batch_info.sb_start_global
        W[i, 1] = batch_info.sb_end_global
        W[i, 2] = batch_info.current_width
        W[i, 3] = batch_info.quantized_width

    return W


def construct_all_batches(
    bim_df: pd.DataFrame,
    hm3_snp_names: List[str],
    batch_size_hm3: int,
    n_quantization_groups: int,
    window_size_bp: int = 1_000_000,
) -> Tuple[List[BatchInfo], np.ndarray]:
    """
    Complete pipeline: construct, quantize, and pad batches for a chromosome.

    Parameters
    ----------
    bim_df : pd.DataFrame
        BIM dataframe for this chromosome
    hm3_snp_names : List[str]
        List of HM3 SNP names for this chromosome
    batch_size_hm3 : int
        Number of HM3 SNPs per batch
    n_quantization_groups : int
        Number of quantization groups (Q)
    window_size_bp : int
        LD window size in base pairs

    Returns
    -------
    batch_infos : List[BatchInfo]
        List of BatchInfo objects
    W : np.ndarray
        Batch matrix (n_batches, 4)
    """
    chromosome = str(bim_df["CHR"].iloc[0])

    # Step 1: Construct batches with reference block boundaries
    batches = construct_chromosome_batches(
        bim_df, hm3_snp_names, batch_size_hm3, window_size_bp
    )

    if len(batches) == 0:
        return [], np.array([])

    # Step 2: Apply quantization and padding
    batch_infos = apply_quantization_and_padding(batches, n_quantization_groups, chromosome)

    # Step 3: Create W matrix
    W = create_batch_matrix(batch_infos)

    logger.info(f"Chromosome {chromosome}: Created {len(batch_infos)} batches")
    logger.info(f"  W matrix shape: {W.shape}")

    return batch_infos, W

"""
Chromosome-wise pipeline for LD score calculation with DP-based batching.

This module orchestrates the complete workflow:
1. Loop through chromosomes
2. Construct batches with DP quantization
3. Load genotypes once per chromosome
4. Process each batch to compute LD scores/weights
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .io import PlinkBEDReader
from .batch_construction import construct_all_batches, BatchInfo
from .compute import compute_batch_weights_numpy, compute_batch_ld_scores_numpy
from .mapping import create_snp_feature_map

logger = logging.getLogger(__name__)


@dataclass
class ChromosomeResult:
    """
    Results for a single chromosome.

    Attributes
    ----------
    chromosome : str
        Chromosome identifier
    hm3_snp_names : List[str]
        Names of HM3 SNPs processed
    ld_scores : np.ndarray
        LD scores for each HM3 SNP, shape (n_hm3_snps,)
    weights : Optional[np.ndarray]
        Weight matrix for features, shape (n_hm3_snps, n_features)
    """

    chromosome: str
    hm3_snp_names: List[str]
    ld_scores: np.ndarray
    weights: Optional[np.ndarray] = None


class LDScorePipeline:
    """
    Pipeline for computing LD scores across chromosomes with DP-based batching.

    Parameters
    ----------
    bfile_prefix_template : str
        Template for PLINK file prefix, e.g., "1000G.EUR.QC.{chr}"
    hm3_dir : str
        Directory containing HM3 SNP lists per chromosome
    batch_size_hm3 : int
        Number of HM3 SNPs per batch
    n_quantization_groups : int
        Number of quantization groups (Q) for DP
    window_size_bp : int
        LD window size in base pairs (default: 1Mb)
    maf_min : float
        Minimum MAF threshold for SNP filtering (default: 0.01)
    chromosomes : Optional[List[int]]
        List of chromosomes to process (default: 1-22)
    """

    def __init__(
        self,
        bfile_prefix_template: str,
        hm3_dir: str,
        batch_size_hm3: int,
        n_quantization_groups: int,
        window_size_bp: int = 1_000_000,
        maf_min: float = 0.01,
        chromosomes: Optional[List[int]] = None,
    ):
        self.bfile_prefix_template = bfile_prefix_template
        self.hm3_dir = Path(hm3_dir)
        self.batch_size_hm3 = batch_size_hm3
        self.n_quantization_groups = n_quantization_groups
        self.window_size_bp = window_size_bp
        self.maf_min = maf_min
        self.chromosomes = chromosomes or list(range(1, 23))

        logger.info("=" * 80)
        logger.info("LD Score Pipeline Configuration")
        logger.info("=" * 80)
        logger.info(f"PLINK template: {bfile_prefix_template}")
        logger.info(f"HM3 directory: {hm3_dir}")
        logger.info(f"Batch size (HM3): {batch_size_hm3}")
        logger.info(f"Quantization groups: {n_quantization_groups}")
        logger.info(f"LD window: {window_size_bp:,} bp")
        logger.info(f"MAF filter: {maf_min}")
        logger.info(f"Chromosomes: {self.chromosomes}")
        logger.info("=" * 80)

    def load_hm3_snps(self, chromosome: int) -> List[str]:
        """
        Load HM3 SNP list for a chromosome.

        Parameters
        ----------
        chromosome : int
            Chromosome number

        Returns
        -------
        List[str]
            List of HM3 SNP names
        """
        # Try common file naming patterns
        possible_paths = [
            self.hm3_dir / f"hm.{chromosome}.snp",
            self.hm3_dir / f"hm3_snps.chr{chromosome}.txt",
            self.hm3_dir / f"hapmap3_snps.chr{chromosome}.txt",
            self.hm3_dir / f"chr{chromosome}.snplist",
            self.hm3_dir / f"w_hm3.snplist.chr{chromosome}",
        ]

        for path in possible_paths:
            if path.exists():
                # Try reading with different formats
                try:
                    snps = pd.read_csv(path, header=None, names=["SNP"])["SNP"].tolist()
                    logger.info(f"Loaded {len(snps)} HM3 SNPs from {path}")
                    return snps
                except:
                    # Try with tab separator
                    try:
                        snps = pd.read_csv(path, sep="\t", header=None, names=["SNP"])["SNP"].tolist()
                        logger.info(f"Loaded {len(snps)} HM3 SNPs from {path}")
                        return snps
                    except:
                        continue

        # If no file found, log warning and return empty
        logger.warning(f"No HM3 SNP file found for chromosome {chromosome}")
        return []

    def process_chromosome(
        self,
        chromosome: int,
        feature_mapping: Optional[pd.DataFrame] = None,
    ) -> Optional[ChromosomeResult]:
        """
        Process a single chromosome.

        Parameters
        ----------
        chromosome : int
            Chromosome number
        feature_mapping : Optional[pd.DataFrame]
            SNP-feature mapping (from create_snp_gene_map)

        Returns
        -------
        Optional[ChromosomeResult]
            Results for this chromosome, or None if no HM3 SNPs found
        """
        logger.info("=" * 80)
        logger.info(f"Processing Chromosome {chromosome}")
        logger.info("=" * 80)

        # Load HM3 SNPs
        hm3_snps = self.load_hm3_snps(chromosome)
        if len(hm3_snps) == 0:
            logger.warning(f"No HM3 SNPs for chromosome {chromosome}, skipping")
            return None

        # Load PLINK data
        bfile_prefix = self.bfile_prefix_template.format(chr=chromosome)
        logger.info(f"Loading PLINK data from: {bfile_prefix}")

        try:
            reader = PlinkBEDReader(
                bfile_prefix,
                maf_min=self.maf_min,
                preload=True,  # Load genotypes into memory
            )
        except FileNotFoundError as e:
            logger.error(f"PLINK files not found for chromosome {chromosome}: {e}")
            return None

        # Construct batches with DP quantization
        logger.info(f"Constructing batches with DP quantization...")
        batch_infos, W = construct_all_batches(
            bim_df=reader.bim,
            hm3_snp_names=hm3_snps,
            batch_size_hm3=self.batch_size_hm3,
            n_quantization_groups=self.n_quantization_groups,
            window_size_bp=self.window_size_bp,
        )

        if len(batch_infos) == 0:
            logger.warning(f"No batches created for chromosome {chromosome}")
            return None

        logger.info(f"Created {len(batch_infos)} batches")
        logger.info(f"W matrix:\n{W}")

        # Process each batch
        all_ld_scores = []
        all_weights = [] if feature_mapping is not None else None
        all_hm3_snp_names = []

        for i, batch_info in enumerate(batch_infos):
            logger.info(f"\nProcessing batch {i+1}/{len(batch_infos)}")
            logger.info(f"  HM3 SNPs: {len(batch_info.hm3_indices)}")
            logger.info(
                f"  Ref block: [{batch_info.sb_start_global}, {batch_info.sb_end_global})"
            )
            logger.info(f"  Width: {batch_info.current_width} -> {batch_info.quantized_width}")

            # Fetch genotypes using padded boundaries
            X_hm3 = reader.genotypes[:, batch_info.hm3_indices]
            ref_indices = np.arange(
                batch_info.sb_start_global, batch_info.sb_end_global
            )
            # Clip to valid range
            ref_indices = ref_indices[ref_indices < reader.m]
            X_ref_block = reader.genotypes[:, ref_indices]

            logger.info(f"  X_hm3 shape: {X_hm3.shape}")
            logger.info(f"  X_ref_block shape: {X_ref_block.shape}")

            # Compute LD scores
            ld_scores = compute_batch_ld_scores_numpy(X_hm3, X_ref_block)
            all_ld_scores.append(ld_scores)

            # Get HM3 SNP names for this batch
            batch_snp_names = reader.bim.iloc[batch_info.hm3_indices]["SNP"].tolist()
            all_hm3_snp_names.extend(batch_snp_names)

            # Compute weights if feature mapping provided
            if feature_mapping is not None:
                # Get ref SNP names
                ref_snp_names = reader.bim.iloc[ref_indices]["SNP"].tolist()

                # Create feature mask
                ref_snp_set = set(ref_snp_names)
                feature_mask = (
                    feature_mapping[feature_mapping["SNP"].isin(ref_snp_set)]
                    .groupby(["SNP", "feature_idx"])
                    .size()
                    .reset_index(name="count")
                )

                # Convert to binary mask matrix
                n_features = feature_mapping["feature_idx"].max() + 1
                feature_mask_matrix = np.zeros(
                    (len(ref_snp_names), n_features), dtype=np.float32
                )

                for _, row in feature_mask.iterrows():
                    snp_name = row["SNP"]
                    if snp_name in ref_snp_set:
                        snp_idx = ref_snp_names.index(snp_name)
                        feature_mask_matrix[snp_idx, int(row["feature_idx"])] = 1.0

                # Compute weights
                weights = compute_batch_weights_numpy(
                    X_hm3, X_ref_block, feature_mask_matrix
                )
                all_weights.append(weights)

                logger.info(f"  LD scores: {ld_scores.shape}")
                logger.info(f"  Weights: {weights.shape}")

        # Combine results
        ld_scores_combined = np.concatenate(all_ld_scores)
        weights_combined = (
            np.vstack(all_weights) if all_weights is not None else None
        )

        logger.info("=" * 80)
        logger.info(f"Chromosome {chromosome} complete")
        logger.info(f"  Total HM3 SNPs processed: {len(all_hm3_snp_names)}")
        logger.info(f"  LD scores shape: {ld_scores_combined.shape}")
        if weights_combined is not None:
            logger.info(f"  Weights shape: {weights_combined.shape}")
        logger.info("=" * 80)

        return ChromosomeResult(
            chromosome=str(chromosome),
            hm3_snp_names=all_hm3_snp_names,
            ld_scores=ld_scores_combined,
            weights=weights_combined,
        )

    def run(
        self,
        feature_annotation_file: Optional[str] = None,
        mapping_strategy: str = "TSS",
        window_size_mapping: int = 100_000,
    ) -> Dict[str, ChromosomeResult]:
        """
        Run pipeline across all chromosomes.

        Parameters
        ----------
        feature_annotation_file : Optional[str]
            Path to feature annotation file (e.g., gene annotation CSV)
        mapping_strategy : str
            SNP-gene mapping strategy ("TSS" or "gene_body")
        window_size_mapping : int
            Window size for TSS mapping (bp)

        Returns
        -------
        Dict[str, ChromosomeResult]
            Results for each chromosome
        """
        logger.info("\n" + "=" * 80)
        logger.info("Starting LD Score Pipeline")
        logger.info("=" * 80 + "\n")

        # Create feature mapping if annotation file provided
        feature_mapping = None
        if feature_annotation_file is not None:
            logger.info("Creating SNP-feature mapping...")
            # This will be done per chromosome in process_chromosome
            pass

        # Process each chromosome
        results = {}

        for chrom in self.chromosomes:
            result = self.process_chromosome(chrom, feature_mapping)
            if result is not None:
                results[str(chrom)] = result

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Complete")
        logger.info("=" * 80)
        logger.info(f"Processed {len(results)} chromosomes")
        total_snps = sum(len(r.hm3_snp_names) for r in results.values())
        logger.info(f"Total HM3 SNPs: {total_snps}")
        logger.info("=" * 80 + "\n")

        return results


def save_results(
    results: Dict[str, ChromosomeResult],
    output_prefix: str,
    save_weights: bool = True,
):
    """
    Save pipeline results to files.

    Parameters
    ----------
    results : Dict[str, ChromosomeResult]
        Results from pipeline
    output_prefix : str
        Output file prefix
    save_weights : bool
        Whether to save weight matrices
    """
    logger.info(f"Saving results to {output_prefix}.*")

    # Combine all chromosomes
    all_data = []

    for chrom, result in results.items():
        for i, snp_name in enumerate(result.hm3_snp_names):
            row = {"CHR": chrom, "SNP": snp_name, "L2": result.ld_scores[i]}
            all_data.append(row)

    # Save LD scores
    df_ldscore = pd.DataFrame(all_data)
    ldscore_file = f"{output_prefix}.l2.ldscore.gz"
    df_ldscore.to_csv(ldscore_file, sep="\t", index=False, compression="gzip")
    logger.info(f"  Saved LD scores: {ldscore_file}")

    # Save weights if available
    if save_weights and any(r.weights is not None for r in results.values()):
        # Combine weights
        all_weights = []
        for result in results.values():
            if result.weights is not None:
                all_weights.append(result.weights)

        if len(all_weights) > 0:
            weights_combined = np.vstack(all_weights)
            weights_file = f"{output_prefix}.weights.npy"
            np.save(weights_file, weights_combined)
            logger.info(f"  Saved weights: {weights_file} (shape: {weights_combined.shape})")

    logger.info("Results saved successfully")

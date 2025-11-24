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
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental import sparse
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

from .io import PlinkBEDReader
from .batch_construction import construct_batches, BatchInfo
from .compute import (
    compute_ld_scores,
    compute_batch_weights_segment_sum,
)
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
    weights : sparse.BCOO
        Sparse weight matrix for features, shape (n_hm3_snps, n_features)
        BCOO format: Batched Coordinate sparse matrix
        Contains feature-stratified LD weights
    n_features : int
        Total number of feature indices (including unmapped feature index)
    ld_scores : Optional[jnp.ndarray]
        LD scores for each HM3 SNP (optional, not computed in weight-only mode)
        Can be derived from weights if needed: ld_scores = weights.sum(axis=1)
    """

    chromosome: str
    hm3_snp_names: List[str]
    weights: sparse.BCOO
    n_features: int
    ld_scores: Optional[jnp.ndarray] = None


class LDScorePipeline:
    """
    Pipeline for computing LD scores across chromosomes.

    Parameters
    ----------
    bfile_prefix_template : str
        Template for PLINK file prefix, e.g., "1000G.EUR.QC.{chr}"
    hm3_dir : str
        Directory containing HM3 SNP lists per chromosome
    batch_size_hm3 : int
        Number of HM3 SNPs per batch
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
        window_size_bp: int = 1_000_000,
        maf_min: float = 0.01,
        chromosomes: Optional[List[int]] = None,
    ):
        self.bfile_prefix_template = bfile_prefix_template
        self.hm3_dir = Path(hm3_dir)
        self.batch_size_hm3 = batch_size_hm3
        self.window_size_bp = window_size_bp
        self.maf_min = maf_min
        self.chromosomes = chromosomes or list(range(1, 23))

        logger.info("=" * 80)
        logger.info("LD Score Pipeline Configuration")
        logger.info("=" * 80)
        logger.info(f"PLINK template: {bfile_prefix_template}")
        logger.info(f"HM3 directory: {hm3_dir}")
        logger.info(f"Batch size (HM3): {batch_size_hm3}")
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

    def _create_bcoo_from_batches(
        self,
        batch_weight_data: List[Dict],
        n_hm3_total: int,
        n_features: int,
    ) -> sparse.BCOO:
        """
        Create BCOO sparse matrix from batch weight data.

        Parameters
        ----------
        batch_weight_data : List[Dict]
            List of batch results, each containing:
            - weights: jnp.ndarray of shape (batch_hm3, n_unique_features)
            - unique_features: jnp.ndarray of feature indices
            - hm3_start_idx: int, starting HM3 index for this batch
            - n_hm3: int, number of HM3 SNPs in this batch
        n_hm3_total : int
            Total number of HM3 SNPs in chromosome
        n_features : int
            Total number of features (F+1, including unmapped)

        Returns
        -------
        sparse.BCOO
            Sparse weight matrix of shape (n_hm3_total, n_features)
        """
        # Collect all non-zero entries
        all_indices = []
        all_values = []

        for batch_data in batch_weight_data:
            weights = batch_data['weights']  # JAX array (batch_hm3, n_unique_features)
            unique_features = batch_data['unique_features']  # JAX array (n_unique_features,)
            hm3_start = batch_data['hm3_start_idx']
            n_hm3 = batch_data['n_hm3']

            # Convert to numpy for indexing
            weights_np = np.array(weights)
            unique_features_np = np.array(unique_features)

            # Find non-zero entries in this batch
            for i in range(n_hm3):
                for j, feature_idx in enumerate(unique_features_np):
                    value = weights_np[i, j]
                    if value != 0:  # Only store non-zero values
                        global_hm3_idx = hm3_start + i
                        all_indices.append([global_hm3_idx, int(feature_idx)])
                        all_values.append(value)

        # Create BCOO sparse matrix directly with JAX
        if len(all_indices) == 0:
            # No non-zero entries
            indices = jnp.zeros((0, 2), dtype=jnp.int32)
            values = jnp.zeros(0, dtype=jnp.float32)
        else:
            indices = jnp.array(all_indices, dtype=jnp.int32)
            values = jnp.array(all_values, dtype=jnp.float32)

        # Create BCOO sparse matrix
        # BCOO format: indices shape (nnz, ndim), data shape (nnz,)
        bcoo = sparse.BCOO(
            (values, indices),
            shape=(n_hm3_total, n_features)
        )

        logger.info(f"Created BCOO matrix: shape={bcoo.shape}, nnz={bcoo.nse}")
        return bcoo

    def process_chromosome(
        self,
        chromosome: int,
        mapping_type: str,
        mapping_data: Union[pd.DataFrame, Dict[str, str]],
        window_size_mapping: int = 0,
        mapping_strategy: str = "score",
    ) -> Optional[ChromosomeResult]:
        """
        Process a single chromosome to compute feature-stratified LD weights.

        Parameters
        ----------
        chromosome : int
            Chromosome number
        mapping_type : str
            Type of feature mapping: "bed" for genomic intervals or "dict" for direct mapping
        mapping_data : Union[pd.DataFrame, Dict[str, str]]
            Feature mapping data (REQUIRED):
            - For "bed": DataFrame with [Feature, Chrom, Start, End, Score, Strand]
            - For "dict": Dictionary {rsid: feature_name}
        window_size_mapping : int
            Window size for spatial mapping (bp), default: 0
        mapping_strategy : str
            Mapping strategy: "score" (max score) or "tss" (closest TSS), default: "score"

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

        # Load PLINK data once
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

        # Create SNP-to-feature mapping (REQUIRED - always created)
        logger.info(f"Creating SNP-feature mapping for chromosome {chromosome}...")

        mapping_vec_np, n_mapped_features = create_snp_feature_map(
            bim_df=reader.bim,
            mapping_type=mapping_type,
            mapping_data=mapping_data,
            window_size=window_size_mapping,
            strategy=mapping_strategy,
        )

        # Convert mapping_vec to JAX array
        mapping_vec = jnp.array(mapping_vec_np, dtype=jnp.int32)

        # Total number of feature indices including unmapped
        # Mapped features: indices 0 to (n_mapped_features - 1)
        # Unmapped SNPs: index n_mapped_features
        n_feature_indices = n_mapped_features + 1

        logger.info(f"  Mapped features: {n_mapped_features}")
        logger.info(f"  Total feature indices (including unmapped): {n_feature_indices}")
        logger.info(f"  Mapping created for {len(reader.bim)} SNPs")

        # Construct batches
        logger.info(f"Constructing batches...")
        batch_infos = construct_batches(
            bim_df=reader.bim,
            hm3_snp_names=hm3_snps,
            batch_size_hm3=self.batch_size_hm3,
            window_size_bp=self.window_size_bp,
        )

        if len(batch_infos) == 0:
            logger.warning(f"No batches created for chromosome {chromosome}")
            return None

        logger.info(f"Created {len(batch_infos)} batches")

        # Process each batch to compute weights
        batch_weight_data = []
        all_hm3_snp_names = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Chr {chromosome} ({len(hm3_snps):,} HM3 SNPs)", total=len(hm3_snps)
            )

            for i, batch_info in enumerate(batch_infos):
                logger.info(f"\nProcessing batch {i+1}/{len(batch_infos)}")
                logger.info(f"  HM3 SNPs: {len(batch_info.hm3_indices)}")
                logger.info(
                    f"  Ref block: [{batch_info.ref_start_idx}, {batch_info.ref_end_idx})"
                )
                ref_width = batch_info.ref_end_idx - batch_info.ref_start_idx
                logger.info(f"  Ref block width: {ref_width} SNPs")

                # Fetch genotypes
                X_hm3 = reader.genotypes[:, batch_info.hm3_indices]
                ref_indices = np.arange(
                    batch_info.ref_start_idx, batch_info.ref_end_idx
                )
                # Clip to valid range
                ref_indices = ref_indices[ref_indices < reader.m]
                X_ref_block = reader.genotypes[:, ref_indices]

                logger.info(f"  X_hm3 shape: {X_hm3.shape}")
                logger.info(f"  X_ref_block shape: {X_ref_block.shape}")

                # Get HM3 SNP names for this batch
                batch_snp_names = reader.bim.iloc[batch_info.hm3_indices]["SNP"].tolist()
                all_hm3_snp_names.extend(batch_snp_names)

                # Get feature indices for reference SNPs in this batch (mapping_vec is already a JAX array)
                block_links = mapping_vec[ref_indices]

                # Compute feature-stratified weights using segment sum
                # Returns: (batch_hm3, n_unique_features), (n_unique_features,)
                weights, unique_features = compute_batch_weights_segment_sum(
                    X_hm3, X_ref_block, block_links
                )

                # Store batch data for sparse matrix construction
                batch_weight_data.append({
                    'weights': weights,
                    'unique_features': unique_features,
                    'hm3_start_idx': len(all_hm3_snp_names) - len(batch_snp_names),
                    'n_hm3': len(batch_snp_names)
                })

                logger.info(f"  Weights shape: {weights.shape}")
                logger.info(f"  Unique features in batch: {len(unique_features)}")

                # Update progress bar by number of HM3 SNPs processed in this batch
                progress.update(task, advance=len(batch_snp_names))

        # Create BCOO sparse weight matrix from batch results
        weights_bcoo = self._create_bcoo_from_batches(
            batch_weight_data, len(all_hm3_snp_names), n_feature_indices
        )

        logger.info("=" * 80)
        logger.info(f"Chromosome {chromosome} complete")
        logger.info(f"  Total HM3 SNPs processed: {len(all_hm3_snp_names)}")
        logger.info(f"  Weights shape: {weights_bcoo.shape} (sparse BCOO)")
        logger.info(f"  Weights nnz: {weights_bcoo.nse:,}")
        logger.info(f"  Sparsity: {100 * (1 - weights_bcoo.nse / (weights_bcoo.shape[0] * weights_bcoo.shape[1])):.2f}%")
        logger.info("=" * 80)

        return ChromosomeResult(
            chromosome=str(chromosome),
            hm3_snp_names=all_hm3_snp_names,
            ld_scores=None,  # Not computed separately
            weights=weights_bcoo,
            n_features=n_feature_indices,
        )

    def run(
        self,
        mapping_type: str,
        mapping_data: Union[pd.DataFrame, Dict[str, str]],
        window_size_mapping: int = 0,
        mapping_strategy: str = "score",
    ) -> Dict[str, ChromosomeResult]:
        """
        Run pipeline across all chromosomes to compute feature-stratified LD weights.

        Parameters
        ----------
        mapping_type : str
            Type of feature mapping: "bed" for genomic intervals or "dict" for direct mapping (REQUIRED)
        mapping_data : Union[pd.DataFrame, Dict[str, str]]
            Feature mapping data (REQUIRED):
            - For "bed": DataFrame with [Feature, Chrom, Start, End, Score, Strand]
            - For "dict": Dictionary {rsid: feature_name}
        window_size_mapping : int
            Window size for spatial mapping (bp), default: 0
        mapping_strategy : str
            Mapping strategy: "score" (max score) or "tss" (closest TSS), default: "score"

        Returns
        -------
        Dict[str, ChromosomeResult]
            Results for each chromosome with BCOO weight matrices
        """
        logger.info("\n" + "=" * 80)
        logger.info("Starting LD Weight Computation Pipeline")
        logger.info("=" * 80)
        logger.info(f"Mapping type: {mapping_type}")
        logger.info(f"Mapping strategy: {mapping_strategy}")
        logger.info(f"Window size: {window_size_mapping:,} bp")
        logger.info("=" * 80 + "\n")

        # Process each chromosome
        results = {}

        for chrom in self.chromosomes:
            # Process chromosome with mapping parameters
            # Mapping is created inside process_chromosome using the PLINK reader
            result = self.process_chromosome(
                chrom,
                mapping_type=mapping_type,
                mapping_data=mapping_data,
                window_size_mapping=window_size_mapping,
                mapping_strategy=mapping_strategy,
            )
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
):
    """
    Save pipeline results (BCOO weight matrices) to files.

    Parameters
    ----------
    results : Dict[str, ChromosomeResult]
        Results from pipeline
    output_prefix : str
        Output file prefix (saves to {output_prefix}.weights.npz)
    """
    logger.info(f"Saving weight matrices to {output_prefix}.*")

    # Combine SNP information from all chromosomes
    all_snp_data = []
    all_bcoo_data = []
    total_hm3 = 0
    max_features = 0

    for chrom, result in results.items():
        bcoo = result.weights
        n_hm3 = len(result.hm3_snp_names)

        # Collect SNP information
        for snp_name in result.hm3_snp_names:
            all_snp_data.append({
                'CHR': chrom,
                'SNP': snp_name,
            })

        # Convert BCOO to numpy arrays for storage
        # bcoo.data: values, bcoo.indices: (nnz, 2) array
        indices = np.array(bcoo.indices)
        values = np.array(bcoo.data)

        # Offset row indices by total_hm3 to stack chromosomes vertically
        indices_offset = indices.copy()
        indices_offset[:, 0] += total_hm3

        all_bcoo_data.append({
            'indices': indices_offset,
            'values': values,
            'chrom': chrom,
            'n_hm3': n_hm3,
        })

        total_hm3 += n_hm3
        max_features = max(max_features, result.n_features)

    # Combine all BCOO data
    all_indices = np.vstack([d['indices'] for d in all_bcoo_data])
    all_values = np.concatenate([d['values'] for d in all_bcoo_data])

    # Save weight matrix as npz (compressed sparse format)
    weights_file = f"{output_prefix}.weights.npz"
    np.savez_compressed(
        weights_file,
        indices=all_indices,
        values=all_values,
        shape=np.array([total_hm3, max_features]),
    )
    logger.info(f"  Saved weights: {weights_file}")
    logger.info(f"    Shape: ({total_hm3:,}, {max_features:,})")
    logger.info(f"    Non-zero entries: {len(all_values):,}")
    logger.info(f"    Sparsity: {100 * (1 - len(all_values) / (total_hm3 * max_features)):.2f}%")

    # Save SNP information for reference
    snp_file = f"{output_prefix}.snps.tsv"
    df_snps = pd.DataFrame(all_snp_data)
    df_snps.to_csv(snp_file, sep="\t", index=False)
    logger.info(f"  Saved SNP info: {snp_file}")

    logger.info("Results saved successfully")

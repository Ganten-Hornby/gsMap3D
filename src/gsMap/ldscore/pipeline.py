"""
Chromosome-wise pipeline for LD score calculation using NumPy/Scipy.

This module orchestrates the complete workflow:
1. Loop through chromosomes
2. Construct batches
3. Load genotypes once per chromosome
4. Process each batch to compute LD weights
5. Save results as AnnData
"""

import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import scipy.sparse
import anndata as ad
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

from .io import PlinkBEDReader
from .batch_construction import construct_batches, BatchInfo
from .compute import (
    compute_ld_scores,
    compute_batch_weights_sparse,
)
from .mapping import create_snp_feature_map
from .config import LDScoreConfig

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
    weights : scipy.sparse.csr_matrix
        Sparse weight matrix for features, shape (n_hm3_snps, n_features)
    feature_names : List[str]
        Names of mapped features (excluding unmapped bin)
    """
    chromosome: str
    hm3_snp_names: List[str]
    weights: scipy.sparse.csr_matrix
    feature_names: List[str]
    ld_scores: Optional[np.ndarray] = None


class LDScorePipeline:
    """
    Pipeline for computing LD scores across chromosomes.
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
        logger.info("LD Score Pipeline Configuration (NumPy/Scipy)")
        logger.info("=" * 80)
        logger.info(f"PLINK template: {bfile_prefix_template}")
        logger.info(f"HM3 directory: {hm3_dir}")
        logger.info(f"Batch size (HM3): {batch_size_hm3}")
        logger.info(f"LD window: {window_size_bp:,} bp")
        logger.info(f"MAF filter: {maf_min}")
        logger.info(f"Chromosomes: {self.chromosomes}")
        logger.info("=" * 80)

    def load_hm3_snps(self, chromosome: int) -> List[str]:
        """Load HM3 SNP list for a chromosome."""
        possible_paths = [
            self.hm3_dir / f"hm.{chromosome}.snp",
            self.hm3_dir / f"hm3_snps.chr{chromosome}.txt",
            self.hm3_dir / f"hapmap3_snps.chr{chromosome}.txt",
            self.hm3_dir / f"chr{chromosome}.snplist",
            self.hm3_dir / f"w_hm3.snplist.chr{chromosome}",
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    snps = pd.read_csv(path, header=None, names=["SNP"])["SNP"].tolist()
                    logger.info(f"Loaded {len(snps)} HM3 SNPs from {path}")
                    return snps
                except:
                    try:
                        snps = pd.read_csv(path, sep="\t", header=None, names=["SNP"])["SNP"].tolist()
                        logger.info(f"Loaded {len(snps)} HM3 SNPs from {path}")
                        return snps
                    except:
                        continue

        logger.warning(f"No HM3 SNP file found for chromosome {chromosome}")
        return []

    def _create_sparse_matrix_from_batches(
        self,
        batch_weight_data: List[Dict],
        n_hm3_total: int,
        n_features: int,
    ) -> scipy.sparse.csr_matrix:
        """
        Concatenate batch results into a single scipy.sparse CSR matrix.
        """
        if not batch_weight_data:
            return scipy.sparse.csr_matrix((n_hm3_total, n_features), dtype=np.float32)

        # batch_weight_data contains dense numpy arrays (or sparse)
        # shape (n_batch_hm3, n_features)
        # We can simply vstack them because they are ordered by HM3 index

        batch_matrices = [b['weights'] for b in batch_weight_data]

        # If any are sparse, use sparse vstack. If dense, we can still use sparse vstack
        # (it handles dense inputs) or convert first.
        # Given n_features can be large, treating as dense might be memory intensive if not needed.
        # But compute_batch_weights_sparse returns dense array currently.
        # We will convert them to sparse rows for storage efficiency.

        sparse_batches = [scipy.sparse.csr_matrix(m) for m in batch_matrices]

        full_matrix = scipy.sparse.vstack(sparse_batches, format='csr')

        # Verify shape
        if full_matrix.shape != (n_hm3_total, n_features):
             logger.warning(f"Matrix shape mismatch: Expected ({n_hm3_total}, {n_features}), got {full_matrix.shape}")

        logger.info(f"Created sparse matrix: shape={full_matrix.shape}, nnz={full_matrix.nnz}")
        return full_matrix

    def process_chromosome(
        self,
        chromosome: int,
        mapping_type: str,
        mapping_data: Union[pd.DataFrame, Dict[str, str]],
        window_size_mapping: int = 0,
        mapping_strategy: str = "score",
    ) -> Optional[ChromosomeResult]:
        """
        Process a single chromosome.
        """
        logger.info("=" * 80)
        logger.info(f"Processing Chromosome {chromosome}")
        logger.info("=" * 80)

        # Load HM3 SNPs
        target_hm3_snps = self.load_hm3_snps(chromosome)
        if len(target_hm3_snps) == 0:
            logger.warning(f"No HM3 SNPs for chromosome {chromosome}, skipping")
            return None

        # Load PLINK data once
        bfile_prefix = self.bfile_prefix_template.format(chr=chromosome)
        logger.info(f"Loading PLINK data from: {bfile_prefix}")

        try:
            reader = PlinkBEDReader(
                bfile_prefix,
                maf_min=self.maf_min,
                preload=True,
            )
        except FileNotFoundError as e:
            logger.error(f"PLINK files not found for chromosome {chromosome}: {e}")
            return None

        # Create SNP-to-feature mapping (SPARSE)
        logger.info(f"Creating SNP-feature mapping for chromosome {chromosome}...")

        # Returns sparse mapping_matrix (M_ref x N_feat+1) and feature names
        mapping_matrix, feature_names = create_snp_feature_map(
            bim_df=reader.bim,
            mapping_type=mapping_type,
            mapping_data=mapping_data,
            window_size=window_size_mapping,
            strategy=mapping_strategy,
        )

        n_mapped_features = len(feature_names)
        n_feature_indices = n_mapped_features + 1  # +1 for unmapped bin

        logger.info(f"  Mapped features: {n_mapped_features}")
        logger.info(f"  Total feature indices: {n_feature_indices}")
        logger.info(f"  Mapping matrix nnz: {mapping_matrix.nnz}")

        # Construct batches (filters HM3 list based on what is in BIM)
        logger.info(f"Constructing batches...")
        batch_infos = construct_batches(
            bim_df=reader.bim,
            hm3_snp_names=target_hm3_snps,
            batch_size_hm3=self.batch_size_hm3,
            window_size_bp=self.window_size_bp,
        )

        if len(batch_infos) == 0:
            logger.warning(f"No batches created for chromosome {chromosome}")
            return None

        # Calculate actual number of SNPs to be processed for the progress bar
        total_snps_to_process = sum(len(b.hm3_indices) for b in batch_infos)
        logger.info(f"Actual HM3 SNPs found in BIM: {total_snps_to_process}")

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
                f"[cyan]Chr {chromosome}", total=total_snps_to_process
            )

            for i, batch_info in enumerate(batch_infos):
                # Fetch genotypes
                X_hm3 = reader.genotypes[:, batch_info.hm3_indices]

                ref_start = batch_info.ref_start_idx
                ref_end = batch_info.ref_end_idx
                # ref_indices = np.arange(ref_start, ref_end)
                # ref_indices = ref_indices[ref_indices < reader.m]
                X_ref_block = reader.genotypes[:, ref_start:ref_end]

                # Get HM3 SNP names for this batch
                batch_snp_names = reader.bim.iloc[batch_info.hm3_indices]["SNP"].tolist()
                all_hm3_snp_names.extend(batch_snp_names)

                # Slice the sparse mapping matrix for the reference block
                # Slicing rows: [ref_indices, :]
                block_mapping = mapping_matrix[ref_start:ref_end, :]

                # Compute weights using sparse mapping
                # Returns (n_hm3_batch, n_features)
                weights = compute_batch_weights_sparse(
                    X_hm3, X_ref_block, block_mapping
                )

                batch_weight_data.append({
                    'weights': weights,
                    # 'unique_features': Not needed anymore as matrix is fixed width
                    'hm3_start_idx': len(all_hm3_snp_names) - len(batch_snp_names),
                    'n_hm3': len(batch_snp_names)
                })

                progress.update(task, advance=len(batch_snp_names))

        # Create Sparse Matrix from batch results
        weights_csr = self._create_sparse_matrix_from_batches(
            batch_weight_data, len(all_hm3_snp_names), n_feature_indices
        )

        return ChromosomeResult(
            chromosome=str(chromosome),
            hm3_snp_names=all_hm3_snp_names,
            weights=weights_csr,
            feature_names=feature_names,
            ld_scores=None,
        )

    def compute_ldscore_weight_matrix(
        self,
        mapping_type: str,
        mapping_data: Union[pd.DataFrame, Dict[str, str]],
        output_dir: str,
        window_size_mapping: int = 0,
        mapping_strategy: str = "score",
    ):
        """
        Run pipeline across all chromosomes, aggregate results, and save to AnnData.
        """
        logger.info("\n" + "=" * 80)
        logger.info("Starting LD Weight Computation Pipeline")
        logger.info("=" * 80)

        results = {}

        for chrom in self.chromosomes:
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
        logger.info("Processing Complete. Saving Results...")
        logger.info("=" * 80)

        if not results:
            logger.warning("No results generated. Skipping save.")
            return

        # --- Save Logic (Moved from save_results) ---

        # Construct output prefix
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_prefix = output_path / "ld_score_weights"

        logger.info(f"Saving results to {output_prefix}.h5ad ...")

        # 1. Collect all SNP info and sparse matrices
        all_snp_names = []
        all_snp_chroms = []
        sparse_matrices = []

        # Get feature names from the first result
        first_result = next(iter(results.values()))
        feature_names = first_result.feature_names
        var_names = feature_names + ["Unmapped"]

        # Sort chromosomes numerically
        sorted_chroms = sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x)

        for chrom in sorted_chroms:
            res = results[chrom]
            all_snp_names.extend(res.hm3_snp_names)
            all_snp_chroms.extend([chrom] * len(res.hm3_snp_names))
            sparse_matrices.append(res.weights)

        # 2. Concatenate matrices
        if not sparse_matrices:
            logger.error("No matrices to concatenate.")
            return

        logger.info("Concatenating sparse matrices...")
        X = scipy.sparse.vstack(sparse_matrices, format='csr')

        # 3. Create Observation DataFrame (SNPs)
        obs = pd.DataFrame({
            'chrom': all_snp_chroms
        }, index=all_snp_names)
        obs.index.name = 'SNP'

        # 4. Create Variable DataFrame (Features)
        var = pd.DataFrame(index=var_names)
        var.index.name = 'Feature'

        # 5. Create AnnData
        logger.info(f"Creating AnnData object: {X.shape[0]} SNPs x {X.shape[1]} Features")
        adata = ad.AnnData(X=X, obs=obs, var=var)

        # 6. Save
        output_file = f"{output_prefix}.h5ad"
        adata.write(output_file)

        logger.info(f"Successfully saved AnnData to {output_file}")


def run_generate_ldscore_weight_matrix(config: LDScoreConfig):
    """
    Entry point function to run the full LD Score pipeline based on configuration.
    """
    # 1. Load Mapping Data
    logger.info(f"Loading mapping data from: {config.mapping_file}")

    if config.mapping_type == 'bed':
        # Assume BED/TSV format for genomic intervals
        mapping_data = pd.read_csv(config.mapping_file, sep=None, engine='python')

    elif config.mapping_type == 'dict':
        df_map = pd.read_csv(config.mapping_file, sep=None, engine='python')

        if 'SNP' in df_map.columns and 'Feature' in df_map.columns:
            mapping_data = dict(zip(df_map['SNP'], df_map['Feature']))
        elif len(df_map.columns) >= 2:
            logger.info("Assuming first column is SNP and second is Feature.")
            mapping_data = dict(zip(df_map.iloc[:, 0], df_map.iloc[:, 1]))
        else:
            raise ValueError(f"Dictionary mapping file {config.mapping_file} must have at least 2 columns.")
    else:
        raise ValueError(f"Unsupported mapping_type: {config.mapping_type}")

    # 2. Initialize Pipeline
    pipeline = LDScorePipeline(
        bfile_prefix_template=config.bfile_root,
        hm3_dir=config.hm3_snp_path,
        batch_size_hm3=config.batch_size_hm3,
        window_size_bp=config.window_size_bp,
        maf_min=config.maf_min,
        chromosomes=config.chromosomes
    )

    # 3. Run Pipeline (Compute & Save)
    pipeline.compute_ldscore_weight_matrix(
        mapping_type=config.mapping_type,
        mapping_data=mapping_data,
        output_dir=config.output_dir,
        window_size_mapping=config.window_size,
        mapping_strategy=config.strategy
    )
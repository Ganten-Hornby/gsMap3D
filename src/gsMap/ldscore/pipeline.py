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
from typing import Dict, List, Optional, Union, Any, Tuple
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
    weights : Union[scipy.sparse.csr_matrix, np.ndarray]
        Weight matrix for features (sparse or dense), shape (n_hm3_snps, n_features)
    feature_names : List[str]
        Names of mapped features
    """
    chromosome: str
    hm3_snp_names: List[str]
    weights: Union[scipy.sparse.csr_matrix, np.ndarray]
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

    def _load_plink_reader(self, chromosome: int) -> Optional[PlinkBEDReader]:
        """Helper to initialize the PLINK reader for a chromosome."""
        bfile_prefix = self.bfile_prefix_template.format(chr=chromosome)
        logger.info(f"Loading PLINK data from: {bfile_prefix}")

        try:
            reader = PlinkBEDReader(
                bfile_prefix,
                maf_min=self.maf_min,
                preload=True,
            )
            return reader
        except FileNotFoundError as e:
            logger.error(f"PLINK files not found for chromosome {chromosome}: {e}")
            return None

    def _create_sparse_matrix_from_batches(
        self,
        batch_weight_data: List[Dict],
        n_hm3_total: int,
        n_features: int,
    ) -> scipy.sparse.csr_matrix:
        """Concatenate batch results into a single scipy.sparse CSR matrix."""
        if not batch_weight_data:
            return scipy.sparse.csr_matrix((n_hm3_total, n_features), dtype=np.float32)

        # batch_weight_data contains dense numpy arrays
        batch_matrices = [b['weights'] for b in batch_weight_data]

        # Convert to sparse CSR for efficient vertical stacking
        sparse_batches = [scipy.sparse.csr_matrix(m) for m in batch_matrices]
        full_matrix = scipy.sparse.vstack(sparse_batches, format='csr')

        if full_matrix.shape != (n_hm3_total, n_features):
             logger.warning(f"Matrix shape mismatch: Expected ({n_hm3_total}, {n_features}), got {full_matrix.shape}")

        logger.info(f"Created sparse matrix: shape={full_matrix.shape}, nnz={full_matrix.nnz}")
        return full_matrix

    def _create_dense_matrix_from_batches(
        self,
        batch_weight_data: List[Dict],
        n_hm3_total: int,
        n_features: int,
    ) -> np.ndarray:
        """Concatenate batch results into a single dense numpy array."""
        if not batch_weight_data:
            return np.zeros((n_hm3_total, n_features), dtype=np.float32)

        batch_matrices = [b['weights'] for b in batch_weight_data]
        full_matrix = np.vstack(batch_matrices)

        if full_matrix.shape != (n_hm3_total, n_features):
             logger.warning(f"Matrix shape mismatch: Expected ({n_hm3_total}, {n_features}), got {full_matrix.shape}")

        logger.info(f"Created dense matrix: shape={full_matrix.shape}")
        return full_matrix

    def _compute_chromosome_weights(
        self,
        chromosome: int,
        reader: PlinkBEDReader,
        target_hm3_snps: List[str],
        mapping_matrix: Union[scipy.sparse.csr_matrix, np.ndarray],
        feature_names: List[str],
        output_format: str = "sparse"
    ) -> Optional[ChromosomeResult]:
        """
        Core logic to compute LD weights for a chromosome given a reader and mapping matrix.

        Parameters
        ----------
        chromosome : int
            Chromosome number.
        reader : PlinkBEDReader
            Initialized PLINK reader.
        target_hm3_snps : List[str]
            List of target HM3 SNPs.
        mapping_matrix : Union[scipy.sparse.csr_matrix, np.ndarray]
            Mapping matrix (SNP x Feature), aligned to reader.bim.
        feature_names : List[str]
            Names of features in mapping_matrix columns.
        output_format : str
            "sparse" or "dense".
        """
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
                # Fetch genotypes for HM3 SNPs in batch
                X_hm3 = reader.genotypes[:, batch_info.hm3_indices]

                # Fetch genotypes for Reference Block
                ref_indices = np.arange(batch_info.ref_start_idx, batch_info.ref_end_idx)
                # Clip indices to be safe
                ref_indices = ref_indices[ref_indices < reader.m]
                X_ref_block = reader.genotypes[:, ref_indices]

                # Track SNP names
                batch_snp_names = reader.bim.iloc[batch_info.hm3_indices]["SNP"].tolist()
                all_hm3_snp_names.extend(batch_snp_names)

                # Slice the Mapping Matrix for the reference block
                # shape: (n_ref_in_block, n_features)
                block_mapping = mapping_matrix[ref_indices, :]

                # Compute Weights: L2 @ Mapping
                # Result is (n_hm3_batch, n_features) - usually dense
                weights = compute_batch_weights_sparse(
                    X_hm3, X_ref_block, block_mapping
                )

                batch_weight_data.append({
                    'weights': weights,
                    'hm3_start_idx': len(all_hm3_snp_names) - len(batch_snp_names),
                    'n_hm3': len(batch_snp_names)
                })

                progress.update(task, advance=len(batch_snp_names))

        # Aggregate results
        n_hm3_total = len(all_hm3_snp_names)
        n_features = len(feature_names)

        if output_format == "sparse":
            weights_out = self._create_sparse_matrix_from_batches(
                batch_weight_data, n_hm3_total, n_features
            )
        else:
            weights_out = self._create_dense_matrix_from_batches(
                batch_weight_data, n_hm3_total, n_features
            )

        return ChromosomeResult(
            chromosome=str(chromosome),
            hm3_snp_names=all_hm3_snp_names,
            weights=weights_out,
            feature_names=feature_names,
            ld_scores=None,
        )

    def process_chromosome(
        self,
        chromosome: int,
        mapping_type: str,
        mapping_data: Union[pd.DataFrame, Dict[str, str]],
        window_size_mapping: int = 0,
        mapping_strategy: str = "score",
    ) -> Optional[ChromosomeResult]:
        """
        Process a single chromosome (Standard Pipeline).
        """
        logger.info("=" * 80)
        logger.info(f"Processing Chromosome {chromosome}")
        logger.info("=" * 80)

        # 1. Load HM3 SNPs
        target_hm3_snps = self.load_hm3_snps(chromosome)
        if not target_hm3_snps:
            return None

        # 2. Load Reader
        reader = self._load_plink_reader(chromosome)
        if not reader:
            return None

        # 3. Create Mapping Matrix
        logger.info(f"Creating SNP-feature mapping for chromosome {chromosome}...")
        mapping_matrix, feature_names = create_snp_feature_map(
            bim_df=reader.bim,
            mapping_type=mapping_type,
            mapping_data=mapping_data,
            window_size=window_size_mapping,
            strategy=mapping_strategy,
        )

        n_mapped_features = len(feature_names)
        # Add Unmapped column check? create_snp_feature_map handles +1 unmapped bin internally
        # feature_names from mapping.py usually excludes "Unmapped" name in list but matrix has +1 col?
        # Let's check mapping.py:
        # It returns `unique_feature_names`. The matrix has `n_features + 1` columns.
        # We should probably verify this alignment.

        # Actually, mapping.py returns unique_feature_names of length F.
        # The matrix has shape F+1.
        # We should append "Unmapped" to feature names here to match matrix width for downstream logic.
        if mapping_matrix.shape[1] == len(feature_names) + 1:
             # Only add if not already present (though list implies strictly mapped names)
             feature_names_full = feature_names + ["Unmapped"]
        else:
             feature_names_full = feature_names

        logger.info(f"  Mapped features: {n_mapped_features}")
        logger.info(f"  Total feature indices: {len(feature_names_full)}")
        logger.info(f"  Mapping matrix nnz: {mapping_matrix.nnz}")

        # 4. Compute Weights
        return self._compute_chromosome_weights(
            chromosome=chromosome,
            reader=reader,
            target_hm3_snps=target_hm3_snps,
            mapping_matrix=mapping_matrix,
            feature_names=feature_names_full,
            output_format="sparse"
        )

    def process_chromosome_annot(
        self,
        chromosome: int,
        annot_file_template: str,
    ) -> Optional[ChromosomeResult]:
        """
        Process a single chromosome using an external annotation file.
        """
        logger.info("=" * 80)
        logger.info(f"Processing Chromosome {chromosome} (Annotation Mode)")
        logger.info("=" * 80)

        # 1. Load HM3 SNPs
        target_hm3_snps = self.load_hm3_snps(chromosome)
        if not target_hm3_snps:
            return None

        # 2. Load Reader
        reader = self._load_plink_reader(chromosome)
        if not reader:
            return None

        # 3. Load Annotation File
        try:
            annot_file = annot_file_template.format(chr=chromosome)
        except KeyError:
            annot_file = annot_file_template.format(chromosome=chromosome)

        if not Path(annot_file).exists():
            logger.error(f"Annotation file not found: {annot_file}")
            return None

        logger.info(f"Loading annotation from: {annot_file}")
        try:
            df_annot = pd.read_csv(annot_file, sep=r"\s+")
        except Exception as e:
            logger.error(f"Failed to read annotation file: {e}")
            return None

        # 4. Clean and Align Annotation
        # Drop metadata columns if present
        cols_to_drop = [c for c in ["CHR", "BP", "CM", "SNP"] if c in df_annot.columns]

        if "SNP" in df_annot.columns:
            # Reindex to match BIM order exactly
            df_annot = df_annot.set_index("SNP")
            # Select only SNPs present in filtered BIM
            df_annot = df_annot.reindex(reader.bim["SNP"], fill_value=0)
        else:
            if len(df_annot) != reader.n_original:
                 logger.error(f"Annotation rows ({len(df_annot)}) != Original PLINK variants ({reader.n_original}). Cannot align without 'SNP' column.")
                 return None
            # If no SNP column but lengths match, we assume alignment.
            # However, reader.bim is filtered. We must filter df_annot similarly.
            # This requires access to reader.G filters which is tricky here.
            # Safer to assume SNP column is present for this pipeline.
            # For now, we assume strict alignment to filtered BIM is required via SNP ID.
            logger.error("Annotation file missing 'SNP' column. Strict alignment required.")
            return None

        # Drop non-feature columns from DataFrame
        feature_df = df_annot.drop(columns=[c for c in ["CHR", "BP", "CM"] if c in df_annot.columns])
        feature_names = feature_df.columns.tolist()
        n_features = len(feature_names)
        logger.info(f"Found {n_features} features in annotation.")

        # QC: Check non-zero proportion
        for feature in feature_names:
            non_zero_count = np.count_nonzero(feature_df[feature].values)
            prop = non_zero_count / len(feature_df)
            if prop < 0.01: # 0.01 percent
                logger.warning(f"Feature '{feature}' has < 1% non-zero entries ({prop:.5%}).")

        # Convert to sparse CSR matrix
        logger.info("Converting annotation to sparse matrix...")
        annot_matrix = scipy.sparse.csr_matrix(feature_df.values, dtype=np.float32)

        # 5. Compute Weights
        return self._compute_chromosome_weights(
            chromosome=chromosome,
            reader=reader,
            target_hm3_snps=target_hm3_snps,
            mapping_matrix=annot_matrix,
            feature_names=feature_names,
            output_format="dense"
        )

    def calculate_annot_ldscore(
        self,
        annot_file_template: str,
        output_dir: str,
        output_prefix: str = "annot_ldscores"
    ):
        """
        Run pipeline to calculate LD scores from an external annotation matrix.
        """
        logger.info("\n" + "=" * 80)
        logger.info("Starting Annotation-based LD Score Computation")
        logger.info(f"Annotation Template: {annot_file_template}")
        logger.info("=" * 80)

        results = {}
        for chrom in self.chromosomes:
            res = self.process_chromosome_annot(chrom, annot_file_template)
            if res:
                results[str(chrom)] = res

        if not results:
            logger.warning("No results generated.")
            return

        self._save_aggregated_results(results, output_dir, output_prefix)

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

        if not results:
            logger.warning("No results generated. Skipping save.")
            return

        self._save_aggregated_results(results, output_dir, "ld_score_weights")

    def _save_aggregated_results(
        self,
        results: Dict[str, ChromosomeResult],
        output_dir: str,
        output_prefix: str
    ):
        """Helper to concatenate and save results."""
        logger.info("\n" + "=" * 80)
        logger.info("Processing Complete. Concatenating and Saving...")
        logger.info("=" * 80)

        # 1. Collect Data
        all_snp_names = []
        all_snp_chroms = []
        matrices = []

        # Assume all chromosomes have same features
        first_res = next(iter(results.values()))
        feature_names = first_res.feature_names

        sorted_chroms = sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x)

        for chrom in sorted_chroms:
            res = results[chrom]
            if res.feature_names != feature_names:
                logger.error(f"Feature mismatch in chromosome {chrom}. Skipping concatenation.")
                continue

            all_snp_names.extend(res.hm3_snp_names)
            all_snp_chroms.extend([chrom] * len(res.hm3_snp_names))
            matrices.append(res.weights)

        # 2. Concatenate
        # Handle sparse vs dense concatenation
        if scipy.sparse.issparse(matrices[0]):
            X_full = scipy.sparse.vstack(matrices, format='csr')
        else:
            X_full = np.vstack(matrices)

        # 3. Create AnnData
        obs = pd.DataFrame({'chrom': all_snp_chroms}, index=all_snp_names)
        obs.index.name = 'SNP'
        var = pd.DataFrame(index=feature_names)
        var.index.name = 'Feature'

        logger.info(f"Creating AnnData object: {X_full.shape[0]} SNPs x {X_full.shape[1]} Features")
        adata = ad.AnnData(X=X_full, obs=obs, var=var)

        # 4. Save
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{output_prefix}.h5ad"

        adata.write(out_file)
        logger.info(f"Successfully saved AnnData to {out_file}")


def run_generate_ldscore_weight_matrix(config: LDScoreConfig):
    """
    Entry point function to run the full LD Score pipeline based on configuration.
    """
    # 1. Load Mapping Data
    logger.info(f"Loading mapping data from: {config.mapping_file}")

    if config.mapping_type == 'bed':
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
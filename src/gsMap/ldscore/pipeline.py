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

    def __init__(self, config: LDScoreConfig):
        self.config = config
        self.hm3_dir = Path(config.hm3_snp_path)

        logger.info("=" * 80)
        logger.info("LD Score Pipeline Configuration (NumPy/Scipy)")
        logger.info("=" * 80)
        logger.info(f"PLINK template: {config.bfile_root}")
        logger.info(f"HM3 directory: {config.hm3_snp_path}")
        logger.info(f"Batch size (HM3): {config.batch_size_hm3}")
        logger.info(f"LD window: {config.window_size_bp:,} bp")
        logger.info(f"MAF filter: {config.maf_min}")
        logger.info(f"Chromosomes: {config.chromosomes}")
        logger.info(f"Output Directory: {config.output_dir}")
        logger.info(f"Output Filename: {config.output_filename}")
        logger.info("=" * 80)

    def run(self):
        """
        Main entry point. Dispatches to specific pipeline mode based on configuration.
        """
        if self.config.annot_file:
            logger.info(f"Mode: Annotation File ({self.config.annot_file})")
            self._run_with_annotation()
        elif self.config.mapping_file:
            logger.info(f"Mode: SNP-Feature Mapping ({self.config.mapping_type})")
            self._run_with_mapping()
        else:
            raise ValueError("Invalid Configuration: Neither 'annot_file' nor 'mapping_file' specified.")

    def _run_with_mapping(self):
        """Run the pipeline using SNP-Feature mapping (BED/Dictionary)."""
        # 1. Load Mapping Data
        mapping_data = self._load_mapping_data()

        results = {}
        for chrom in self.config.chromosomes:
            result = self._process_chromosome_from_mapping(
                chrom,
                mapping_data=mapping_data
            )
            if result is not None:
                results[str(chrom)] = result

        if not results:
            logger.warning("No results generated. Skipping save.")
            return

        self._save_aggregated_results(results, self.config.output_dir, self.config.output_filename)

    def _run_with_annotation(self):
        """Run the pipeline using external annotation matrices."""
        results = {}
        for chrom in self.config.chromosomes:
            result = self._process_chromosome_from_annotation(chrom)
            if result:
                results[str(chrom)] = result

        if not results:
            logger.warning("No results generated.")
            return

        self._save_aggregated_results(results, self.config.output_dir, self.config.output_filename)

    def _load_mapping_data(self) -> Union[pd.DataFrame, Dict[str, str]]:
        """Helper to load mapping file based on config."""
        logger.info(f"Loading mapping data from: {self.config.mapping_file}")

        if self.config.mapping_type == 'bed':
            return pd.read_csv(self.config.mapping_file, sep=None, engine='python')

        elif self.config.mapping_type == 'dict':
            df_map = pd.read_csv(self.config.mapping_file, sep=None, engine='python')

            if 'SNP' in df_map.columns and 'Feature' in df_map.columns:
                return dict(zip(df_map['SNP'], df_map['Feature']))
            elif len(df_map.columns) >= 2:
                logger.info("Assuming first column is SNP and second is Feature.")
                return dict(zip(df_map.iloc[:, 0], df_map.iloc[:, 1]))
            else:
                raise ValueError(f"Dictionary mapping file {self.config.mapping_file} must have at least 2 columns.")
        else:
            raise ValueError(f"Unsupported mapping_type: {self.config.mapping_type}")

    def _load_hm3_snps(self, chromosome: int) -> List[str]:
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
                    continue

        logger.warning(f"No HM3 SNP file found for chromosome {chromosome}")
        return []

    def _load_plink_reader(self, chromosome: int) -> Optional[PlinkBEDReader]:
        """Helper to initialize the PLINK reader for a chromosome."""
        bfile_prefix = self.config.bfile_root.format(chr=chromosome)
        logger.info(f"Loading PLINK data from: {bfile_prefix}")

        try:
            reader = PlinkBEDReader(
                bfile_prefix,
                maf_min=self.config.maf_min,
                preload=True,
            )
            return reader
        except FileNotFoundError as e:
            logger.error(f"PLINK files not found for chromosome {chromosome}: {e}")
            return None

    def _process_chromosome_from_mapping(
        self,
        chromosome: int,
        mapping_data: Union[pd.DataFrame, Dict[str, str]],
    ) -> Optional[ChromosomeResult]:
        """Process a single chromosome using mapping rules (BED/Dict)."""
        logger.info("=" * 80)
        logger.info(f"Processing Chromosome {chromosome}")
        logger.info("=" * 80)

        target_hm3_snps = self._load_hm3_snps(chromosome)
        if not target_hm3_snps:
            return None

        reader = self._load_plink_reader(chromosome)
        if not reader:
            return None

        logger.info(f"Creating SNP-feature mapping for chromosome {chromosome}...")
        # create_snp_feature_map now returns (matrix, names, df)
        mapping_matrix, feature_names, mapping_df = create_snp_feature_map(
            bim_df=reader.bim,
            mapping_type=self.config.mapping_type,
            mapping_data=mapping_data,
            window_size=self.config.window_size,
            strategy=self.config.strategy,
        )

        # Save Curated Mapping if it exists (BED type)
        if mapping_df is not None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            mapping_out_file = output_dir / f"{self.config.output_filename}.chr{chromosome}.mapping.csv"
            mapping_df.to_csv(mapping_out_file, index=False)
            logger.info(f"  Saved SNP-Feature mapping ({len(mapping_df)} rows) to: {mapping_out_file}")

        # Ensure feature names align with matrix width (handle Unmapped bin)
        if mapping_matrix.shape[1] == len(feature_names) + 1:
             feature_names_full = feature_names + ["Unmapped"]
        else:
             feature_names_full = feature_names

        logger.info(f"  Features: {len(feature_names_full)} | Matrix nnz: {mapping_matrix.nnz}")

        return self._compute_chromosome_weights(
            chromosome=chromosome,
            reader=reader,
            target_hm3_snps=target_hm3_snps,
            mapping_matrix=mapping_matrix,
            feature_names=feature_names_full,
            output_format="sparse"
        )

    def _process_chromosome_from_annotation(
        self,
        chromosome: int,
    ) -> Optional[ChromosomeResult]:
        """Process a single chromosome using external annotation files."""
        logger.info("=" * 80)
        logger.info(f"Processing Chromosome {chromosome} (Annotation Mode)")
        logger.info("=" * 80)

        target_hm3_snps = self._load_hm3_snps(chromosome)
        if not target_hm3_snps:
            return None

        reader = self._load_plink_reader(chromosome)
        if not reader:
            return None

        # Load Annotation File
        try:
            annot_file = self.config.annot_file.format(chr=chromosome)
        except KeyError:
            annot_file = self.config.annot_file.format(chromosome=chromosome)

        if not Path(annot_file).exists():
            logger.error(f"Annotation file not found: {annot_file}")
            return None

        logger.info(f"Loading annotation from: {annot_file}")
        try:
            df_annot = pd.read_csv(annot_file, sep=r"\s+")
        except Exception as e:
            logger.error(f"Failed to read annotation file: {e}")
            return None

        # Drop metadata columns
        cols_to_drop = [c for c in ["CHR", "BP", "CM", "SNP"] if c in df_annot.columns]

        if "SNP" in df_annot.columns:
            # Align to filtered BIM
            df_annot = df_annot.set_index("SNP")
            df_annot = df_annot.reindex(reader.bim["SNP"], fill_value=0)
        else:
            if len(df_annot) != reader.n_original:
                 logger.error(f"Annotation rows mismatch. Missing 'SNP' column preventing alignment.")
                 return None
            logger.error("Annotation file missing 'SNP' column. Strict alignment required.")
            return None

        # Drop non-feature columns
        feature_df = df_annot.drop(columns=[c for c in ["CHR", "BP", "CM"] if c in df_annot.columns])
        feature_names = feature_df.columns.tolist()

        # QC
        for feature in feature_names:
            if np.count_nonzero(feature_df[feature].values) / len(feature_df) < 0.0001:
                logger.warning(f"Feature '{feature}' has < 0.01% non-zero entries.")

        logger.info("Converting annotation to sparse matrix...")
        annot_matrix = scipy.sparse.csr_matrix(feature_df.values, dtype=np.float32)

        return self._compute_chromosome_weights(
            chromosome=chromosome,
            reader=reader,
            target_hm3_snps=target_hm3_snps,
            mapping_matrix=annot_matrix,
            feature_names=feature_names,
            output_format="dense"
        )

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
        Core logic: Batches -> Load Genotypes -> Slice Mapping -> Compute Weights.
        """
        logger.info(f"Constructing batches...")
        batch_infos = construct_batches(
            bim_df=reader.bim,
            hm3_snp_names=target_hm3_snps,
            batch_size_hm3=self.config.batch_size_hm3,
            window_size_bp=self.config.window_size_bp,
        )

        if len(batch_infos) == 0:
            logger.warning(f"No batches created for chromosome {chromosome}")
            return None

        total_snps = sum(len(b.hm3_indices) for b in batch_infos)
        logger.info(f"Actual HM3 SNPs found in BIM: {total_snps}")

        batch_weight_data = []
        all_hm3_snp_names = []

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn()
        ) as progress:
            task = progress.add_task(f"[cyan]Chr {chromosome}", total=total_snps)

            for i, batch_info in enumerate(batch_infos):
                # Genotypes
                X_hm3 = reader.genotypes[:, batch_info.hm3_indices]

                ref_indices = np.arange(batch_info.ref_start_idx, batch_info.ref_end_idx)
                ref_indices = ref_indices[ref_indices < reader.m]
                X_ref_block = reader.genotypes[:, ref_indices]

                # Metadata
                batch_snp_names = reader.bim.iloc[batch_info.hm3_indices]["SNP"].tolist()
                all_hm3_snp_names.extend(batch_snp_names)

                # Slice Mapping
                block_mapping = mapping_matrix[ref_indices, :]

                # Compute
                weights = compute_batch_weights_sparse(X_hm3, X_ref_block, block_mapping)

                batch_weight_data.append({
                    'weights': weights,
                    'hm3_start_idx': len(all_hm3_snp_names) - len(batch_snp_names),
                    'n_hm3': len(batch_snp_names)
                })
                progress.update(task, advance=len(batch_snp_names))

        # Aggregate
        n_hm3_total = len(all_hm3_snp_names)
        n_features = len(feature_names)

        if output_format == "sparse":
            weights_out = self._create_sparse_matrix_from_batches(batch_weight_data, n_hm3_total, n_features)
        else:
            weights_out = self._create_dense_matrix_from_batches(batch_weight_data, n_hm3_total, n_features)

        return ChromosomeResult(
            chromosome=str(chromosome),
            hm3_snp_names=all_hm3_snp_names,
            weights=weights_out,
            feature_names=feature_names
        )

    def _create_sparse_matrix_from_batches(self, batch_data: List[Dict], n_rows: int, n_cols: int) -> scipy.sparse.csr_matrix:
        """Helper to stack batch results into sparse CSR."""
        if not batch_data:
            return scipy.sparse.csr_matrix((n_rows, n_cols), dtype=np.float32)
        matrices = [scipy.sparse.csr_matrix(b['weights']) for b in batch_data]
        return scipy.sparse.vstack(matrices, format='csr')

    def _create_dense_matrix_from_batches(self, batch_data: List[Dict], n_rows: int, n_cols: int) -> np.ndarray:
        """Helper to stack batch results into dense numpy array."""
        if not batch_data:
            return np.zeros((n_rows, n_cols), dtype=np.float32)
        return np.vstack([b['weights'] for b in batch_data])

    def _save_aggregated_results(
        self,
        results: Dict[str, ChromosomeResult],
        output_dir: str,
        output_filename: str
    ):
        """Helper to concatenate and save results."""
        logger.info("\n" + "=" * 80)
        logger.info("Processing Complete. Concatenating and Saving...")
        logger.info("=" * 80)

        all_snp_names = []
        all_snp_chroms = []
        matrices = []

        # Check consistency
        first_res = next(iter(results.values()))
        feature_names = first_res.feature_names

        sorted_chroms = sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x)

        for chrom in sorted_chroms:
            res = results[chrom]
            if res.feature_names != feature_names:
                logger.error(f"Feature mismatch in chromosome {chrom}. Skipping.")
                continue

            all_snp_names.extend(res.hm3_snp_names)
            all_snp_chroms.extend([chrom] * len(res.hm3_snp_names))
            matrices.append(res.weights)

        # Concatenate
        if scipy.sparse.issparse(matrices[0]):
            X_full = scipy.sparse.vstack(matrices, format='csr')
        else:
            X_full = np.vstack(matrices)

        # AnnData
        obs = pd.DataFrame({'chrom': all_snp_chroms}, index=all_snp_names)
        obs.index.name = 'SNP'
        var = pd.DataFrame(index=feature_names)
        var.index.name = 'Feature'

        logger.info(f"Creating AnnData object: {X_full.shape[0]} SNPs x {X_full.shape[1]} Features")
        adata = ad.AnnData(X=X_full, obs=obs, var=var)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{output_filename}.h5ad"

        adata.write(out_file)
        logger.info(f"Successfully saved AnnData to {out_file}")
"""
Chromosome-wise pipeline for LD score calculation using NumPy/Scipy.

This module orchestrates the complete workflow:
1. Loop through chromosomes
2. Construct batches
3. Load genotypes once per chromosome
4. Process each batch to compute LD weights
5. Save results as AnnData
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyranges as pr
import scipy.sparse
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from gsMap.config import LDScoreConfig

from .batch_construction import construct_batches
from .compute import (
    compute_batch_weights_sparse,
    compute_ld_scores,
)
from .io import PlinkBEDReader
from .mapping import create_snp_feature_map

logger = logging.getLogger(__name__)


@dataclass
class ChromosomeResult:
    """
    Results for a single chromosome.

    Attributes
    ----------
    hm3_snp_names : List[str]
        Names of HM3 SNPs processed
    weights : Union[scipy.sparse.csr_matrix, np.ndarray]
        Weight matrix for features (sparse or dense), shape (n_hm3_snps, n_features)
    feature_names : List[str]
        Names of mapped features
    """
    hm3_snp_names: list[str]
    hm3_snp_chr: list[int]
    hm3_snp_bp: list[int]
    weights: scipy.sparse.csr_matrix | np.ndarray
    feature_names: list[str]
    mapping_df: pd.DataFrame | None = None
    ld_scores: np.ndarray | None = None


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
        logger.info(f"LD window: {config.ld_wind} {config.ld_unit}")
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

    def _load_mapping_data(self) -> pd.DataFrame | dict[str, str]:
        """Helper to load mapping file based on config."""
        logger.info(f"Loading mapping data from: {self.config.mapping_file}")

        if self.config.mapping_type == 'bed':
            # Use pyranges to read standard BED file
            try:
                bed_pr = pr.read_bed(str(self.config.mapping_file))
                bed_df = bed_pr.df

                # Convert pyranges BED format to expected format
                # Standard BED columns: Chromosome, Start, End, Name, Score, Strand
                # Expected format: Feature, Chromosome, Start, End, [Score], [Strand]

                if 'Name' not in bed_df.columns:
                    logger.error("BED file must contain a 'Name' column (4th column in BED6 format)")
                    logger.error("Required format: standard BED6 (chr, start, end, name, score, strand)")
                    logger.error("Note: BED file should NOT have a header line")
                    sys.exit(1)

                # Rename 'Name' to 'Feature' for internal use
                bed_df = bed_df.rename(columns={'Name': 'Feature'})

                # Ensure required columns exist
                required_cols = ['Chromosome', 'Start', 'End', 'Feature']
                if not all(col in bed_df.columns for col in required_cols):
                    logger.error("BED file missing required columns after parsing")
                    logger.error("Required format: standard BED6 (chr, start, end, name, score, strand)")
                    logger.error("Note: BED file should NOT have a header line")
                    sys.exit(1)

                logger.info(f"Successfully loaded BED file: {len(bed_df)} features")
                logger.info(f"Columns: {list(bed_df.columns)}")
                return bed_df

            except Exception as e:
                logger.error(f"Failed to read BED file: {e}")
                logger.error("Required format: standard BED6 (chr, start, end, name, score, strand)")
                logger.error("Note: BED file should NOT have a header line")
                sys.exit(1)

        elif self.config.mapping_type == 'dict':
            df_map = pd.read_csv(self.config.mapping_file, sep=None, engine='python')

            if 'SNP' in df_map.columns and 'Feature' in df_map.columns:
                return dict(zip(df_map['SNP'], df_map['Feature'], strict=False))
            elif len(df_map.columns) >= 2:
                logger.info("Assuming first column is SNP and second is Feature.")
                return dict(zip(df_map.iloc[:, 0], df_map.iloc[:, 1], strict=False))
            else:
                raise ValueError(f"Dictionary mapping file {self.config.mapping_file} must have at least 2 columns.")
        else:
            raise ValueError(f"Unsupported mapping_type: {self.config.mapping_type}")

    def _load_hm3_snps(self, chromosome: int) -> list[str]:
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

    def _load_plink_reader(self, chromosome: int) -> PlinkBEDReader | None:
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
        mapping_data: pd.DataFrame | dict[str, str],
    ) -> ChromosomeResult | None:
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
            feature_window_size=self.config.feature_window_size,
            strategy=self.config.strategy,
        )

        # Save Curated Mapping if it exists (BED type)
        if mapping_df is not None:
             logger.debug(f"  Captured SNP-Feature mapping ({len(mapping_df)} rows) for chromosome {chromosome}")

        # Ensure feature names align with matrix width (handle Unmapped bin)
        if mapping_matrix.shape[1] == len(feature_names) + 1:
             feature_names_full = feature_names + ["Unmapped"]
        else:
             feature_names_full = feature_names

        logger.info(f"  Features: {len(feature_names_full)} | Matrix nnz: {mapping_matrix.nnz}")

        if self.config.calculate_w_ld:
             self._compute_w_ld(chromosome, target_hm3_snps)

        return self._compute_chromosome_weights(
            chromosome=chromosome,
            reader=reader,
            target_hm3_snps=target_hm3_snps,
            mapping_matrix=mapping_matrix,
            feature_names=feature_names_full,
            mapping_df=mapping_df,
            output_format="sparse"
        )

    def _process_chromosome_from_annotation(
        self,
        chromosome: int,
    ) -> ChromosomeResult | None:
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
        [c for c in ["CHR", "BP", "CM", "SNP"] if c in df_annot.columns]

        if "SNP" in df_annot.columns:
            # Align to filtered BIM
            df_annot = df_annot.set_index("SNP")
            df_annot = df_annot.reindex(reader.bim["SNP"], fill_value=0)
        else:
            logger.warning("Annotation file missing 'SNP' column. Thin annotation format detected, assuming strict alignment.")

            if len(df_annot) == reader.m_original:
                # Filter df_annot according to filtered SNPs using snp_ids_original
                logger.info(f"Filtering annotation from {len(df_annot)} to {reader.m} SNPs based on MAF/QC filters")
                df_annot.index = reader.snp_ids_original
                df_annot = df_annot.reindex(reader.bim["SNP"], fill_value=0)
            else:
                logger.error(f"Annotation rows mismatch. Missing 'SNP' column preventing alignment. For the thin format, annotation rows ({len(df_annot)}) must match SNP count ({reader.m_original}) in the plink panel.")
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

        if self.config.calculate_w_ld:
             self._compute_w_ld(chromosome, target_hm3_snps)

        return self._compute_chromosome_weights(
            chromosome=chromosome,
            reader=reader,
            target_hm3_snps=target_hm3_snps,
            mapping_matrix=annot_matrix,
            feature_names=feature_names,
            mapping_df=None,
            output_format="dense"
        )

    def _compute_chromosome_weights(
        self,
        chromosome: int,
        reader: PlinkBEDReader,
        target_hm3_snps: list[str],
        mapping_matrix: scipy.sparse.csr_matrix | np.ndarray,
        feature_names: list[str],
        mapping_df: pd.DataFrame | None = None,
        output_format: str = "sparse"
    ) -> ChromosomeResult | None:
        """
        Core logic: Batches -> Load Genotypes -> Slice Mapping -> Compute Weights.
        """
        logger.info("Constructing batches...")
        batch_infos = construct_batches(
            bim_df=reader.bim,
            hm3_snp_names=target_hm3_snps,
            batch_size_hm3=self.config.batch_size_hm3,
            ld_wind=self.config.ld_wind,
            ld_unit=self.config.ld_unit,
        )

        if len(batch_infos) == 0:
            logger.warning(f"No batches created for chromosome {chromosome}")
            return None

        total_snps = sum(len(b.hm3_indices) for b in batch_infos)
        logger.info(f"Actual HM3 SNPs found in BIM: {total_snps}")

        batch_weight_data = []
        all_hm3_snp_names = []
        all_hm3_snp_chr = []
        all_hm3_snp_bp = []

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
                batch_bim = reader.bim.iloc[batch_info.hm3_indices]
                batch_snp_names = batch_bim["SNP"].tolist()
                all_hm3_snp_names.extend(batch_snp_names)
                all_hm3_snp_chr.extend(batch_bim["CHR"].tolist())
                all_hm3_snp_bp.extend(batch_bim["BP"].tolist())

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
            hm3_snp_names=all_hm3_snp_names,
            hm3_snp_chr=all_hm3_snp_chr,
            hm3_snp_bp=all_hm3_snp_bp,
            weights=weights_out,
            feature_names=feature_names,
            mapping_df=mapping_df
        )

    def _create_sparse_matrix_from_batches(self, batch_data: list[dict], n_rows: int, n_cols: int) -> scipy.sparse.csr_matrix:
        """Helper to stack batch results into sparse CSR."""
        if not batch_data:
            return scipy.sparse.csr_matrix((n_rows, n_cols), dtype=np.float32)
        matrices = [scipy.sparse.csr_matrix(b['weights']) for b in batch_data]
        return scipy.sparse.vstack(matrices, format='csr')

    def _create_dense_matrix_from_batches(self, batch_data: list[dict], n_rows: int, n_cols: int) -> np.ndarray:
        """Helper to stack batch results into dense numpy array."""
        if not batch_data:
            return np.zeros((n_rows, n_cols), dtype=np.float32)
        return np.vstack([b['weights'] for b in batch_data])

    def _compute_w_ld(self, chromosome: int, hm3_snps: list[str]):
        """
        Compute 'w_ld' (weighted LD scores) for a chromosome.
        w_ld is typically the LD score of a SNP computed against the set of regression SNPs (HM3 SNPs).
        """
        logger.info(f"Computing w_ld for chromosome {chromosome}...")

        # 1. Initialize Reader restricted to HM3 SNPs
        bfile_prefix = self.config.bfile_root.format(chr=chromosome)
        try:
            # We filter for HM3 SNPs specifically
            reader = PlinkBEDReader(
                bfile_prefix,
                maf_min=self.config.maf_min,
                keep_snps=hm3_snps,
                preload=True,
            )
        except Exception as e:
            logger.error(f"Failed to load PLINK for w_ld (chk {chromosome}): {e}")
            return

        # 2. Construct Batches (using the filtered BIM)
        # Note: reader.bim now ONLY contains HM3 SNPs (and those passing MAF)
        # We use all available SNPs in the reader as targets
        available_hm3 = reader.bim["SNP"].tolist()

        batch_infos = construct_batches(
            bim_df=reader.bim,
            hm3_snp_names=available_hm3,
            batch_size_hm3=self.config.batch_size_hm3,
            ld_wind=self.config.ld_wind,
            ld_unit=self.config.ld_unit,
        )

        logger.info(f"  w_ld: Processing {len(batch_infos)} batches for {len(available_hm3)} SNPs")

        w_ld_values = []
        w_ld_snps = []

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn()
        ) as progress:
            task = progress.add_task(f"[magenta]w_ld Chr {chromosome}", total=len(available_hm3))

            for batch in batch_infos:
                # Genotypes (filtered to HM3)
                X_hm3 = reader.genotypes[:, batch.hm3_indices]

                # Reference block is also from the filtered reader (so it's HM3 only)
                ref_indices = np.arange(batch.ref_start_idx, batch.ref_end_idx)
                # Clip to valid range (should be handled by construct_batches but safe to double check)
                ref_indices = ref_indices[ref_indices < reader.m]
                X_ref = reader.genotypes[:, ref_indices]

                # Compute LD Scores (L2 sum)
                ld_scores_batch = compute_ld_scores(X_hm3, X_ref)

                w_ld_values.append(ld_scores_batch)

                batch_snps = reader.bim.iloc[batch.hm3_indices]["SNP"].tolist()
                w_ld_snps.extend(batch_snps)

                progress.update(task, advance=len(batch.hm3_indices))

        # 3. Aggregate and Save
        if w_ld_values:
            w_ld_arr = np.concatenate(w_ld_values)

            # Match with metadata
            df_w_ld = reader.bim[reader.bim["SNP"].isin(w_ld_snps)].copy()
            # Ensure order
            df_w_ld = df_w_ld.set_index("SNP").reindex(w_ld_snps).reset_index()
            df_w_ld["L2"] = w_ld_arr

            # Select columns
            out_cols = ["CHR", "SNP", "BP", "CM", "L2"]
            # Ensure columns exist (CM might be missing or 0)
            if "CM" not in df_w_ld.columns:
                 df_w_ld["CM"] = 0.0

            df_w_ld = df_w_ld[out_cols]

            # Determine output path
            w_ld_base = Path(self.config.w_ld_dir) if self.config.w_ld_dir else Path(self.config.output_dir) / "w_ld"
            w_ld_base.mkdir(parents=True, exist_ok=True)

            out_file = w_ld_base / f"weights.{chromosome}.l2.ldscore.gz"
            df_w_ld.to_csv(out_file, sep="\t", index=False, compression="gzip")
            logger.info(f"  Saved w_ld to: {out_file}")

    def _save_aggregated_results(
        self,
        results: dict[str, ChromosomeResult],
        output_dir: str,
        output_filename: str
    ):
        """Helper to concatenate and save results."""
        logger.info("\n" + "=" * 80)
        logger.info("Processing Complete. Concatenating and Saving...")
        logger.info("=" * 80)

        all_snp_names = []
        all_snp_chr = [] # numeric CHR from BIM
        all_snp_bp = []  # numeric BP from BIM
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
            all_snp_chr.extend(res.hm3_snp_chr)
            all_snp_bp.extend(res.hm3_snp_bp)
            matrices.append(res.weights)

        # Concatenate
        if scipy.sparse.issparse(matrices[0]):
            X_full = scipy.sparse.vstack(matrices, format='csr')
        else:
            X_full = np.vstack(matrices)

        # AnnData
        obs = pd.DataFrame({
            'CHR': all_snp_chr,
            'BP': all_snp_bp
        }, index=all_snp_names)
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

        # Save Combined Mapping CSV if available
        all_mapping_dfs = [res.mapping_df for res in results.values() if res.mapping_df is not None]
        if all_mapping_dfs:
            combined_mapping_df = pd.concat(all_mapping_dfs, ignore_index=True)
            mapping_out_file = out_path / f"{output_filename}.csv"
            combined_mapping_df.to_csv(mapping_out_file, index=False)
            logger.info(f"Successfully saved combined SNP-Feature mapping to {mapping_out_file}")

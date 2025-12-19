#!/usr/bin/env python
"""
Test script for the ldscore module using 1000G reference panel.

This script:
1. Converts gencode CSV to BED format
2. Runs the LD score pipeline with TSS strategy
3. Saves results to output directory
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gsMap.ldscore.pipeline import LDScorePipeline
from gsMap.config import LDScoreConfig
import jax
jax.config.update("jax_compilation_cache_dir", None)
# set device as cpu
jax.config.update('jax_platform_name', 'cpu')
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s | %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_gencode_csv_to_bed(csv_path: str, strategy: str = "tss") -> pd.DataFrame:
    """
    Convert gencode CSV file to BED format for SNP-feature mapping.

    Parameters
    ----------
    csv_path : str
        Path to gencode CSV file with columns:
        [gene_id, gene_name, gene_type, Chromosome, Start, End, Strand]
    strategy : str
        Mapping strategy: 'tss' or 'score'

    Returns
    -------
    pd.DataFrame
        BED format DataFrame with columns:
        ['Feature', 'Chromosome', 'Start', 'End', 'Score', 'Strand']
    """
    logger.info(f"Loading gencode CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"Original CSV shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Create BED format DataFrame
    bed_df = pd.DataFrame({
        'Feature': df['gene_name'],  # Use gene name as feature
        'Chromosome': df['Chromosome'],
        'Start': df['Start'],
        'End': df['End'],
        'Score': 1.0,  # Default score (can be modified based on importance)
        'Strand': df['Strand']
    })

    # Clean chromosome names (remove 'chr' prefix if present)
    # The PLINK BIM files typically use numeric chromosomes
    bed_df['Chromosome'] = bed_df['Chromosome'].str.replace('chr', '')

    # Filter to autosomal chromosomes (1-22)
    bed_df = bed_df[bed_df['Chromosome'].isin([str(i) for i in range(1, 23)])]

    logger.info(f"BED format shape after filtering: {bed_df.shape}")
    logger.info(f"Unique chromosomes: {sorted(bed_df['Chromosome'].unique())}")
    logger.info(f"Unique features (genes): {bed_df['Feature'].nunique()}")

    return bed_df


def test_ldscore_weight_matrix():
    """Main test function."""

    # ========================================================================
    # Configuration
    # ========================================================================

    # Base paths
    data_dir = Path("/mnt/d/01_Project/01_Research/202312_gsMap/data/gsMap_dev_data/online_resource/gsMap_resource")

    # Input files
    csv_file = data_dir / "genome_annotation/gtf/gencode.v46lift37.basic.annotation.protein_coding.gene.csv"
    bfile_template = str(data_dir / "LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC.{chr}")
    hm3_dir = str(data_dir / "LDSC_resource/hapmap3_snps")

    # Output directory
    output_dir = Path("/mnt/d/01_Project/01_Research/202312_gsMap/src/gsMap/test_ldscore_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mapping parameters
    strategy = "tss"  # Use TSS (transcription start site) strategy
    window_size = 100000  # 100kb window around genes

    # Pipeline parameters
    chromosomes = [22]  # Test with chromosome 22 first (smallest)
    # chromosomes = list(range(1, 23))  # Uncomment to run all chromosomes
    batch_size_hm3 = 50

    logger.info("=" * 80)
    logger.info("LD Score Test Configuration")
    logger.info("=" * 80)
    logger.info(f"CSV file: {csv_file}")
    logger.info(f"BFILE template: {bfile_template}")
    logger.info(f"HM3 directory: {hm3_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Window size: {window_size:,} bp")
    logger.info(f"Test chromosomes: {chromosomes}")
    logger.info("=" * 80)

    # ========================================================================
    # Step 1: Convert CSV to BED format
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Converting CSV to BED format")
    logger.info("=" * 80)

    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        sys.exit(1)

    bed_df = convert_gencode_csv_to_bed(str(csv_file), strategy=strategy)

    # Save BED file for reference
    bed_output = output_dir / "gencode.protein_coding.bed"
    bed_df.to_csv(bed_output, sep='\t', index=False)
    logger.info(f"Saved BED file to: {bed_output}")

    # ========================================================================
    # Step 2: Initialize and Run LD Score Pipeline
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Running LD Score Pipeline")
    logger.info("=" * 80)

    # Create configuration
    config = LDScoreConfig(
        workdir=output_dir.parent,
        project_name=output_dir.name,
        bfile_root=bfile_template,
        hm3_snp_path=hm3_dir,
        output_dir=str(output_dir),
        mapping_type="bed",
        mapping_file=str(bed_output),
        feature_window_size=window_size,
        strategy=strategy,
        ld_wind=1.0,  # LD window size
        ld_unit="CM",
        maf_min=0.01,
        chromosomes=chromosomes,
        batch_size_hm3=batch_size_hm3,
        calculate_w_ld = True
    )

    # Initialize pipeline and run
    pipeline = LDScorePipeline(config)
    pipeline.run()



if __name__ == "__main__":
    try:
        test_ldscore_weight_matrix()
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)

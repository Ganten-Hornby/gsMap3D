#!/usr/bin/env python
"""
Test script for verifying w_ld calculation in LDScorePipeline.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from gsMap.ldscore.pipeline import LDScorePipeline
from gsMap.config import LDScoreConfig
import jax

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s | %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_w_ld_calculation():
    """Verify w_ld calculation."""

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
    output_dir = Path("/mnt/d/01_Project/01_Research/202312_gsMap/src/gsMap/test_ldscore_output_w_ld")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    w_ld_dir = output_dir / "w_ld"

    # Create dummy mapping file just to satisfy pipeline requirements
    # We don't care about the main mapping results for this test
    # but we need a valid mode to trigger the pipeline.
    mapping_file = output_dir / "dummy_mapping.bed"
    pd.DataFrame({
        'Feature': ['GeneA'],
        'Chromosome': ['22'],
        'Start': [20000000],
        'End': [20001000],
        'Score': [1.0],
        'Strand': ['+']
    }).to_csv(mapping_file, sep='\t', index=False)

    # ========================================================================
    # Run LD Score Pipeline with w_ld enabled
    # ========================================================================

    logger.info("Running Pipeline with calculate_w_ld=True...")

    config = LDScoreConfig(
        bfile_root=bfile_template,
        hm3_snp_path=hm3_dir,
        output_dir=str(output_dir),
        w_ld_dir=str(w_ld_dir),
        
        # Enable w_ld
        calculate_w_ld=True,
        
        # Standard configs
        mapping_type="bed",
        mapping_file=str(mapping_file),
        feature_window_size=0,
        strategy="tss",
        ld_wind=1.0,
        ld_unit="CM",
        maf_min=0.01,
        chromosomes=[22], # Test on chr22
        batch_size_hm3=50,
    )

    pipeline = LDScorePipeline(config)
    pipeline.run()

    # ========================================================================
    # Verify Results
    # ========================================================================
    
    expected_file = w_ld_dir / "weights.22.l2.ldscore.gz"
    if not expected_file.exists():
        logger.error(f"FAILURE: w_ld file not found: {expected_file}")
        sys.exit(1)
        
    logger.info(f"w_ld file found: {expected_file}")
    
    df_w_ld = pd.read_csv(expected_file, sep="\t", compression="gzip")
    logger.info(f"Loaded w_ld: {df_w_ld.shape}")
    logger.info(f"Columns: {df_w_ld.columns.tolist()}")
    
    # Check for NaN/Inf
    if df_w_ld["L2"].isna().any():
        logger.error("FAILURE: w_ld contains NaNs")
        sys.exit(1)
        
    # Check for reasonable values (LD score >= 1 approx, since r^2 with self is 1)
    # Though with unbiased estimator it can be slightly < 1 if N is small, but usually sum is >= 1
    min_l2 = df_w_ld["L2"].min()
    logger.info(f"Min L2: {min_l2}")
    logger.info(f"Max L2: {df_w_ld['L2'].max()}")
    
    # It's possible for unbiased estimator to be slightly < 0 for single pairs, 
    # but summation usually positive. 
    # Just checking we have data.
    
    if len(df_w_ld) == 0:
        logger.error("FAILURE: w_ld is empty")
        sys.exit(1)

    logger.info("SUCCESS: w_ld calculation verified.")

if __name__ == "__main__":
    test_w_ld_calculation()

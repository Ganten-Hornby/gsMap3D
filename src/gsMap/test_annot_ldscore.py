#!/usr/bin/env python
"""
Test script for calculating LD scores from baseline annotation files.

This script:
1. Initializes the LD Score Pipeline
2. Loads baseline annotation files (e.g., baseline.{chr}.annot.gz)
3. Calculates LD scores for each annotation
4. Saves results to output directory
"""

import logging
import sys
from pathlib import Path
import jax

# Configure JAX
jax.config.update("jax_compilation_cache_dir", None)
jax.config.update('jax_platform_name', 'cpu')

from gsMap.ldscore.pipeline import LDScorePipeline
from gsMap.config import LDScoreConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s | %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_calculate_annot_ldscore():
    """Main test function for annotation-based LD scores."""

    # ========================================================================
    # Configuration
    # ========================================================================

    # Base paths
    data_dir = Path("/mnt/d/01_Project/01_Research/202312_gsMap/data/gsMap_dev_data/online_resource/gsMap_resource")
    annot_dir = Path("/mnt/d/01_Project/01_Research/202312_gsMap/data/gsMap_dev_data/online_resource/gsMap_additional_annotation")

    # Input files
    bfile_template = str(data_dir / "LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC.{chr}")
    hm3_dir = str(data_dir / "LDSC_resource/hapmap3_snps")
    annot_template = str(annot_dir / "baseline.{chr}.annot.gz")

    # Output directory
    output_dir = Path("/mnt/d/01_Project/01_Research/202312_gsMap/src/gsMap/test_ldscore_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline parameters
    chromosomes = [22]  # Test with chromosome 22 first (smallest)
    batch_size_hm3 = 50
    window_size_bp = 1_000_000  # 1Mb LD window
    maf_min = 0.01

    logger.info("=" * 80)
    logger.info("Annotation LD Score Test Configuration")
    logger.info("=" * 80)
    logger.info(f"BFILE template: {bfile_template}")
    logger.info(f"HM3 directory: {hm3_dir}")
    logger.info(f"Annotation template: {annot_template}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"LD window: {window_size_bp:,} bp")
    logger.info(f"MAF filter: {maf_min}")
    logger.info(f"Test chromosomes: {chromosomes}")
    logger.info(f"Batch size (HM3): {batch_size_hm3}")
    logger.info("=" * 80)

    # ========================================================================
    # Initialize and Run LD Score Pipeline
    # ========================================================================

    logger.info("\n" + "=" * 80)
    logger.info("Running Annotation LD Score Pipeline")
    logger.info("=" * 80)

    # Create configuration with annotation file
    config = LDScoreConfig(
        workdir=output_dir.parent,
        project_name=output_dir.name,
        bfile_root=bfile_template,
        hm3_snp_path=hm3_dir,
        output_dir=str(output_dir),
        annot_file=annot_template,  # Use annotation file mode
        ld_wind=1.0,
        ld_unit="CM",
        maf_min=maf_min,
        chromosomes=chromosomes,
        batch_size_hm3=batch_size_hm3,
    )

    # Initialize pipeline and run
    pipeline = LDScorePipeline(config)
    pipeline.run()

    logger.info("\n" + "=" * 80)
    logger.info("Test Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}/annot_ldscores.h5ad")


if __name__ == "__main__":
    try:
        test_calculate_annot_ldscore()
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)

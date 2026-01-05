
import logging
import time
from pathlib import Path
import yaml

from gsMap.config import QuickModeConfig
from gsMap.find_latent import run_find_latent_representation
from gsMap.config.find_latent_config import check_find_latent_done
from gsMap.latent2gene import run_latent_to_gene
from gsMap.config.latent2gene_config import check_latent2gene_done
from gsMap.spatial_ldsc.spatial_ldsc_multiple_sumstats import run_spatial_ldsc
from gsMap.spatial_ldsc.spatial_ldsc_jax import run_spatial_ldsc_jax
from gsMap.config.spatial_ldsc_config import check_spatial_ldsc_done
from gsMap.cauchy_combination_test import run_Cauchy_combination
from gsMap.config.cauchy_config import check_cauchy_done
from gsMap.report import run_report
from gsMap.config.report_config import check_report_done

logger = logging.getLogger("gsMap.pipeline")

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m {int(seconds % 60)}s"


def run_quick_mode(config: QuickModeConfig):
    """
    Run the Quick Mode pipeline.
    """
    logger.info("Starting Quick Mode pipeline")
    pipeline_start_time = time.time()
    
    steps = ["find_latent", "latent2gene", "spatial_ldsc", "cauchy", "report"]
    try:
        start_idx = steps.index(config.start_step)
    except ValueError:
        raise ValueError(f"Invalid start_step: {config.start_step}. Must be one of {steps}")
        
    stop_idx = len(steps) - 1
    if config.stop_step:
        try:
            stop_idx = steps.index(config.stop_step)
        except ValueError:
            raise ValueError(f"Invalid stop_step: {config.stop_step}. Must be one of {steps}")
            
    if start_idx > stop_idx:
        raise ValueError(f"start_step ({config.start_step}) must be before or equal to stop_step ({config.stop_step})")

    # Step 1: Find Latent Representations
    if start_idx <= 0 <= stop_idx:
        logger.info("=== Step 1: Find Latent Representations ===")
        start_time = time.time()
        
        if check_find_latent_done(config):
            logger.info(f"Find latent representations already done (verified via {config.find_latent_metadata_path}). Skipping...")
        else:
            run_find_latent_representation(config.find_latent_config)
            
        logger.info(f"Step 1 completed in {format_duration(time.time() - start_time)}")

    # Step 2: Latent to Gene
    if start_idx <= 1 <= stop_idx:
        logger.info("=== Step 2: Latent to Gene Mapping ===")
        start_time = time.time()
        
        if check_latent2gene_done(config):
             logger.info("Latent to gene mapping already done. Skipping...")
        else:
             run_latent_to_gene(config.latent2gene_config)
            
        logger.info(f"Step 2 completed in {format_duration(time.time() - start_time)}")
        
    # Get lists of traits to process
    if not config.sumstats_config_dict:
        # Check if we should warn? Only if running step 3,4,5
        if start_idx <= 4 and stop_idx >= 2:
             logger.warning("No summary statistics provided. Steps requiring GWAS data (Spatial LDSC, Cauchy, Report) may fail or do nothing if relying on them.")
    
    traits_to_process = config.sumstats_config_dict

    # Step 3: Spatial LDSC
    if start_idx <= 2 <= stop_idx:
        logger.info("=== Step 3: Spatial LDSC ===")
        start_time = time.time()
        

        traits_remaining = {}
        for trait_name, sumstats_path in traits_to_process.items():
            if check_spatial_ldsc_done(config, trait_name):
                logger.info(f"Spatial LDSC result already exists for {trait_name}. Skipping...")
            else:
                traits_remaining[trait_name] = sumstats_path
        
        if not traits_remaining:
            logger.info("All traits have been processed for Spatial LDSC. Skipping step...")
        else:
            sldsc_config = config.spatial_ldsc_config
            # Update config to run only remaining traits
            sldsc_config.sumstats_config_dict = traits_remaining
            run_spatial_ldsc_jax(sldsc_config)

                
        logger.info(f"Step 3 completed in {format_duration(time.time() - start_time)}")
        
    # Step 4: Cauchy Combination
    if start_idx <= 3 <= stop_idx:
        logger.info("=== Step 4: Cauchy Combination ===")
        start_time = time.time()

        for trait_name in traits_to_process:
            logger.info(f"--- Processing Cauchy for {trait_name} ---")
            if check_cauchy_done(config, trait_name):
                 logger.info(f"Cauchy result already exists for {trait_name}. Skipping...")
            else:
                cauchy_config = config.cauchy_config
                cauchy_config.trait_name = trait_name
                run_Cauchy_combination(cauchy_config)

        logger.info(f"Step 4 completed in {format_duration(time.time() - start_time)}")

    # Step 5: Report
    if start_idx <= 4 <= stop_idx:
        logger.info("=== Step 5: Generate Report ===")
        start_time = time.time()
        
        if check_report_done(config, "full_report"):
            logger.info("Report already exists. Skipping...")
        else:

            run_report(config,)
            
        logger.info(f"Step 5 completed in {format_duration(time.time() - start_time)}")

    logger.info(f"Pipeline completed successfully in {format_duration(time.time() - pipeline_start_time)}")

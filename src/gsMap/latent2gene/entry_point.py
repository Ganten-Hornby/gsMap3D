"""
Main entry point for the latent2gene subpackage
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

from .rank_calculator import RankCalculator
from .marker_scores import MarkerScoreCalculator
from ..config import LatentToGeneConfig
from .memmap_io import MemMapDense

logger = logging.getLogger(__name__)


def run_latent_to_gene(config: LatentToGeneConfig) -> Dict[str, Any]:
    """
    Main entry point for latent to gene conversion
    
    This function orchestrates the complete pipeline:
    1. Calculate ranks and concatenate latent representations
    2. Calculate marker scores for each cell type
    
    Args:
        config: LatentToGeneConfig object with all necessary parameters
        
    Returns:
        Dictionary with paths to all outputs:
            - concatenated_latent_adata: Path to concatenated latent representations
            - rank_memmap: Path to rank memory map file
            - mean_frac: Path to mean expression fraction
            - marker_scores: Path to marker scores memory map
            - metadata: Path to metadata YAML file
    """
    
    logger.info("=" * 60)
    logger.info("Starting latent to gene conversion pipeline")
    logger.info("=" * 60)
    logger.info(f"Using configuration: {config}")

    
    # Setup output directory using config paths
    output_dir = Path(config.latent2gene_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if all outputs already exist using config paths
    expected_outputs = {
        "concatenated_latent_adata": Path(config.concatenated_latent_adata_path),
        "rank_memmap": Path(config.rank_memmap_path),
        "mean_frac": Path(config.mean_frac_path),
        "marker_scores": Path(config.marker_scores_memmap_path),
        "metadata": Path(config.latent2gene_metadata_path),
        "rank_meta": Path(config.rank_memmap_path).with_suffix('.meta.yaml'),
        "marker_scores_meta": Path(config.marker_scores_memmap_path).with_suffix('.meta.yaml')
    }
    
    if all(Path(p).exists() for p in expected_outputs.values()):
        logger.info("All outputs already exist. Checking completion status...")
        

        # Check rank memmap completion using the new class method
        rank_memmap_complete, rank_meta = MemMapDense.check_complete(expected_outputs["rank_memmap"])
        if not rank_memmap_complete:
            if rank_meta:
                logger.warning("Rank memmap exists but is not marked as complete")
            else:
                logger.warning("Could not read rank memmap metadata")
        
        # Check marker scores memmap completion using the new class method
        marker_scores_complete, marker_meta = MemMapDense.check_complete(expected_outputs["marker_scores"])
        if not marker_scores_complete:
            if marker_meta:
                logger.warning("Marker scores memmap exists but is not marked as complete")
            else:
                logger.warning("Could not read marker scores memmap metadata")
        
        if rank_memmap_complete and marker_scores_complete:
            logger.info("All memory maps are properly completed. Loading metadata...")
            with open(expected_outputs["metadata"], 'r') as f:
                existing_metadata = yaml.safe_load(f)
            logger.info(f"Found existing complete results for {existing_metadata.get('n_cells', 'unknown')} cells "
                       f"and {existing_metadata.get('n_genes', 'unknown')} genes")
            return {k: str(v) for k, v in expected_outputs.items()}
        else:
            logger.warning("Memory maps exist but are not properly completed. Re-running pipeline...")

    # Step 1: Calculate ranks and concatenate
    logger.info("\n" + "-" * 40)
    logger.info("Step 1: Rank calculation and concatenation")
    logger.info("-" * 40)
    
    rank_calculator = RankCalculator(config)
    
    # Use sample_h5ad_dict from config (already validated in config.__post_init__)
    logger.info(f"Found {len(config.sample_h5ad_dict)} samples to process")
    
    rank_outputs = rank_calculator.calculate_ranks_and_concatenate(
        sample_h5ad_dict=config.sample_h5ad_dict,
        annotation_key=config.annotation,
        data_layer=config.data_layer,
    )
    
    # Step 2: Calculate marker scores
    logger.info("\n" + "-" * 40)
    logger.info("Step 2: Marker score calculation")
    logger.info("-" * 40)
    
    marker_calculator = MarkerScoreCalculator(config)
    
    marker_scores_path = marker_calculator.calculate_marker_scores(
        adata_path=rank_outputs["concatenated_latent_adata"],
        rank_memmap_path=rank_outputs["rank_memmap"],
        mean_frac_path=rank_outputs["mean_frac"],
        output_path=expected_outputs["marker_scores"]
    )
    
    # Convert config to dict with all Path objects as strings
    config_dict = config.to_dict_with_paths_as_strings()
    
    # Create overall metadata
    metadata = {
        "config": config_dict,
        "outputs": {
            "concatenated_latent_adata": str(rank_outputs["concatenated_latent_adata"]),
            "rank_memmap": str(rank_outputs["rank_memmap"]),
            "mean_frac": str(rank_outputs["mean_frac"]),
            "marker_scores": str(marker_scores_path)
        },
        "n_sections": len(config.sample_h5ad_dict)
    }
    
    # Save overall metadata in YAML format
    with open(expected_outputs["metadata"], 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("Latent to gene conversion complete!")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("=" * 60)
    
    return metadata["outputs"]

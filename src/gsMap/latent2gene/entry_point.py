"""
Main entry point for the latent2gene subpackage
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel

from gsMap.config import DatasetType, LatentToGeneConfig, MarkerScoreCrossSliceStrategy

from .gene_blocks import calculate_gene_block_marker_scores, gene_block_store_matches_inputs
from .marker_scores import MarkerScoreCalculator
from .memmap_io import MemMapDense
from .rank_calculator import (
    RankCalculator,
    rank_outputs_match_inputs,
    rank_provenance_path,
    write_rank_provenance,
)

logger = logging.getLogger(__name__)
console = Console()


def run_latent_to_gene(
    config: LatentToGeneConfig,
    *,
    materialize_marker_scores: bool = True,
) -> dict[str, Any]:
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

    # Setup output directory using config paths
    output_dir = Path(config.latent2gene_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_h5ad_dict = config.sample_h5ad_dict
    if sample_h5ad_dict is None:
        raise ValueError("Latent-to-gene inputs have not been resolved")

    # Check if all outputs already exist using config paths
    expected_outputs = {
        "concatenated_latent_adata": Path(config.concatenated_latent_adata_path),
        "rank_memmap": Path(config.rank_memmap_path),
        "mean_frac": Path(config.mean_frac_path),
        "metadata": Path(config.latent2gene_metadata_path),
        "rank_meta": Path(config.rank_memmap_path).with_suffix(".meta.json"),
        "rank_provenance": rank_provenance_path(config.rank_memmap_path),
    }
    if materialize_marker_scores:
        expected_outputs.update(
            {
                "marker_scores": Path(config.marker_scores_memmap_path),
                "marker_scores_meta": Path(config.marker_scores_memmap_path).with_suffix(
                    ".meta.json"
                ),
            }
        )
    else:
        expected_outputs.update(
            {
                "marker_score_blocks": Path(config.marker_score_blocks_dir),
                "marker_score_blocks_manifest": Path(config.marker_score_blocks_manifest_path),
                "rank_row_min": Path(config.rank_row_min_path),
            }
        )

    if all(Path(p).exists() for p in expected_outputs.values()):
        logger.info("All outputs already exist. Checking completion status...")

        rank_outputs_current = rank_outputs_match_inputs(
            sample_h5ad_dict=sample_h5ad_dict,
            rank_memmap_path=expected_outputs["rank_memmap"],
            mean_frac_path=expected_outputs["mean_frac"],
            concatenated_adata_path=expected_outputs["concatenated_latent_adata"],
            annotation_key=config.annotation,
            data_layer=config.data_layer,
            min_cells_per_type=config.min_cells_per_type,
        )
        if not rank_outputs_current:
            logger.warning("Rank outputs do not match the current latent H5AD inputs")

        if materialize_marker_scores:
            marker_scores_complete, marker_meta = MemMapDense.check_complete(
                expected_outputs["marker_scores"]
            )
            if not marker_scores_complete:
                if marker_meta:
                    logger.warning("Marker scores memmap exists but is not marked as complete")
                else:
                    logger.warning("Could not read marker scores memmap metadata")
        else:
            marker_scores_complete = gene_block_store_matches_inputs(
                expected_outputs["marker_score_blocks_manifest"],
                rank_memmap_path=expected_outputs["rank_memmap"],
                mean_frac_path=expected_outputs["mean_frac"],
                marker_metadata_path=expected_outputs["concatenated_latent_adata"],
            )

        if rank_outputs_current and marker_scores_complete:
            logger.info("All memory maps are properly completed. Loading metadata...")
            with open(expected_outputs["metadata"]) as f:
                existing_metadata = yaml.unsafe_load(f)
            logger.info(
                f"Found existing complete results for {existing_metadata.get('n_cells', 'unknown')} cells "
                f"and {existing_metadata.get('n_genes', 'unknown')} genes"
            )
            return {k: str(v) for k, v in expected_outputs.items()}
        else:
            logger.warning(
                "Memory maps exist but are not properly completed. Re-running pipeline..."
            )

    # Step 1: Calculate ranks and concatenate
    console.print(
        Panel(
            "[bold cyan]Step 1: Rank calculation and concatenation[/bold cyan]",
            border_style="cyan",
        )
    )

    rank_calculator = RankCalculator(config)

    # Use sample_h5ad_dict from config (already validated in config.__post_init__)
    logger.info(f"Found {len(sample_h5ad_dict)} samples to process")

    rank_outputs = rank_calculator.calculate_ranks_and_concatenate(
        sample_h5ad_dict=sample_h5ad_dict,
        annotation_key=config.annotation,
        data_layer=config.data_layer,
    )

    # Step 2: Calculate marker scores
    console.print(
        Panel("[bold cyan]Step 2: Marker score calculation[/bold cyan]", border_style="cyan")
    )

    marker_calculator = MarkerScoreCalculator(config)

    if materialize_marker_scores:
        marker_scores_path = marker_calculator.calculate_marker_scores(
            adata_path=rank_outputs["concatenated_latent_adata"],
            rank_memmap_path=rank_outputs["rank_memmap"],
            mean_frac_path=rank_outputs["mean_frac"],
            output_path=expected_outputs["marker_scores"],
        )
        marker_outputs = {"marker_scores": str(marker_scores_path)}
    else:
        if config.memmap_tmp_dir is None:
            raise ValueError(
                "memmap_tmp_dir is required for gene-block marker scores; "
                "use node-local storage such as /data"
            )
        marker_calculator.prepare_homogeneous_neighbors(
            adata_path=rank_outputs["concatenated_latent_adata"],
            rank_memmap_path=rank_outputs["rank_memmap"],
        )
        # Neighbor preparation intentionally rewrites the concatenated AnnData.
        # Persist that valid enriched signature before the first gene block so
        # an interrupted block run resumes instead of rebuilding all ranks.
        write_rank_provenance(
            sample_h5ad_dict=sample_h5ad_dict,
            rank_memmap_path=rank_outputs["rank_memmap"],
            mean_frac_path=rank_outputs["mean_frac"],
            concatenated_adata_path=rank_outputs["concatenated_latent_adata"],
            annotation_key=config.annotation,
            data_layer=config.data_layer,
            min_cells_per_type=config.min_cells_per_type,
        )
        pooled_by_slice = (
            config.dataset_type == DatasetType.SPATIAL_3D
            and config.cross_slice_marker_score_strategy
            in {
                MarkerScoreCrossSliceStrategy.PER_SLICE_POOL,
                MarkerScoreCrossSliceStrategy.HIERARCHICAL_POOL,
            }
        )
        n_slices = 1 + 2 * config.n_adjacent_slices if pooled_by_slice else 1
        manifest_path = calculate_gene_block_marker_scores(
            rank_memmap_path=rank_outputs["rank_memmap"],
            rank_row_min_path=config.rank_row_min_path,
            mean_frac_path=rank_outputs["mean_frac"],
            neighbor_adata_path=rank_outputs["concatenated_latent_adata"],
            output_dir=config.marker_score_blocks_dir,
            manifest_path=config.marker_score_blocks_manifest_path,
            gene_block_size=config.marker_score_gene_block_size,
            spot_batch_size=config.mkscore_batch_size,
            n_slices=n_slices,
            homogeneous_neighbors=config.homogeneous_neighbors,
            hierarchical_pool=(
                config.cross_slice_marker_score_strategy
                == MarkerScoreCrossSliceStrategy.HIERARCHICAL_POOL
            ),
            no_expression_fraction=config.no_expression_fraction,
            tmp_dir=config.memmap_tmp_dir,
        )
        marker_outputs = {
            "marker_score_blocks": str(config.marker_score_blocks_dir),
            "marker_score_blocks_manifest": str(manifest_path),
            "rank_row_min": str(config.rank_row_min_path),
        }

    # Homogeneous-neighbor preparation intentionally enriches the concatenated
    # AnnData in place. Refresh the rank-output signature only after that write
    # and after marker-score calculation completes successfully.
    write_rank_provenance(
        sample_h5ad_dict=sample_h5ad_dict,
        rank_memmap_path=rank_outputs["rank_memmap"],
        mean_frac_path=rank_outputs["mean_frac"],
        concatenated_adata_path=rank_outputs["concatenated_latent_adata"],
        annotation_key=config.annotation,
        data_layer=config.data_layer,
        min_cells_per_type=config.min_cells_per_type,
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
            **marker_outputs,
        },
        "n_sections": len(sample_h5ad_dict),
    }

    # Save overall metadata in YAML format
    with open(expected_outputs["metadata"], "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    logger.info(f"All outputs saved to: {output_dir}")

    return metadata["outputs"]

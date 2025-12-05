"""
JAX-optimized implementation of spatial LDSC.
"""

import gc
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Tuple, List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import psutil
from jax import jit, vmap
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

import anndata as ad
from ..config import SpatialLDSCConfig
from ..latent2gene.memmap_io import MemMapDense
from ..utils.regression_read import _read_ref_ld_v2, _read_sumstats, _read_w_ld
from .ldscore_quick_mode import SpatialLDSCProcessor

logger = logging.getLogger("gsMap.spatial_ldsc_jax")

# Configure JAX for optimal performance and memory efficiency
jax.config.update('jax_enable_x64', False)  # Use float32 for speed and memory efficiency

# Platform selection - comment/uncomment as needed
# jax.config.update('jax_platform_name', 'cpu')  # Force CPU usage
# jax.config.update('jax_platform_name', 'gpu')  # Force GPU usage

# Memory configuration for environments with limited resources
# os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')
# ============================================================================
# Memory monitoring
# ============================================================================

def log_memory_usage(message=""):
    """Log current memory usage."""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        rss_gb = mem_info.rss / 1024**3
        logger.debug(f"Memory usage {message}: {rss_gb:.2f} GB")
        return rss_gb
    except:
        return 0.0


# ============================================================================
# Core computational functions
# ============================================================================

def prepare_snp_data_for_blocks(data: dict, n_blocks: int) -> dict:
    """Prepare SNP-related data arrays for equal-sized blocks."""
    if 'chisq' in data:
        n_snps = len(data['chisq'])
    elif 'N' in data:
        n_snps = len(data['N'])
    else:
        raise ValueError("Cannot determine number of SNPs from data")
    
    block_size = n_snps // n_blocks
    n_snps_used = block_size * n_blocks
    n_dropped = n_snps - n_snps_used
    
    if n_dropped > 0:
        logger.info(f"Truncating SNP data: dropping {n_dropped} SNPs "
                   f"({n_dropped/n_snps*100:.3f}%) for {n_blocks} blocks of size {block_size}")
    
    truncated = {}
    snp_keys = ['baseline_ld_sum', 'w_ld', 'chisq', 'N']
    
    for key, value in data.items():
        if key in snp_keys and isinstance(value, (np.ndarray, jnp.ndarray)):
            truncated[key] = value[:n_snps_used]
        elif key == 'baseline_ld':
            truncated[key] = value.iloc[:n_snps_used]
        else:
            truncated[key] = value
    
    truncated['block_size'] = block_size
    truncated['n_blocks'] = n_blocks
    truncated['n_snps_used'] = n_snps_used
    truncated['n_snps_original'] = n_snps
    
    return truncated

@jax.profiler.annotate_function
@partial(jit, static_argnums=(0, 1))
def process_chunk_jit(n_blocks: int,
                      batch_size: int,
                      spatial_ld: jnp.ndarray,
                      baseline_ld_sum: jnp.ndarray, 
                      chisq: jnp.ndarray,
                      N: jnp.ndarray,
                      baseline_ann: jnp.ndarray,
                      w_ld: jnp.ndarray,
                      Nbar: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Process an entire chunk of spots with JIT compilation and batch processing.
    Processes spots in batches to reduce memory usage.
    """
    def process_single_spot(spot_ld):
        """Process a single spot."""
        # Compute initial weights
        with jax.profiler.StepTraceAnnotation("weight_computation"):
            x_tot = spot_ld + baseline_ld_sum
            
            # Aggregate for weight calculation
            hsq = 10000.0 * (jnp.mean(chisq) - 1.0) / jnp.mean(x_tot * N)
            hsq = jnp.clip(hsq, 0.0, 1.0)
            
            # Compute weights efficiently
            ld_clip = jnp.maximum(x_tot, 1.0)
            w_ld_clip = jnp.maximum(w_ld, 1.0)
            c = hsq * N / 10000.0
            weights = jnp.sqrt(1.0 / (2 * jnp.square(1.0 + c * ld_clip) * w_ld_clip))
            
            # Scale weights
            weights = weights.reshape(-1, 1)
            weights_scaled = weights / jnp.sum(weights)
        
        # Apply weights and combine features
        with jax.profiler.StepTraceAnnotation("feature_preparation"):
            x_focal = jnp.concatenate([
                (spot_ld.reshape(-1, 1) * weights_scaled),
                (baseline_ann * weights_scaled)
            ], axis=1)
            y_weighted = chisq.reshape(-1, 1) * weights_scaled
            
            # Reshape for block computation
            n_snps_used = x_focal.shape[0]
            block_size = n_snps_used // n_blocks
            
            x_blocks = x_focal.reshape(n_blocks, block_size, -1)
            y_blocks = y_weighted.reshape(n_blocks, block_size, -1)
        
        # Compute block values
        with jax.profiler.StepTraceAnnotation("block_computation"):
            xty_blocks = jnp.einsum('nbp,nb->np', x_blocks, y_blocks.squeeze())
            xtx_blocks = jnp.einsum('nbp,nbq->npq', x_blocks, x_blocks)
        
        # Jackknife regression
        with jax.profiler.StepTraceAnnotation("jackknife_regression"):
            xty_total = jnp.sum(xty_blocks, axis=0)
            xtx_total = jnp.sum(xtx_blocks, axis=0)
            est = jnp.linalg.solve(xtx_total, xty_total)
            
            # Delete-one estimates using vectorized solve
            xty_del = xty_total - xty_blocks
            xtx_del = xtx_total - xtx_blocks
            delete_ests = jnp.linalg.solve(xtx_del, xty_del[..., None]).squeeze(-1)
            
            # Pseudovalues and standard error
            pseudovalues = n_blocks * est - (n_blocks - 1) * delete_ests
            jknife_est = jnp.mean(pseudovalues, axis=0)
            jknife_cov = jnp.cov(pseudovalues.T, ddof=1) / n_blocks
            jknife_se = jnp.sqrt(jnp.diag(jknife_cov))
        
        # Return spatial coefficient (first element)
        return jknife_est[0] / Nbar, jknife_se[0] / Nbar
    
    # Process in batches to reduce memory usage
    n_spots = spatial_ld.shape[1]
    
    if batch_size == 0 or batch_size >= n_spots:
        # Process all spots at once (batch_size=0 means no batching)
        with jax.profiler.StepTraceAnnotation("vmap_all_spots"):
            betas, ses = vmap(process_single_spot, in_axes=1, out_axes=0)(spatial_ld)
    else:
        # Process in smaller batches
        betas_list = []
        ses_list = []
        
        with jax.profiler.StepTraceAnnotation("batch_processing"):
            for start_idx in range(0, n_spots, batch_size):
                end_idx = min(start_idx + batch_size, n_spots)
                batch_ld = spatial_ld[:, start_idx:end_idx]
                
                with jax.profiler.StepTraceAnnotation(f"vmap_batch_{start_idx}_{end_idx}"):
                    batch_betas, batch_ses = vmap(process_single_spot, in_axes=1, out_axes=0)(batch_ld)
                betas_list.append(batch_betas)
                ses_list.append(batch_ses)
        
        with jax.profiler.StepTraceAnnotation("concatenate_results"):
            betas = jnp.concatenate(betas_list)
            ses = jnp.concatenate(ses_list)
    
    return betas, ses


@partial(jit, static_argnums=(0,))
def process_chunk_batched_jit(n_blocks: int,
                               spatial_ld: jnp.ndarray,
                               baseline_ld_sum: jnp.ndarray,
                               chisq: jnp.ndarray,
                               N: jnp.ndarray,
                               baseline_ann: jnp.ndarray,
                               w_ld: jnp.ndarray,
                               Nbar: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Process an entire chunk of spots with JIT compilation and BATCHED matrix operations.

    OPTIMIZATION: Uses batched matrix operations instead of vmap to improve GPU utilization.
    All spots are processed simultaneously using efficient matrix operations.

    Args:
        n_blocks: Number of jackknife blocks
        spatial_ld: (n_snps, n_spots) array of spatial LD scores
        baseline_ld_sum: (n_snps,) baseline LD scores summed
        chisq: (n_snps,) chi-squared statistics
        N: (n_snps,) sample sizes
        baseline_ann: (n_snps, n_baseline_features) baseline annotations
        w_ld: (n_snps,) regression weights
        Nbar: Average sample size

    Returns:
        betas: (n_spots,) regression coefficients
        ses: (n_spots,) standard errors
    """
    n_snps, n_spots = spatial_ld.shape
    n_baseline_features = baseline_ann.shape[1]

    # Compute x_tot for all spots: (n_snps, n_spots)
    x_tot = spatial_ld + baseline_ld_sum.reshape(-1, 1)

    # Compute hsq for each spot: (n_spots,)
    # hsq = 10000 * (mean(chisq) - 1) / mean(x_tot * N)
    N_expanded = N.reshape(-1, 1)  # (n_snps, 1)
    x_tot_N = x_tot * N_expanded  # (n_snps, n_spots)
    mean_chisq = jnp.mean(chisq)
    mean_x_tot_N = jnp.mean(x_tot_N, axis=0)  # (n_spots,)
    hsq = 10000.0 * (mean_chisq - 1.0) / mean_x_tot_N  # (n_spots,)
    hsq = jnp.clip(hsq, 0.0, 1.0)

    # Compute weights for all spots: (n_snps, n_spots)
    ld_clip = jnp.maximum(x_tot, 1.0)
    w_ld_clip = jnp.maximum(w_ld.reshape(-1, 1), 1.0)
    c = (hsq.reshape(1, -1) * N_expanded) / 10000.0  # (n_snps, n_spots)
    weights = jnp.sqrt(1.0 / (2 * jnp.square(1.0 + c * ld_clip) * w_ld_clip))

    # Normalize weights per spot
    weights_sum = jnp.sum(weights, axis=0, keepdims=True)  # (1, n_spots)
    weights_scaled = weights / weights_sum  # (n_snps, n_spots)

    # Prepare features for all spots
    # x_focal shape: (n_snps, n_spots, 1 + n_baseline_features)
    spatial_weighted = (spatial_ld * weights_scaled)[..., None]  # (n_snps, n_spots, 1)
    baseline_weighted = baseline_ann[:, None, :] * weights_scaled[..., None]  # (n_snps, n_spots, n_baseline)
    x_focal = jnp.concatenate([spatial_weighted, baseline_weighted], axis=2)

    # y_weighted: (n_snps, n_spots, 1)
    y_weighted = (chisq.reshape(-1, 1) * weights_scaled)[..., None]

    # Reshape for block computation
    block_size = n_snps // n_blocks
    n_snps_used = block_size * n_blocks

    # Truncate to block-aligned size
    x_focal = x_focal[:n_snps_used]
    y_weighted = y_weighted[:n_snps_used]

    # Reshape: (n_blocks, block_size, n_spots, n_features)
    x_blocks = x_focal.reshape(n_blocks, block_size, n_spots, -1)
    y_blocks = y_weighted.reshape(n_blocks, block_size, n_spots, 1)

    # Compute block XtY and XtX for all spots simultaneously
    # xty_blocks: (n_blocks, n_spots, n_features)
    xty_blocks = jnp.einsum('nbsf,nbs->nsf', x_blocks, y_blocks.squeeze(-1))

    # xtx_blocks: (n_blocks, n_spots, n_features, n_features)
    xtx_blocks = jnp.einsum('nbsf,nbsg->nsfg', x_blocks, x_blocks)

    # Total across blocks
    xty_total = jnp.sum(xty_blocks, axis=0)  # (n_spots, n_features)
    xtx_total = jnp.sum(xtx_blocks, axis=0)  # (n_spots, n_features, n_features)

    # Solve for all spots: (n_spots, n_features)
    est = jnp.linalg.solve(xtx_total, xty_total[..., None]).squeeze(-1)

    # Delete-one estimates: (n_blocks, n_spots, n_features)
    xty_del = xty_total - xty_blocks  # (n_blocks, n_spots, n_features)
    xtx_del = xtx_total - xtx_blocks  # (n_blocks, n_spots, n_features, n_features)
    delete_ests = jnp.linalg.solve(xtx_del, xty_del[..., None]).squeeze(-1)

    # Pseudovalues: (n_blocks, n_spots, n_features)
    pseudovalues = n_blocks * est - (n_blocks - 1) * delete_ests

    # Jackknife estimates per spot
    jknife_est = jnp.mean(pseudovalues, axis=0)  # (n_spots, n_features)

    # Jackknife covariance for each spot
    # Center pseudovalues
    pseudo_centered = pseudovalues - jknife_est  # broadcast (n_blocks, n_spots, n_features)

    # Covariance: (n_spots, n_features, n_features)
    jknife_cov = jnp.einsum('nsf,nsg->sfg', pseudo_centered, pseudo_centered) / (n_blocks * (n_blocks - 1))

    # Extract diagonal for SE: (n_spots, n_features)
    jknife_se = jnp.sqrt(jnp.diagonal(jknife_cov, axis1=1, axis2=2))

    # Return spatial coefficient (first feature) for all spots
    return jknife_est[:, 0] / Nbar, jknife_se[:, 0] / Nbar



# ============================================================================
# Data loading and preparation
# ============================================================================

def load_and_prepare_data(config: SpatialLDSCConfig, 
                         trait_name: str,
                         sumstats_file: str) -> Tuple[dict, pd.Index]:
    """Load and prepare all data for a single trait."""
    logger.info(f"Loading data for {trait_name}...")
    
    log_memory_usage("before loading data")
    
    # Load weights and baseline LD scores
    w_ld = _read_w_ld(config.w_file)
    w_ld.set_index("SNP", inplace=True)
    
    # Use ldscore_save_dir which is set to quick_mode_resource_dir when in quick mode
    baseline_ld_path = f"{config.ldscore_save_dir}/baseline/baseline."
    baseline_ld = _read_ref_ld_v2(baseline_ld_path)
    
    log_memory_usage("after loading baseline")
    
    # Find common SNPs
    common_snps = baseline_ld.index.intersection(w_ld.index)
    
    # Load and process summary statistics
    sumstats = _read_sumstats(fh=sumstats_file, alleles=False, dropna=False)
    sumstats.set_index("SNP", inplace=True)
    sumstats = sumstats.astype(np.float32)
    
    # Filter by chi-squared
    chisq_max = config.chisq_max
    if chisq_max is None:
        chisq_max = max(0.001 * sumstats.N.max(), 80)
    sumstats["chisq"] = sumstats.Z ** 2

    # Calculate genomic control lambda (λGC) before filtering
    lambda_gc = np.median(sumstats.chisq) / 0.4559364
    logger.info(f"Lambda GC (genomic control λ): {lambda_gc:.4f}")

    sumstats = sumstats[sumstats.chisq < chisq_max]
    logger.info(f"Filtered to {len(sumstats)} SNPs with chi^2 < {chisq_max}")
    
    # Find common SNPs with sumstats
    common_snps = common_snps.intersection(sumstats.index)
    logger.info(f"Common SNPs: {len(common_snps)}")
    
    if len(common_snps) < 200000:
        logger.warning(f"WARNING: Only {len(common_snps)} common SNPs")
    
    # Get SNP positions
    snp_positions = baseline_ld.index.get_indexer(common_snps)
    
    # Subset all data to common SNPs
    baseline_ld = baseline_ld.loc[common_snps]
    w_ld = w_ld.loc[common_snps]
    sumstats = sumstats.loc[common_snps]
    
    # Load additional baseline if needed
    if config.use_additional_baseline_annotation:
        # Use ldscore_save_dir which points to the correct directory
        additional_path = f"{config.ldscore_save_dir}/additional_baseline/baseline."
        additional_ld = _read_ref_ld_v2(additional_path)
        additional_ld = additional_ld.loc[common_snps]
        baseline_ld = pd.concat([baseline_ld, additional_ld], axis=1)
    
    # Prepare data dictionary
    data = {
        'baseline_ld': baseline_ld,
        'baseline_ld_sum': baseline_ld.sum(axis=1).values.astype(np.float32),
        'w_ld': w_ld.LD_weights.values.astype(np.float32),
        'sumstats': sumstats,
        'chisq': sumstats.chisq.values.astype(np.float32),
        'N': sumstats.N.values.astype(np.float32),
        'Nbar': np.float32(sumstats.N.mean()),
        'snp_positions': snp_positions
    }
    
    return data, common_snps


def wrapper_of_process_chunk_jit(*args, **kwargs):
    """Wrapper to call the JIT-compiled process_chunk_jit function."""
    # return process_chunk_jit(*args, **kwargs)
    return process_chunk_batched_jit(*args, **kwargs)

def load_marker_scores_memmap_format(config: SpatialLDSCConfig) -> ad.AnnData:
    """
    Load marker scores memmap and wrap it in an AnnData object with metadata
    from a reference h5ad file using configuration.

    Args:
        config: SpatialLDSCConfig containing paths and settings

    Returns:
        AnnData object with X backed by the memory map
    """
    memmap_path = Path(config.marker_scores_memmap_path)
    metadata_path = Path(config.concatenated_latent_adata_path)
    tmp_dir = config.memmap_tmp_dir
    mode = 'r'  # Read-only mode for loading

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # check complete
    is_complete, _ = MemMapDense.check_complete(memmap_path)
    if not is_complete:
        raise ValueError(f"Marker score at {memmap_path} is incomplete or corrupted. Please recompute.")

    # Load metadata source in backed mode
    logger.info(f"Loading metadata from {metadata_path}")
    src_adata = ad.read_h5ad(metadata_path, backed='r')

    # Determine shape from metadata
    shape = (src_adata.n_obs, src_adata.n_vars)

    # Initialize MemMapDense
    mm = MemMapDense(
        memmap_path,
        shape=shape,
        mode=mode,
        tmp_dir=tmp_dir
    )

    logger.info("Constructing AnnData wrapper...")
    adata = ad.AnnData(
        X=mm.memmap,
        obs=src_adata.obs.copy(),
        var=src_adata.var.copy(),
        uns=src_adata.uns.copy(),
        obsm=src_adata.obsm.copy(),
        varm=src_adata.varm.copy()
    )

    # Attach the manager to allow access to MemMapDense methods
    adata.uns['memmap_manager'] = mm

    return adata


def generate_expected_output_filename(config: SpatialLDSCConfig, trait_name: str) -> str:

    base_name = f"{config.project_name}_{trait_name}"

    # If we have cell indices range, include it in filename
    if config.cell_indices_range:
        start_cell, end_cell = config.cell_indices_range
        return f"{base_name}_cells_{start_cell}_{end_cell}.csv.gz"

    # If sample filter is set, filename will include sample info
    # but we can't predict exact start/end without loading data
    # For now, just check the simple complete case
    if config.sample_filter:
        # Return None to indicate we can't reliably predict the filename
        # and should proceed with processing
        return None

    # Default case: complete coverage
    return f"{base_name}.csv.gz"


def log_existing_result_statistics(result_path: Path, trait_name: str):

    try:
        # Read the existing result
        logger.info(f"Reading existing result from: {result_path}")
        df = pd.read_csv(result_path, compression='gzip')

        n_spots = len(df)
        bonferroni_threshold = 0.05 / n_spots
        n_bonferroni_sig = (df['p'] < bonferroni_threshold).sum()

        # FDR correction
        reject, _, _, _ = multipletests(
            df['p'], alpha=0.001, method='fdr_bh'
        )
        n_fdr_sig = reject.sum()

        logger.info("=" * 70)
        logger.info(f"EXISTING RESULT SUMMARY - {trait_name}")
        logger.info("=" * 70)
        logger.info(f"Total spots: {n_spots:,}")
        logger.info(f"Max -log10(p): {df['neg_log10_p'].max():.2f}")
        logger.info("-" * 70)
        logger.info(f"Nominally significant (p < 0.05): {(df['p'] < 0.05).sum():,}")
        logger.info(f"Bonferroni threshold: {bonferroni_threshold:.2e}")
        logger.info(f"Bonferroni significant: {n_bonferroni_sig:,}")
        logger.info(f"FDR significant (alpha=0.001): {n_fdr_sig:,}")
        logger.info("=" * 70)

    except Exception as e:
        logger.warning(f"Could not read existing result statistics: {e}")


# ============================================================================
# Main entry point
# ============================================================================

def run_spatial_ldsc_jax(config: SpatialLDSCConfig):
    """
    Run spatial LDSC for all traits in config.sumstats_config_dict.
    """
    if config.marker_score_format not in ["memmap", "h5ad"]:
        raise NotImplementedError(f"Marker score format '{config.marker_score_format}' is not supported. Only 'memmap' and 'h5ad' are supported.")

    traits_to_process = list(config.sumstats_config_dict.items())
    if not traits_to_process:
        raise ValueError("No traits to process. config.sumstats_config_dict is empty.")

    # Create output directory
    output_dir = config.ldsc_save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of loader threads based on platform
    n_loader_threads = 10 if jax.default_backend() == 'gpu' else 2

    # Load marker scores once (format-agnostic)
    logger.info(f"Loading marker scores (format: {config.marker_score_format})...")
    marker_score_adata = None
    
    try:
        if config.marker_score_format == "memmap":
            marker_score_adata = load_marker_scores_memmap_format(config)
        elif config.marker_score_format == "h5ad":
            if not config.marker_score_h5ad_path:
                raise ValueError("marker_score_h5ad_path must be provided when marker_score_format is 'h5ad'")
            
            h5ad_path = Path(config.marker_score_h5ad_path)
            if not h5ad_path.exists():
                raise FileNotFoundError(f"Marker score H5AD file not found: {h5ad_path}")
            
            logger.info(f"Loading marker scores from H5AD: {h5ad_path}")
            marker_score_adata = ad.read_h5ad(h5ad_path, backed='r')
        
        processor = None

        try:
            for idx, (trait_name, sumstats_file) in enumerate(traits_to_process):
                logger.info("=" * 70)
                logger.info(f"Running Spatial LDSC (JAX Implementation)")
                logger.info(f"Project: {config.project_name}, Trait: {trait_name} ({idx+1}/{len(traits_to_process)})")
                if config.sample_filter:
                    logger.info(f"Sample filter: {config.sample_filter}")
                if config.cell_indices_range:
                    logger.info(f"Cell indices range: {config.cell_indices_range}")
                logger.info("=" * 70)

                # Check if output already exists
                expected_filename = generate_expected_output_filename(config, trait_name)
                if expected_filename is not None:
                    expected_output_path = output_dir / expected_filename
                    if expected_output_path.exists():
                        logger.info(f"Output file already exists: {expected_output_path}")
                        logger.info(f"Skipping trait {trait_name} ({idx+1}/{len(traits_to_process)})")

                        # Log statistics from existing result
                        log_existing_result_statistics(expected_output_path, trait_name)
                        continue

                # Load and prepare trait-specific data
                data, common_snps = load_and_prepare_data(config, trait_name, sumstats_file)
                data_truncated = prepare_snp_data_for_blocks(data, config.n_blocks)

                if processor is None:
                    # First trait: create processor
                    logger.debug("Initializing processor...")
                    processor = SpatialLDSCProcessor(
                        config=config,
                        trait_name=trait_name,
                        data_truncated=data_truncated,
                        output_dir=output_dir,
                        marker_score_adata=marker_score_adata,
                        n_loader_threads=n_loader_threads
                    )
                else:
                    # Subsequent traits: reset state while keeping memmap/adata loaded
                    logger.debug("Reusing processor...")
                    processor.reset_for_new_trait(trait_name, data_truncated, output_dir)

                # Process all chunks for current trait
                start_time = time.time()
                processor.process_all_chunks(wrapper_of_process_chunk_jit)

                elapsed_time = time.time() - start_time
                h, rem = divmod(elapsed_time, 3600)
                m, s = divmod(rem, 60)
                logger.info(f"Trait {trait_name} completed in {int(h)}h {int(m)}m {s:.2f}s")

        finally:
            # Cleanup once: close memmap/adata if needed
            if marker_score_adata is not None:
                logger.info("Closing marker score resources...")
                # If it's our MemMap wrapper, close it explicitly
                if 'memmap_manager' in marker_score_adata.uns:
                    marker_score_adata.uns['memmap_manager'].close()
                # If it's backed AnnData, close the file
                if marker_score_adata.isbacked:
                    marker_score_adata.file.close()

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise


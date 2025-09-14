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
from typing import Tuple, List

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import psutil
from jax import jit, vmap
from scipy.stats import norm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

import anndata as ad
from ..config import SpatialLDSCConfig
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


# ============================================================================
# Main entry point
# ============================================================================

def run_spatial_ldsc_single_trait(config: SpatialLDSCConfig,
                                 trait_name: str,
                                 sumstats_file: str) -> pd.DataFrame:
    """
    Run spatial LDSC for a single trait using the unified processor.
    
    Args:
        config: Configuration object
        trait_name: Name of the trait
        sumstats_file: Path to summary statistics file
    
    Returns:
        Merged DataFrame with results
    """
    logger.info("=" * 70)
    logger.info(f"Running Spatial LDSC (Unified Processor Version)")
    logger.info(f"Project: {config.project_name}, Trait: {trait_name}")
    if config.sample_filter:
        logger.info(f"Filtering by sample: {config.sample_filter}")
    if config.cell_indices_range:
        logger.info(f"Cell indices range: {config.cell_indices_range}")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = config.ldsc_save_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    data, common_snps = load_and_prepare_data(config, trait_name, sumstats_file)
    data_truncated = prepare_snp_data_for_blocks(data, config.n_blocks)
    
    if config.marker_score_format == "memmap":
        # Import the unified processor

        # Determine number of loader threads based on platform
        if jax.default_backend() == 'gpu':
            n_loader_threads = 10
        else:
            n_loader_threads = 2
        
        # Create processor instance
        processor = SpatialLDSCProcessor(
            config=config,
            trait_name=trait_name,
            data_truncated=data_truncated,
            output_dir=output_dir,
            n_loader_threads=n_loader_threads
        )
        
        # Process all chunks
        start_time = time.time()
        try:
            merged_df = processor.process_all_chunks(process_chunk_jit)
        finally:
            # Clean up resources
            processor.cleanup()
        
        elapsed_time = time.time() - start_time
        h, rem = divmod(elapsed_time, 3600)
        m, s = divmod(rem, 60)
        logger.info(f"Processing completed in {int(h)}h {int(m)}m {s:.2f}s")
        
        return merged_df

    else:
        # not implemented yet for feather mode
        raise NotImplementedError("Feather mode is not implemented yet.")

def run_spatial_ldsc_jax(config: SpatialLDSCConfig):
    """
    Wrapper for compatibility with existing code.
    Processes all traits from config.sumstats_config_dict.
    """
    for trait_name, sumstats_file in config.sumstats_config_dict.items():
        run_spatial_ldsc_single_trait(config, trait_name, sumstats_file)


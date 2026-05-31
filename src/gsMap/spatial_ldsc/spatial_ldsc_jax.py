"""
JAX-optimized implementation of spatial LDSC.
"""

import logging
import time
from functools import partial
from pathlib import Path

import anndata as ad
import jax
import jax.numpy as jnp
from jax import jit, vmap

from gsMap.config import SpatialLDSCConfig

from .io import (
    FeatherAnnData,
    generate_expected_output_filename,
    load_common_resources,
    load_common_resources_direct_ldscore,
    load_marker_scores_memmap_format,
    log_existing_result_statistics,
)
from .ldscore_quick_mode import SpatialLDSCProcessor

logger = logging.getLogger("gsMap.spatial_ldsc_jax")

# Configure JAX for optimal performance and memory efficiency
jax.config.update("jax_enable_x64", False)  # Use float32 for speed and memory efficiency

# Platform selection - comment/uncomment as needed
# jax.config.update('jax_platform_name', 'cpu')  # Force CPU usage
# jax.config.update('jax_platform_name', 'gpu')  # Force GPU usage

# Memory configuration for environments with limited resources
# os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')

# ============================================================================
# Core computational functions
# ============================================================================


@jax.profiler.annotate_function
@partial(jit, static_argnums=(0, 1))
def process_chunk_jit(
    n_blocks: int,
    batch_size: int,
    spatial_ld: jnp.ndarray,
    baseline_ld_sum: jnp.ndarray,
    chisq: jnp.ndarray,
    N: jnp.ndarray,
    baseline_ann: jnp.ndarray,
    w_ld: jnp.ndarray,
    Nbar: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
            x_focal = jnp.concatenate(
                [(spot_ld.reshape(-1, 1) * weights_scaled), (baseline_ann * weights_scaled)],
                axis=1,
            )
            y_weighted = chisq.reshape(-1, 1) * weights_scaled

            # Reshape for block computation
            n_snps_used = x_focal.shape[0]
            block_size = n_snps_used // n_blocks

            x_blocks = x_focal.reshape(n_blocks, block_size, -1)
            y_blocks = y_weighted.reshape(n_blocks, block_size, -1)

        # Compute block values
        with jax.profiler.StepTraceAnnotation("block_computation"):
            xty_blocks = jnp.einsum("nbp,nb->np", x_blocks, y_blocks.squeeze())
            xtx_blocks = jnp.einsum("nbp,nbq->npq", x_blocks, x_blocks)

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
                    batch_betas, batch_ses = vmap(process_single_spot, in_axes=1, out_axes=0)(
                        batch_ld
                    )
                betas_list.append(batch_betas)
                ses_list.append(batch_ses)

        with jax.profiler.StepTraceAnnotation("concatenate_results"):
            betas = jnp.concatenate(betas_list)
            ses = jnp.concatenate(ses_list)

    return betas, ses


@partial(jit, static_argnums=(0,))
def process_chunk_batched_lowmem_jit(
    n_blocks: int,
    spatial_ld: jnp.ndarray,
    baseline_ld_sum: jnp.ndarray,
    chisq: jnp.ndarray,
    N: jnp.ndarray,
    baseline_ann: jnp.ndarray,
    w_ld: jnp.ndarray,
    Nbar: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Process an entire chunk with batched normal-equation accumulation.

    This is algebraically equivalent to constructing the weighted design tensor
    [spatial_ld, baseline_ann] * weights for every spot, but avoids materializing
    the large (n_snps, n_spots, n_features) tensor.
    """
    n_snps, n_spots = spatial_ld.shape
    n_baseline = baseline_ann.shape[1]

    x_tot = spatial_ld + baseline_ld_sum.reshape(-1, 1)

    N_expanded = N.reshape(-1, 1)
    mean_chisq = jnp.mean(chisq)
    mean_x_tot_N = jnp.mean(x_tot * N_expanded, axis=0)
    hsq = 10000.0 * (mean_chisq - 1.0) / mean_x_tot_N
    hsq = jnp.clip(hsq, 0.0, 1.0)

    ld_clip = jnp.maximum(x_tot, 1.0)
    w_ld_clip = jnp.maximum(w_ld.reshape(-1, 1), 1.0)
    c = (hsq.reshape(1, -1) * N_expanded) / 10000.0
    weights = jnp.sqrt(1.0 / (2 * jnp.square(1.0 + c * ld_clip) * w_ld_clip))
    weights_scaled = weights / jnp.sum(weights, axis=0, keepdims=True)
    weights_sq = jnp.square(weights_scaled)

    block_size = n_snps // n_blocks
    n_snps_used = block_size * n_blocks

    spatial_blocks = spatial_ld[:n_snps_used].reshape(n_blocks, block_size, n_spots)
    baseline_blocks = baseline_ann[:n_snps_used].reshape(n_blocks, block_size, n_baseline)
    chisq_blocks = chisq[:n_snps_used].reshape(n_blocks, block_size)
    weights_sq_blocks = weights_sq[:n_snps_used].reshape(n_blocks, block_size, n_spots)

    # Weighted normal equations for each jackknife block and spot.
    # Feature 0 is the spatial LD score; remaining features are baseline terms.
    spatial_xty = jnp.einsum(
        "nbs,nb,nbs->ns", spatial_blocks, chisq_blocks, weights_sq_blocks
    )
    baseline_xty = jnp.einsum(
        "nbk,nb,nbs->nsk", baseline_blocks, chisq_blocks, weights_sq_blocks
    )
    xty_blocks = jnp.concatenate([spatial_xty[..., None], baseline_xty], axis=-1)

    spatial_spatial = jnp.einsum(
        "nbs,nbs,nbs->ns", spatial_blocks, spatial_blocks, weights_sq_blocks
    )
    spatial_baseline = jnp.einsum(
        "nbs,nbk,nbs->nsk", spatial_blocks, baseline_blocks, weights_sq_blocks
    )
    baseline_baseline = jnp.einsum(
        "nbk,nbl,nbs->nskl", baseline_blocks, baseline_blocks, weights_sq_blocks
    )

    top_row = jnp.concatenate(
        [spatial_spatial[..., None, None], spatial_baseline[..., None, :]], axis=-1
    )
    bottom_rows = jnp.concatenate([spatial_baseline[..., :, None], baseline_baseline], axis=-1)
    xtx_blocks = jnp.concatenate([top_row, bottom_rows], axis=-2)

    xty_total = jnp.sum(xty_blocks, axis=0)
    xtx_total = jnp.sum(xtx_blocks, axis=0)

    est = jnp.linalg.solve(xtx_total, xty_total[..., None]).squeeze(-1)

    xty_del = xty_total - xty_blocks
    xtx_del = xtx_total - xtx_blocks
    delete_ests = jnp.linalg.solve(xtx_del, xty_del[..., None]).squeeze(-1)

    spatial_pseudovalues = n_blocks * est[:, 0] - (n_blocks - 1) * delete_ests[:, :, 0]
    spatial_jknife_est = jnp.mean(spatial_pseudovalues, axis=0)
    spatial_centered = spatial_pseudovalues - spatial_jknife_est
    spatial_jknife_se = jnp.sqrt(
        jnp.sum(jnp.square(spatial_centered), axis=0) / (n_blocks * (n_blocks - 1))
    )

    return spatial_jknife_est / Nbar, spatial_jknife_se / Nbar


@partial(jit, static_argnums=(0,))
def process_chunk_batched_fullsolve_jit(
    n_blocks: int,
    spatial_ld: jnp.ndarray,
    baseline_ld_sum: jnp.ndarray,
    chisq: jnp.ndarray,
    N: jnp.ndarray,
    baseline_ann: jnp.ndarray,
    w_ld: jnp.ndarray,
    Nbar: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Process an entire chunk with batched matrix operations.

    This kernel materializes the weighted design tensor. It is fastest on GPUs
    when the chunk fits comfortably in device memory because XLA can fuse the
    tensor construction with the block normal-equation reductions.
    """
    n_snps, n_spots = spatial_ld.shape

    x_tot = spatial_ld + baseline_ld_sum.reshape(-1, 1)

    N_expanded = N.reshape(-1, 1)
    x_tot_N = x_tot * N_expanded
    mean_chisq = jnp.mean(chisq)
    mean_x_tot_N = jnp.mean(x_tot_N, axis=0)
    hsq = 10000.0 * (mean_chisq - 1.0) / mean_x_tot_N
    hsq = jnp.clip(hsq, 0.0, 1.0)

    ld_clip = jnp.maximum(x_tot, 1.0)
    w_ld_clip = jnp.maximum(w_ld.reshape(-1, 1), 1.0)
    c = (hsq.reshape(1, -1) * N_expanded) / 10000.0
    weights = jnp.sqrt(1.0 / (2 * jnp.square(1.0 + c * ld_clip) * w_ld_clip))

    weights_sum = jnp.sum(weights, axis=0, keepdims=True)
    weights_scaled = weights / weights_sum

    spatial_weighted = (spatial_ld * weights_scaled)[..., None]
    baseline_weighted = baseline_ann[:, None, :] * weights_scaled[..., None]
    x_focal = jnp.concatenate([spatial_weighted, baseline_weighted], axis=2)

    y_weighted = (chisq.reshape(-1, 1) * weights_scaled)[..., None]

    block_size = n_snps // n_blocks
    n_snps_used = block_size * n_blocks

    x_focal = x_focal[:n_snps_used]
    y_weighted = y_weighted[:n_snps_used]

    x_blocks = x_focal.reshape(n_blocks, block_size, n_spots, -1)
    y_blocks = y_weighted.reshape(n_blocks, block_size, n_spots, 1)

    xty_blocks = jnp.einsum("nbsf,nbs->nsf", x_blocks, y_blocks.squeeze(-1))
    xtx_blocks = jnp.einsum("nbsf,nbsg->nsfg", x_blocks, x_blocks)

    xty_total = jnp.sum(xty_blocks, axis=0)
    xtx_total = jnp.sum(xtx_blocks, axis=0)

    est = jnp.linalg.solve(xtx_total, xty_total[..., None]).squeeze(-1)

    xty_del = xty_total - xty_blocks
    xtx_del = xtx_total - xtx_blocks
    delete_ests = jnp.linalg.solve(xtx_del, xty_del[..., None]).squeeze(-1)

    pseudovalues = n_blocks * est - (n_blocks - 1) * delete_ests
    jknife_est = jnp.mean(pseudovalues, axis=0)

    pseudo_centered = pseudovalues - jknife_est
    spatial_jknife_se = jnp.sqrt(
        jnp.sum(jnp.square(pseudo_centered[:, :, 0]), axis=0) / (n_blocks * (n_blocks - 1))
    )

    return jknife_est[:, 0] / Nbar, spatial_jknife_se / Nbar


@partial(jit, static_argnums=(0,))
def process_chunk_batched_materialized_jit(
    n_blocks: int,
    spatial_ld: jnp.ndarray,
    baseline_ld_sum: jnp.ndarray,
    chisq: jnp.ndarray,
    N: jnp.ndarray,
    baseline_ann: jnp.ndarray,
    w_ld: jnp.ndarray,
    Nbar: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Fast materialized kernel that solves only the spatial coefficient.

    Given weighted normal equations for [spatial, baseline], the spatial
    coefficient is computed by residualizing against the baseline block:

        beta = (d - b' C^-1 e) / (a - b' C^-1 b)

    This avoids solving all coefficients for every delete-one jackknife
    replicate while preserving the same regression target.
    """
    n_snps, n_spots = spatial_ld.shape

    x_tot = spatial_ld + baseline_ld_sum.reshape(-1, 1)

    N_expanded = N.reshape(-1, 1)
    x_tot_N = x_tot * N_expanded
    mean_chisq = jnp.mean(chisq)
    mean_x_tot_N = jnp.mean(x_tot_N, axis=0)
    hsq = 10000.0 * (mean_chisq - 1.0) / mean_x_tot_N
    hsq = jnp.clip(hsq, 0.0, 1.0)

    ld_clip = jnp.maximum(x_tot, 1.0)
    w_ld_clip = jnp.maximum(w_ld.reshape(-1, 1), 1.0)
    c = (hsq.reshape(1, -1) * N_expanded) / 10000.0
    weights = jnp.sqrt(1.0 / (2 * jnp.square(1.0 + c * ld_clip) * w_ld_clip))

    weights_sum = jnp.sum(weights, axis=0, keepdims=True)
    weights_scaled = weights / weights_sum

    spatial_weighted = (spatial_ld * weights_scaled)[..., None]
    baseline_weighted = baseline_ann[:, None, :] * weights_scaled[..., None]
    x_focal = jnp.concatenate([spatial_weighted, baseline_weighted], axis=2)

    y_weighted = (chisq.reshape(-1, 1) * weights_scaled)[..., None]

    block_size = n_snps // n_blocks
    n_snps_used = block_size * n_blocks

    x_focal = x_focal[:n_snps_used]
    y_weighted = y_weighted[:n_snps_used]

    x_blocks = x_focal.reshape(n_blocks, block_size, n_spots, -1)
    y_blocks = y_weighted.reshape(n_blocks, block_size, n_spots, 1)

    xty_blocks = jnp.einsum("nbsf,nbs->nsf", x_blocks, y_blocks.squeeze(-1))
    xtx_blocks = jnp.einsum("nbsf,nbsg->nsfg", x_blocks, x_blocks)

    xty_total = jnp.sum(xty_blocks, axis=0)
    xtx_total = jnp.sum(xtx_blocks, axis=0)

    def spatial_coef(xtx, xty):
        a = xtx[..., 0, 0]
        b = xtx[..., 1:, 0]
        c_mat = xtx[..., 1:, 1:]
        d = xty[..., 0]
        e = xty[..., 1:]
        rhs = jnp.stack([e, b], axis=-1)
        solved = jnp.linalg.solve(c_mat, rhs)
        c_inv_e = solved[..., 0]
        c_inv_b = solved[..., 1]
        numerator = d - jnp.sum(b * c_inv_e, axis=-1)
        denominator = a - jnp.sum(b * c_inv_b, axis=-1)
        return numerator / denominator

    est_spatial = spatial_coef(xtx_total, xty_total)

    xty_del = xty_total - xty_blocks
    xtx_del = xtx_total - xtx_blocks
    delete_spatial = spatial_coef(xtx_del, xty_del)

    spatial_pseudovalues = n_blocks * est_spatial - (n_blocks - 1) * delete_spatial
    spatial_jknife_est = jnp.mean(spatial_pseudovalues, axis=0)
    spatial_centered = spatial_pseudovalues - spatial_jknife_est
    spatial_jknife_se = jnp.sqrt(
        jnp.sum(jnp.square(spatial_centered), axis=0) / (n_blocks * (n_blocks - 1))
    )

    return spatial_jknife_est / Nbar, spatial_jknife_se / Nbar


def _estimate_available_device_gb() -> float | None:
    """Best-effort estimate of JAX device memory available to this process."""
    try:
        stats = jax.devices()[0].memory_stats()
        if not stats:
            return None
        limit = stats.get("bytes_limit")
        in_use = stats.get("bytes_in_use", 0)
        if limit is None:
            return None
        return max(limit - in_use, 0) / (1024**3)
    except Exception:  # noqa: BLE001
        return None


@partial(jit, static_argnums=(0,))
def process_chunk_batched_jit(
    n_blocks: int,
    spatial_ld: jnp.ndarray,
    baseline_ld_sum: jnp.ndarray,
    chisq: jnp.ndarray,
    N: jnp.ndarray,
    baseline_ann: jnp.ndarray,
    w_ld: jnp.ndarray,
    Nbar: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward-compatible alias for the fastest batched implementation."""
    return process_chunk_batched_fullsolve_jit(
        n_blocks, spatial_ld, baseline_ld_sum, chisq, N, baseline_ann, w_ld, Nbar
    )


def wrapper_of_process_chunk_jit(*args, **kwargs):
    """Wrapper to call the JIT-compiled processing function.

    Automatically selects between the fully-batched implementation
    (process_chunk_batched_jit) and the memory-efficient per-spot vmap
    implementation (process_chunk_jit) based on estimated peak GPU/CPU memory.

    Args layout (positional, matching process_chunk_batched_jit):
        args[0]: n_blocks (int)
        args[1]: spatial_ld  (n_snps, n_spots)
        args[2]: baseline_ld_sum  (n_snps,)
        args[3]: chisq  (n_snps,)
        args[4]: N  (n_snps,)
        args[5]: baseline_ann  (n_snps, n_baseline_features)
        args[6]: w_ld  (n_snps,)
        args[7]: Nbar  (float)
    """
    n_blocks = args[0]
    spatial_ld = args[1]
    baseline_ann = args[5]

    n_snps = spatial_ld.shape[0]
    n_spots = spatial_ld.shape[1]
    # +1 because spatial_ld is concatenated with baseline_ann inside the JIT fn
    n_features = 1 + baseline_ann.shape[1]

    # Peak memory estimate: x_blocks shape = (n_snps, n_spots, n_features) × float32
    est_gb = n_snps * n_spots * n_features * 4 / (1024**3)

    # Use the fast materialized kernel when the estimated allocation fits.
    # The previous fixed 2 GB GPU cap forced A100-class devices into the slow
    # one-cell fallback for moderate chunks. Scale the cap with the actual JAX
    # memory limit while keeping a conservative ceiling for other workloads.
    is_gpu = jax.default_backend() == "gpu"
    if is_gpu:
        available_gb = _estimate_available_device_gb()
        _SAFE_GB = 2.0 if available_gb is None else min(max(available_gb * 0.30, 2.0), 10.0)
    else:
        _SAFE_GB = 8.0
    if est_gb <= _SAFE_GB:
        return process_chunk_batched_fullsolve_jit(*args, **kwargs)

    if is_gpu:
        return process_chunk_batched_lowmem_jit(*args, **kwargs)

    # Insert batch_size=1 between n_blocks and spatial_ld for the CPU fallback.
    return process_chunk_jit(n_blocks, 1, *args[1:], **kwargs)


# ============================================================================
# Main entry point
# ============================================================================


def run_spatial_ldsc_jax(config: SpatialLDSCConfig):
    """
    Run spatial LDSC for all traits in config.sumstats_config_dict.
    """
    traits_to_process = list(config.sumstats_config_dict.items())
    if not traits_to_process:
        raise ValueError("No traits to process. config.sumstats_config_dict is empty.")

    # Create output directory
    output_dir = config.ldsc_save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of loader threads based on platform
    n_loader_threads = 10 if jax.default_backend() == "gpu" else 2

    marker_score_adata = None

    try:
        if config.ldscore_h5ad_path is not None:
            # ── Direct ldscore input mode ────────────────────────────────────
            # Bypasses LD score computation;
            # baseline annotations come from additional_baseline_h5ad_path_list.
            logger.info("Using direct ldscore input mode (pre-computed SNP × Cell h5ad).")
            baseline_ld, w_ld, ldscore_adata = load_common_resources_direct_ldscore(config)

            logger.debug("Initializing processor (direct ldscore mode)...")
            processor = SpatialLDSCProcessor(
                config=config,
                output_dir=output_dir,
                marker_score_adata=None,
                snp_gene_weight_adata=None,
                baseline_ld=baseline_ld,
                w_ld=w_ld,
                n_loader_threads=n_loader_threads,
                ldscore_adata=ldscore_adata,
            )
        else:
            # ── Weight-matrix mode ───────────────────────
            if config.marker_score_format not in ["memmap", "h5ad", "feather"]:
                raise NotImplementedError(
                    f"Marker score format '{config.marker_score_format}' is not supported. "
                    "Only 'memmap', 'h5ad', and 'feather' are supported."
                )

            # Load marker scores once (format-agnostic)
            logger.info(f"Loading marker scores (format: {config.marker_score_format})...")

            if config.marker_score_format == "memmap":
                marker_score_adata = load_marker_scores_memmap_format(config)

            elif config.marker_score_format == "feather":
                feather_path = Path(config.marker_score_feather_path)
                logger.info(f"Loading marker scores from Feather: {feather_path}")
                marker_score_adata = FeatherAnnData(
                    feather_path, index_col="HUMAN_GENE_SYM", transpose=True
                )

            elif config.marker_score_format == "h5ad":
                if not config.marker_score_h5ad_path:
                    raise ValueError(
                        "marker_score_h5ad_path must be provided when marker_score_format is 'h5ad'"
                    )

                h5ad_path = Path(config.marker_score_h5ad_path)
                if not h5ad_path.exists():
                    raise FileNotFoundError(f"Marker score H5AD file not found: {h5ad_path}")

                logger.info(f"Loading marker scores from H5AD: {h5ad_path}")
                marker_score_adata = ad.read_h5ad(h5ad_path, backed="r")

            # Load common resources once (baseline, weights, snp_gene_weights)
            baseline_ld, w_ld, snp_gene_weight_adata = load_common_resources(config)

            # Initialize processor with common resources
            logger.debug("Initializing processor (weight-matrix mode)...")
            processor = SpatialLDSCProcessor(
                config=config,
                output_dir=output_dir,
                marker_score_adata=marker_score_adata,
                snp_gene_weight_adata=snp_gene_weight_adata,
                baseline_ld=baseline_ld,
                w_ld=w_ld,
                n_loader_threads=n_loader_threads,
            )

        try:
            for idx, (trait_name, sumstats_file) in enumerate(traits_to_process):
                try:
                    logger.info("=" * 70)
                    logger.info("Running Spatial LDSC (JAX Implementation)")
                    logger.info(
                        f"Project: {config.project_name}, Trait: {trait_name} ({idx + 1}/{len(traits_to_process)})"
                    )
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
                            logger.info(
                                f"Skipping trait {trait_name} ({idx + 1}/{len(traits_to_process)})"
                            )

                            # Log statistics from existing result
                            log_existing_result_statistics(expected_output_path, trait_name)
                            continue

                    # Setup processor for current trait
                    processor.setup_trait(trait_name, sumstats_file)

                    # Process all chunks for current trait
                    start_time = time.time()
                    processor.process_all_chunks(wrapper_of_process_chunk_jit)

                    elapsed_time = time.time() - start_time
                    h, rem = divmod(elapsed_time, 3600)
                    m, s = divmod(rem, 60)
                    logger.info(f"Trait {trait_name} completed in {int(h)}h {int(m)}m {s:.2f}s")
                except Exception as e:
                    logger.error(f"Error processing trait {trait_name}: {e}")
                    continue  # Move on to the next trait even if this one fails

        finally:
            # Cleanup once: close memmap/adata if needed
            if marker_score_adata is not None:
                logger.info("Closing marker score resources...")
                # If it's our MemMap wrapper, close it explicitly
                if (
                    config.marker_score_format == "memmap"
                    and "memmap_manager" in marker_score_adata.uns
                ):
                    marker_score_adata.uns["memmap_manager"].close()
                # If it's backed AnnData, close the file
                if config.marker_score_format == "h5ad" and marker_score_adata.isbacked:
                    marker_score_adata.file.close()

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise

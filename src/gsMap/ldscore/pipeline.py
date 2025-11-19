import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from tqdm import tqdm
import gc
import logging

from .config import LDScoreConfig
from .io import PlinkBEDReader, load_omics_features
from .mapping import create_snp_feature_map
from .utils import get_block_limits
from .compute import compute_batch_weights, prepare_padding, prepare_vector_padding

logger = logging.getLogger(__name__)


def get_quantized_width(required_width: int, bin_step: int = 128) -> int:
    """Return the next multiple of bin_step >= required_width to minimize JIT recompilation."""
    return int(np.ceil(required_width / bin_step) * bin_step)


def run_generate_ldscore(config: LDScoreConfig):
    # 1. Load Metadata
    logger.info(f"Initializing LD Score generation for {config.bfile_root}")
    reader = PlinkBEDReader(config.bfile_root)

    logger.info(f"Loading HM3 SNPs from {config.hm3_snp_path}")
    hm3_df = pd.read_csv(config.hm3_snp_path, sep="\t")

    # Filter HM3 SNPs to those present in the BIM file to avoid errors
    # We assume the BIM SNP IDs are unique for simplicity here
    valid_hm3_mask = hm3_df['SNP'].isin(reader.bim['SNP'])
    if (~valid_hm3_mask).any():
        logger.warning(f"{(~valid_hm3_mask).sum()} HM3 SNPs not found in reference panel. They will be skipped.")
        hm3_df = hm3_df[valid_hm3_mask].reset_index(drop=True)

    # 2. Omics Feature Setup
    feature_names = []
    if config.omics_h5ad_path:
        feature_names = load_omics_features(config.omics_h5ad_path)
        logger.info(f"Loaded {len(feature_names)} omics features")

    # Load Mapping Input
    mapping_data = {}
    if config.mapping_type == 'bed':
        if not config.mapping_file:
            raise ValueError("mapping_file is required for 'bed' mapping type")
        logger.info(f"Loading mapping file: {config.mapping_file}")
        mapping_data = pd.read_csv(config.mapping_file, sep="\t")
        # Ensure necessary columns exist
        required_cols = ['Feature', 'Chromosome', 'Start', 'End']
        if not all(col in mapping_data.columns for col in required_cols):
            raise ValueError(f"Mapping file missing required columns: {required_cols}")
    else:
        # Placeholder for dict loading if implemented
        pass

    # 3. Create Global Map (SNP -> Feature)
    logger.info("Mapping SNPs to Features...")
    snp_feature_map, num_features = create_snp_feature_map(
        reader.bim,
        config.mapping_type,
        mapping_data,
        window_size=config.window_size,
        strategy=config.strategy
    )
    logger.info(f"Total features mapped: {num_features}")

    # 4. Iterate Chromosomes
    chrom_list = range(1, 23) if config.chromosomes == 'all' else config.chromosomes
    bcoo_matrices = []

    for chrom in chrom_list:
        chrom_str = str(chrom)
        logger.info(f"Processing Chromosome {chrom}...")

        # Get indices for current chromosome
        # Note: PlinkBEDReader loads the whole BIM. We work with global indices.
        ref_indices_chrom = np.where(reader.bim['CHR'].astype(str) == chrom_str)[0]

        if len(ref_indices_chrom) == 0:
            logger.warning(f"No Reference SNPs found for Chr {chrom}. Skipping.")
            continue

        coords_ref = reader.bim.iloc[ref_indices_chrom]['BP'].values

        # Offset to convert chromosome-relative indices to global indices
        chrom_offset = ref_indices_chrom[0]

        # Filter HM3 for this chromosome
        hm3_chrom = hm3_df[hm3_df['CHR'].astype(str) == chrom_str]
        # TODO should only keep HM3 SNPs that are also in the reference panel
        coords_hm3 = hm3_chrom['BP'].values
        m_hm3_chrom = len(hm3_chrom)


        if m_hm3_chrom == 0:
            logger.info(f"No HM3 SNPs for Chr {chrom}. Skipping.")
            continue

        # 5. Pre-calculate Block Limits (Local Indices relative to Chromosome)
        # left_limits and right_limits are indices into coords_ref
        left_limits, right_limits = get_block_limits(coords_ref, coords_hm3, config.window_size)

        # 6. Batch Processing
        chrom_weights = []

        # Processing loop
        for start_idx in tqdm(range(0, m_hm3_chrom, config.batch_size_hm3), desc=f"Chr {chrom} Batches"):
            end_idx = min(start_idx + config.batch_size_hm3, m_hm3_chrom)

            # Indices of HM3 SNPs in this batch (relative to chromosome list)
            batch_indices_local = np.arange(start_idx, end_idx)

            # Identify the "Super Block" for this batch
            # Min left index and Max right index in the reference panel for this batch
            sb_start_local = left_limits[batch_indices_local].min()
            sb_end_local = right_limits[batch_indices_local].max()

            # Calculate relative limits for the mask
            # For a SNP i, its valid window starts at left_limits[i] - sb_start_local inside the superblock
            relative_starts = left_limits[batch_indices_local] - sb_start_local
            relative_ends = right_limits[batch_indices_local] - sb_start_local

            # Convert local chromosome indices to global reader indices
            sb_start_global = chrom_offset + sb_start_local
            sb_end_global = chrom_offset + sb_end_local

            # Actual width needed
            current_width = sb_end_global - sb_start_global

            # Quantize width to minimize JIT recompilation
            # We pad the reference block to this width
            target_width = get_quantized_width(current_width, bin_step=256)

            # 1. Load HM3 Genotypes (Target)
            # Map HM3 IDs to global indices in BIM
            # Optimization: We could use a pre-computed map, but `isin` + `where` works for now
            batch_snps = hm3_chrom.iloc[batch_indices_local]['SNP'].values
            # This lookup is slow inside a loop; practically we should pre-map,
            # but keeping it simple for the framework structure
            hm3_global_indices = reader.bim.index[reader.bim['SNP'].isin(batch_snps)].values

            # Sort to ensure alignment if needed, though HM3 is usually sorted
            hm3_global_indices.sort()

            X_hm3 = reader.get_genotypes(hm3_global_indices)  # (N, Batch)
            X_hm3 = jnp.array(X_hm3)

            # 2. Load Reference Super Block
            # We load the contiguous range [sb_start_global, sb_end_global)
            # get_genotypes supports arbitrary indices, so we generate range
            ref_range_indices = np.arange(sb_start_global, sb_end_global)
            X_ref_block = reader.get_genotypes(ref_range_indices)  # (N, current_width)
            X_ref_block = jnp.array(X_ref_block)

            # 3. Pad Reference Block to Quantized Width
            X_ref_padded = prepare_padding(X_ref_block, target_width)

            # 4. Get Feature Mapping for Super Block
            # Slice global map
            batch_mapping_raw = snp_feature_map[sb_start_global:sb_end_global]
            batch_mapping = jnp.array(batch_mapping_raw)

            # Pad mapping vector with "garbage bin" index (num_features)
            batch_mapping_padded = prepare_vector_padding(
                batch_mapping, target_width, fill_value=num_features
            )

            # 5. Compute Weights
            # Transfer to JAX device and compute
            relative_starts_jax = jnp.array(relative_starts)
            relative_ends_jax = jnp.array(relative_ends)

            w_matrix = compute_batch_weights(
                X_ref_padded,
                X_hm3,
                batch_mapping_padded,
                relative_starts_jax,
                relative_ends_jax,
                num_features
            )

            # Convert to BCOO (Batched Coordinate format) for sparse storage
            # Thresholding small values (near zero) can save space,
            # but standard LDSC keeps all L2 scores.
            chrom_weights.append(sparse.BCOO.fromdense(w_matrix))

            # Memory Cleanup
            del X_hm3, X_ref_block, X_ref_padded
            if start_idx % 1000 == 0:
                gc.collect()
                jax.clear_caches()

        # Concatenate Batch Results for Chromosome
        if chrom_weights:
            chrom_final = sparse.bcoo_concatenate(chrom_weights, dimension=0)
            bcoo_matrices.append(chrom_final)
            logger.info(f"Chromosome {chrom} complete. Shape: {chrom_final.shape}")
        else:
            logger.warning(f"No results generated for Chromosome {chrom}")

    # 8. Final Concatenation
    if bcoo_matrices:
        final_weight_matrix = sparse.bcoo_concatenate(bcoo_matrices, dimension=0)
        logger.info(f"Global LD Score Matrix Computation Complete. Final Shape: {final_weight_matrix.shape}")
        return final_weight_matrix
    else:
        logger.error("No weights computed across any chromosomes.")
        return None
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
from .utils import get_block_limits, dynamic_programming_quantization
from .compute import compute_batch_weights


logger = logging.getLogger(__name__)

def run_generate_ldscore(config: LDScoreConfig):
    # 1. Load Data Metadata
    logger.info("Loading Metadata...")
    reader = PlinkBEDReader(config.bfile_root)
    hm3_df = pd.read_csv(config.hm3_snp_path, sep="\t")  # Assuming format
    #TODO: Validate HM3 SNPs against reference, only keep intersecting SNPs

    # 2. Omics Feature Setup
    if config.omics_h5ad_path:
        feature_names = load_omics_features(config.omics_h5ad_path)
        # Construct dummy mapping data if needed or load external
        # For this general framework, we assume the mapping input is separate

    # Load Mapping Input (BED or Dict)
    # This is a placeholder for reading the specific mapping file format
    if config.mapping_type == 'bed':
        mapping_data = pd.read_csv(config.mapping_file, sep="\t")
    else:
        # Load dict pickle or json
        mapping_data = {}

        # 3. Create Global Map (SNP -> Feature)
    logger.info("Mapping SNPs to Features...")
    snp_feature_map, num_features = create_snp_feature_map(
        reader.bim,
        config.mapping_type,
        mapping_data,
        window_size=config.window_size,
        strategy=config.strategy
    )

    # 4. Iterate Chromosomes
    chrom_list = range(1, 23) if config.chromosomes == 'all' else config.chromosomes
    bcoo_matrices = []

    for chrom in chrom_list:
        logger.info(f"Processing Chromosome {chrom}...")

        # Filter Reference and HM3 for this Chromosome
        # (Assuming reader can handle subsetting or we load chr specific files)
        # For pseudo-code simplicity, assume indices are retrieved:
        ref_indices_chrom = np.where(reader.bim['CHR'] == chrom)[0]
        coords_ref = reader.bim.iloc[ref_indices_chrom]['BP'].values

        hm3_chrom = hm3_df[hm3_df['CHR'] == chrom]
        coords_hm3 = hm3_chrom['BP'].values
        m_hm3_chrom = len(hm3_chrom)

        if m_hm3_chrom == 0: continue

        # 5. Pre-calculate Blocks
        left_limits, right_limits = get_block_limits(coords_ref, coords_hm3, config.window_size)

        # 6. Quantize Block Lengths (DP Optimization)
        raw_lengths = right_limits - left_limits
        quantized_lengths = dynamic_programming_quantization(raw_lengths, config.quantization_num_bins)

        # 7. Batch Processing
        chrom_weights = []  # List to hold batch results

        for start_idx in tqdm(range(0, m_hm3_chrom, config.batch_size_hm3)):
            end_idx = min(start_idx + config.batch_size_hm3, m_hm3_chrom)
            current_batch_size = end_idx - start_idx

            # Determine Block range for this batch
            # We take the union of blocks for SNPs in this batch
            # To keep logic simple for JIT, we take the max quantized length in this batch
            batch_q_len = quantized_lengths[start_idx:end_idx].max()

            # We need to load the Reference Genotype Window
            # Since each SNP has a different left/right, efficiently loading matrix A
            # requires handling the offsets.
            # OPTIMIZATION: To vectorize, we define a 'super-block' for the batch
            # covering min(left) to max(right) and mask, or use the quantized length padding.

            # Simplified Approach:
            # 1. Load HM3 Batch (N x Batch)
            batch_hm3_indices = hm3_chrom.iloc[start_idx:end_idx].index  # Need mapping to global ref index
            X_hm3 = reader.get_genotypes(batch_hm3_indices)
            X_hm3 = jnp.array(X_hm3)

            # 2. For each SNP in batch, the block is different.
            # This implies A is not a single matrix but a collection.
            # Standard LDscore uses a sliding window.
            # To use matrix multiplication A.T @ B, A must be common or structured.

            # Correction to standard LDSC Matrix algorithm:
            # Usually, we iterate Blocks of Reference SNPs.
            # Here, we iterate HM3 SNPs.
            # If we want A @ B, A must be the Reference Matrix.
            # We define a Super-Block for the batch:
            batch_global_left = left_limits[start_idx:end_idx].min()
            batch_global_right = batch_global_left + batch_q_len  # Approximation for vectorized load

            # Load Ref Super Block
            X_ref_block = reader.get_genotypes(np.arange(batch_global_left, batch_global_right))
            X_ref_block = jnp.array(X_ref_block)

            # Get mapping for this block
            batch_mapping = snp_feature_map[batch_global_left:batch_global_right]
            batch_mapping = jnp.array(batch_mapping)

            # Compute
            # Result: (current_batch_size, num_features)
            w_matrix = compute_batch_weights(
                X_ref_block,
                X_hm3,
                batch_mapping,
                num_features
            )

            chrom_weights.append(sparse.BCOO.fromdense(w_matrix))

            # Cleanup
            del X_hm3, X_ref_block
            if start_idx % 10 == 0: gc.collect()

        # Concatenate Batch Results
        chrom_final = sparse.bcoo_concatenate(chrom_weights, dimension=0)
        bcoo_matrices.append(chrom_final)

    # 8. Final Concatenation
    final_weight_matrix = sparse.bcoo_concatenate(bcoo_matrices, dimension=0)

    logger.info("Done.")
    return final_weight_matrix
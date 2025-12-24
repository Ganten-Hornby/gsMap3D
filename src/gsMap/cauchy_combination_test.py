import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
from scipy import stats
from scipy.stats import fisher_exact
import statsmodels.api as sm
from rich.progress import Progress

from gsMap.config.cauchy_config import CauchyCombinationConfig

logger = logging.getLogger(__name__)

def load_ldsc(path):
    logger.debug(f'Loading {path}')
    df = pd.read_csv(path)
    df['log10_p'] = -np.log10(df['p'])
    # Clean up spot index
    df['spot'] = df['spot'].astype(str).str.replace(r'\.0$', '', regex=True)
    df.set_index('spot', inplace=True)
    # drop nan
    df = df.dropna()
    return df

def load_ldsc_with_key(key_path_tuple):
    """Helper function to load LDSC and return with its key"""
    key, path = key_path_tuple
    df = load_ldsc(path)
    # Select log10_p and rename with the key
    df_subset = df[['log10_p']].rename(columns={'log10_p': key})
    return key, df_subset

def join_ldsc_results(paths_dict, columns_to_keep=['log10_p'], max_workers=None):
    """
    Load and join LDSC results from multiple paths using ProcessPoolExecutor.
    Each log10_p column is renamed with the dictionary key.
    """
    dfs_dict = {}

    # Use ProcessPoolExecutor for parallel loading
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_key = {
            executor.submit(load_ldsc_with_key, (key, path)): key
            for key, path in paths_dict.items()
        }

        # Collect results as they complete
        for future in as_completed(future_to_key):
            key, df_subset = future.result()
            dfs_dict[key] = df_subset

    # Maintain original order
    dfs_to_join = [dfs_dict[key] for key in paths_dict.keys()]

    # OPTIMIZED JOIN: Use pd.concat which is much faster than sequential joins
    df_merged = pd.concat(dfs_to_join, axis=1, join='inner', sort=False)

    return df_merged

def _acat_test(pvalues: np.ndarray, weights=None):
    """
    Aggregated Cauchy Association Test (ACAT)
    """
    if np.any(np.isnan(pvalues)):
        raise ValueError("Cannot have NAs in the p-values.")
    if np.any((pvalues > 1) | (pvalues < 0)):
        raise ValueError("P-values must be between 0 and 1.")
    
    # Handle exact 0 or 1
    if np.any(pvalues == 0):
        return 0.0
    if np.any(pvalues == 1):
        if np.all(pvalues == 1):
            return 1.0
        # If mixed 1s and <1s, 1s contribute nothing to the sum of tans likely, 
        # but let's follow the standard logic: tan((0.5-1)*pi) -> tan(-0.5pi) -> -inf.
        # This implementation handles small p-values carefully but large ones (near 1) 
        # result in negative large stats which is fine (large CDF).
        pass

    if weights is None:
        weights = np.full(len(pvalues), 1 / len(pvalues))
    else:
        if len(weights) != len(pvalues):
            raise Exception("Length of weights and p-values differs.")
        if any(weights < 0):
            raise Exception("All weights must be positive.")
        weights = np.array(weights) / np.sum(weights)

    is_small = pvalues < 1e-16
    is_large = ~is_small

    cct_stat = 0.0
    if np.any(is_small):
         cct_stat += np.sum((weights[is_small] / pvalues[is_small]) / np.pi)
    
    if np.any(is_large):
        cct_stat += np.sum(weights[is_large] * np.tan((0.5 - pvalues[is_large]) * np.pi))

    if cct_stat > 1e15:
        pval = (1 / cct_stat) / np.pi
    else:
        pval = 1 - sp.stats.cauchy.cdf(cct_stat)

    return pval

def remove_outliers_IQR(data, threshold_factor=3.0):
    """
    Remove outliers using IQR method on log10 p-values.
    p_values_filtered = p_values[p_values_log < median_log + 3 * iqr_log]
    """
    if len(data) == 0:
        return data, np.array([], dtype=bool)

    median = np.median(data)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    
    cutoff = median + threshold_factor * iqr
    
    # Filter: keep values less than cutoff
    mask = data < cutoff
    data_passed = data[mask]
    
    return data_passed, mask

def process_trait(trait, anno_data, all_data, annotation, annotation_col):
    """
    Process a single trait for a given annotation: calculate Cauchy combination p-value.
    """
    # Calculate significance threshold (Bonferroni correction)
    sig_threshold = 0.05 / len(all_data)

    # Get p-values for this annotation and trait
    if trait not in anno_data.columns:
        logger.warning(f"Trait {trait} not found in data columns.")
        return None

    log10p = anno_data[trait].values
    log10p, mask = remove_outliers_IQR(log10p)
    p_values = 10 ** (-log10p)  # convert from log10(p) to p

    # Calculate Cauchy combination and mean/median metrics
    if len(p_values) == 0:
        p_cauchy_val = 1.0
        p_median_val = 1.0
        top_95_mean = 0.0
    else:
        p_cauchy_val = _acat_test(p_values)
        p_median_val = np.median(p_values)
        
        # Calculate top 5% mean of -log10pvalue (referred to as top_95_mean)
        sorted_log10p = np.sort(log10p)
        n_top_05 = int(np.ceil(len(sorted_log10p) * 0.05))
        if n_top_05 > 0:
            top_95_mean = np.mean(sorted_log10p[-n_top_05:])
        else:
            top_95_mean = 0.0

    # Calculate significance statistics
    sig_spots_in_anno = np.sum(p_values < sig_threshold)
    total_spots_in_anno = len(p_values)

    return {
        'trait': trait,
        'annotation_name': annotation_col,
        'annotation': annotation,
        'p_cauchy': p_cauchy_val,
        'p_median': p_median_val,
        'top_95_mean': top_95_mean,
        'sig_spots': sig_spots_in_anno,
        'total_spots': total_spots_in_anno,
    }

def run_cauchy_on_dataframe(df, annotation_col, trait_cols=None, extra_group_col=None):
    """
    Run Cauchy combination test on a dataframe.
    
    Args:
        df: DataFrame containing log10(p) values and annotation column.
        annotation_col: Name of column containing annotations.
        trait_cols: List of trait columns. If None, uses all columns except annotation_col.
        extra_group_col: Optional extra column to group by (e.g., 'sample_name').
        
    Returns:
        DataFrame with results.
    """
    if trait_cols is None:
        trait_cols = [c for c in df.columns if c not in [annotation_col, extra_group_col] and c != 'spot']

    all_results = []
    
    # Define the groups
    group_cols = [annotation_col]
    if extra_group_col:
        group_cols.append(extra_group_col)
    
    # Pre-calculate groups to avoid repeated filtering
    grouped = df.groupby(group_cols, observed=True)
    
    for trait in trait_cols:
        def process_one_group(group_key):
            df_group = grouped.get_group(group_key)
            
            # Usually enrichment is within the same context.
            # If extra_group_col is sample_name, background should be other annotations in the SAME sample.
            if extra_group_col:
                # Group key is (anno, extra)
                anno, extra = group_key
                df_background = df[df[extra_group_col] == extra]
                # We need all_data for process_trait to calculate background
                res = process_trait(trait, df_group, df_background, anno, annotation_col)
                if res:
                    res[extra_group_col] = extra
            else:
                # Group key is just (anno,)
                anno = group_key[0] if isinstance(group_key, tuple) else group_key
                res = process_trait(trait, df_group, df, anno, annotation_col)
            
            return res

        group_keys = list(grouped.groups.keys())
        
        # Use rich progress bar
        with Progress(transient=True) as progress:
            task = progress.add_task(f"[green]Processing {trait}...", total=len(group_keys))
            
            # Parallelize over groups
            with ThreadPoolExecutor(max_workers=None) as executor:
                # Submit all tasks
                future_to_group = {executor.submit(process_one_group, g): g for g in group_keys}
                
                for future in as_completed(future_to_group):
                    res = future.result()
                    if res is not None:
                        all_results.append(res)
                    progress.advance(task)

    if not all_results:
        return pd.DataFrame()
        
    combined_results = pd.DataFrame(all_results)
    sort_cols = ['trait', 'p_cauchy']
    if extra_group_col:
        sort_cols = [extra_group_col] + sort_cols
        
    combined_results.sort_values(by=sort_cols, inplace=True)
    
    return combined_results


def get_trait_files_dict(config: CauchyCombinationConfig) -> dict[str, Path]:
    """
    Discover LDSC result files and return a dictionary mapping trait names to file paths.
    """
    if config.trait_name is None:
        logger.info(f"Trait name not specified. Scanning {config.ldsc_save_dir} for spatial LDSC results...")
        # Pattern: {self.project_name}_{trait_name}.csv.gz
        pattern = f"{config.project_name}_*.csv.gz"
        ldsc_files = list(config.ldsc_save_dir.glob(pattern))
        
        if not ldsc_files:
            logger.warning(f"No spatial LDSC result files found matching pattern '{pattern}' in {config.ldsc_save_dir}")
            return {}

        traits_dict = {}
        for f in ldsc_files:
             # Extract trait name: projectname_{trait}.csv.gz
             fname = f.name
             trait = fname[len(config.project_name)+1:].replace(".csv.gz", "")
             traits_dict[trait] = f
        
        logger.info(f"Found {len(traits_dict)} traits: {list(traits_dict.keys())}")
        return traits_dict
    else:
        ldsc_input_file = config.get_ldsc_result_file(config.trait_name)
        if not ldsc_input_file.exists():
            logger.warning(f"LDSC result file not found for {config.trait_name}: {ldsc_input_file}")
            return {}
        return {config.trait_name: ldsc_input_file}

def run_Cauchy_combination(config: CauchyCombinationConfig):
    # 1. Discover traits and load combined LDSC results
    traits_dict = get_trait_files_dict(config)
    if not traits_dict:
        return

    logger.info(f"Joining LDSC results for {len(traits_dict)} traits...")
    df_combined = join_ldsc_results(traits_dict)
    
    # 2. Add Annotation & Metadata
    logger.info(f"------Loading annotations...")
    latent_adata_path = config.concatenated_latent_adata_path
    if not latent_adata_path.exists():
        raise FileNotFoundError(f"Latent adata with annotations not found at: {latent_adata_path}")
    
    logger.info(f"Loading metadata from {latent_adata_path}")
    adata = sc.read_h5ad(latent_adata_path)
    
    # Check for all requested annotation columns
    for anno in config.annotation_list:
        if anno not in adata.obs.columns:
            raise ValueError(f"Annotation column '{anno}' not found in adata.obs.")
    
    # Check for sample_name column
    sample_col = 'sample_name'
    if sample_col not in adata.obs.columns:
        if 'batch_id' in adata.obs.columns:
             sample_col = 'batch_id'
             logger.warning(f"'sample_name' not found in adata.obs, using 'batch_id' instead.")
        else:
             logger.warning(f"Neither 'sample_name' nor 'batch_id' found in adata.obs. Sample-level Cauchy will be skipped.")
             sample_col = None

    # Filter to common spots
    common_cells = np.intersect1d(df_combined.index, adata.obs_names)
    logger.info(f"Found {len(common_cells)} common spots between LDSC results and metadata.")
    if len(common_cells) == 0:
        logger.warning(f"No common spots found. Skipping...")
        return
        
    df_combined = df_combined.loc[common_cells].copy()
    
    # Add metadata columns
    metadata_cols = config.annotation_list.copy()
    if sample_col:
        metadata_cols.append(sample_col)
    
    for col in metadata_cols:
        df_combined[col] = adata.obs.loc[common_cells, col].values

    # 3. Save combined data to parquet
    logger.info(f"Saving combined data to {config.combined_parquet_path}")
    df_to_save = df_combined.reset_index().rename(columns={'index': 'spot'})
    df_to_save.to_parquet(config.combined_parquet_path)

    # 4. Process each annotation and each trait
    trait_cols = list(traits_dict.keys())
    for annotation_col in config.annotation_list:
        logger.info(f"=== Processing Cauchy combination for annotation: {annotation_col} ===")
        
        # Run Cauchy Combination (Annotation Level)
        result_df = run_cauchy_on_dataframe(df_combined, 
                                          annotation_col=annotation_col, 
                                          trait_cols=trait_cols)
        
        # Save results per trait
        for trait_name in trait_cols:
            trait_result = result_df[result_df['trait'] == trait_name]
            output_file = config.get_cauchy_result_file(trait_name, annotation=annotation_col, all_samples=True)
            trait_result.to_csv(output_file, index=False)

        # Run Cauchy Combination (Sample-Annotation level)
        if sample_col:
            sample_result_df = run_cauchy_on_dataframe(df_combined, 
                                                      annotation_col=annotation_col, 
                                                      trait_cols=trait_cols,
                                                      extra_group_col=sample_col)
            
            for trait_name in trait_cols:
                trait_sample_result = sample_result_df[sample_result_df['trait'] == trait_name]
                sample_output_file = config.get_cauchy_result_file(trait_name, annotation=annotation_col, all_samples=False)
                trait_sample_result.to_csv(sample_output_file, index=False)
    
    logger.info("Cauchy combination processing completed.")

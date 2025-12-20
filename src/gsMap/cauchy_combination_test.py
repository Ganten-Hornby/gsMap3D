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
    Process a single trait for a given annotation: calculate Cauchy combination p-value
    and Fisher's exact test for enrichment.
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

    # Calculate Cauchy combination and median
    if len(p_values) == 0:
        p_cauchy_val = 1.0
        p_median_val = 1.0
    else:
        p_cauchy_val = _acat_test(p_values)
        p_median_val = np.median(p_values)

    # Calculate significance statistics
    sig_spots_in_anno = np.sum(p_values < sig_threshold)
    total_spots_in_anno = len(p_values)

    # Get p-values for other annotations (background)
    other_annotations_mask = all_data[annotation_col] != annotation
    
    other_p_values = 10 ** (-all_data.loc[other_annotations_mask, trait].values)
    sig_spots_elsewhere = np.sum(other_p_values < sig_threshold)
    total_spots_elsewhere = len(other_p_values)

    # Odds ratio calculation using Fisher's exact test
    try:
        # Create contingency table
        # [[Sig In, Non-Sig In],
        #  [Sig Out, Non-Sig Out]]
        contingency_table = np.array([
            [sig_spots_in_anno, total_spots_in_anno - sig_spots_in_anno],
            [sig_spots_elsewhere, total_spots_elsewhere - sig_spots_elsewhere]
        ])

        # Calculate odds ratio and p-value using Fisher's exact test
        odds_ratio, p_value = fisher_exact(contingency_table)

        # if odds_ratio is infinite, set it to a large number
        if odds_ratio == np.inf:
            odds_ratio = 1e4

        # Calculate confidence intervals
        table = sm.stats.Table2x2(contingency_table)
        conf_int = table.oddsratio_confint()
        ci_low, ci_high = conf_int
    except Exception as e:
        # Handle calculation errors
        odds_ratio = 0
        p_value = 1
        ci_low, ci_high = 0, 0
        # logger.warning(f"Fisher's exact test failed for {trait} in {annotation}: {e}")

    return {
        'trait': trait,
        'p_cauchy': p_cauchy_val,
        'p_median': p_median_val,
        'odds_ratio': odds_ratio,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_odds_ratio': p_value,
        'sig_spots': sig_spots_in_anno,
        'total_spots': total_spots_in_anno,
        'sig_ratio': sig_spots_in_anno / total_spots_in_anno if total_spots_in_anno > 0 else 0,
        'overall_sig_spots': sig_spots_in_anno + sig_spots_elsewhere,
        'overall_spots': total_spots_in_anno + total_spots_elsewhere
    }

def run_cauchy_on_dataframe(df, annotation_col, trait_cols=None):
    """
    Run Cauchy combination test on a dataframe.
    
    Args:
        df: DataFrame containing log10(p) values and annotation column.
        annotation_col: Name of column containing annotations.
        trait_cols: List of trait columns. If None, uses all columns except annotation_col.
        
    Returns:
        DataFrame with results.
    """
    results_dict = {}
    annotations = df[annotation_col].unique()
    
    if trait_cols is None:
        trait_cols = [c for c in df.columns if c != annotation_col]

    all_results = []
    
    for trait in trait_cols:
        # Create partial function for this trait
        # We need to compute calculate_trait_for_anno for each annotation
        
        def process_one_anno(anno):
            # Because ThreadPoolExecutor shares memory, we can slice df here without much overhead
            # provided df is not being written to.
            df_anno = df[df[annotation_col] == anno]
            return process_trait(trait, df_anno, df, anno, annotation_col)

        # Use rich progress bar
        with Progress(transient=True) as progress:
            task = progress.add_task(f"[green]Processing {trait}...", total=len(annotations))
            
            # Parallelize over annotations
            with ThreadPoolExecutor(max_workers=None) as executor:
                # Submit all tasks
                future_to_anno = {executor.submit(process_one_anno, anno): anno for anno in annotations}
                
                for future in as_completed(future_to_anno):
                    res = future.result()
                    if res is not None:
                        all_results.append(res)
                    progress.advance(task)

    if not all_results:
        return pd.DataFrame()
        
    combined_results = pd.DataFrame(all_results)
    # Sort or organize if needed? The original code sorted by p_cauchy within annotation.
    # Here we have a flat list. We can sort by trait and p_cauchy.
    combined_results.sort_values(by=['trait', 'p_cauchy'], inplace=True)
    
    return combined_results


def run_Cauchy_combination(config: CauchyCombinationConfig):
    # 1. Load the LDSC results
    logger.info(f"------Loading LDSC results from {config.ldsc_save_dir}...")
    
    # We will load data for the specified trait
    ldsc_input_file = config.get_ldsc_result_file(config.trait_name)
    if not ldsc_input_file.exists():
        raise FileNotFoundError(f"LDSC result file not found: {ldsc_input_file}")
    
    ldsc_df = load_ldsc(ldsc_input_file)
    
    # Rename the log10_p column to the trait name to match expected format for run_cauchy_on_dataframe
    ldsc_df.rename(columns={'log10_p': config.trait_name}, inplace=True)
    trait_cols = [config.trait_name]
    
    # 2. Add Annotation
    logger.info(f"------Loading annotations...")
    
    # In the new design, we use concatenated_latent_adata_path for annotations
    # (ConfigWithAutoPaths provides this)
    latent_adata_path = config.concatenated_latent_adata_path
    
    if not latent_adata_path.exists():
        # Fallback logic if needed, or error out
        # Try finding it in workdir if not standard
        fallback = Path(f"{config.workdir}/{config.project_name}/latent_to_gene/concatenated_latent_adata.h5ad")
        if fallback.exists():
            latent_adata_path = fallback
        else:
             raise FileNotFoundError(f"Latent adata with annotations not found at: {latent_adata_path}")
    
    logger.info(f"Loading metadata from {latent_adata_path}")
    adata = sc.read_h5ad(latent_adata_path)
    
    # Check for annotation column
    if config.annotation not in adata.obs.columns:
        raise ValueError(f"Annotation column '{config.annotation}' not found in adata.obs.")
    
    # Join annotation to LDSC results
    # Ensure indices match (spots)
    common_cells = np.intersect1d(ldsc_df.index, adata.obs_names)
    logger.info(f"Found {len(common_cells)} common spots between LDSC results and annotation.")
    
    if len(common_cells) == 0:
        raise ValueError("No common spots found between LDSC results and latent adata.")
    
    ldsc_df = ldsc_df.loc[common_cells]
    
    # Add annotation column
    ldsc_df[config.annotation] = adata.obs.loc[common_cells, config.annotation].values
    
    # 3. Run Cauchy Combination
    logger.info(f"------Running Cauchy Combination Test...")
    result_df = run_cauchy_on_dataframe(ldsc_df, 
                                      annotation_col=config.annotation, 
                                      trait_cols=trait_cols)
    
    # 4. Save Results
    output_file = config.output_file
    logger.info(f"------Saving results to {output_file}...")
    
    # Format columns as expected by downstream or previous versions if needed
    # Previous columns were: p_cauchy, p_median, n_cell, annotation
    # New columns include odds_ratio etc. keeping them is good.
    
    result_df.to_csv(output_file, compression='gzip', index=False)
    logger.info("Done.")

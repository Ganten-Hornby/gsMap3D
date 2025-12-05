import glob
import logging
import os
from pathlib import Path
from typing import Tuple, List, Optional, Union

import pandas as pd
import numpy as np
import anndata as ad

import pyarrow.feather as feather
from statsmodels.stats.multitest import multipletests

from ..config import SpatialLDSCConfig
from ..latent2gene.memmap_io import MemMapDense

logger = logging.getLogger("gsMap.spatial_ldsc.io")


def _read_sumstats(fh, alleles=False, dropna=False):
    """Parse GWAS summary statistics."""
    logger.info(f"Reading summary statistics from {fh} ...")

    # Determine compression type
    compression = None
    if fh.endswith("gz"):
        compression = "gzip"
    elif fh.endswith("bz2"):
        compression = "bz2"

    # Define columns and dtypes
    dtype_dict = {"SNP": str, "Z": float, "N": float, "A1": str, "A2": str}
    usecols = ["SNP", "Z", "N"]
    if alleles:
        usecols += ["A1", "A2"]

    # Read the file
    try:
        sumstats = pd.read_csv(
            fh,
            sep=r"\s+",
            na_values=".",
            usecols=usecols,
            dtype=dtype_dict,
            compression=compression,
        )
    except (AttributeError, ValueError) as e:
        logger.error(f"Failed to parse sumstats file: {str(e.args)}")
        raise ValueError("Improperly formatted sumstats file: " + str(e.args)) from e

    # Drop NA values if specified
    if dropna:
        sumstats = sumstats.dropna(how="any")

    logger.info(f"Read summary statistics for {len(sumstats)} SNPs.")

    # Drop duplicates
    m = len(sumstats)
    sumstats = sumstats.drop_duplicates(subset="SNP")
    if m > len(sumstats):
        logger.info(f"Dropped {m - len(sumstats)} SNPs with duplicated rs numbers.")

    return sumstats


def _read_chr_files(base_path, suffix, expected_count=22):
    """Read chromosome files using glob pattern matching."""
    # Create the pattern to search for files
    file_pattern = f"{base_path}[1-9]*{suffix}*"

    # Find all matching files
    all_files = glob.glob(file_pattern)

    # Extract chromosome numbers
    chr_files = []
    for file in all_files:
        try:
            # Extract the chromosome number from filename
            file_name = os.path.basename(file)
            base_name = os.path.basename(base_path)
            chr_part = file_name.replace(base_name, "").split(suffix)[0]
            chr_num = int(chr_part)
            if 1 <= chr_num <= expected_count:
                chr_files.append((chr_num, file))
        except (ValueError, IndexError):
            continue

    # Check if we have the expected number of chromosome files
    if len(chr_files) != expected_count:
        logger.warning(
            f"❗ SEVERE WARNING ❗ Expected {expected_count} chromosome files, but found {len(chr_files)}! "
            f"⚠️ For human GWAS data, all 22 autosomes must be present. Please verify your input files."
        )

    # Sort by chromosome number and return file paths
    chr_files.sort()
    return [file for _, file in chr_files]


def _read_file(file_path):
    """Read a file based on its format/extension."""
    try:
        if file_path.endswith(".feather"):
            return pd.read_feather(file_path)
        elif file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        elif file_path.endswith(".gz"):
            return pd.read_csv(file_path, compression="gzip", sep="\t")
        elif file_path.endswith(".bz2"):
            return pd.read_csv(file_path, compression="bz2", sep="\t")
        else:
            return pd.read_csv(file_path, sep="\t")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {str(e)}")
        raise


def _read_ref_ld_v2(ld_file):
    """Read reference LD scores for all chromosomes."""
    suffix = ".l2.ldscore"
    logger.info(f"Reading LD score annotations from {ld_file}[1-22]{suffix}...")

    # Get the chromosome files
    chr_files = _read_chr_files(ld_file, suffix)

    # Read and concatenate all files
    df_list = [_read_file(file) for file in chr_files]

    if not df_list:
        logger.error(f"No LD score files found matching pattern: {ld_file}*{suffix}*")
        raise FileNotFoundError(f"No LD score files found matching pattern: {ld_file}*{suffix}*")

    ref_ld = pd.concat(df_list, axis=0)
    logger.info(f"Loaded {len(ref_ld)} SNPs from LD score files")

    # Set SNP as index
    if "index" in ref_ld.columns:
        ref_ld.rename(columns={"index": "SNP"}, inplace=True)
    if "SNP" in ref_ld.columns:
        ref_ld.set_index("SNP", inplace=True)

    return ref_ld


def _read_w_ld(w_file):
    """Read LD weights for all chromosomes."""
    suffix = ".l2.ldscore"
    logger.info(f"Reading LD score annotations from {w_file}[1-22]{suffix}...")

    # Get the chromosome files
    chr_files = _read_chr_files(w_file, suffix)

    if not chr_files:
        logger.error(f"No LD score files found matching pattern: {w_file}*{suffix}*")
        raise FileNotFoundError(f"No LD score files found matching pattern: {w_file}*{suffix}*")

    # Read and process each file
    w_array = []
    for file in chr_files:
        x = _read_file(file)

        # Sort if possible
        if "CHR" in x.columns and "BP" in x.columns:
            x = x.sort_values(by=["CHR", "BP"])

        # Drop unnecessary columns
        columns_to_drop = ["MAF", "CM", "Gene", "TSS", "CHR", "BP"]
        columns_to_drop = [col for col in columns_to_drop if col in x.columns]
        if columns_to_drop:
            x = x.drop(columns=columns_to_drop, axis=1)

        w_array.append(x)

    # Concatenate and set column names
    w_ld = pd.concat(w_array, axis=0)
    logger.info(f"Loaded {len(w_ld)} SNPs from LD weight files")

    # Set column names
    w_ld.columns = (
        ["SNP", "LD_weights"] + list(w_ld.columns[2:])
        if len(w_ld.columns) > 2
        else ["SNP", "LD_weights"]
    )

    return w_ld


# ============================================================================
# Memory monitoring
# ============================================================================
import psutil

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

def generate_expected_output_filename(config: SpatialLDSCConfig, trait_name: str) -> Optional[str]:

    base_name = f"{config.project_name}_{trait_name}"

    # If we have cell indices range, include it in filename
    if config.cell_indices_range:
        start_cell, end_cell = config.cell_indices_range
        return f"{base_name}_cells_{start_cell}_{end_cell}.csv.gz"

    # If sample filter is set, filename will include sample info
    # but we can't predict exact start/end without loading data
    # For now, just check the simple complete case
    # If using sample filter, we might not be able to easily predict output name
    # without knowing the sample filtering results. 
    # But usually we don't skip in that case or logic is handled by caller.
    if config.sample_filter:
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




class LazyFeatherX:
    """
    A proxy for the 'X' matrix that slices directly from the Feather file
    via memory mapping without loading the full data.
    """

    def __init__(self, arrow_table, feature_names):
        self.table = arrow_table
        self.feature_names = feature_names
        # Standard AnnData shape: (n_obs, n_vars)
        self.shape = (self.table.num_rows, len(self.feature_names))

    def __getitem__(self, key):
        """
        Handles slicing: adata.X[start:end], adata.X[start:end, :], etc.
        """
        # Normalize the key to always be a tuple of (row_slice, col_slice)
        if not isinstance(key, tuple):
            row_key = key
            col_key = slice(None)  # Select all columns
        else:
            row_key, col_key = key

        # --- 1. Handle Row Slicing ---
        if isinstance(row_key, slice):
            start = row_key.start or 0
            stop = row_key.stop or self.shape[0]
            step = row_key.step or 1

            # Calculate length based on step (simplified for step=1)
            # For complex steps, we might need more logic, but basic slicing:
            if step == 1:
                length = stop - start
                sliced_table = self.table.slice(offset=start, length=length)
            else:
                # Fallback for stepped slicing: Read range, then step in pandas
                # (Slightly less efficient but works)
                length = stop - start
                sliced_table = self.table.slice(offset=start, length=length)
                # We will handle the stepping after conversion

        elif isinstance(row_key, int):
            # Single row request
            sliced_table = self.table.slice(offset=row_key, length=1)
        else:
            raise NotImplementedError("Only slice objects (start:stop) or integers are supported for rows.")

        # --- 2. Handle Column Slicing ---
        final_cols = self.feature_names

        # Helper to map integer indices to column names
        def get_col_names_by_indices(indices):
            return [self.feature_names[i] for i in indices]

        if isinstance(col_key, slice):
            final_cols = self.feature_names[col_key]
            sliced_table = sliced_table.select(final_cols)

        elif isinstance(col_key, (list, np.ndarray)):
            # Check if it's integers or strings
            if len(col_key) > 0:
                if isinstance(col_key[0], (int, np.integer)):
                    final_cols = get_col_names_by_indices(col_key)
                else:
                    # Assume strings
                    final_cols = col_key
            sliced_table = sliced_table.select(final_cols)

        elif isinstance(col_key, int):
            final_cols = [self.feature_names[col_key]]
            sliced_table = sliced_table.select(final_cols)

        # --- 3. Materialize to NumPy ---
        # FIX: We convert to Pandas first, because pyarrow.Table
        # doesn't have a direct to_numpy() for 2D structures.
        df = sliced_table.to_pandas()

        # If we had a row step > 1, apply it now on the small DataFrame
        if isinstance(row_key, slice) and row_key.step and row_key.step != 1:
            df = df.iloc[::row_key.step]

        return df.to_numpy()


class FeatherAnnData:
    """
    A minimal AnnData-like class backed by a Feather file.
    Mimics the behavior of anndata.AnnData without loading X into memory.
    """

    def __init__(self, file_path, index_col=None):
        # 1. Open with memory mapping (Zero RAM usage for data)
        self._table = feather.read_table(file_path, memory_map=True)

        # 2. Setup Index (Obs Names) and Columns (Var Names)
        all_cols = self._table.column_names

        if index_col:
            # Load the index column specifically
            self.obs_names = self._table.column(index_col).to_pylist()
            # The variables (genes) are all columns MINUS the index column
            self.var_names = [c for c in all_cols if c != index_col]
        else:
            # Fallback: Assume all columns are genes
            self.obs_names = [str(i) for i in range(self._table.num_rows)]
            self.var_names = all_cols

        # 3. Setup Metadata DataFrames
        self.obs = pd.DataFrame(index=self.obs_names)
        self.var = pd.DataFrame(index=self.var_names)

        # 4. Setup Attributes (n_obs, n_vars)
        self.n_obs = self._table.num_rows
        self.n_vars = len(self.var_names)

        # 5. Setup the Lazy X
        # We pass the table and the valid gene columns
        self.X = LazyFeatherX(self._table, self.var_names)

        # 6. Setup Shape
        self.shape = (self.n_obs, self.n_vars)

    def __repr__(self):
        return (f"FeatherAnnData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}\n"
                f"    obs: {list(self.obs.columns)}\n"
                f"    var: {list(self.var.columns)}\n"
                f"    uns: (Empty)\n"
                f"    obsm: (Empty)\n"
                f"    varm: (Empty)\n"
                f"    layers: (Empty)\n"
                f"    Backing: PyArrow Memory Mapping (Read-Only)")


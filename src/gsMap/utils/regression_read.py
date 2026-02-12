import glob
import logging
import os

import pandas as pd

logger = logging.getLogger("gsMap.utils.regression_read")


def _read_sumstats(fh, alleles=False, dropna=False):
    """Parse GWAS summary statistics."""
    fh = str(fh)
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
    file_path = str(file_path)
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
            x = x.drop(columns=columns_to_drop)

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

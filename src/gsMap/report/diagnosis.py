import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def convert_z_to_p(gwas_data):
    """Convert Z-scores to P-values."""
    z_scores = pd.to_numeric(gwas_data["Z"], errors="coerce").to_numpy()
    gwas_data["Z"] = z_scores
    gwas_data["P"] = norm.sf(np.abs(z_scores)) * 2
    min_p_value = 1e-300
    gwas_data["P"] = gwas_data["P"].clip(lower=min_p_value)
    return gwas_data


def load_gwas_data(sumstats_file, *, usecols=None, calculate_p: bool = True):
    """Load GWAS columns with the fastest available parser."""

    started = time.perf_counter()
    source = Path(sumstats_file)
    parser = "pyarrow"
    try:
        gwas_data = pd.read_csv(
            source,
            compression="gzip",
            sep="\t",
            usecols=usecols,
            engine="pyarrow",
        )
    except (ImportError, ValueError):
        parser = "pandas"
        logger.debug(f"PyArrow could not parse {source}; falling back to pandas")
        gwas_data = pd.read_csv(
            source,
            compression="gzip",
            sep="\t",
            usecols=usecols,
        )

    required_columns = {"SNP", "Z"}
    missing_columns = required_columns.difference(gwas_data.columns)
    if missing_columns:
        raise ValueError(
            f"GWAS file {source} is missing required columns: {sorted(missing_columns)}"
        )

    logger.info(
        f"Loaded {len(gwas_data):,} GWAS rows from {source.name} with {parser} "
        f"in {time.perf_counter() - started:.2f}s"
    )
    return convert_z_to_p(gwas_data) if calculate_p else gwas_data


def filter_snps(gwas_data_with_gene_annotation_sort, SUBSAMPLE_SNP_NUMBER):
    """Filter the SNPs based on significance levels."""
    pass_suggestive_line_mask = gwas_data_with_gene_annotation_sort["P"] < 1e-5
    pass_suggestive_line_number = pass_suggestive_line_mask.sum()

    if pass_suggestive_line_number > SUBSAMPLE_SNP_NUMBER:
        snps2plot = gwas_data_with_gene_annotation_sort[pass_suggestive_line_mask].SNP
        logger.info(
            f"To reduce the number of SNPs to plot, only {snps2plot.shape[0]} SNPs with P < 1e-5 are plotted."
        )
    else:
        snps2plot = gwas_data_with_gene_annotation_sort.head(SUBSAMPLE_SNP_NUMBER).SNP
        logger.info(
            f"To reduce the number of SNPs to plot, {snps2plot.shape[0]} SNPs with "
            f"the smallest P-values are plotted."
        )

    return snps2plot

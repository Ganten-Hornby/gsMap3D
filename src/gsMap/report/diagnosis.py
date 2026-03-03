import logging

import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def convert_z_to_p(gwas_data):
    """Convert Z-scores to P-values."""
    gwas_data["P"] = norm.sf(abs(gwas_data["Z"])) * 2
    min_p_value = 1e-300
    gwas_data["P"] = gwas_data["P"].clip(lower=min_p_value)
    return gwas_data


def load_gwas_data(sumstats_file):
    """Load and process GWAS data."""
    logger.info("Loading and processing GWAS data...")
    gwas_data = pd.read_csv(sumstats_file, compression="gzip", sep="\t")
    return convert_z_to_p(gwas_data)


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
            f"To reduce the number of SNPs to plot, only {SUBSAMPLE_SNP_NUMBER} SNPs with the smallest P-values are plotted."
        )

    return snps2plot

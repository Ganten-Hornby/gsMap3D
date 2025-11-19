"""
I/O utilities for loading genomic data in the LD score framework.

This module provides optimized readers for PLINK binary files and omics feature matrices.
"""

from pathlib import Path
from typing import Optional, Tuple
import logging

import bitarray as ba
import numpy as np
import pandas as pd
import pyranges as pr

logger = logging.getLogger(__name__)


class PlinkBEDReader:
    """
    Optimized reader for PLINK binary files.

    Reads genotypes directly into NumPy arrays suitable for JAX processing.
    Handles PLINK BED format with proper bitarray decoding and standardization.

    Attributes
    ----------
    bfile : str
        Base filename prefix for PLINK files
    bim : pd.DataFrame
        BIM file data (SNP information)
    fam : pd.DataFrame
        FAM file data (individual information)
    m : int
        Number of SNPs
    n : int
        Number of individuals
    bed_path : str
        Path to the BED file
    bytes_per_snp : int
        Bytes required per SNP in the BED file
    nru : int
        Rounded-up number of individuals for bitarray alignment
    _bedcode : dict
        Mapping for PLINK genotype encoding
    """

    def __init__(self, bfile_prefix: str):
        """
        Initialize PlinkBEDReader from a PLINK file prefix.

        Parameters
        ----------
        bfile_prefix : str
            PLINK file prefix (without .bed/.bim/.fam extension)

        Raises
        ------
        OSError
            If BED file has invalid magic number or SNP-major mode is not used
        FileNotFoundError
            If any of the required PLINK files are missing
        """
        self.bfile = bfile_prefix
        self.bim = self._load_bim(f"{bfile_prefix}.bim")
        self.fam = self._load_fam(f"{bfile_prefix}.fam")
        self.m = len(self.bim)
        self.n = len(self.fam)

        # BED file specifications
        self.bed_path = f"{bfile_prefix}.bed"
        self.bytes_per_snp = ((self.n + 3) // 4)

        # Calculate rounded-up number of individuals for bitarray alignment
        e = (4 - self.n % 4) if self.n % 4 != 0 else 0
        self.nru = self.n + e

        # PLINK BED encoding:
        # 00 -> homozygous major (0 copies of minor allele)
        # 01 -> missing genotype
        # 10 -> heterozygous (1 copy of minor allele)
        # 11 -> homozygous minor (2 copies of minor allele)
        self._bedcode = {
            2: ba.bitarray("11"),
            9: ba.bitarray("10"),  # 9 represents missing
            1: ba.bitarray("01"),
            0: ba.bitarray("00"),
        }

        # Validate BED header
        self._validate_bed_header()

    def _validate_bed_header(self) -> None:
        """
        Validate BED file header for magic number and SNP-major mode.

        Raises
        ------
        OSError
            If magic number is invalid or file is not in SNP-major mode
        FileNotFoundError
            If BED file does not exist
        """
        if not Path(self.bed_path).exists():
            raise FileNotFoundError(f"BED file not found: {self.bed_path}")

        with open(self.bed_path, "rb") as f:
            magic_number = ba.bitarray(endian="little")
            magic_number.fromfile(f, 2)
            bed_mode = ba.bitarray(endian="little")
            bed_mode.fromfile(f, 1)

            # Check magic number: expected 0x1b6c (in little-endian bitarray format)
            if magic_number != ba.bitarray("0011011011011000"):
                raise OSError(f"Invalid PLINK .bed magic number in {self.bed_path}")

            # Check SNP-major mode: expected 0x01
            if bed_mode != ba.bitarray("10000000"):
                raise OSError(f"PLINK .bed file must be in SNP-major mode: {self.bed_path}")

        logger.info(f"Validated BED file: {self.bed_path} ({self.m} SNPs × {self.n} individuals)")

    @staticmethod
    def _load_bim(path: str) -> pd.DataFrame:
        """
        Load BIM file containing SNP information.

        Parameters
        ----------
        path : str
            Path to the BIM file

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: CHR, SNP, CM, BP, A1, A2
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"BIM file not found: {path}")

        cols = ["CHR", "SNP", "CM", "BP", "A1", "A2"]
        return pd.read_csv(path, sep="\t", header=None, names=cols)

    @staticmethod
    def _load_fam(path: str) -> pd.DataFrame:
        """
        Load FAM file containing individual information.

        Parameters
        ----------
        path : str
            Path to the FAM file

        Returns
        -------
        pd.DataFrame
            DataFrame with column: IID (individual ID)
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"FAM file not found: {path}")

        return pd.read_csv(path, sep=r"\s+", header=None, usecols=[1], names=["IID"])

    def get_genotypes(self, snp_indices: np.ndarray, standardize: bool = True) -> np.ndarray:
        """
        Read specific SNPs and return standardized genotype matrix.

        Parameters
        ----------
        snp_indices : np.ndarray
            Indices of SNPs to read (0-based)
        standardize : bool, optional
            Whether to standardize genotypes to mean 0, variance 1 (default: True)

        Returns
        -------
        np.ndarray
            Genotype matrix of shape (n_individuals, n_snps) with dtype float32.
            Missing values are imputed with the mean before standardization.

        Notes
        -----
        PLINK BED encoding:
        - 00 -> 0 (homozygous major)
        - 01 -> 9 (missing, coded as 9)
        - 10 -> 1 (heterozygous)
        - 11 -> 2 (homozygous minor)
        """
        n_snps_to_read = len(snp_indices)

        if n_snps_to_read == 0:
            return np.zeros((self.n, 0), dtype=np.float32)

        # Validate indices
        if np.any(snp_indices < 0) or np.any(snp_indices >= self.m):
            raise ValueError(f"SNP indices must be in range [0, {self.m})")

        # Read genotypes for requested SNPs
        genotypes = []

        with open(self.bed_path, "rb") as f:
            for snp_idx in snp_indices:
                # Seek to SNP position (skip 3-byte header + previous SNPs)
                f.seek(3 + int(snp_idx) * 2 * self.nru)

                # Read bitarray for this SNP
                z = ba.bitarray(endian="little")
                z.fromfile(f, 2 * self.nru)

                # Decode genotypes using PLINK encoding
                # Extract every other bit (PLINK uses 2 bits per genotype)
                A = z[0::2]  # First bit of each genotype pair
                B = z[1::2]  # Second bit of each genotype pair

                # Decode to genotype values:
                # 00 (A=0, B=0) -> 0 (hom major)
                # 01 (A=1, B=0) -> 9 (missing)
                # 10 (A=0, B=1) -> 1 (het)
                # 11 (A=1, B=1) -> 2 (hom minor)
                genotype_vals = np.array(z.decode(self._bedcode), dtype=np.float32)
                genotype_vals = genotype_vals.reshape(self.nru)[: self.n]  # Trim padding

                genotypes.append(genotype_vals)

        # Stack into matrix: (n_individuals, n_snps)
        X = np.column_stack(genotypes)

        # Standardize if requested
        if standardize:
            X = self._standardize_genotypes(X)

        return X

    @staticmethod
    def _standardize_genotypes(X: np.ndarray) -> np.ndarray:
        """
        Standardize genotypes to mean 0, variance 1.

        Missing values (coded as 9) are imputed with the mean before standardization.

        Parameters
        ----------
        X : np.ndarray
            Genotype matrix of shape (n_individuals, n_snps)

        Returns
        -------
        np.ndarray
            Standardized genotype matrix
        """
        X_std = np.zeros_like(X, dtype=np.float32)

        for j in range(X.shape[1]):
            snp = X[:, j].copy()

            # Identify non-missing genotypes
            non_missing = snp != 9

            if np.sum(non_missing) == 0:
                # All missing - set to 0
                X_std[:, j] = 0
                continue

            # Calculate mean from non-missing values
            mean_val = np.mean(snp[non_missing])

            # Impute missing with mean
            snp[~non_missing] = mean_val

            # Calculate standard deviation
            std_val = np.std(snp)

            # Standardize (handle monomorphic SNPs)
            if std_val > 0:
                X_std[:, j] = (snp - mean_val) / std_val
            else:
                # Monomorphic SNP - set to 0
                X_std[:, j] = 0

        return X_std


def load_omics_features(h5ad_path: str) -> list[str]:
    """
    Load omics feature names from an AnnData H5AD file.

    Parameters
    ----------
    h5ad_path : str
        Path to the H5AD file

    Returns
    -------
    list[str]
        List of feature names (genes, peaks, etc.)

    Raises
    ------
    FileNotFoundError
        If the H5AD file does not exist
    """
    import anndata

    if not Path(h5ad_path).exists():
        raise FileNotFoundError(f"H5AD file not found: {h5ad_path}")

    adata = anndata.read_h5ad(h5ad_path, backed="r")
    feature_names = adata.var_names.tolist()

    # Close file handle
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    logger.info(f"Loaded {len(feature_names)} features from {h5ad_path}")

    return feature_names
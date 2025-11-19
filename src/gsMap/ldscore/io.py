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
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class PlinkBEDReader:
    """
    Optimized reader for PLINK binary files with filtering and pre-loading.

    Loads and filters genotypes during initialization, storing them as a
    standardized numpy array for efficient batch access. Follows the pattern
    from the original PlinkBEDFile implementation.

    Attributes
    ----------
    bfile : str
        Base filename prefix for PLINK files
    bim : pd.DataFrame
        BIM file data (SNP information, filtered)
    fam : pd.DataFrame
        FAM file data (individual information)
    m : int
        Number of SNPs (after filtering)
    n : int
        Number of individuals
    genotypes : np.ndarray
        Pre-loaded and standardized genotype matrix (n_individuals, m_snps)
    maf : np.ndarray
        Minor allele frequency for each SNP
    freq : np.ndarray
        Allele frequency for each SNP
    """

    def __init__(
        self,
        bfile_prefix: str,
        maf_min: Optional[float] = None,
        keep_snps: Optional[list[str]] = None,
        preload: bool = True
    ):
        """
        Initialize PlinkBEDReader with optional filtering.

        Parameters
        ----------
        bfile_prefix : str
            PLINK file prefix (without .bed/.bim/.fam extension)
        maf_min : float, optional
            Minimum MAF threshold for SNP filtering (default: None, no filtering)
        keep_snps : list[str], optional
            List of SNP IDs to keep (default: None, keep all)
        preload : bool, optional
            Whether to pre-load genotypes into memory (default: True)

        Raises
        ------
        OSError
            If BED file has invalid magic number or SNP-major mode is not used
        FileNotFoundError
            If any of the required PLINK files are missing
        """
        self.bfile = bfile_prefix
        self.bed_path = f"{bfile_prefix}.bed"

        # Load metadata
        logger.info(f"Loading PLINK files from: {bfile_prefix}")
        self.bim = self._load_bim(f"{bfile_prefix}.bim")
        self.fam = self._load_fam(f"{bfile_prefix}.fam")

        self.m_original = len(self.bim)
        self.n = len(self.fam)

        # Calculate alignment parameters
        e = (4 - self.n % 4) if self.n % 4 != 0 else 0
        self.nru = self.n + e
        self.bytes_per_snp = ((self.n + 3) // 4)

        # PLINK BED encoding
        self._bedcode = {
            2: ba.bitarray("11"),  # homozygous minor
            9: ba.bitarray("10"),  # missing
            1: ba.bitarray("01"),  # heterozygous
            0: ba.bitarray("00"),  # homozygous major
        }

        # Validate and load BED file
        self._validate_bed_header()
        logger.info("Loading all genotype data into memory...")
        self.geno_bitarray = self._read_all_genotypes()

        # Calculate MAF and filter SNPs
        logger.info("Calculating MAF and filtering SNPs...")
        self._calculate_maf()
        self._apply_filters(maf_min=maf_min, keep_snps=keep_snps)

        # Pre-load genotypes if requested
        self.genotypes = None
        if preload:
            logger.info(f"Pre-loading and standardizing {self.m} SNPs...")
            self.genotypes = self._load_and_standardize_all()
            logger.info(f"✓ Genotypes ready: {self.genotypes.shape}")

        logger.info(f"PlinkBEDReader initialized: {self.m} SNPs × {self.n} individuals")

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

        logger.info(f"Validated BED file: {self.bed_path} ({self.m_original} SNPs × {self.n} individuals)")

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

    def _read_all_genotypes(self) -> ba.bitarray:
        """
        Read entire BED file into bitarray.

        Returns
        -------
        ba.bitarray
            Complete genotype bitarray for all SNPs
        """
        with open(self.bed_path, "rb") as f:
            # Read header
            magic_number = ba.bitarray(endian="little")
            magic_number.fromfile(f, 2)
            bed_mode = ba.bitarray(endian="little")
            bed_mode.fromfile(f, 1)

            # Read all genotype data
            geno = ba.bitarray(endian="little")
            geno.fromfile(f)

        # Verify length
        expected_len = 2 * self.m_original * self.nru
        if len(geno) != expected_len:
            raise OSError(
                f"BED file has {len(geno)} bits, expected {expected_len} "
                f"for {self.m_original} SNPs × {self.nru} aligned individuals"
            )

        return geno

    def _calculate_maf(self) -> None:
        """
        Calculate MAF and allele frequencies for all SNPs.

        Updates self.maf, self.freq, and self.valid_snp arrays.
        """
        self.freq = np.zeros(self.m_original, dtype=np.float32)
        self.maf = np.zeros(self.m_original, dtype=np.float32)
        self.valid_snp = np.ones(self.m_original, dtype=bool)

        for j in range(self.m_original):
            # Extract bitarray for this SNP
            z = self.geno_bitarray[2 * j * self.nru : 2 * (j + 1) * self.nru]
            A = z[0::2]  # First bit
            B = z[1::2]  # Second bit

            # Count genotypes
            a = A.count()  # Count of 1s in first bit
            b = B.count()  # Count of 1s in second bit
            c = (A & B).count()  # Count of both 1s (homozygous minor)

            # Calculate allele frequency
            major_ct = b + c  # Number of copies of major allele
            n_nomiss = self.n - a + c  # Number of non-missing genotypes

            if n_nomiss > 0:
                f = major_ct / (2 * n_nomiss)
                self.freq[j] = f
                self.maf[j] = min(f, 1 - f)

                # Mark as invalid if all genotypes are missing
                if n_nomiss < self.n:
                    het_miss_ct = a + b - 2 * c
                    if het_miss_ct >= self.n:
                        self.valid_snp[j] = False
            else:
                # All missing - invalid SNP
                self.valid_snp[j] = False

    def _apply_filters(
        self,
        maf_min: Optional[float] = None,
        keep_snps: Optional[list[str]] = None
    ) -> None:
        """
        Apply SNP filters based on MAF and keep list.

        Parameters
        ----------
        maf_min : float, optional
            Minimum MAF threshold
        keep_snps : list[str], optional
            List of SNP IDs to keep
        """
        # Start with valid SNPs
        keep_mask = self.valid_snp.copy()

        # Apply MAF filter
        if maf_min is not None and maf_min > 0:
            maf_mask = self.maf >= maf_min
            keep_mask &= maf_mask
            n_removed = np.sum(~maf_mask & self.valid_snp)
            if n_removed > 0:
                logger.info(f"Filtered {n_removed} SNPs with MAF < {maf_min}")

        # Apply keep list
        if keep_snps is not None:
            keep_set = set(keep_snps)
            snp_mask = self.bim['SNP'].isin(keep_set)
            keep_mask &= snp_mask
            n_removed = np.sum(~snp_mask & self.valid_snp)
            if n_removed > 0:
                logger.info(f"Filtered {n_removed} SNPs not in keep list")

        # Count invalid SNPs
        n_invalid = np.sum(~keep_mask)
        if n_invalid > 0:
            logger.info(f"Total SNPs filtered: {n_invalid}/{self.m_original}")

        # Filter genotype bitarray
        filtered_geno = ba.bitarray()
        for j in np.arange(self.m_original)[keep_mask]:
            filtered_geno += self.geno_bitarray[
                2 * j * self.nru : 2 * (j + 1) * self.nru
            ]

        # Update state
        self.geno_bitarray = filtered_geno
        self.bim = self.bim.loc[keep_mask].reset_index(drop=True)
        self.freq = self.freq[keep_mask]
        self.maf = self.maf[keep_mask]
        self.m = len(self.bim)

        # Add MAF to BIM dataframe
        self.bim['MAF'] = self.maf

    def _load_and_standardize_all(self) -> jnp.ndarray:
        """
        Load all genotypes and standardize them using JAX-accelerated operations.

        Returns
        -------
        jnp.ndarray
            Standardized genotype matrix of shape (n_individuals, m_snps)

        Notes
        -----
        Uses JAX for GPU/CPU-accelerated matrix operations:
        - Replace missing (9) with NaN for masked operations
        - Compute means/stds using nanmean/nanstd
        - Impute and standardize using broadcasting
        - Returns JAX array for pure JAX pipeline
        """
        # Decode all genotypes at once (numpy for bitarray decode)
        decoded = np.array(self.geno_bitarray.decode(self._bedcode), dtype=np.float32)

        # Reshape to (m_snps, nru) then extract (m_snps, n)
        genotypes_raw = decoded.reshape((self.m, self.nru))[:, :self.n]

        # Transpose to (n_individuals, m_snps)
        X_np = genotypes_raw.T.copy()

        # Convert to JAX array for accelerated computation
        X = jnp.array(X_np, dtype=jnp.float32)

        # Replace missing values (9) with NaN for masked operations
        X = jnp.where(X == 9, jnp.nan, X)

        # Compute column-wise statistics (ignoring NaNs)
        # Shape: (m_snps,)
        means = jnp.nanmean(X, axis=0)
        stds = jnp.nanstd(X, axis=0)

        # Handle edge cases
        # 1. All-missing columns: set mean to 0
        all_missing = jnp.isnan(means)
        means = jnp.where(all_missing, 0.0, means)
        stds = jnp.where(all_missing, 1.0, stds)  # Avoid division by zero

        # 2. Monomorphic SNPs: set std to 1 to avoid division by zero
        monomorphic = (stds == 0)
        stds = jnp.where(monomorphic, 1.0, stds)

        # Impute missing values with column means using broadcasting
        # For each column, fill NaNs with the column's mean
        nan_mask = jnp.isnan(X)

        # Use where to replace NaNs with column means
        # Broadcasting means across rows
        X = jnp.where(nan_mask, means[jnp.newaxis, :], X)

        # Standardize: (X - mean) / std using broadcasting
        # means and stds have shape (m_snps,), broadcasting along columns
        X_std = (X - means[jnp.newaxis, :]) / stds[jnp.newaxis, :]

        # Set all-missing and monomorphic columns to zero
        edge_case_mask = all_missing | monomorphic
        X_std = jnp.where(edge_case_mask[jnp.newaxis, :], 0.0, X_std)

        # Return JAX array directly (no numpy conversion)
        return X_std

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

    def get_genotypes(
        self,
        snp_indices: Optional[np.ndarray] = None,
        snp_names: Optional[list[str]] = None
    ) -> jnp.ndarray:
        """
        Get genotype matrix for specified SNPs.

        If genotypes were pre-loaded during initialization, returns subset
        from memory. Otherwise, reads from disk.

        Parameters
        ----------
        snp_indices : np.ndarray, optional
            Indices of SNPs to retrieve (0-based). If None, returns all SNPs.
        snp_names : list[str], optional
            SNP IDs to retrieve. If provided, snp_indices is ignored.

        Returns
        -------
        jnp.ndarray
            Standardized genotype matrix of shape (n_individuals, n_snps)

        Raises
        ------
        ValueError
            If indices are out of bounds or SNP names not found
        RuntimeError
            If genotypes were not pre-loaded and on-demand reading is needed
        """
        # Handle SNP names
        if snp_names is not None:
            snp_set = set(snp_names)
            mask = self.bim['SNP'].isin(snp_set)
            if not np.any(mask):
                raise ValueError(f"None of the requested SNPs found in BIM file")
            snp_indices = np.where(mask)[0]
            logger.info(f"Found {len(snp_indices)}/{len(snp_names)} requested SNPs")

        # Use pre-loaded genotypes if available
        if self.genotypes is not None:
            if snp_indices is None:
                return self.genotypes
            else:
                # Validate indices
                if np.any(snp_indices < 0) or np.any(snp_indices >= self.m):
                    raise ValueError(f"SNP indices must be in range [0, {self.m})")
                return self.genotypes[:, snp_indices]
        else:
            raise RuntimeError(
                "Genotypes were not pre-loaded. Initialize with preload=True "
                "or call _load_and_standardize_all() first."
            )

    def get_all_genotypes(self) -> jnp.ndarray:
        """
        Get all standardized genotypes.

        Returns
        -------
        jnp.ndarray
            Complete genotype matrix of shape (n_individuals, m_snps)

        Raises
        ------
        RuntimeError
            If genotypes were not pre-loaded
        """
        if self.genotypes is None:
            raise RuntimeError(
                "Genotypes were not pre-loaded. Initialize with preload=True."
            )
        return self.genotypes


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
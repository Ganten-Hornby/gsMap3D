"""
I/O utilities for loading genomic data in the LD score framework.

This module provides optimized readers for PLINK binary files and omics feature matrices,
leveraging pandas-plink for efficient I/O and standardizing data with NumPy.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import logging

import numpy as np
import pandas as pd
import xarray as xr
from pandas_plink import read_plink1_bin

logger = logging.getLogger(__name__)


class PlinkBEDReader:
    """
    Optimized reader for PLINK binary files using pandas-plink and Xarray.

    Loads genotypes lazily via Dask/Xarray, applies matrix-based MAF calculation
    and filtering, and converts to standardized NumPy arrays.

    Attributes
    ----------
    bfile : str
        Base filename prefix for PLINK files
    G : xr.DataArray
        The underlying xarray DataArray (samples x variants) containing genotypes (0, 1, 2, nan)
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
    """

    def __init__(
        self,
        bfile_prefix: str,
        maf_min: Optional[float] = None,
        keep_snps: Optional[List[str]] = None,
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
            Whether to pre-load and standardize genotypes into memory (default: True)
        """
        self.bfile = bfile_prefix

        # Construct paths
        bed_path = f"{bfile_prefix}.bed"
        bim_path = f"{bfile_prefix}.bim"
        fam_path = f"{bfile_prefix}.fam"

        # Validate existence
        if not (Path(bed_path).exists() and Path(bim_path).exists() and Path(fam_path).exists()):
             raise FileNotFoundError(f"One or more PLINK files missing for prefix: {bfile_prefix}")

        logger.info(f"Loading PLINK files from: {bfile_prefix}")

        # Load using pandas-plink
        # This returns an xarray DataArray with dask backing (lazy loading)
        # Shape is (sample, variant)
        self.G = read_plink1_bin(bed_path, bim_path, fam_path, verbose=False)

        # Initial dimensions
        self.n_original = self.G.sizes['sample']
        self.m_original = self.G.sizes['variant']
        logger.info(f"Loaded metadata: {self.m_original} SNPs × {self.n_original} individuals")

        # Calculate MAF using matrix operations (lazy execution via dask)
        logger.info("Calculating MAF using matrix operations...")
        self.maf = self._calculate_maf()

        # Apply filters to the xarray DataArray
        self._apply_filters(maf_min=maf_min, keep_snps=keep_snps)

        # Update dimensions after filtering
        self.n = self.G.sizes['sample']
        self.m = self.G.sizes['variant']

        # Extract metadata DataFrames from xarray coordinates for compatibility
        self._sync_metadata()

        # Pre-load genotypes if requested
        self.genotypes = None
        if preload:
            logger.info(f"Pre-loading and standardizing {self.m} SNPs...")
            self.genotypes = self._load_and_standardize_all()
            logger.info(f"✓ Genotypes ready: {self.genotypes.shape}")

        logger.info(f"PlinkBEDReader initialized: {self.m} SNPs × {self.n} individuals")

    def _calculate_maf(self) -> xr.DataArray:
        """
        Calculate Minor Allele Frequency using matrix operations on the DataArray.
        """
        # Calculate mean across samples (axis 0), ignoring NaNs
        # The result 'freq_a1' represents the frequency of the A1 allele (coded as 2)
        freq_a1 = self.G.mean(dim="sample", skipna=True) / 2.0

        # Calculate MAF: min(f, 1-f)
        maf = xr.ufuncs.minimum(freq_a1, 1.0 - freq_a1)

        return maf.compute()

    def _apply_filters(
        self,
        maf_min: Optional[float] = None,
        keep_snps: Optional[List[str]] = None
    ) -> None:
        """
        Apply SNP filters directly to the xarray DataArray.
        """
        # 1. Create a boolean mask for variants
        variant_ids = self.G.variant.values
        mask = np.ones(len(variant_ids), dtype=bool)

        # 2. Apply MAF filter
        if maf_min is not None and maf_min > 0:
            maf_mask = (self.maf >= maf_min).values
            mask &= maf_mask

            n_removed_maf = np.sum(~maf_mask)
            if n_removed_maf > 0:
                logger.info(f"Filtered {n_removed_maf} SNPs with MAF < {maf_min}")

        # 3. Apply Keep List filter
        if keep_snps is not None:
            current_snps = self.G.snp.values
            keep_set = set(keep_snps)
            snp_mask = np.isin(current_snps, list(keep_set))

            mask &= snp_mask
            n_removed_snp = np.sum(~snp_mask)
            if n_removed_snp > 0:
                logger.info(f"Filtered {n_removed_snp} SNPs not in keep list")

        # 4. Filter the main DataArray
        n_before = self.G.sizes['variant']
        self.G = self.G.isel(variant=mask)
        self.maf = self.maf[mask]

        n_after = self.G.sizes['variant']
        if n_before != n_after:
            logger.info(f"Total SNPs filtered: {n_before - n_after}/{n_before}")

    def _sync_metadata(self):
        """Extract filtered BIM and FAM dataframes from xarray coordinates."""
        # Use standard PLINK column names: CHR, SNP, CM, BP, A1, A2
        # pandas-plink maps 'a0' and 'a1'. Usually a1 is the effect allele in PLINK 1.9 usage,
        # but conventions vary. We map 'a1' -> 'A1', 'a0' -> 'A2'.
        self.bim = pd.DataFrame({
            'CHR': self.G.chrom.values,
            'SNP': self.G.snp.values,
            'CM': self.G.cm.values,
            'BP': self.G.pos.values,
            'A1': self.G.a1.values,
            'A2': self.G.a0.values,
            'i': np.arange(self.m)
        })
        self.bim['MAF'] = self.maf.values

        # pandas-plink stores FAM info in coordinates
        self.fam = pd.DataFrame({
            'fid': self.G.fid.values,
            'iid': self.G.iid.values,
            'father': self.G.father.values,
            'mother': self.G.mother.values,
            'gender': self.G.gender.values,
            'trait': self.G.trait.values
        })

    def _load_and_standardize_all(self) -> np.ndarray:
        """
        Load genotypes from Dask into memory, convert to NumPy, and standardize.

        Returns
        -------
        np.ndarray
            Standardized genotype matrix of shape (n_individuals, m_snps)
        """
        logger.info("Reading filtered genotype matrix into memory...")
        X = self.G.values.astype(np.float32)

        # 3. Compute stats (ignoring NaNs)
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)

        # 4. Handle edge cases
        all_missing = np.isnan(means)
        means = np.where(all_missing, 0.0, means)
        stds = np.where(all_missing, 1.0, stds)

        # Monomorphic SNPs
        monomorphic = (stds == 0)
        stds = np.where(monomorphic, 1.0, stds)

        # 5. Impute and Standardize
        # Impute missing values with column means
        nan_mask = np.isnan(X)
        # Broadcasting means: (m,) -> (1, m) to match (n, m)
        X = np.where(nan_mask, means[np.newaxis, :], X)

        # Standardize: (X - mean) / std
        X_std = (X - means[np.newaxis, :]) / stds[np.newaxis, :]

        # Zero out edge cases
        edge_case_mask = all_missing | monomorphic
        X_std = np.where(edge_case_mask[np.newaxis, :], 0.0, X_std)

        return X_std

    def get_genotypes(
        self,
        snp_indices: Optional[np.ndarray] = None,
        snp_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Get genotype matrix for specified SNPs.
        """
        if snp_names is not None:
            # Find indices based on SNP IDs
            current_snps = self.bim['SNP'].values
            mask = np.isin(current_snps, snp_names)
            if not np.any(mask):
                raise ValueError(f"None of the requested SNPs found")
            snp_indices = np.where(mask)[0]

        if self.genotypes is not None:
            if snp_indices is None:
                return self.genotypes
            return self.genotypes[:, snp_indices]
        else:
            raise RuntimeError(
                "Genotypes were not pre-loaded. Initialize with preload=True."
            )

    def get_all_genotypes(self) -> np.ndarray:
        if self.genotypes is None:
            raise RuntimeError("Genotypes were not pre-loaded.")
        return self.genotypes
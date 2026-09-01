"""PLINK I/O and variant QC for LD-score calculations."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin

logger = logging.getLogger(__name__)


def biallelic_snp_mask(allele0: np.ndarray, allele1: np.ndarray) -> np.ndarray:
    """Return a mask for distinct, single-nucleotide A/C/G/T allele pairs."""
    allele0 = np.char.upper(np.asarray(allele0).astype(str))
    allele1 = np.char.upper(np.asarray(allele1).astype(str))
    if allele0.shape != allele1.shape:
        raise ValueError("allele arrays must have identical shapes")
    bases = np.array(["A", "C", "G", "T"])
    return np.isin(allele0, bases) & np.isin(allele1, bases) & (allele0 != allele1)


@dataclass(frozen=True)
class VariantStatistics:
    """Per-variant statistics calculated from hard-called diploid genotypes."""

    called_samples: np.ndarray
    mac: np.ndarray
    maf: np.ndarray
    means: np.ndarray
    stds: np.ndarray


def calculate_variant_statistics(genotypes: np.ndarray) -> VariantStatistics:
    """Calculate called-sample count, MAC, MAF, mean, and population SD.

    Missing genotypes must be represented by ``NaN``. MAC is calculated from
    the number of observed alleles for each variant, rather than assuming every
    individual was called.
    """
    if genotypes.ndim != 2:
        raise ValueError("genotypes must be a two-dimensional sample-by-variant matrix")

    called_samples = np.sum(~np.isnan(genotypes), axis=0, dtype=np.int64)
    allele_sum = np.nansum(genotypes, axis=0, dtype=np.float64)
    called_alleles = 2 * called_samples

    minor_allele_count = np.minimum(allele_sum, called_alleles - allele_sum)
    # PLINK BED stores hard calls, so rounding protects against floating-point
    # summation noise without changing the definition of MAC.
    mac = np.rint(np.maximum(minor_allele_count, 0)).astype(np.int64)

    maf = np.zeros(genotypes.shape[1], dtype=np.float64)
    np.divide(mac, called_alleles, out=maf, where=called_alleles > 0)

    means = np.zeros(genotypes.shape[1], dtype=np.float32)
    np.divide(
        allele_sum,
        called_samples,
        out=means,
        where=called_samples > 0,
        casting="unsafe",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        stds = np.nanstd(genotypes, axis=0).astype(np.float32, copy=False)

    return VariantStatistics(
        called_samples=called_samples,
        mac=mac,
        maf=maf,
        means=means,
        stds=stds,
    )


class PlinkBEDReader:
    """Load, QC, and standardize one chromosome from PLINK 1 binary files.

    The BED payload is materialized exactly once. The previous implementation
    decoded it repeatedly for MAF, basic QC, and standardization; on network
    storage those redundant passes dominated chromosome runtime.
    """

    def __init__(
        self,
        bfile_prefix: str,
        snps_only: bool = True,
        mac_min: int | None = 5,
        maf_min: float | None = None,
        keep_snps: list[str] | None = None,
        preload: bool = True,
    ):
        """Initialize the reader.

        Parameters
        ----------
        bfile_prefix
            PLINK prefix without ``.bed/.bim/.fam``.
        snps_only
            Retain only biallelic single-nucleotide A/C/G/T variants.
        mac_min
            Retain variants with MAC strictly greater than this value. The
            default therefore implements ``MAC > 5``. ``None`` disables it.
        maf_min
            Optional inclusive MAF threshold. Disabled by default.
        keep_snps
            Optional SNP-ID allow-list, applied before reading the BED payload.
        preload
            Store the standardized genotype matrix on ``self.genotypes``.
            Metadata and QC still require one eager BED read when false.
        """
        if mac_min is not None and mac_min < 0:
            raise ValueError("mac_min must be non-negative or None")
        if maf_min is not None and not 0 <= maf_min <= 0.5:
            raise ValueError("maf_min must be between 0 and 0.5 or None")

        self.bfile = bfile_prefix
        bed_path = f"{bfile_prefix}.bed"
        bim_path = f"{bfile_prefix}.bim"
        fam_path = f"{bfile_prefix}.fam"
        if not all(Path(path).is_file() for path in (bed_path, bim_path, fam_path)):
            raise FileNotFoundError(f"One or more PLINK files missing for prefix: {bfile_prefix}")

        reader_started = time.perf_counter()
        logger.info("Loading PLINK files from: %s", bfile_prefix)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*delim_whitespace.*", category=FutureWarning
            )
            self.G = read_plink1_bin(bed_path, bim_path, fam_path, verbose=False)

        self.n_original = self.G.sizes["sample"]
        self.m_original = self.G.sizes["variant"]
        self.snp_ids_original = pd.Index(self.G.snp.values)
        logger.info("Loaded metadata: %d SNPs × %d individuals", self.m_original, self.n_original)

        metadata_mask = np.ones(self.m_original, dtype=bool)
        if snps_only:
            snp_mask = biallelic_snp_mask(self.G.a0.values, self.G.a1.values)
            metadata_mask &= snp_mask
            logger.info(
                "SNP-only filter before BED decoding: retained %d; removed %d indels/non-SNPs",
                int(snp_mask.sum()),
                int((~snp_mask).sum()),
            )
        if keep_snps is not None:
            allow_mask = self.snp_ids_original.isin(set(keep_snps))
            metadata_mask &= allow_mask
            logger.info(
                "SNP allow-list before BED decoding: matched %d of %d variants",
                int(allow_mask.sum()),
                self.m_original,
            )
        if not np.all(metadata_mask):
            self.G = self.G.isel(variant=metadata_mask)
        if self.G.sizes["variant"] == 0:
            raise ValueError("No variants remain after SNP-only/allow-list filtering")

        logger.info(
            "Reading genotype matrix once for QC and standardization: %d SNPs",
            self.G.sizes["variant"],
        )
        decode_started = time.perf_counter()
        raw_genotypes = np.asarray(self.G.values, dtype=np.float32)
        decode_seconds = time.perf_counter() - decode_started
        statistics_started = time.perf_counter()
        statistics = calculate_variant_statistics(raw_genotypes)
        statistics_seconds = time.perf_counter() - statistics_started

        valid = (statistics.called_samples > 0) & np.isfinite(statistics.stds)
        valid &= statistics.stds > 0
        basic_qc_removed = int((~valid).sum())

        if mac_min is not None:
            mac_pass = statistics.mac > mac_min
            logger.info(
                "MAC > %d: retained %d; removed %d",
                mac_min,
                int(mac_pass.sum()),
                int((~mac_pass).sum()),
            )
            valid &= mac_pass

        if maf_min is not None:
            maf_pass = statistics.maf >= maf_min
            logger.info(
                "MAF >= %.6g: retained %d; removed %d",
                maf_min,
                int(maf_pass.sum()),
                int((~maf_pass).sum()),
            )
            valid &= maf_pass

        if basic_qc_removed:
            logger.info("Basic QC removed %d monomorphic/all-missing variants", basic_qc_removed)
        if not np.any(valid):
            raise ValueError("No variants remain after MAC/MAF/basic-QC filtering")

        if np.all(valid):
            filtered_genotypes = raw_genotypes
        else:
            filtered_genotypes = np.ascontiguousarray(raw_genotypes[:, valid])
        self.G = self.G.isel(variant=valid)
        self.maf = statistics.maf[valid]
        self.mac = statistics.mac[valid]
        self.called_samples = statistics.called_samples[valid]
        self.n = self.G.sizes["sample"]
        self.m = self.G.sizes["variant"]
        self._sync_metadata()

        self.genotypes: np.ndarray | None = None
        standardize_started = time.perf_counter()
        if preload:
            means = statistics.means[valid]
            stds = statistics.stds[valid]
            missing = np.isnan(filtered_genotypes)
            if np.any(missing):
                np.copyto(filtered_genotypes, means[None, :], where=missing)
            filtered_genotypes -= means
            filtered_genotypes /= stds
            self.genotypes = np.ascontiguousarray(filtered_genotypes, dtype=np.float32)
        standardize_seconds = time.perf_counter() - standardize_started

        logger.info(
            "PlinkBEDReader initialized: %d SNPs × %d individuals in %.2fs "
            "(BED decode %.2fs; statistics %.2fs; standardize %.2fs)",
            self.m,
            self.n,
            time.perf_counter() - reader_started,
            decode_seconds,
            statistics_seconds,
            standardize_seconds,
        )

    def _sync_metadata(self) -> None:
        self.bim = pd.DataFrame(
            {
                "CHR": self.G.chrom.values,
                "SNP": self.G.snp.values,
                "CM": self.G.cm.values,
                "BP": self.G.pos.values,
                "A1": self.G.a1.values,
                "A2": self.G.a0.values,
                "i": np.arange(self.m),
                "MAF": self.maf,
                "MAC": self.mac,
                "N_CALLED": self.called_samples,
            }
        )
        self.fam = pd.DataFrame(
            {
                "fid": self.G.fid.values,
                "iid": self.G.iid.values,
                "father": self.G.father.values,
                "mother": self.G.mother.values,
                "gender": self.G.gender.values,
                "trait": self.G.trait.values,
            }
        )

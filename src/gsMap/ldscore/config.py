"""
Configuration dataclasses for the general LD score framework.

This module defines the configuration structure for LD score calculation,
including paths, mapping strategies, and computational parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class LDScoreConfig:
    """
    Configuration for LD score weight matrix calculation.

    This dataclass encapsulates all parameters needed for the modular LD score
    framework, including genomic data paths, SNP-feature mapping strategies,
    and JAX acceleration settings.

    Attributes
    ----------
    bfile_root : str
        PLINK file prefix for reference panel genotypes
    hm3_snp_path : str
        Path to HapMap3 SNP list file
    output_dir : str
        Directory for saving output files
    omics_h5ad_path : str, optional
        Path to omics feature matrix in H5AD format
    mapping_type : str
        Type of SNP-feature mapping ('bed' or 'dict')
    mapping_file : str, optional
        Path to mapping file (BED format or dictionary)
    window_size : int
        Genomic window size in base pairs for BED mapping (default: 0)
    strategy : str
        Mapping strategy: 'score', 'tss', or 'distance' (default: 'score')
    remove_repeats : bool
        Whether to enforce unique SNP-feature assignments (default: True)
    chromosomes : Union[str, List[int]]
        Chromosomes to process: 'all' or list of chromosome numbers
    batch_size_hm3 : int
        Batch size for HapMap3 SNPs in JAX processing (default: 1024)
    use_gpu : bool
        Whether to use GPU acceleration via JAX (default: False)
    quantization_num_bins : int
        Number of discrete window sizes for JIT quantization (default: 20)

    Examples
    --------
    >>> config = LDScoreConfig(
    ...     bfile_root="/data/1000G_EUR/chr",
    ...     hm3_snp_path="/data/hapmap3_snps.txt",
    ...     output_dir="/results/ldscore",
    ...     mapping_type="bed",
    ...     window_size=100000,
    ...     use_gpu=True
    ... )
    """
    # Paths
    bfile_root: str  # Reference panel prefix
    hm3_snp_path: str  # Path to HM3 SNP list
    output_dir: str

    # Omics Input
    omics_h5ad_path: Optional[str] = None

    # Mapping Input
    # 'bed' or 'dict'
    mapping_type: str = "bed"
    mapping_file: Optional[str] = None

    # Mapping Strategy parameters
    window_size: int = 0  # bp window
    strategy: str = "score"  # 'score', 'tss', 'distance'
    remove_repeats: bool = True

    # Computation
    chromosomes: Union[str, List[int]] = "all"
    batch_size_hm3: int = 1024

    # JAX/Performance
    use_gpu: bool = False
    quantization_num_bins: int = 20  # How many discrete window sizes to allow for JIT
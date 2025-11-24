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

    # Computation
    chromosomes: Union[str, List[int]] = "all"
    batch_size_hm3: int = 50

    # JAX/Performance
    use_gpu: bool = False

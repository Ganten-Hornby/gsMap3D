from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict
from pathlib import Path


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
    remove_repeats: bool = True

    # Computation
    chromosomes: Union[str, List[int]] = "all"
    batch_size_hm3: int = 1024

    # JAX/Performance
    use_gpu: bool = False
    quantization_num_bins: int = 20  # How many discrete window sizes to allow for JIT
"""
Configuration dataclasses for the general LD score framework.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class LDScoreConfig:
    """
    Configuration for LD Score pipeline.
    """

    # Paths
    bfile_root: str  # Reference panel prefix template (e.g., 'data/1000G.{chr}')
    hm3_snp_path: str  # Path to HM3 SNP list
    output_dir: str
    output_filename: str = "ld_score_weights" # Prefix for output files

    # Omics Input
    omics_h5ad_path: Optional[str] = None

    # Mapping Input (Strategy A/B)
    # 'bed' or 'dict'
    mapping_type: str = "bed"
    mapping_file: Optional[str] = None

    # Annotation Input (Strategy C - Direct Annotation Matrix)
    # Template for annotation files (e.g., 'baseline.{chr}.annot.gz')
    annot_file: Optional[str] = None

    # Mapping Strategy parameters
    window_size: int = 0  # bp window for mapping (e.g. TSS window)
    strategy: str = "score"  # 'score', 'tss', 'center', 'allow_repeat'

    # LD Calculation parameters
    window_size_bp: int = 1_000_000  # LD window size
    maf_min: float = 0.01  # Minimum MAF filter

    # Computation
    chromosomes: Union[str, List[int]] = "all"
    batch_size_hm3: int = 50

    def __post_init__(self):
        """
        Post-initialization processing:
        1. Parse chromosome string to list.
        2. Fix PLINK file template.
        3. Validate file existence.
        """
        # 1. Parse Chromosomes
        if self.chromosomes == "all":
            self.chromosomes = list(range(1, 23))
        elif isinstance(self.chromosomes, str):
            # Handle string input like "1,2,22"
            self.chromosomes = [int(x) for x in self.chromosomes.split(',')]

        # 2. Handle PLINK Prefix Template
        # Ensure bfile_root has {chr} placeholder if it's treated as a template
        if "{chr}" not in self.bfile_root and "{chromosome}" not in self.bfile_root:
            # If it looks like a prefix "data/chr", append "{chr}"
            # Careful not to double append if user meant "data/1000G" -> "data/1000G.{chr}"
            # A safe heuristic: append ".{chr}" or "{chr}" depending on trailing characters
            logger.info(f"Appending placeholder to bfile_root: {self.bfile_root} -> {self.bfile_root}.{{chr}}")
            self.bfile_root = f"{self.bfile_root}.{{chr}}"

        # 3. Validate PLINK Files
        logger.info("Validating PLINK binary files based on template...")
        missing_chroms = []

        for chrom in self.chromosomes:
            try:
                # Try formatting with 'chr' first (standard)
                prefix = self.bfile_root.format(chr=chrom)
            except KeyError:
                # Fallback to 'chromosome'
                prefix = self.bfile_root.format(chromosome=chrom)

            bed_path = Path(f"{prefix}.bed")
            if not bed_path.exists():
                missing_chroms.append(str(chrom))

        if missing_chroms:
            logger.warning(
                f"PLINK .bed files missing for {len(missing_chroms)} chromosomes: {', '.join(missing_chroms)}. "
                "These will be skipped during processing."
            )
        else:
            logger.info(f"Confirmed all PLINK files exist for {len(self.chromosomes)} chromosomes.")
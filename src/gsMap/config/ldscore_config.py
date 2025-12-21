"""
Configuration dataclasses for the general LD score framework.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Annotated
import logging
import typer

from gsMap.config.base import ConfigWithAutoPaths

logger = logging.getLogger(__name__)


@dataclass
class LDScoreConfig:
    """
    Configuration for LD Score pipeline.
    """

    # Paths
    bfile_root: Annotated[str, typer.Option(
        help="Reference panel prefix template (e.g., 'data/1000G.{chr}')"
    )]

    hm3_snp_path: Annotated[Path, typer.Option(
        help="Path to HM3 SNP list",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )]

    output_dir: Annotated[Optional[Path], typer.Option(
        help="Output directory. If None, uses {workdir}/{project_name}/generate_ldscore"
    )] = None

    output_filename: Annotated[str, typer.Option(
        help="Prefix for output files"
    )] = "ld_score_weights"

    # Omics Input
    omics_h5ad_path: Annotated[Optional[Path], typer.Option(
        help="Path to omics H5AD file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

    # Mapping Input (Strategy A/B)
    mapping_type: Annotated[str, typer.Option(
         help="Mapping type: 'bed' or 'dict'"
    )] = "bed"

    mapping_file: Annotated[Optional[Path], typer.Option(
        help="Path to mapping file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

    # Annotation Input (Strategy C - Direct Annotation Matrix)
    annot_file: Annotated[Optional[str], typer.Option(
        help="Template for annotation files (e.g., 'baseline.{chr}.annot.gz')"
    )] = None

    # Mapping Strategy parameters
    feature_window_size: Annotated[int, typer.Option(
        help="bp window for mapping (e.g. TSS window)"
    )] = 0

    strategy: Annotated[str, typer.Option(
        help="Strategy: 'score', 'tss', 'center', 'allow_repeat'"
    )] = "score"

    # LD Calculation parameters
    ld_wind: Annotated[float, typer.Option(
        help="LD window size"
    )] = 1.0

    ld_unit: Annotated[str, typer.Option(
        help="LD unit: 'SNP', 'KB', 'CM'"
    )] = "CM"

    maf_min: Annotated[float, typer.Option(
        help="Minimum MAF filter"
    )] = 0.01

    # Computation
    chromosomes: Annotated[str, typer.Option(
        help="Chromosomes to process. 'all' uses 1-22 autosomes, or provide a comma-separated list of chromosomes (e.g., '1,2,3')"
    )] = "all"

    batch_size_hm3: Annotated[int, typer.Option(
        help="Batch size for HM3 SNPs"
    )] = 50

    # w_ld Calculation
    calculate_w_ld: Annotated[bool, typer.Option(
        help="Whether to calculate w_ld"
    )] = False

    w_ld_dir: Annotated[Optional[Path], typer.Option(
        help="Directory for w_ld outputs"
    )] = None

    def __post_init__(self):
        """
        Post-initialization processing:
        1. Initialize base auto-paths.
        2. Set default output_dir if not provided.
        3. Parse chromosome string to list.
        4. Fix PLINK file template.
        5. Validate file existence.
        """
        # 1. Base auto-paths (sets project_dir)
        super().__post_init__()

        # 2. Set default output_dir
        if self.output_dir is None:
            self.output_dir = self.ldscore_save_dir
            logger.info(f"Using default output directory: {self.output_dir}")
        else:
            self.output_dir = Path(self.output_dir)

        # 3. Parse Chromosomes
        if self.chromosomes == "all":
            self.chromosomes = list(range(1, 23))
        elif isinstance(self.chromosomes, str):
            # Handle string input like "1,2,22"
            self.chromosomes = [int(x) for x in self.chromosomes.split(',')]
        # Else it's already a list or properly set

        # 4. Handle PLINK Prefix Template
        # Ensure bfile_root has {chr} placeholder if it's treated as a template
        if "{chr}" not in self.bfile_root and "{chromosome}" not in self.bfile_root:
            logger.info(f"Appending placeholder to bfile_root: {self.bfile_root} -> {self.bfile_root}.{{chr}}")
            self.bfile_root = f"{self.bfile_root}.{{chr}}"

        # 5. Validate PLINK Files
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


@dataclass
class GenerateLDScoreConfig(ConfigWithAutoPaths):
    """Configuration for generating LD scores."""
    
    # Required from parent
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]

    chrom: Annotated[str, typer.Option(
        help='Chromosome id (1-22) or "all"'
    )]
    
    bfile_root: Annotated[str, typer.Option(
        help="Root path for genotype plink bfiles (.bim, .bed, .fam)"
    )]
    
    gtf_annotation_file: Annotated[Path, typer.Option(
        help="Path to GTF annotation file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )]

    sample_name: Annotated[str, typer.Option(
        help="Name of the sample"
    )] = None

    keep_snp_root: Optional[str] = None  # Internal field
    
    gene_window_size: Annotated[int, typer.Option(
        help="Gene window size in base pairs",
        min=1000,
        max=1000000
    )] = 50000
    
    enhancer_annotation_file: Annotated[Optional[Path], typer.Option(
        help="Path to enhancer annotation file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None
    
    snp_multiple_enhancer_strategy: Annotated[str, typer.Option(
        help="Strategy for handling multiple enhancers per SNP",
        case_sensitive=False
    )] = "max_mkscore"
    
    gene_window_enhancer_priority: Annotated[Optional[str], typer.Option(
        help="Priority between gene window and enhancer annotations"
    )] = None
    
    additional_baseline_annotation: Annotated[Optional[str], typer.Option(
        help="Path of additional baseline annotations"
    )] = None
    
    spots_per_chunk: Annotated[int, typer.Option(
        help="Number of spots per chunk",
        min=100,
        max=10000
    )] = 1000
    
    ld_wind: Annotated[int, typer.Option(
        help="LD window size",
        min=1,
        max=10
    )] = 1
    
    ld_unit: Annotated[str, typer.Option(
        help="Unit for LD window",
        case_sensitive=False
    )] = "CM"
    
    # Additional fields
    ldscore_save_format: str = "feather"
    save_pre_calculate_snp_gene_weight_matrix: bool = False
    baseline_annotation_dir: Optional[str] = None
    SNP_gene_pair_dir: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.show_config("Generate LDScore Configuration")


def check_ldscore_done(config: GenerateLDScoreConfig) -> bool:
    """
    Check if generate_ldscore step is done.
    """
    # Assuming it's done if w_ld directory exists and has files
    w_ld_dir = Path(config.ldscore_save_dir) / "w_ld"
    if not w_ld_dir.exists():
        return False
    
    # Check if there are any .l2.ldscore.gz files
    return any(w_ld_dir.glob("*.l2.ldscore.gz"))

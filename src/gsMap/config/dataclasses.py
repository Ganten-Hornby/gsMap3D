"""
Configuration dataclasses for gsMap commands.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated, List, Literal
from collections import OrderedDict
import logging
import typer

from .base import ConfigWithAutoPaths
from .utils import (
    configure_jax_platform,
    get_anndata_shape,
    inspect_h5ad_structure,
    validate_h5ad_structure,
    process_h5ad_inputs,
    verify_homolog_file_format
)

# Import module-specific configs for orchestration
from gsMap.find_latent.config import FindLatentRepresentationsConfig
from gsMap.latent2gene.config import LatentToGeneConfig
from gsMap.spatial_ldsc.config import SpatialLDSCConfig


logger = logging.getLogger("gsMap.config")


@dataclass
class RunAllModeConfig(ConfigWithAutoPaths):
    """Configuration for running the complete gsMap pipeline in a single command."""
    
    # Required from parent
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]

    sample_name: Annotated[str, typer.Option(
        help="Scientific name of the sample"
    )]
    
    trait_name: Annotated[str, typer.Option(
        help="Name of the trait for GWAS analysis"
    )]
    
    sumstats_file: Annotated[Path, typer.Option(
        help="Path to GWAS summary statistics file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )]
    
    h5ad_yaml: Annotated[Path, typer.Option(
        help="YAML file with sample names and h5ad paths",
        exists=True,
        file_okay=True,
        dir_okay=False
    )]
    
    annotation: Annotated[str, typer.Option(
        help="Annotation of cell type in adata.obs to use"
    )]

    # Additional options
    use_gpu: Annotated[bool, typer.Option(
        "--use-gpu/--no-gpu",
        help="Use GPU for JAX computations"
    )] = True

    @property
    def find_latent_config(self) -> FindLatentRepresentationsConfig:
        return FindLatentRepresentationsConfig(
            workdir=self.workdir,
            project_name=self.project_name,
            h5ad_yaml=self.h5ad_yaml,
            annotation=self.annotation
        )

    @property
    def latent2gene_config(self) -> LatentToGeneConfig:
        return LatentToGeneConfig(
            workdir=self.workdir,
            project_name=self.project_name,
            annotation=self.annotation,
            use_gpu=self.use_gpu
        )

    @property
    def spatial_ldsc_config(self) -> SpatialLDSCConfig:
        return SpatialLDSCConfig(
            workdir=self.workdir,
            project_name=self.project_name,
            trait_name=self.trait_name,
            sumstats_file=self.sumstats_file
        )


@dataclass
class CauchyCombinationConfig(ConfigWithAutoPaths):
    """Configuration for Cauchy combination test."""
    
    trait_name: Annotated[str, typer.Option(
        help="Name of the trait being analyzed"
    )]
    
    annotation: Annotated[str, typer.Option(
        help="Name of the annotation in adata.obs to use"
    )]
    
    sample_name_list: Annotated[Optional[str], typer.Option(
        help="Space-separated list of sample names"
    )] = None
    
    output_file: Annotated[Optional[Path], typer.Option(
        help="Path to save the combined Cauchy results"
    )] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Handle sample_name_list
        if self.sample_name_list and isinstance(self.sample_name_list, str):
            self.sample_name_list = self.sample_name_list.split()
        
        if self.sample_name is not None:
             # This assumes sample_name is added to ConfigWithAutoPaths or used as intended
             # For now, following original logic:
             self.sample_name_list = [self.sample_name]
             if self.output_file is None:
                self.output_file = Path(
                    f"{self.cauchy_save_dir}/{self.project_name}_single_sample_{self.sample_name}_{self.trait_name}.Cauchy.csv.gz"
                )
        else:
            if not self.sample_name_list:
                raise ValueError("At least one sample name must be provided via sample_name or sample_name_list.")
            
            if self.output_file is None:
                self.output_file = Path(
                    f"{self.cauchy_save_dir}/{self.project_name}_all_samples_{self.trait_name}.Cauchy.csv.gz"
                )


@dataclass
class CreateSliceMeanConfig:
    """Configuration for creating slice mean from multiple h5ad files."""
    
    slice_mean_output_file: Annotated[Path, typer.Option(
        help="Path to the output file for the slice mean"
    )]
    
    sample_name_list: Annotated[str, typer.Option(
        help="Space-separated list of sample names"
    )]
    
    h5ad_list: Annotated[str, typer.Option(
        help="Space-separated list of h5ad file paths"
    )]
    
    # Optional parameters
    h5ad_yaml: Annotated[Optional[Path], typer.Option(
        help="Path to YAML file with sample names and h5ad paths",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None
    
    homolog_file: Annotated[Optional[Path], typer.Option(
        help="Path to homologous gene conversion file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None
    
    data_layer: Annotated[str, typer.Option(
        help="Data layer for gene expression"
    )] = "counts"
    
    species: Optional[str] = None
    h5ad_dict: Optional[dict] = None
    
    def __post_init__(self):
        # Parse lists if provided as strings
        if isinstance(self.sample_name_list, str):
            self.sample_name_list = self.sample_name_list.split()
        if isinstance(self.h5ad_list, str):
            self.h5ad_list = self.h5ad_list.split()
        
        if self.h5ad_list is None and self.h5ad_yaml is None:
            raise ValueError("At least one of --h5ad_list or --h5ad_yaml must be provided.")
        
        import yaml
        if self.h5ad_yaml is not None:
            if isinstance(self.h5ad_yaml, (str, Path)):
                logger.info(f"Reading h5ad yaml file: {self.h5ad_yaml}")
                with open(self.h5ad_yaml) as f:
                    h5ad_dict = yaml.safe_load(f)
            else:
                h5ad_dict = self.h5ad_yaml
        elif self.sample_name_list and self.h5ad_list:
            logger.info("Reading sample name list and h5ad list")
            h5ad_dict = dict(zip(self.sample_name_list, self.h5ad_list, strict=False))
        else:
            raise ValueError(
                "Please provide either h5ad_yaml or both sample_name_list and h5ad_list."
            )
        
        # Check if sample names are unique
        assert len(h5ad_dict) == len(set(h5ad_dict)), "Sample names must be unique."
        assert len(h5ad_dict) > 1, "At least two samples are required."
        
        logger.info(f"Input h5ad files: {h5ad_dict}")
        
        # Check if all files exist
        self.h5ad_dict = {}
        for sample_name, h5ad_file in h5ad_dict.items():
            h5ad_file = Path(h5ad_file)
            if not h5ad_file.exists():
                raise FileNotFoundError(f"{h5ad_file} does not exist.")
            self.h5ad_dict[sample_name] = h5ad_file
        
        self.slice_mean_output_file = Path(self.slice_mean_output_file)
        self.slice_mean_output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Verify homolog file format if provided
        if self.homolog_file is not None:
            verify_homolog_file_format(self)


@dataclass
class FormatSumstatsConfig:
    """Configuration for formatting GWAS summary statistics."""
    
    sumstats: Annotated[Path, typer.Option(
        help="Path to the summary statistics file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )]
    
    out: Annotated[Path, typer.Option(
        help="Path to save the formatted summary statistics"
    )]
    
    # Optional parameters
    n_col: Annotated[Optional[str], typer.Option(
        help="Column name for sample size"
    )] = None
    
    n_val: Annotated[Optional[float], typer.Option(
        help="Constant sample size value"
    )] = None
    
    snp_col: Annotated[str, typer.Option(
        help="Column name for SNP ID"
    )] = "SNP"
    
    a1_col: Annotated[str, typer.Option(
        help="Column name for effect allele"
    )] = "A1"
    
    a2_col: Annotated[str, typer.Option(
        help="Column name for other allele"
    )] = "A2"
    
    p_col: Annotated[str, typer.Option(
        help="Column name for p-value"
    )] = "P"
    
    signed_sumstats: Annotated[Optional[str], typer.Option(
        help="Column name for signed summary statistics (e.g., Z, BETA, OR)"
    )] = None
    
    def __post_init__(self):
        if self.n_col is None and self.n_val is None:
            raise ValueError("One of --n-col or --n-val must be provided.")
        if self.n_col is not None and self.n_val is not None:
            raise ValueError("Only one of --n-col or --n-val must be provided.")


@dataclass
class DiagnosisConfig(ConfigWithAutoPaths):
    """Configuration for diagnosis command."""
    # Placeholder for diagnosis config fields
    pass


@dataclass
class VisualizeConfig(ConfigWithAutoPaths):
    """Configuration for visualization command."""
    # Placeholder for visualization config fields
    pass


@dataclass
class ThreeDCombineConfig(ConfigWithAutoPaths):
    """Configuration for 3D combine command."""
    # Placeholder for 3D combine config fields
    pass


@dataclass
class RunLinkModeConfig(ConfigWithAutoPaths):
    """Configuration for running gsMap with linked mode."""
    # Placeholder for link mode config fields
    pass


@dataclass
class gsMapPipelineConfig(ConfigWithAutoPaths):
    """Unified configuration for the complete gsMap pipeline"""

    # Component configurations
    find_latent: FindLatentRepresentationsConfig = None
    latent2gene: LatentToGeneConfig = None
    spatial_ldsc: SpatialLDSCConfig = None

    def __post_init__(self):
        super().__post_init__()
        # Initialize component configs if they weren't provided
        if self.find_latent is None:
            self.find_latent = FindLatentRepresentationsConfig(
                workdir=self.workdir,
                project_name=self.project_name
            )
        if self.latent2gene is None:
            self.latent2gene = LatentToGeneConfig(
                workdir=self.workdir,
                project_name=self.project_name
            )
        if self.spatial_ldsc is None:
            self.spatial_ldsc = SpatialLDSCConfig(
                workdir=self.workdir,
                project_name=self.project_name
            )
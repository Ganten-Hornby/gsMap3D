"""
Configuration for Quick Mode pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Annotated, List, Literal, OrderedDict, Dict

import typer
import logging
import yaml

from gsMap.config.base import ConfigWithAutoPaths
# Use relative imports to avoid circular dependency
from .find_latent_config import FindLatentRepresentationsConfig
from .latent2gene_config import LatentToGeneConfig
from .ldscore_config import GenerateLDScoreConfig
from .spatial_ldsc_config import SpatialLDSCConfig
from .report_config import ReportConfig
from .cauchy_config import CauchyCombinationConfig
from gsMap.config.utils import process_h5ad_inputs

logger = logging.getLogger("gsMap.config")

@dataclass
class QuickModeConfig(ConfigWithAutoPaths):
    """Configuration for running the complete gsMap pipeline in a single command."""

    # ------------------------------------------------------------------------
    # Pipeline Control
    # ------------------------------------------------------------------------
    start_step: Annotated[str, typer.Option(
        help="Step to start execution from (find_latent, latent2gene, spatial_ldsc, cauchy, report)",
        case_sensitive=False
    )] = "find_latent"

    stop_step: Annotated[Optional[str], typer.Option(
        help="Step to stop execution at (inclusive)",
        case_sensitive=False
    )] = None

    # ------------------------------------------------------------------------
    # Find Latent Representations Parameters
    # ------------------------------------------------------------------------
    h5ad_path: Annotated[Optional[List[Path]], typer.Option(
        help="Space-separated list of h5ad file paths. Sample names are derived from file names without suffix.",
        exists=True,
        file_okay=True,
    )] = None

    h5ad_yaml: Annotated[Path, typer.Option(
        help="YAML file with sample names and h5ad paths",
        exists=True,
        file_okay=True,
        dir_okay=False,
    )] = None

    h5ad_list_file: Annotated[Optional[Path], typer.Option(
        help="Each row is a h5ad file path, sample name is the file name without suffix",
        exists=True,
        file_okay=True,
        dir_okay=False,
    )] = None

    data_layer: Annotated[str, typer.Option(
        help="Gene expression raw counts data layer in h5ad layers, e.g., 'count', 'counts'. Otherwise use 'X' for adata.X"
    )] = "X"

    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm storing spatial coordinates"
    )] = "spatial"

    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation of cell type in adata.obs to use"
    )] = None

    n_neighbors: Annotated[int, typer.Option(
        help="Number of neighbors for LGCN",
        min=1,
        max=50
    )] = 10

    n_cell_training: Annotated[int, typer.Option(
        help="Number of cells used for training",
        min=1000,
        max=1000000
    )] = 100000

    homolog_file: Annotated[Optional[Path], typer.Option(
        help="Path to homologous gene conversion file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None

    high_quality_cell_qc: Annotated[bool, typer.Option(
        "--high-quality-cell-qc/--no-high-quality-cell-qc",
        help="Enable/disable high quality cell QC based on module scores. If enabled, it will likewise enable high_quality_neighbor_filter in latent2gene."
    )] = True

    # ------------------------------------------------------------------------
    # Latent To Gene Parameters
    # ------------------------------------------------------------------------
    dataset_type: Annotated[str, typer.Option(
        help="Type of dataset: scRNA (uses KNN on latent space), spatial2D (2D spatial), or spatial3D (multi-slice)",
        case_sensitive=False
    )] = 'spatial2D'

    latent_representation_niche: Annotated[str, typer.Option(
        help="Key for spatial niche embedding in obsm"
    )] = "emb_niche"

    latent_representation_cell: Annotated[str, typer.Option(
        help="Key for cell identity embedding in obsm"
    )] = "emb_cell"

    num_neighbour_spatial: Annotated[int, typer.Option(
        help="k1: Number of spatial neighbors in it's own slice for spatial dataset",
        min=10,
        max=500
    )] = 201

    num_homogeneous: Annotated[int, typer.Option(
        help="k3: Number of homogeneous neighbors per cell (for spatial) or KNN neighbors (for scRNA-seq)",
        min=1,
        max=100
    )] = 21

    n_adjacent_slices: Annotated[int, typer.Option(
        help="Number of slices to search above and below for 3D data",
        min=0,
        max=5
    )] = 1

    k_adjacent: Annotated[int, typer.Option(
        help="Number of neighbors to find on each adjacent slice for 3D data",
        min=10,
        max=100
    )] = 100

    memmap_tmp_dir: Annotated[Optional[Path], typer.Option(
        help="Temporary directory for memory-mapped files to improve I/O performance on slow filesystems.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )] = None


    # ------------------------------------------------------------------------
    # Spatial LDSC Parameters
    # ------------------------------------------------------------------------
    trait_name: Annotated[Optional[str], typer.Option(
        help="Name of the trait for GWAS analysis. Required if sumstats_file is provided."
    )] = None

    sumstats_file: Annotated[Optional[Path], typer.Option(
        help="Path to GWAS summary statistics file. Only one of sumstats_file or sumstats_config_file should be provided.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

    sumstats_config_file: Annotated[Optional[Path], typer.Option(
        help="Path to sumstats config file (YAML) mapping trait names to sumstats files. Only one of sumstats_file or sumstats_config_file should be provided.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

    snp_gene_weight_adata_path: Annotated[Path, typer.Option(
        help="Path to the SNP-gene weight matrix (H5AD format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

    w_ld_dir: Annotated[Optional[Path], typer.Option(
        help="Directory containing the weights files (w_ld)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )] = None

    # ------------------------------------------------------------------------
    # Report Parameters
    # ------------------------------------------------------------------------
    top_corr_genes: Annotated[int, typer.Option(
        help="Number of top correlated genes to display",
        min=1,
        max=500
    )] = 50

    selected_genes: Annotated[Optional[str], typer.Option(
        help="Comma-separated list of specific genes to include"
    )] = None

    # ------------------------------------------------------------------------
    # Common / Optimization
    # ------------------------------------------------------------------------
    use_gpu: Annotated[bool, typer.Option(
        "--use-gpu/--no-gpu",
        help="Use GPU for JAX computations"
    )] = True

    num_processes: Annotated[int, typer.Option(
        help="Number of processes for parallel execution",
        min=1
    )] = 4

    # Internal state
    sample_h5ad_dict: Optional[OrderedDict] = None
    sumstats_config_dict: Dict[str, Path] = field(default_factory=dict)


    def __post_init__(self):
        super().__post_init__()
        
        # Process h5ad inputs if provided (for FindLatent step)
        input_options = {
            'h5ad_yaml': ('h5ad_yaml', 'yaml'),
            'h5ad_path': ('h5ad_path', 'list'),
            'h5ad_list_file': ('h5ad_list_file', 'file'),
        }
        has_input = any(getattr(self, k) is not None for k in ['h5ad_path', 'h5ad_yaml', 'h5ad_list_file'])
        if has_input:
            self.sample_h5ad_dict = process_h5ad_inputs(self, input_options)
            
        # Process sumstats inputs
        if self.sumstats_file is None and self.sumstats_config_file is None:
            # We don't raise error immediately because user might only run find_latent or latent2gene
             pass
        elif self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError(
                "Only one of sumstats_file and sumstats_config_file must be provided."
            )
        elif self.sumstats_file is not None:
             if self.trait_name is None:
                 raise ValueError("trait_name must be provided if sumstats_file is provided.")
             self.sumstats_config_dict = {self.trait_name: self.sumstats_file}
        elif self.sumstats_config_file is not None:
             if self.trait_name is not None:
                 raise ValueError("trait_name must not be provided if sumstats_config_file is provided.")
             with open(self.sumstats_config_file) as f:
                config_loaded = yaml.load(f, Loader=yaml.FullLoader)
                for t_name, s_file in config_loaded.items():
                    self.sumstats_config_dict[t_name] = Path(s_file)

    @property
    def find_latent_config(self) -> FindLatentRepresentationsConfig:
        return FindLatentRepresentationsConfig(
            workdir=self.workdir,
            project_name=self.project_name,
            h5ad_path=self.h5ad_path,
            h5ad_yaml=self.h5ad_yaml,
            h5ad_list_file=self.h5ad_list_file,
            data_layer=self.data_layer,
            spatial_key=self.spatial_key,
            annotation=self.annotation,
            n_neighbors=self.n_neighbors,
            n_cell_training=self.n_cell_training,
            homolog_file=self.homolog_file,
            high_quality_cell_qc=self.high_quality_cell_qc,
            use_gpu=False, # FindLatent typically uses Torch/PyG. Default is used.
        )

    @property
    def latent2gene_config(self) -> LatentToGeneConfig:
        return LatentToGeneConfig(
            workdir=self.workdir,
            project_name=self.project_name,
            dataset_type=self.dataset_type,
            annotation=self.annotation,
            latent_representation_niche=self.latent_representation_niche,
            latent_representation_cell=self.latent_representation_cell,
            num_neighbour_spatial=self.num_neighbour_spatial,
            num_homogeneous=self.num_homogeneous,
            n_adjacent_slices=self.n_adjacent_slices,
            k_adjacent=self.k_adjacent,
            memmap_tmp_dir=self.memmap_tmp_dir,
            find_neighbor_within_high_quality=self.high_quality_cell_qc,
            use_gpu=self.use_gpu,
            compute_workers=self.num_processes,
        )

    @property
    def spatial_ldsc_config(self) -> SpatialLDSCConfig:        
        return SpatialLDSCConfig(
            workdir=self.workdir,
            project_name=self.project_name,
            trait_name=self.trait_name,
            sumstats_file=self.sumstats_file,
            sumstats_config_file=self.sumstats_config_file,
            w_ld_dir=self.w_ld_dir, 
            snp_gene_weight_adata_path=self.snp_gene_weight_adata_path,
            use_jax=self.use_gpu, 
            num_processes=self.num_processes,
            memmap_tmp_dir=self.memmap_tmp_dir,
        )

    def get_report_config(self, trait_name: str, sumstats_file: Path) -> ReportConfig:
        return ReportConfig(
            workdir=self.workdir,
            project_name=self.project_name,
            sample_name=self.project_name,
            trait_name=trait_name,
            annotation=self.annotation or "cluster", 
            sumstats_file=sumstats_file,
            top_corr_genes=self.top_corr_genes,
            selected_genes=self.selected_genes,
        )

    def get_cauchy_config(self, trait_name: str) -> CauchyCombinationConfig:
        return CauchyCombinationConfig(
            workdir=self.workdir,
            project_name=self.project_name,
            trait_name=trait_name,
            annotation=self.annotation,
            sample_name_list=[self.project_name],
        )

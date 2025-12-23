"""
Configuration for latent to gene mapping.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Annotated, List, OrderedDict, Literal
import yaml
import logging
import typer

from gsMap.config.base import ConfigWithAutoPaths
from gsMap.config.utils import (
    configure_jax_platform,
    process_h5ad_inputs,
    validate_h5ad_structure,
    get_anndata_shape
)

logger = logging.getLogger("gsMap.config")

class DatasetType(str, Enum):
    SCRNA_SEQ = 'scRNA'
    SPATIAL_2D = 'spatial2D'
    SPATIAL_3D = 'spatial3D'

class MarkerScoreCrossSliceStrategy(str, Enum):
    SIMILARITY_ONLY = 'similarity_only'
    WEIGHTED_MEAN_POOLING = 'weighted_mean_pooling'
    MAX_POOLING = 'max_pooling'

@dataclass
class LatentToGeneComputeConfig:
    """Compute configuration for latent-to-gene step."""

    use_gpu: Annotated[bool, typer.Option(
        "--use-gpu/--no-gpu",
        help="Use GPU for JAX computations (requires sufficient GPU memory)"
    )] = True

    memmap_tmp_dir: Annotated[Optional[Path], typer.Option(
        help="Temporary directory for memory-mapped files to improve I/O performance on slow filesystems. "
             "If provided, memory maps will be copied to this directory for faster random access during computation.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )] = None

    # Batch sizes
    rank_batch_size: int = 500
    mkscore_batch_size: int = 500
    find_homogeneous_batch_size: int = 100
    rank_write_interval: int = 10

    # Worker configurations
    rank_read_workers: Annotated[int, typer.Option(
        help="Number of parallel reader threads for rank memory map",
        min=1,
        max=16
    )] = 10

    mkscore_compute_workers: Annotated[int, typer.Option(
        help="Number of parallel compute threads for marker score calculation",
        min=1,
        max=16
    )] = 4

    mkscore_write_workers: Annotated[int, typer.Option(
        help="Number of parallel writer threads for marker scores",
        min=1,
        max=16
    )] = 4

    compute_input_queue_size: Annotated[int, typer.Option(
        help="Maximum size of compute input queue (multiplier of mkscore_compute_workers)",
        min=1,
        max=10
    )] = 5

    writer_queue_size: Annotated[int, typer.Option(
        help="Maximum size of writer input queue",
        min=10,
        max=500
    )] = 100

@dataclass
class LatentToGeneCoreConfig:

    dataset_type: Annotated[DatasetType, typer.Option(
        help="Type of dataset: scRNA (uses KNN on latent space), spatial2D (2D spatial), or spatial3D (multi-slice)",
        case_sensitive=False
    )] = 'spatial2D'

    # --------input h5ad file paths which have the latent representations
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

    sample_h5ad_dict: Optional[OrderedDict] = None

    # --------input h5ad obs, obsm, layers keys

    annotation: Annotated[Optional[str], typer.Option(
        help="Cell type annotation in adata.obs to use. This would constrain finding homogeneous spots within each cell type"
    )] = None

    data_layer: Annotated[str, typer.Option(
        help="Gene expression raw counts data layer in h5ad layers, e.g., 'count', 'counts'. Other wise use 'X' for adata.X"
    )] = "X"

    latent_representation_niche: Annotated[Optional[str], typer.Option(
        help="Key for spatial niche embedding in obsm"
    )] = None

    latent_representation_cell: Annotated[str, typer.Option(
        help="Key for cell identity embedding in obsm"
    )] = "emb_cell"

    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm"
    )] = "spatial"

    # --------parameters for finding homogeneous spots

    spatial_neighbors: Annotated[int, typer.Option(
        help="k1: Number of spatial neighbors in it's own slice for spatial dataset",
        min=10,
        max=500
    )] = 201

    homogeneous_neighbors: Annotated[int, typer.Option(
        help="k3: Number of homogeneous neighbors per cell (for spatial) or KNN neighbors (for scRNA-seq)",
        min=1,
        max=100
    )] = 21

    cell_embedding_similarity_threshold: Annotated[float, typer.Option(
        help="Minimum similarity threshold for cell embedding.",
        min=0.0,
        max=1.0
    )] = 0.0

    spatial_domain_similarity_threshold: Annotated[float, typer.Option(
        help="Minimum similarity threshold for spatial domain embedding.",
        min=0.0,
        max=1.0
    )] = 0.5

    no_expression_fraction: Annotated[bool, typer.Option(
        "--no-expression-fraction",
        help="Skip expression fraction filtering"
    )] = False

    # --------3D slice-aware neighbor search parameters
    adjacent_slice_spatial_neighbors: Annotated[int, typer.Option(
        help="Number of spatial neighbors to find on each adjacent slice for 3D data",
        min=10,
        max=500
    )] = 100

    n_adjacent_slices: Annotated[int, typer.Option(
        help="Number of adjacent slices to search above and below (± n_adjacent_slices) in 3D space for each focal spot. Padding will be applied automatically.",
        min=0,
        max=5
    )] = 1

    cross_slice_marker_score_strategy: Annotated[MarkerScoreCrossSliceStrategy, typer.Option(
        help="Strategy for computing marker scores across slices in spatial3D datasets. "
             "'similarity_only': Select top homogeneous neighbors from all slices combined based on similarity scores. "
             "'weighted_mean_pooling': Select fixed number of homogeneous neighbors per slice, compute weighted average using similarity as weights. "
             "'max_pooling': Select fixed number of homogeneous neighbors per slice, take maximum marker score across slices.",
        case_sensitive=False
    )] = MarkerScoreCrossSliceStrategy.WEIGHTED_MEAN_POOLING

    high_quality_neighbor_filter: Annotated[bool, typer.Option(
        "--high-quality-neighbor-filter/--no-high-quality-filter",
        help="Only find neighbors within high quality cells (requires High_quality column in obs)"
    )] = False

    fix_cross_slice_homogenous_neighbors: bool = False

    # Performance options
    # Minimum number of cells per cell type in that dataset to be used for finding homogeneous neighbors
    min_cells_per_type: int | None = None

    enable_profiling: bool = False


@dataclass
class LatentToGeneConfig(LatentToGeneComputeConfig, LatentToGeneCoreConfig, ConfigWithAutoPaths):
    """Configuration for latent to gene mapping.
    
    Inherits compute/IO fields from LatentToGeneComputeConfig.
    """

    @property
    def total_homogeneous_neighbor_per_cell(self):
        return self.homogeneous_neighbors * (1 + 2 * self.n_adjacent_slices)

    def __post_init__(self):
        """Initialize and validate configuration"""
        super().__post_init__()

        # Step 1: Configure JAX platform
        configure_jax_platform(self.use_gpu)

        # Step 2: Process and validate h5ad inputs
        self._process_h5ad_inputs()

        # Step 3: Configure dataset-specific parameters first
        self._configure_dataset_parameters()

        # Step 4: Set up validation fields and validate structure (after dataset config)
        self._setup_and_validate_fields()

        self.show_config("Latent to Gene Configuration")

    def _process_h5ad_inputs(self):
        """Process h5ad inputs from various sources"""

        # Define input options
        input_options = {
            'h5ad_yaml': ('h5ad_yaml', 'yaml'),
            'h5ad_path': ('h5ad_path', 'list'),
            'h5ad_list_file': ('h5ad_list_file', 'file'),
        }

        # Process h5ad inputs
        self.sample_h5ad_dict = process_h5ad_inputs(self, input_options)

        # Auto-detect from latent directory if no inputs provided
        if not self.sample_h5ad_dict:
            self._auto_detect_h5ad_files()

        # Validate at least one sample exists
        if len(self.sample_h5ad_dict) == 0:
            raise ValueError("No valid samples found in the provided input")

        logger.info(f"Loaded and validated {len(self.sample_h5ad_dict)} samples")

    def _auto_detect_h5ad_files(self):
        """Auto-detect h5ad files from latent directory"""
        if self.find_latent_metadata_path.exists():
            import yaml
            with open(self.find_latent_metadata_path, 'r') as f:
                find_latent_metadata = yaml.safe_load(f)
            self.sample_h5ad_dict = OrderedDict(
                {sample_name: Path(latent_file)
                 for sample_name, latent_file in
                 find_latent_metadata['outputs']['latent_files'].items()
                 })
            # assert all files exist
            for sample_name, latent_file in self.sample_h5ad_dict.items():
                if not latent_file.exists():
                    raise FileNotFoundError(f"Latent file not found for sample '{sample_name}': {latent_file}")
            logger.info(
                f"Auto-detected {len(self.sample_h5ad_dict)} samples from find_latent_metadata_path: {self.find_latent_metadata_path}")
        else:
            self.sample_h5ad_dict = OrderedDict()
            latent_dir = self.latent_dir
            logger.info(f"No input options provided. Auto-detecting h5ad files from latent directory: {latent_dir}")

            # Look for latent files with different naming patterns
            latent_files = list(latent_dir.glob("*_latent_adata.h5ad"))
            if not latent_files:
                latent_files = list(latent_dir.glob("*_add_latent.h5ad"))

            if not latent_files:
                raise ValueError(
                    f"No h5ad files found in latent directory {latent_dir}. "
                    f"Please run the find latent representation first. "
                    f"Or provide one of: h5ad_yaml, h5ad_path, or h5ad_list_file, which points to h5ad files which contain the latent embedding."
                )

            # Extract sample names from file names
            for latent_file in latent_files:
                sample_name = self._extract_sample_name(latent_file)
                self.sample_h5ad_dict[sample_name] = latent_file

            # sort by sample name
            self.sample_h5ad_dict = OrderedDict(sorted(self.sample_h5ad_dict.items()))

        logger.info(f"Auto-detected {len(self.sample_h5ad_dict)} samples from latent directory")

    def _extract_sample_name(self, latent_file):
        """Extract sample name from latent file path"""
        filename = latent_file.stem

        # Remove known suffixes
        suffixes_to_remove = ["_latent_adata", "_add_latent"]
        for suffix in suffixes_to_remove:
            if filename.endswith(suffix):
                return filename[:-len(suffix)]

        return filename

    def _setup_and_validate_fields(self):
        """Set up required/optional fields and validate h5ad structure"""
        # Define required fields
        required_fields = {
            'latent_representation_cell': ('obsm', self.latent_representation_cell,
                                           'Latent representation of cell identity'),
            'spatial_key': ('obsm', self.spatial_key, 'Spatial key'),
        }

        # Add annotation as required if provided
        if self.annotation:
            required_fields['annotation'] = ('obs', self.annotation, 'Annotation')

        # Add niche representation as required if provided
        if self.latent_representation_niche:
            required_fields['latent_representation_niche'] = (
                'obsm',
                self.latent_representation_niche,
                'Latent representation of spatial niche'
            )

        # Add High_quality as required if find_neighbor_within_high_quality is enabled
        if self.high_quality_neighbor_filter:
            required_fields['High_quality'] = ('obs', 'High_quality', 'High quality cell indicator')

        # Validate h5ad structure
        validate_h5ad_structure(self.sample_h5ad_dict, required_fields)

    def _configure_dataset_parameters(self):
        """Configure parameters based on dataset type"""
        self.min_cells_per_type = self.homogeneous_neighbors if self.min_cells_per_type is None else min(self.min_cells_per_type, self.homogeneous_neighbors)

        if self.dataset_type == DatasetType.SPATIAL_2D:
            self._configure_spatial_2d()
        elif self.dataset_type == DatasetType.SPATIAL_3D:
            self._configure_spatial_3d()
        elif self.dataset_type == DatasetType.SCRNA_SEQ:
            self._configure_scrna_seq()

    def _configure_spatial_2d(self):
        """Configure parameters for spatial 2D datasets"""
        # spatial2D can have multiple slices but doesn't search across them
        if self.n_adjacent_slices != 0:
            self.n_adjacent_slices = 0
            logger.info(
                "Dataset type is spatial2D. This will only search homogeneous neighbors within each 2D slice (no cross-slice search). Setting adjacent_slices=0.")

        if self.latent_representation_niche is None:
            logger.warning("latent_representation_niche is not provided. Spatial domain similarity will not be used.")

        assert self.homogeneous_neighbors <= self.spatial_neighbors, \
            f"homogeneous_neighbors ({self.homogeneous_neighbors}) must be <= spatial_neighbors ({self.spatial_neighbors}) for spatial2D datasets"

    def _configure_spatial_3d(self):
        """Configure parameters for spatial 3D datasets"""
        if self.n_adjacent_slices == 0:
            raise ValueError(
                "Dataset type is spatial3D, but adjacent_slices=0. "
                "You must set adjacent_slices to 1 or higher to enable cross-slice search. "
                "If you don't want cross-slice search, use dataset_type='spatial2D' instead."
            )

        if self.latent_representation_niche is None:
            logger.warning("latent_representation_niche is not provided. Spatial domain similarity will not be used.")

        assert self.adjacent_slice_spatial_neighbors <= self.spatial_neighbors, \
            f"adjacent_slice_neighbors ({self.adjacent_slice_spatial_neighbors}) must be <= spatial_neighbors ({self.spatial_neighbors})"
        assert self.homogeneous_neighbors <= self.adjacent_slice_spatial_neighbors, \
            f"homogeneous_neighbors ({self.homogeneous_neighbors}) must be <= adjacent_slice_neighbors ({self.adjacent_slice_spatial_neighbors})"

        n_slices = 1 + self.n_adjacent_slices  # only focal + above slices
        assert n_slices <= len(self.sample_h5ad_dict), \
            f"3D Cross slice search requires at least {n_slices} slices (1 focal + {self.n_adjacent_slices} above or {self.n_adjacent_slices} below). " \
            f"Only {len(self.sample_h5ad_dict)} samples provided. Please provide more slices or reduce adjacent_slices."

        logger.info(
            f"Dataset type is spatial3D, using adjacent_slices={self.n_adjacent_slices} for cross-slice search")
        logger.info(f"The Z axis order of slices is determined by the h5ad input order. Currently, the order is: ")
        logger.info(f"{' -> '.join(list(self.sample_h5ad_dict.keys()))}")

        # Adjust num_homogeneous based on adjacent slices and strategy
        # Only multiply for 'similarity_only' strategy (original behavior)
        # For 'mean_pooling' and 'max_pooling', num_homogeneous represents per-slice count

        homogeneous_neighbors = self.homogeneous_neighbors
        n_adjacent_slices = self.n_adjacent_slices
        # Check if we should use fix number of homogeneous neighbors per slice
        if self.cross_slice_marker_score_strategy in [
            MarkerScoreCrossSliceStrategy.WEIGHTED_MEAN_POOLING,
            MarkerScoreCrossSliceStrategy.MAX_POOLING
        ]:

            self.fix_cross_slice_homogenous_neighbors = True
            logger.info(
                f"Using {self.cross_slice_marker_score_strategy.value} strategy with fixed number of homogeneous neighbors per adjacent slice: {self.homogeneous_neighbors} per slice.")


        elif self.cross_slice_marker_score_strategy == MarkerScoreCrossSliceStrategy.SIMILARITY_ONLY:
            logger.info(
                f"Using similarity_only strategy, will select top homogeneous neighbors from all adjacent slices based on similarity scores. Each adjacent slice can contribute variable number of homogeneous neighbors.")

        logger.info(
            f"Each focal cell will select {homogeneous_neighbors * (1 + 2 * n_adjacent_slices) = } total homogeneous neighbors across {(1 + 2 * n_adjacent_slices) = } slices.")

    def _configure_scrna_seq(self):
        """Configure parameters for scRNA-seq datasets"""
        self.n_adjacent_slices = 0
        self.spatial_key = None
        self.latent_representation_niche = None

def check_latent2gene_done(config: LatentToGeneConfig) -> bool:
    """
    Check if latent2gene step is done by verifying validity of metadata and output files.
    """
    from gsMap.latent2gene.memmap_io import MemMapDense

    expected_outputs = {
        "concatenated_latent_adata": Path(config.concatenated_latent_adata_path),
        "rank_memmap": Path(config.rank_memmap_path),
        "mean_frac": Path(config.mean_frac_path),
        "marker_scores": Path(config.marker_scores_memmap_path),
        "metadata": Path(config.latent2gene_metadata_path),
    }

    if not all(p.exists() for p in expected_outputs.values()):
        return False

    try:
        # Check rank memmap completion
        rank_memmap_complete, _ = MemMapDense.check_complete(expected_outputs["rank_memmap"])
        if not rank_memmap_complete:
            return False

        # Check marker scores memmap completion
        marker_scores_complete, _ = MemMapDense.check_complete(expected_outputs["marker_scores"])
        if not marker_scores_complete:
            return False

        # Check metadata
        with open(expected_outputs["metadata"], 'r') as f:
            metadata = yaml.unsafe_load(f)

        if 'outputs' not in metadata:
            return False

        return True
    except Exception as e:
        logger.warning(f"Error checking latent2gene results: {e}")
        return False

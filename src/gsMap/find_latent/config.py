"""
Configuration for finding latent representations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Annotated, List, OrderedDict
import logging
import typer

from gsMap.config.base import ConfigWithAutoPaths
from gsMap.config.utils import (
    process_h5ad_inputs,
    validate_h5ad_structure,
    verify_homolog_file_format
)

logger = logging.getLogger("gsMap.config")

@dataclass
class FindLatentRepresentationsConfig(ConfigWithAutoPaths):
    """Configuration for finding latent representations."""

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

    h5ad_list_file: Annotated[Optional[str], typer.Option(
        help="Each row is a h5ad file path, sample name is the file name without suffix",
        exists=True,
        file_okay=True,
        dir_okay=False,
    )] = None

    sample_h5ad_dict: Optional[OrderedDict] = None

    data_layer: Annotated[str, typer.Option(
        help="Gene expression raw counts data layer in h5ad layers, e.g., 'count', 'counts'. Other wise use 'X' for adata.X"
    )] = "X"

    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm storing spatial coordinates"
    )] = "spatial"

    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation of cell type in adata.obs to use"
    )] = None

    # Feature extraction parameters
    n_neighbors: Annotated[int, typer.Option(
        help="Number of neighbors for LGCN",
        min=1,
        max=50
    )] = 10

    K: Annotated[int, typer.Option(
        help="Graph convolution depth for LGCN",
        min=1,
        max=10
    )] = 3

    feat_cell: Annotated[int, typer.Option(
        help="Number of top variable features to retain",
        min=100,
        max=10000
    )] = 2000

    pearson_residual: Annotated[bool, typer.Option(
        "--pearson-residual",
        help="Take the residuals of the input data"
    )] = False

    # Model parameters
    hidden_size: Annotated[int, typer.Option(
        help="Units in the first hidden layer",
        min=32,
        max=512
    )] = 128

    embedding_size: Annotated[int, typer.Option(
        help="Size of the latent embedding layer",
        min=8,
        max=128
    )] = 32

    # Transformer parameters
    use_tf: Annotated[bool, typer.Option(
        "--use-tf",
        help="Enable transformer module"
    )] = False

    module_dim: Annotated[int, typer.Option(
        help="Dimensionality of transformer modules",
        min=10,
        max=100
    )] = 30

    hidden_gmf: Annotated[int, typer.Option(
        help="Hidden units for global mean feature extractor",
        min=32,
        max=512
    )] = 128

    n_modules: Annotated[int, typer.Option(
        help="Number of transformer modules",
        min=4,
        max=64
    )] = 16

    nhead: Annotated[int, typer.Option(
        help="Number of attention heads in transformer",
        min=1,
        max=16
    )] = 4

    n_enc_layer: Annotated[int, typer.Option(
        help="Number of transformer encoder layers",
        min=1,
        max=8
    )] = 2

    # Training parameters
    distribution: Annotated[str, typer.Option(
        help="Distribution type for loss calculation",
        case_sensitive=False
    )] = "nb"

    n_cell_training: Annotated[int, typer.Option(
        help="Number of cells used for training",
        min=1000,
        max=1000000
    )] = 100000

    batch_size: Annotated[int, typer.Option(
        help="Batch size for training",
        min=32,
        max=4096
    )] = 1024

    itermax: Annotated[int, typer.Option(
        help="Maximum number of training iterations",
        min=10,
        max=1000
    )] = 100

    patience: Annotated[int, typer.Option(
        help="Early stopping patience",
        min=1,
        max=50
    )] = 10

    two_stage: Annotated[bool, typer.Option(
        "--two-stage/--single-stage",
        help="Tune the cell embeddings based on the provided annotation"
    )] = True

    do_sampling: Annotated[bool, typer.Option(
        "--do-sampling/--no-sampling",
        help="Down-sampling cells in training"
    )] = True

    homolog_file: Annotated[Optional[Path], typer.Option(
        help="Path to homologous gene conversion file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None

    species: Optional[str] = None

    latent_representation_niche: Annotated[str, typer.Option(
        help="Key for spatial niche embedding in obsm"
    )] = "emb_niche"

    latent_representation_cell: Annotated[str, typer.Option(
        help="Key for cell identity embedding in obsm"
    )] = "emb_cell"

    def __post_init__(self):
        super().__post_init__()

        # Define input options
        input_options = {
            'h5ad_yaml': ('h5ad_yaml', 'yaml'),
            'h5ad_path': ('h5ad_path', 'list'),
            'h5ad_list_file': ('h5ad_list_file', 'file'),
        }

        # Process h5ad inputs
        self.sample_h5ad_dict = process_h5ad_inputs(self, input_options)

        if not self.sample_h5ad_dict:
            raise ValueError(
                "At least one of h5ad_yaml, h5ad_path, h5ad_list_file, or spe_file_list must be provided"
            )

        # Define required and optional fields for validation
        required_fields = {
            'data_layer': ('layers', self.data_layer, 'Data layer'),
            'spatial_key': ('obsm', self.spatial_key, 'Spatial key'),
        }

        # Add annotation as required if provided
        if self.annotation:
            required_fields['annotation'] = ('obs', self.annotation, 'Annotation')

        # Validate h5ad structure
        validate_h5ad_structure(self.sample_h5ad_dict, required_fields)

        # Log final sample count
        logger.info(f"Loaded and validated {len(self.sample_h5ad_dict)} samples")

        # Check if at least one sample is provided
        if len(self.sample_h5ad_dict) == 0:
            raise ValueError("No valid samples found in the provided input")

        # Verify homolog file format if provided
        verify_homolog_file_format(self)

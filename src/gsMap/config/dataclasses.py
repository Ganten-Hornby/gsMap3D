"""
Configuration dataclasses for gsMap commands.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from .base import ConfigWithAutoPaths

# Import module-specific configs for orchestration
from .find_latent_config import FindLatentRepresentationsConfig
from .latent2gene_config import LatentToGeneConfig
from .spatial_ldsc_config import SpatialLDSCConfig
from .utils import (
    verify_homolog_file_format,
)

logger = logging.getLogger("gsMap.config")


@dataclass
class CreateSliceMeanConfig:
    """Configuration for creating slice mean from multiple h5ad files."""

    slice_mean_output_file: Annotated[
        Path,
        typer.Option(
            help="Path to the output file for the slice mean", dir_okay=False, resolve_path=True
        ),
    ]

    sample_name_list: Annotated[
        str | list[str], typer.Option(help="Space-separated list of sample names")
    ]

    h5ad_list: Annotated[
        str | list[str], typer.Option(help="Space-separated list of h5ad file paths")
    ]

    # Optional parameters
    h5ad_yaml: Annotated[
        Path | None,
        typer.Option(
            help="Path to YAML file with sample names and h5ad paths",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None

    homolog_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to homologous gene conversion file",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None

    data_layer: Annotated[str, typer.Option(help="Data layer for gene expression")] = "counts"

    species: str | None = None
    h5ad_dict: dict | None = None

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
            if isinstance(self.h5ad_yaml, str | Path):
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
class DiagnosisConfig(ConfigWithAutoPaths):
    """Configuration for diagnosis command."""

    trait_name: Annotated[str, typer.Option(help="Name of the trait")]
    annotation: Annotated[str, typer.Option(help="Annotation layer name")]

    sumstats_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to GWAS summary statistics file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    top_corr_genes: Annotated[
        int, typer.Option(help="Number of top correlated genes to display")
    ] = 50
    selected_genes: Annotated[
        str | None, typer.Option(help="Comma-separated list of specific genes to include")
    ] = None

    fig_width: Annotated[int | None, typer.Option(help="Width of figures")] = None
    fig_height: Annotated[int | None, typer.Option(help="Height of figures")] = None
    point_size: Annotated[int | None, typer.Option(help="Point size")] = None

    plot_type: Annotated[str, typer.Option(help="Plot type (gsMap, manhattan, GSS, all)")] = "all"
    plot_origin: Annotated[
        str, typer.Option(help="Plot origin for spatial plots (upper or lower)")
    ] = "upper"

    @property
    def customize_fig(self) -> bool:
        return any([self.fig_width, self.fig_height, self.point_size])

    @property
    def hdf5_with_latent_path(self) -> Path:
        return self.concatenated_latent_adata_path

    @property
    def mkscore_feather_path(self) -> Path:
        # Fallback to a default name in the latent2gene directory
        return self.latent2gene_dir / f"{self.project_name}_marker_score.feather"


@dataclass
class VisualizeConfig(ConfigWithAutoPaths):
    """Configuration for visualization command."""

    trait_name: Annotated[str, typer.Option(help="Name of the trait")]
    annotation: Annotated[str | None, typer.Option(help="Annotation layer name")] = None

    fig_title: Annotated[str | None, typer.Option(help="Title for the figure")] = None
    fig_style: Annotated[str, typer.Option(help="Style of the figures (light/dark)")] = "light"
    point_size: Annotated[int | None, typer.Option(help="Point size")] = None
    fig_width: Annotated[int, typer.Option(help="Figure width")] = 800
    fig_height: Annotated[int, typer.Option(help="Figure height")] = 600

    output_dir: Annotated[Path | None, typer.Option(help="Directory to save output files")] = None
    hdf5_with_latent_path: Annotated[
        Path | None, typer.Option(help="Path to HDF5 with latent")
    ] = None
    plot_origin: Annotated[
        str, typer.Option(help="Plot origin for spatial plots (upper or lower)")
    ] = "upper"

    def __post_init__(self):
        super().__post_init__()
        if self.hdf5_with_latent_path is None:
            self.hdf5_with_latent_path = self.concatenated_latent_adata_path
        if self.output_dir is None:
            self.output_dir = self.get_report_dir(self.trait_name)


@dataclass
class ThreeDCombineConfig:
    workdir: str
    trait_name: str | None = None
    adata_3d: str | None = None
    project_name: str | None = None
    st_id: str | None = None
    annotation: str | None = None
    spatial_key: str = "spatial"
    cmap: str | None = None
    point_size: float = 0.01
    background_color: str = "white"
    n_snapshot: int = 200
    show_outline: bool = False
    save_mp4: bool = False
    save_gif: bool = False

    def __post_init__(self):
        if self.workdir is None:
            raise ValueError("workdir must be provided.")
        work_dir = Path(self.workdir)
        if self.project_name is not None:
            self.project_dir = work_dir / self.project_name
        else:
            self.project_dir = work_dir


@dataclass
class RunLinkModeConfig(ConfigWithAutoPaths):
    """Configuration for running gsMap with linked mode."""

    # Placeholder for link mode config fields
    pass


@dataclass
class gsMapPipelineConfig(ConfigWithAutoPaths):
    """Unified configuration for the complete gsMap pipeline"""

    # Component configurations
    find_latent: FindLatentRepresentationsConfig | None = None
    latent2gene: LatentToGeneConfig | None = None
    spatial_ldsc: SpatialLDSCConfig | None = None

    def __post_init__(self):
        super().__post_init__()
        # Initialize component configs if they weren't provided
        if self.find_latent is None:
            self.find_latent = FindLatentRepresentationsConfig(
                workdir=self.workdir, project_name=self.project_name
            )
        if self.latent2gene is None:
            self.latent2gene = LatentToGeneConfig(
                workdir=self.workdir, project_name=self.project_name
            )
        if self.spatial_ldsc is None:
            self.spatial_ldsc = SpatialLDSCConfig(
                workdir=self.workdir, project_name=self.project_name
            )

"""
Configuration for spatial LD score regression.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Annotated, List, Literal, Dict, Any
import logging
import typer
import yaml

from gsMap.config.base import ConfigWithAutoPaths

logger = logging.getLogger("gsMap.config")

@dataclass
class SpatialLDSCComputeConfig:
    """Compute configuration for spatial LDSC step."""
    __display_in_quick_mode_cli__ = True

    use_gpu: Annotated[bool, typer.Option(
        "--use-gpu/--no-gpu",
        help="Use GPU for JAX-accelerated spatial LDSC implementation"
    ), {"__display_in_quick_mode_cli__": True}] = True

    memmap_tmp_dir: Annotated[Optional[Path], typer.Option(
        help="Temporary directory for memory-mapped files to improve I/O performance on slow filesystems. "
             "If provided, memory maps will be copied to this directory for faster random access during computation.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )] = None

    ldsc_read_workers: Annotated[int, typer.Option(
        help="Number of read workers",
        min=1
    )] = 10

    ldsc_compute_workers: Annotated[int, typer.Option(
        help="Number of compute workers for LDSC regression",
        min=1
    )] = 10

    spots_per_chunk_quick_mode: Annotated[int, typer.Option(
        help="Number of spots per chunk in quick mode",
        min=1
    )] = 50

    device_ids: Annotated[Optional[str], typer.Option(
        help="Comma-separated list of GPU device IDs to use (e.g., '0,1'). "
             "If None, uses the first available GPU or the default JAX device."
    )] = None

    platform: Optional[str] = None
    devices: Optional[List[Any]] = None

@dataclass
class GWASSumstatsConfig:
    """Configuration for GWAS summary statistics."""
    trait_name: Annotated[Optional[str], typer.Option(
        help="Name of the trait for GWAS analysis"
    )] = None

    sumstats_file: Annotated[Optional[Path], typer.Option(
        help="Path to GWAS summary statistics file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

    sumstats_config_file: Annotated[Optional[Path], typer.Option(
        help="Path to sumstats config file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

    sumstats_config_dict: Dict[str, Path] = field(default_factory=dict)

    @property
    def trait_name_list(self) -> List[str]:
        """Return the list of trait names to process."""
        return list(self.sumstats_config_dict.keys())

    def __post_init__(self):
        self._init_sumstats()

    def _init_sumstats(self):
        """
        Process sumstats input options and populate sumstats_config_dict.

        Args:
            self: Configuration object with sumstats_file, sumstats_config_file, and trait_name fields.

        Returns:
            Dict mapping trait names to sumstats file paths.
        """
        sumstats_config_dict = {}

        if self.sumstats_file is None and self.sumstats_config_file is None:
            raise ValueError("One of sumstats_file and sumstats_config_file must be provided.")
        if self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError(
                "Only one of sumstats_file and sumstats_config_file must be provided."
            )
        if self.sumstats_file is not None and self.trait_name is None:
            raise ValueError("trait_name must be provided if sumstats_file is provided.")
        if self.sumstats_config_file is not None and self.trait_name is not None:
            raise ValueError(
                "trait_name must not be provided if sumstats_config_file is provided."
            )
        # load the sumstats self file
        if self.sumstats_config_file is not None:
            # get the directory of the config file to resolve relative paths
            config_dir = Path(self.sumstats_config_file).parent
            with open(self.sumstats_config_file) as f:
                config_loaded = yaml.load(f, Loader=yaml.FullLoader)
            for _trait_name, sumstats_file in config_loaded.items():
                s_path = Path(sumstats_file)
                if not s_path.is_absolute():
                    s_path = config_dir / s_path
                sumstats_config_dict[_trait_name] = s_path.resolve()
        # load the sumstats file
        elif self.sumstats_file is not None:
            sumstats_config_dict[self.trait_name] = Path(self.sumstats_file)
        else:
            raise ValueError("One of sumstats_file and sumstats_config_file must be provided.")

        for sumstats_file in sumstats_config_dict.values():
            assert Path(sumstats_file).exists(), f"{sumstats_file} does not exist."

        self.sumstats_config_dict = sumstats_config_dict


@dataclass
class SpatialLDSCCoreConfig(GWASSumstatsConfig):
    """Core configuration for spatial LDSC."""
    w_ld_dir: Annotated[Optional[Path], typer.Option(
        help="Directory containing the weights files (w_ld)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )] = None

    additional_baseline_h5ad_path_list: Annotated[List[Path], typer.Option(
        help="List of additional baseline h5ad paths",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = field(default_factory=list)


    chisq_max: Annotated[Optional[int], typer.Option(
        help="Maximum chi-square value"
    )] = None

    cell_indices_range: Annotated[Optional[tuple[int, int]], typer.Option(
        help="0-based range [start, end) of cell indices to process"
    )] = None

    sample_filter: Annotated[Optional[str], typer.Option(
        help="Filter processing to a specific sample"
    )] = None

    n_blocks: Annotated[int, typer.Option(
        help="Number of jackknife blocks",
        min=1
    )] = 200

    # spots_per_chunk_quick_mode is inherited from SpatialLDSCComputeConfig

    snp_gene_weight_adata_path: Annotated[Path, typer.Option(
        help="Path to the SNP-gene weight matrix (H5AD format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

    # use_gpu is inherited from SpatialLDSCComputeConfig

    marker_score_feather_path: Annotated[Optional[Path], typer.Option(
        help="Path to marker score feather file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

    marker_score_h5ad_path: Annotated[Optional[Path], typer.Option(
        help="Path to marker score h5ad file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None

    marker_score_format: Annotated[Optional[Literal["memmap", "feather", "h5ad"]], typer.Option(
        help="Format of marker scores"
    )] = None



@dataclass
class SpatialLDSCConfig(SpatialLDSCCoreConfig, SpatialLDSCComputeConfig, ConfigWithAutoPaths):
    """Spatial LDSC Configuration"""

    def __post_init__(self):
        super().__post_init__()


        # Import here to avoid circular imports
        from gsMap.config.utils import configure_jax_platform, get_anndata_shape

        # Configure JAX platform if use_gpu is enabled
        self.platform, self.devices = configure_jax_platform(self.use_gpu, self.device_ids)


        # Auto-detect marker_score_format if not specified
        if self.marker_score_format is None:
            if self.marker_score_feather_path is not None:
                self.marker_score_format = "feather"
                logger.info("Auto-detected marker_score_format as 'feather' based on marker_score_feather_path")
            elif self.marker_score_h5ad_path is not None:
                self.marker_score_format = "h5ad"
                logger.info("Auto-detected marker_score_format as 'h5ad' based on marker_score_h5ad_path")
            else:
                self.marker_score_format = "memmap"
                logger.info("Using default marker_score_format 'memmap'")

        # Validate cell_indices_range is 0-based
        if self.cell_indices_range is not None:
            # Validate exclusivity between sample_filter and cell_indices_range

            if self.sample_filter is not None:
                raise ValueError(
                    "Only one of sample_filter or cell_indices_range can be provided, not both. "
                    "Use sample_filter to filter by sample, or cell_indices_range to process specific cell indices."
                )

            start, end = self.cell_indices_range

            # Check that indices are 0-based
            if start < 0:
                raise ValueError(f"cell_indices_range start must be >= 0, got {start}")
            if start == 1:
                logger.warning(
                    "cell_indices_range appears to be 1-based (start=1). "
                    "Please ensure indices are 0-based. Adjusting start to 0."
                )
                start = 0

            # Check that start < end
            if start >= end:
                raise ValueError(f"cell_indices_range start ({start}) must be less than end ({end})")

            # Validate against actual data shape based on marker score format
            if self.marker_score_format == "memmap":
                # For memmap format, check the concatenated latent adata
                adata_path = Path(self.workdir) / self.project_name / "latent2gene" / "concatenated_latent_adata.h5ad"
                shape = get_anndata_shape(str(adata_path))
                if shape is not None:
                    n_obs, _ = shape
                    if end > n_obs:
                        logger.warning(
                            f"cell_indices_range end ({end}) exceeds number of observations ({n_obs}). "
                            f"Setting end to {n_obs}."
                        )
                        end = n_obs
            elif self.marker_score_format == "h5ad":
                # For h5ad format, check the provided h5ad path
                adata_path = Path(self.marker_score_h5ad_path)
                assert adata_path.exists(), f"Marker score h5ad not found at {adata_path}."
                shape = get_anndata_shape(str(adata_path))
                if shape is not None:
                    n_obs, _ = shape
                    if end > n_obs:
                        logger.warning(
                            f"cell_indices_range end ({end}) exceeds number of observations ({n_obs}). "
                            f"Setting end to {n_obs}."
                        )
                        end = n_obs
            elif self.marker_score_format == "feather":
                # For feather format, validate the path exists
                feather_path = Path(self.marker_score_feather_path)
                assert feather_path.exists(), f"Marker score feather file not found at {feather_path}."

                # Use pyarrow to get the number of rows and validate end
                try:
                    import pyarrow.feather as feather
                    # Read metadata without loading full data
                    feather_table = feather.read_table(str(feather_path), memory_map=True, columns=[])
                    n_obs = feather_table.num_rows
                    if end > n_obs:
                        logger.warning(
                            f"cell_indices_range end ({end}) exceeds number of rows ({n_obs}) in feather file. "
                            f"Setting end to {n_obs}."
                        )
                        end = n_obs
                except ImportError:
                    logger.warning(
                        "pyarrow not available. Cannot validate cell_indices_range against feather file. "
                        "Install pyarrow to enable validation."
                    )
                except Exception as e:
                    logger.warning(f"Could not read feather file metadata: {e}")

            # Update cell_indices_range with validated values
            self.cell_indices_range = (start, end)
            logger.info(f"Processing cell_indices_range: [{start}, {end})")

        if self.snp_gene_weight_adata_path is None:
            raise ValueError("snp_gene_weight_adata_path must be provided.")

        # Handle w_ld_dir
        if self.w_ld_dir is None:
            w_ld_dir = Path(self.ldscore_save_dir) / "w_ld"
            if w_ld_dir.exists():
                self.w_ld_dir = w_ld_dir
                logger.info(f"Using weights directory generated in the generate_ldscore step: {self.w_ld_dir}")
            else:
                raise ValueError(
                    "No w_ld_dir provided and no weights directory found in generate_ldscore output. "
                    "Either provide --w-ld-dir or run generate_ldscore first."
                )
        else:
            logger.info(f"Using provided weights directory: {self.w_ld_dir}")

        self.show_config(SpatialLDSCConfig)


def check_spatial_ldsc_done(config: SpatialLDSCConfig, trait_name: str) -> bool:
    """
    Check if spatial_ldsc step is done for a specific trait.
    """
    result_file = config.get_ldsc_result_file(trait_name)
    return result_file.exists()

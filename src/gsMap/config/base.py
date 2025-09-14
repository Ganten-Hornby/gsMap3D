"""
Base configuration classes and utilities for gsMap.
"""
from dataclasses import dataclass, asdict
from functools import wraps
from pathlib import Path
from typing import Optional, Annotated, List, Dict, Any
import typer
import logging
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console

def config_logger():
    logger = logging.getLogger("gsMap")
    # clean up existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set logger to DEBUG to capture all messages
    logger.setLevel(logging.DEBUG)
    
    # Create rich console handler for INFO level messages
    console = Console()
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True
    )
    rich_handler.setLevel(logging.INFO)
    rich_handler.setFormatter(
        logging.Formatter("{levelname:.5s} | {name} - {message}", style="{")
    )
    logger.addHandler(rich_handler)
    
    # Create file handler for DEBUG level messages with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"gsMap_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "[{asctime}] {levelname:.5s} | {name}:{funcName}:{lineno} - {message}", 
            style="{"
        )
    )
    logger.addHandler(file_handler)
    
    # Log the setup
    logger.info(f"Logging configured - console: INFO+, file: DEBUG+ -> {log_file}")
    
    return logger

config_logger()

def ensure_path_exists(func):
    """Decorator to ensure path exists when accessing properties."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, Path):
            if result.suffix:
                result.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            else:  # It's a directory path
                result.mkdir(parents=True, exist_ok=True, mode=0o755)
        return result
    return wrapper

class ConfigWithAutoPaths:
    """Base configuration class with automatic path generation."""

    # Required from parent
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]

    project_name: Annotated[str, typer.Option(
        help="Name of the project"
    )]

    def __post_init__(self):
        if self.workdir is None:
            raise ValueError('workdir must be provided.')
        work_dir = Path(self.workdir)
        self.project_dir = work_dir / self.project_name
    
    def to_dict_with_paths_as_strings(self) -> Dict[str, Any]:
        """
        Convert the config object to a dictionary with all Path objects converted to strings.
        Also handles OrderedDict with Path values (like sample_h5ad_dict).
        
        Returns:
            Dictionary representation of the config with all Path objects as strings
        """
        # Convert config to dict
        config_dict = asdict(self)
        
        # Convert sample_h5ad_dict (OrderedDict with Path values) to proper format
        if hasattr(self, 'sample_h5ad_dict') and self.sample_h5ad_dict:
            sample_h5ad_dict_str = {k: str(v) for k, v in self.sample_h5ad_dict.items()}
            config_dict['sample_h5ad_dict'] = sample_h5ad_dict_str
        
        # Convert all Path objects in config to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, dict):
                config_dict[key] = {k: str(v) if isinstance(v, Path) else v for k, v in value.items()}
            elif isinstance(value, list):
                config_dict[key] = [str(v) if isinstance(v, Path) else v for v in value]
        
        return config_dict

    ## ---- Find latent representation paths
    @property
    @ensure_path_exists
    def latent_dir(self) -> Path:
        return self.project_dir / "find_latent_representations"

    @property
    def model_path(self) -> Path:
        return self.latent_dir / 'LGCN_model/gsMap_LGCN_.pt'

    @property
    def find_latent_metadata_path(self) -> Path:
        return self.latent_dir / 'find_latent_metadata.yaml'

    ## ---- Latent to gene paths

    @property
    @ensure_path_exists
    def latent2gene_dir(self) -> Path:
        """Directory for latent to gene outputs"""
        return self.project_dir / "latent_to_gene"
    
    @property
    def concatenated_latent_adata_path(self) -> Path:
        """Path to concatenated latent representations"""
        return self.latent2gene_dir / "concatenated_latent_adata.h5ad"
    
    @property
    def rank_memmap_path(self) -> Path:
        """Path to rank zarr file"""
        return self.latent2gene_dir / "ranks.dat"
    
    @property
    def mean_frac_path(self) -> Path:
        """Path to mean expression fraction parquet"""
        return self.latent2gene_dir / "mean_frac.parquet"
    
    @property
    def marker_scores_memmap_path(self) -> Path:
        """Path to marker scores zarr"""
        return self.latent2gene_dir / "marker_scores.dat"
    
    @property
    def latent2gene_metadata_path(self) -> Path:
        """Path to latent2gene metadata YAML"""
        return self.latent2gene_dir / "metadata.yaml"

    ## ---- LD score paths


    ## ---- Spatial LDSC paths

    @property
    @ensure_path_exists
    def ldsc_save_dir(self) -> Path:
        """Directory for spatial LDSC results"""
        return self.project_dir / "spatial_ldsc"

    @ensure_path_exists
    def get_ldsc_result_file(self, trait_name: str) -> Path:
        return Path(f"{self.ldsc_save_dir}/{self.project_name}_{trait_name}.csv.gz")
    #
    # #
    # # @property
    # # @ensure_path_exists
    # # def ldscore_save_dir(self) -> Path:
    # #     return Path(f"{self.workdir}/{self.sample_name}/generate_ldscore")
    #
    # @property
    # @ensure_path_exists
    # def cauchy_save_dir(self) -> Path:
    #     return Path(f"{self.workdir}/{self.project_name}/cauchy_combination")
    #
    # @ensure_path_exists
    # def get_report_dir(self, trait_name: str) -> Path:
    #     return Path(f"{self.workdir}/{self.project_name}/report/{trait_name}")
    #
    # def get_gsMap_report_file(self, trait_name: str) -> Path:
    #     return (
    #         self.get_report_dir(trait_name) / f"{self.project_name}_{trait_name}_gsMap_Report.html"
    #     )



    #
    #
    # @property
    # @ensure_path_exists
    # def mkscore_feather_path(self) -> Path:
    #     return Path(f'{self.project_dir}/latent_to_gene/mk_score/{self.sample_name}_gene_marker_score.feather')
    #
    # @property
    # @ensure_path_exists
    # def tuned_mkscore_feather_path(self) -> Path:
    #     return Path(f'{self.project_dir}/latent_to_gene/mk_score_pooling/{self.sample_name}_gene_marker_score.feather')
    #
    # @property
    # @ensure_path_exists
    # def ldscore_save_dir(self) -> Path:
    #     return Path(f'{self.project_dir}/generate_ldscore/{self.sample_name}')
    #
    # @property
    # @ensure_path_exists
    # def ldsc_save_dir(self) -> Path:
    #     return Path(f'{self.project_dir}/spatial_ldsc/{self.sample_name}')
    
    @property
    @ensure_path_exists
    def cauchy_save_dir(self) -> Path:
        return self.project_dir / "cauchy_combination"
        return Path(f'{self.project_dir}/cauchy_combination/{self.sample_name}')
    
    @ensure_path_exists
    def get_report_dir(self, trait_name: str) -> Path:
        return Path(f'{self.project_dir}/report/{self.sample_name}/{trait_name}')
    
    def get_gsMap_report_file(self, trait_name: str) -> Path:
        return (
            self.get_report_dir(trait_name)
            / f"{self.sample_name}_{trait_name}_gsMap_Report.html"
        )
    
    @ensure_path_exists
    def get_manhattan_html_plot_path(self, trait_name: str) -> Path:
        return Path(
            f'{self.project_dir}/report/{self.sample_name}/{trait_name}/manhattan_plot/{self.sample_name}_{trait_name}_Diagnostic_Manhattan_Plot.html')
    
    @ensure_path_exists
    def get_GSS_plot_dir(self, trait_name: str) -> Path:
        return Path(f'{self.project_dir}/report/{self.sample_name}/{trait_name}/GSS_plot')
    
    def get_GSS_plot_select_gene_file(self, trait_name: str) -> Path:
        return self.get_GSS_plot_dir(trait_name) / "plot_genes.csv"
    
    @ensure_path_exists
    def get_ldsc_result_file(self, trait_name: str) -> Path:
        return Path(f"{self.ldsc_save_dir}/{self.sample_name}_{trait_name}.csv.gz")
    
    @ensure_path_exists
    def get_cauchy_result_file(self, trait_name: str, all_samples: bool = False) -> Path:
        if all_samples:
            return Path(
                f"{self.cauchy_save_dir}/{self.project_name}_all_samples_{trait_name}.Cauchy.csv.gz"
            )
        else:
            return Path(
                f"{self.cauchy_save_dir}/{self.project_name}_single_sample_{self.sample_name}_{trait_name}.Cauchy.csv.gz"
            )
    
    @ensure_path_exists
    def get_gene_diagnostic_info_save_path(self, trait_name: str) -> Path:
        return Path(
            f'{self.project_dir}/report/{self.sample_name}/{trait_name}/{self.sample_name}_{trait_name}_Gene_Diagnostic_Info.csv')
    
    @ensure_path_exists
    def get_gsMap_plot_save_dir(self, trait_name: str) -> Path:
        return Path(f'{self.project_dir}/report/{self.sample_name}/{trait_name}/gsMap_plot')
    
    def get_gsMap_html_plot_save_path(self, trait_name: str) -> Path:
        return (
            self.get_gsMap_plot_save_dir(trait_name)
            / f"{self.sample_name}_{trait_name}_gsMap_plot.html"
        )
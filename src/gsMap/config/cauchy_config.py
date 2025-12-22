from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated, List
import typer

from .base import ConfigWithAutoPaths

@dataclass
class CauchyCombinationConfig(ConfigWithAutoPaths):
    """Configuration for Cauchy combination test."""
    
    trait_name: Annotated[str, typer.Option(
        help="Name of the trait being analyzed"
    )]
    
    annotation: Annotated[str, typer.Option(
        help="Name of the annotation in adata.obs to use"
    )]

    sample_name_list: Annotated[Optional[List[str]], typer.Option(
        help="List of sample names to include"
    )] = None

    def __post_init__(self):
        super().__post_init__()
        self.show_config("Cauchy Combination Configuration")

    @property
    def output_file(self) -> Path:
        """Annotation-level result file (aggregated all samples)"""
        return self.get_cauchy_result_file(self.trait_name, all_samples=True)

    @property
    def sample_output_file(self) -> Path:
        """Sample-annotation pair level result file"""
        return self.get_cauchy_result_file(self.trait_name, all_samples=False)


def check_cauchy_done(config: CauchyCombinationConfig, trait_name: str) -> bool:
    """
    Check if cauchy step is done for a specific trait.
    Checks both annotation-level and sample-level results.
    """
    anno_result = config.get_cauchy_result_file(trait_name, all_samples=True)
    sample_result = config.get_cauchy_result_file(trait_name, all_samples=False)
    return anno_result.exists() and sample_result.exists()

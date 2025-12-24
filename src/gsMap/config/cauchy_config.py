from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated, List
import typer

from .base import ConfigWithAutoPaths

@dataclass
class CauchyCombinationConfig(ConfigWithAutoPaths):
    """Configuration for Cauchy combination test."""
    
    annotation: Annotated[str, typer.Option(
        help="Name of the annotation in adata.obs to use",
    )]

    trait_name: Annotated[Optional[str], typer.Option(
        help="Name of the trait being analyzed. If None, all available traits will be processed."
    )] = None


    def __post_init__(self):
        super().__post_init__()
        self.show_config("Cauchy Combination Configuration")


def check_cauchy_done(config: ConfigWithAutoPaths, trait_name: str) -> bool:
    """
    Check if cauchy step is done for a specific trait.
    Checks both annotation-level and sample-level results.
    """
    anno_result = config.get_cauchy_result_file(trait_name, annotation=annotation, all_samples=True)
    sample_result = config.get_cauchy_result_file(trait_name, annotation=annotation, all_samples=False)
    return anno_result.exists() and sample_result.exists()

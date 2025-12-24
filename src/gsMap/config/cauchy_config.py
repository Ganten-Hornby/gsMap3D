from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated, List
import typer

from .base import ConfigWithAutoPaths

@dataclass
class CauchyCombinationConfig(ConfigWithAutoPaths):
    """Configuration for Cauchy combination test."""
    
    annotation: Annotated[Optional[str], typer.Option(
        help="Name of the annotation in adata.obs to use",
    )] = None

    cauchy_annotations: Annotated[Optional[List[str]], typer.Option(
        help="List of annotations in adata.obs to use",
    )] = field(default_factory=list)

    trait_name: Annotated[Optional[str], typer.Option(
        help="Name of the trait being analyzed. If None, all available traits will be processed."
    )] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.annotation and not self.cauchy_annotations:
            raise ValueError("At least one of 'annotation' or 'cauchy_annotations' must be provided.")
        
        # Build unique list of annotations
        from collections import OrderedDict
        self.annotation_list = []
        if self.annotation:
            self.annotation_list.append(self.annotation)
        if self.cauchy_annotations:
            self.annotation_list.extend(self.cauchy_annotations)
        self.annotation_list = list(OrderedDict.fromkeys(self.annotation_list))
        
        self.show_config("Cauchy Combination Configuration")


def check_cauchy_done(config: CauchyCombinationConfig, trait_name: str) -> bool:
    """
    Check if cauchy step is done for a specific trait.
    Checks both annotation-level and sample-level results for all configured annotations.
    """
    if not config.annotation_list:
        return False
        
    for annotation in config.annotation_list:
        anno_result = config.get_cauchy_result_file(trait_name, annotation=annotation, all_samples=True)
        sample_result = config.get_cauchy_result_file(trait_name, annotation=annotation, all_samples=False)
        if not (anno_result.exists() and sample_result.exists()):
            return False
    return True

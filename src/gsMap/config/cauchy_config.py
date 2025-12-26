from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Annotated, List, Dict, Any, Union
import logging
import yaml
import typer

from .base import ConfigWithAutoPaths
from .spatial_ldsc_config import GWASSumstatsConfig


@dataclass
class CauchyCombinationConfig(GWASSumstatsConfig,ConfigWithAutoPaths):
    """Cauchy Combination Configuration"""

    annotation: Annotated[Optional[str], typer.Option(
        help="Name of the annotation in adata.obs to use",
    )] = None

    cauchy_annotations: Annotated[Optional[List[str]], typer.Option(
        help="List of annotations in adata.obs to use",
    )] = field(default_factory=list)

    annotation_list: List[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self._init_annotation_list()
        self.show_config(CauchyCombinationConfig)

    def _init_annotation_list(self):
        """Build the unique list of annotations from 'annotation' and 'cauchy_annotations'."""
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



    @property
    def ldsc_traits_result_path_dict(self) -> Dict[str, Path]:
        """
        Discover LDSC result files for the configured traits and return a dictionary mapping trait names to file paths.
        """
        traits_dict = {}
        
        for trait in self.trait_name_list:
            ldsc_input_file = self.get_ldsc_result_file(trait)
            if ldsc_input_file.exists():
                traits_dict[trait] = ldsc_input_file
            else:
                raise FileNotFoundError(f"LDSC result file not found for {trait}: {ldsc_input_file}")

        if not traits_dict:
            raise FileNotFoundError(f"No valid LDSC result files found for the specified traits in {self.ldsc_save_dir}")
            
        return traits_dict


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

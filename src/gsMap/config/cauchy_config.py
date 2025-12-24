from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Annotated, List, Dict, Any, Union
import logging
import yaml
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

    sumstats_config_file: Annotated[Optional[Path], typer.Option(
        help="Path to sumstats config file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )] = None


    def __post_init__(self):
        super().__post_init__()
        if not self.annotation and not self.cauchy_annotations:
            raise ValueError("At least one of 'annotation' or 'cauchy_annotations' must be provided.")
        
        if self.trait_name is None and self.sumstats_config_file is None:
             raise ValueError("At least one of 'trait_name' or 'sumstats_config_file' must be provided.")
        
        self._trait_names = []
        # Load the sumstats config file if provided
        if self.sumstats_config_file is not None:
            with open(self.sumstats_config_file) as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
            self._trait_names.extend(list(config_data.keys()))
        
        # Add single trait if provided
        if self.trait_name is not None:
            if self.trait_name not in self._trait_names:
                self._trait_names.append(self.trait_name)
        
        # Build unique list of annotations
        from collections import OrderedDict
        self.annotation_list = []
        if self.annotation:
            self.annotation_list.append(self.annotation)
        if self.cauchy_annotations:
            self.annotation_list.extend(self.cauchy_annotations)
        self.annotation_list = list(OrderedDict.fromkeys(self.annotation_list))
        
        self.show_config("Cauchy Combination Configuration")

    @property
    def trait_name_list(self) -> List[str]:
        """Return the list of trait names to process."""
        return self._trait_names

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

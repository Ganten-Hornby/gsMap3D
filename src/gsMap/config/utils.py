"""
Utility functions for gsMap configuration and data validation.
"""

import logging
from pathlib import Path
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Tuple
import h5py
import psutil
import yaml

logger = logging.getLogger("gsMap.config")

def _log_device_info(devices: List[Any]):
    """Helper to log memory specs correctly for CPU vs GPU."""
    for d in devices:
        if d.device_kind == "cpu":
            # Using psutil for meaningful CPU memory reporting
            mem_gb = psutil.virtual_memory().total / 1e9
            mem_type = "System RAM"
        else:
            # VRAM reporting for GPUs
            mem_stats = d.memory_stats()
            mem_gb = mem_stats.get('bytes_limit', 0) / 1e9
            mem_type = "VRAM"
            
        logger.info(f"Initialized JAX Device {d.id}: {d.device_kind} ({mem_type}: {mem_gb:.2f} GB)")
        
def configure_jax_platform(
    use_accelerator: bool = True, 
    device_ids: Optional[str] = None,
    preallocate: bool = True
) -> Tuple[str, List[int]]:
    try:
        import jax
        from jax import config as jax_config
        import os

        # 1. Memory Management
        if not preallocate:
            # Must be set before JAX starts its first operation
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # 2. Forced CPU Fallback
        if not use_accelerator:
            jax_config.update('jax_platforms', 'cpu')
            devices = jax.devices('cpu')
            _log_device_info(devices)
            return 'cpu', [d.id for d in devices]

        # 3. Corrected Backend Detection
        try:
            # This is the stable modern replacement for get_backend()
            best_platform = jax.default_backend() 
        except Exception:
            best_platform = 'cpu'

        # 4. Device Filtering
        available_devices = jax.devices(best_platform)
        selected_devices = available_devices
        
        if device_ids and best_platform != 'cpu':
            try:
                requested_ids = [int(i.strip()) for i in device_ids.split(',')]
                # Filter based on the .id attribute
                selected_devices = [d for d in available_devices if d.id in requested_ids]
                
                if not selected_devices:
                    logger.warning(f"IDs {requested_ids} not found. Defaulting to all {best_platform}s.")
                    selected_devices = available_devices
            except (ValueError, TypeError):
                logger.error(f"Malformed device_ids: {device_ids}")

        # Finalize
        jax_config.update('jax_platforms', best_platform)
        _log_device_info(selected_devices)
        
        return best_platform, [d.id for d in selected_devices]

    except ImportError:
        logger.error("JAX/jaxlib not found.")
        raise

def get_anndata_shape(h5ad_path: str):
    """Get the shape (n_obs, n_vars) of an AnnData file without loading it."""
    with h5py.File(h5ad_path, 'r') as f:
        # 1. Verify it's a valid AnnData file by checking metadata
        if f.attrs.get('encoding-type') != 'anndata':
            logger.error(f"File '{h5ad_path}' does not appear to be a valid AnnData file.")
            return None

        # 2. Determine n_obs and n_vars from the primary metadata sources
        if 'obs' not in f or 'var' not in f:
            logger.error("AnnData file is missing 'obs' or 'var' group.")
            return None

        # Get the name of the index column from attributes
        obs_index_key = f['obs'].attrs.get('_index', None)
        var_index_key = f['var'].attrs.get('_index', None)

        if not obs_index_key or obs_index_key not in f['obs']:
            logger.error("Could not determine index for 'obs'.")
            return None
        if not var_index_key or var_index_key not in f['var']:
            logger.error("Could not determine index for 'var'.")
            return None

        # The shape is the length of these index arrays
        obs_obj = f['obs'][obs_index_key]
        if isinstance(obs_obj, h5py.Group):
            obs_obj = obs_obj['categories']
        n_obs = obs_obj.shape[0]

        var_obj  = f['var'][var_index_key]
        if isinstance(var_obj, h5py.Group):
            var_obj = var_obj['categories']
        n_vars = var_obj.shape[0]

        return n_obs, n_vars

def inspect_h5ad_structure(filename):
    """
    Inspect the structure of an h5ad file without loading data.
    
    Returns dict with keys present in each slot.
    """
    structure = {}
    
    with h5py.File(filename, 'r') as f:
        # Check main slots
        slots = ['obs', 'var', 'obsm', 'varm', 'obsp', 'varp', 'uns', 'layers', 'X', 'raw']
        
        for slot in slots:
            if slot in f:
                if slot in ['obsm', 'varm', 'obsp', 'varp', 'layers', 'uns']:
                    # These are groups containing multiple keys
                    structure[slot] = list(f[slot].keys())
                elif slot in ['obs', 'var']:
                    # These are dataframes - get column names
                    if 'column-order' in f[slot].attrs:
                        structure[slot] = list(f[slot].attrs['column-order'])
                    else:
                        structure[slot] = list(f[slot].keys())
                else:
                    # X, raw - just note they exist
                    structure[slot] = True
    
    return structure

def validate_h5ad_structure(sample_h5ad_dict, required_fields, optional_fields=None):
    """
    Validate h5ad files have required structure.
    
    Args:
        sample_h5ad_dict: OrderedDict of {sample_name: h5ad_path}
        required_fields: Dict of {field_name: (slot, field_key, error_msg_template)}
            e.g., {'spatial': ('obsm', 'spatial', 'Spatial key')}
        optional_fields: Dict of {field_name: (slot, field_key)} for fields to warn about
    
    Returns:
        None, raises ValueError if required fields are missing
    """
    for sample_name, h5ad_path in sample_h5ad_dict.items():
        if not h5ad_path.exists():
            raise FileNotFoundError(f"H5AD file not found for sample '{sample_name}': {h5ad_path}")
        
        # Inspect h5ad structure
        structure = inspect_h5ad_structure(h5ad_path)
        
        # Check required fields
        for field_name, (slot, field_key, error_msg) in required_fields.items():
            if field_key is None:  # Skip if field not specified
                continue
                
            # Special handling for data_layer
            if field_name == 'data_layer' and field_key != 'X':
                if 'layers' not in structure or field_key not in structure.get('layers', []):
                    raise ValueError(
                        f"Data layer '{field_key}' not found in layers for sample '{sample_name}'. "
                        f"Available layers: {structure.get('layers', [])}"
                    )
            elif field_name == 'data_layer' and field_key == 'X':
                if 'X' not in structure:
                    raise ValueError(f"X matrix not found in h5ad file for sample '{sample_name}'")
            else:
                # Standard validation for obsm, obs, etc.
                if slot not in structure or field_key not in structure.get(slot, []):
                    available = structure.get(slot, [])
                    raise ValueError(
                        f"{error_msg} '{field_key}' not found in {slot} for sample '{sample_name}'. "
                        f"Available keys in {slot}: {available}"
                    )
        
        # Check optional fields (warn only)
        if optional_fields:
            for field_name, (slot, field_key) in optional_fields.items():
                if field_key is None:  # Skip if field not specified
                    continue
                    
                if slot not in structure or field_key not in structure.get(slot, []):
                    available = structure.get(slot, [])
                    logger.warning(
                        f"Optional field '{field_key}' not found in {slot} for sample '{sample_name}'. "
                        f"Available keys in {slot}: {available}"
                    )

def process_h5ad_inputs(config, input_options):
    """
    Process h5ad input options and create sample_h5ad_dict.
    
    Args:
        config: Configuration object with h5ad input fields
        input_options: Dict mapping option names to (field_name, processing_type)
            e.g., {'h5ad_yaml': ('h5ad_yaml', 'yaml'), 
                   'h5ad': ('h5ad', 'list'),
                   'h5ad_list_file': ('h5ad_list_file', 'file')}
    
    Returns:
        OrderedDict of {sample_name: h5ad_path}
    """

    if config.sample_h5ad_dict  is not None and len(config.sample_h5ad_dict) > 0:
        logger.info("Using pre-defined sample_h5ad_dict from configuration")
        return OrderedDict(config.sample_h5ad_dict)

    sample_h5ad_dict = OrderedDict()
    
    # Check which options are provided
    options_provided = []
    for option_name, (field_name, _) in input_options.items():
        if hasattr(config, field_name) and getattr(config, field_name):
            options_provided.append(option_name)
    
    # Ensure at most one option is provided
    if len(options_provided) > 1:
        assert False, (
            f"At most one input option can be provided. Got {len(options_provided)}: {', '.join(options_provided)}. "
            f"Please provide only one of: {', '.join(input_options.keys())}"
        )
    
    # Process the provided input option
    for option_name, (field_name, processing_type) in input_options.items():
        field_value = getattr(config, field_name, None)
        if not field_value:
            continue
            
        if processing_type == 'yaml':
            logger.info(f"Using {option_name}: {field_value}")
            yaml_file_path = Path(field_value)
            yaml_parent_dir = yaml_file_path.parent
            with open(field_value) as f:
                h5ad_data = yaml.safe_load(f)
                for sample_name, h5ad_path in h5ad_data.items():
                    h5ad_path = Path(h5ad_path)
                    # Resolve relative paths relative to yaml file location
                    if not h5ad_path.is_absolute():
                        h5ad_path = yaml_parent_dir / h5ad_path
                    sample_h5ad_dict[sample_name] = h5ad_path
                    
        elif processing_type == 'list':
            logger.info(f"Using {option_name} with {len(field_value)} files")
            for h5ad_path in field_value:
                h5ad_path = Path(h5ad_path)
                sample_name = h5ad_path.stem
                if sample_name in sample_h5ad_dict:
                    logger.warning(f"Duplicate sample name: {sample_name}, will be overwritten")
                sample_h5ad_dict[sample_name] = h5ad_path
                
        elif processing_type == 'file':
            logger.info(f"Using {option_name}: {field_value}")
            list_file_path = Path(field_value)
            list_file_parent_dir = list_file_path.parent
            with open(field_value) as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        h5ad_path = Path(line)
                        # Resolve relative paths relative to list file location
                        if not h5ad_path.is_absolute():
                            h5ad_path = list_file_parent_dir / h5ad_path
                        sample_name = h5ad_path.stem
                        if sample_name in sample_h5ad_dict:
                            logger.warning(f"Duplicate sample name: {sample_name}, will be overwritten")
                        sample_h5ad_dict[sample_name] = h5ad_path
        break
    
    return sample_h5ad_dict

def verify_homolog_file_format(config):
    if config.homolog_file is not None:
        logger.info(
            f"User provided homolog file to map gene names to human: {config.homolog_file}"
        )
        # check the format of the homolog file
        with open(config.homolog_file) as f:
            first_line = f.readline().strip()
            _n_col = len(first_line.split())
            if _n_col != 2:
                raise ValueError(
                    f"Invalid homolog file format. Expected 2 columns, first column should be other species gene name, second column should be human gene name. "
                    f"Got {_n_col} columns in the first line."
                )
            else:
                first_col_name, second_col_name = first_line.split()
                config.species = first_col_name
                logger.info(
                    f"Homolog file provided and will map gene name from column1:{first_col_name} to column2:{second_col_name}"
                )
    else:
        logger.info("No homolog file provided. Run in human mode.")

"""Bind encoder weights to their ordered input genes and preprocessing contract."""

import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch

from .section_data import cell_qc_summary


def feature_hash(genes):
    return hashlib.sha256(json.dumps(genes, ensure_ascii=True).encode()).hexdigest()


def input_contract(config):
    """Conservative reuse guard, not a content checksum of the source H5AD files."""
    fields = (
        "dataset_type",
        "data_layer",
        "feat_cell",
        "K",
        "n_neighbors",
        "spatial_key",
        "graph_mode",
        "graph_radius",
        "graph_sigma",
        "graph_z_scale",
        "distribution",
        "use_tf",
        "annotation",
        "do_sampling",
        "n_cell_training",
        "random_seed",
        "cell_qc_min_genes",
        "cell_qc_min_counts",
    )
    inputs = []
    for name, filename in config.sample_h5ad_dict.items():
        path = Path(filename).resolve()
        stat = path.stat()
        inputs.append(
            {
                "section": str(name),
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "parameters": {
            name: getattr(value, "value", value)
            for name in fields
            for value in [getattr(config, name, None)]
        },
        "ordered_inputs": inputs,
        "encoder_input_transform": "identity",
        "gene_exclusion_regex": "^(HB.-|MT-)",
        "gene_names_make_unique": True,
        "hvg_qc_version": 2,
    }


def make_feature_schema(genes, config, is_count_data, audit):
    genes = list(genes)
    return {
        "version": 1,
        "hvg_order": genes,
        "feature_sha256": feature_hash(genes),
        "input_is_count_data": bool(is_count_data),
        "input_contract": input_contract(config),
        "hvg_selection": audit,
        "cell_qc": cell_qc_summary(config),
        "run_config": config.to_dict_with_paths_as_strings(),
        "software_versions": {
            name: version(name) for name in ("scanpy", "numpy", "pandas", "torch")
        },
    }


def validate_feature_schema(schema):
    if not isinstance(schema, dict) or schema.get("version") != 1:
        raise ValueError("Missing or unsupported checkpoint feature schema")
    genes = schema.get("hvg_order")
    if (
        not isinstance(genes, list)
        or not genes
        or not all(isinstance(gene, str) for gene in genes)
        or len(set(genes)) != len(genes)
        or schema.get("feature_sha256") != feature_hash(genes)
        or not isinstance(schema.get("input_contract"), dict)
        or not isinstance(schema.get("input_is_count_data"), bool)
    ):
        raise ValueError("Invalid checkpoint ordered HVGs or feature hash")


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != 1:
        raise ValueError(
            "Checkpoint has no verified ordered HVGs (legacy state_dict). "
            "Archive the old run and train in a new output directory; it will not be overwritten."
        )
    validate_feature_schema(payload.get("feature_schema"))
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint has no model state_dict")
    return payload


def validate_input_contract(schema, config):
    validate_feature_schema(schema)
    if schema["input_contract"] != input_contract(config):
        raise ValueError(
            "Checkpoint input sections/order, source file stats or preprocessing differ. "
            "Use the original inputs/config or a new output directory; refusing automatic retraining."
        )


def load_checkpoint_state(model, path, expected_schema):
    payload = read_checkpoint(path)
    validate_feature_schema(expected_schema)
    schema = payload["feature_schema"]
    for key in ("hvg_order", "feature_sha256", "input_contract", "input_is_count_data", "cell_qc"):
        if schema[key] != expected_schema[key]:
            raise ValueError(f"Checkpoint feature mismatch: {key}")
    return model.load_compatible_state_dict(payload["state_dict"])


def save_checkpoint(model, path, schema):
    """Publish weights and their feature definition together, never as separate files."""
    validate_feature_schema(schema)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as f:
            temporary = f.name
            torch.save(
                {
                    "checkpoint_version": 1,
                    "state_dict": model.state_dict(),
                    "feature_schema": schema,
                },
                f,
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)

#!/usr/bin/env python3
"""Attach the E11.5 spatial-domain annotation to the derived graph-ready H5AD."""

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def load_aligned_annotation(
    h5ad_path: Path, metadata_path: Path
) -> tuple[pd.Categorical, dict[str, int]]:
    with h5py.File(h5ad_path, "r") as handle:
        obs_group = handle["obs"]
        index_key = obs_group.attrs["_index"]
        if isinstance(index_key, bytes):
            index_key = index_key.decode("utf-8")
        obs_names = decode_strings(obs_group[index_key][...])

    metadata = pd.read_csv(
        metadata_path,
        usecols=lambda column: column in {"Unnamed: 0", "spatial_domain"},
        dtype={"spatial_domain": "string"},
    )
    if list(metadata.columns) != ["Unnamed: 0", "spatial_domain"]:
        raise ValueError(f"Unexpected metadata columns: {metadata.columns.tolist()}")
    metadata_ids = metadata.iloc[:, 0].astype(str)
    if metadata_ids.duplicated().any():
        raise ValueError("Metadata observation IDs are not unique")
    if pd.Index(obs_names).has_duplicates:
        raise ValueError("H5AD observation IDs are not unique")
    if len(metadata_ids) != len(obs_names):
        raise ValueError(
            f"Observation count mismatch: metadata={len(metadata_ids):,}, h5ad={len(obs_names):,}"
        )

    metadata_index = pd.Index(metadata_ids)
    order = metadata_index.get_indexer(obs_names)
    if np.any(order < 0):
        missing = obs_names[order < 0][:5].tolist()
        raise ValueError(f"Metadata is missing H5AD observation IDs, examples={missing}")

    aligned = pd.Categorical(metadata["spatial_domain"].array.take(order))
    counts = {
        str(label): int(count)
        for label, count in zip(
            aligned.categories,
            np.bincount(aligned.codes[aligned.codes >= 0], minlength=len(aligned.categories)),
            strict=True,
        )
    }
    return aligned, counts


def write_categorical_column(h5ad_path: Path, column: str, values: pd.Categorical) -> None:
    pending = f"__pending_{column}__"
    with h5py.File(h5ad_path, "r+") as handle:
        obs_group = handle["obs"]
        column_order = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in obs_group.attrs["column-order"]
        ]
        if column in obs_group or column in column_order:
            raise ValueError(f"H5AD already contains obs[{column!r}]")
        if pending in obs_group:
            del obs_group[pending]

        group = obs_group.create_group(pending)
        group.attrs["encoding-type"] = "categorical"
        group.attrs["encoding-version"] = "0.2.0"
        group.attrs["ordered"] = bool(values.ordered)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        categories = group.create_dataset(
            "categories", data=np.asarray(values.categories, dtype=object), dtype=string_dtype
        )
        categories.attrs["encoding-type"] = "string-array"
        categories.attrs["encoding-version"] = "0.2.0"

        codes = group.create_dataset(
            "codes",
            data=np.asarray(values.codes, dtype=np.int8),
            chunks=True,
            compression="gzip",
            compression_opts=1,
        )
        codes.attrs["encoding-type"] = "array"
        codes.attrs["encoding-version"] = "0.2.0"
        handle.flush()

        obs_group.move(pending, column)
        obs_group.attrs["column-order"] = np.asarray([*column_order, column], dtype=object)
        handle.flush()


def main() -> None:
    args = parse_args()
    categorical, counts = load_aligned_annotation(args.h5ad, args.metadata)
    write_categorical_column(args.h5ad, "spatial_domain", categorical)

    checked = ad.read_h5ad(args.h5ad, backed="r")
    try:
        observed = checked.obs["spatial_domain"]
        if len(observed) != len(categorical):
            raise ValueError("Written spatial_domain column has the wrong length")
        missing = int(observed.isna().sum())
        observed_counts = {str(key): int(value) for key, value in observed.value_counts().items()}
        if observed_counts != counts:
            raise ValueError("Written spatial_domain category counts do not match metadata")
    finally:
        checked.file.close()

    audit = {
        "h5ad": str(args.h5ad.resolve()),
        "metadata": str(args.metadata.resolve()),
        "alignment": "observation_id",
        "column": "spatial_domain",
        "n_observations": len(categorical),
        "n_missing": missing,
        "n_categories": len(categorical.categories),
        "category_counts": counts,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()

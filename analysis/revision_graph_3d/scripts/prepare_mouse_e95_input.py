#!/usr/bin/env python3
"""Add explicit physical-section labels to a read-only-derived E9.5 H5AD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import anndata as ad
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spatial-key", default="3d_align_spatial")
    parser.add_argument("--section-key", default="section_id")
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def validate_output(
    output: Path,
    *,
    spatial_key: str,
    section_key: str,
    expected_z: np.ndarray,
) -> None:
    adata = ad.read_h5ad(output, backed="r")
    try:
        if spatial_key not in adata.obsm:
            raise ValueError(f"prepared H5AD lacks obsm[{spatial_key!r}]")
        if section_key not in adata.obs:
            raise ValueError(f"prepared H5AD lacks obs[{section_key!r}]")
        observed_z = adata.obs[section_key].astype(str).to_numpy().astype(np.float64)
        if not np.array_equal(observed_z, expected_z):
            raise ValueError("section labels do not exactly reproduce the stored z coordinate")
    finally:
        adata.file.close()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("the source H5AD is read-only; output must be a different path")
    if not source.is_file():
        raise FileNotFoundError(source)

    with h5py.File(source, "r") as handle:
        spatial_path = f"obsm/{args.spatial_key}"
        if spatial_path not in handle:
            raise KeyError(f"source H5AD lacks {spatial_path!r}")
        coords = handle[spatial_path]
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"{spatial_path} must have exactly three XYZ columns")
        z = np.asarray(coords[:, 2], dtype=np.float64)

    if not np.isfinite(z).all():
        raise ValueError("stored z coordinates contain non-finite values")
    section_z, section_codes = np.unique(z, return_inverse=True)
    if section_z.size > np.iinfo(np.int16).max:
        raise ValueError("too many physical sections for int16 categorical codes")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        validate_output(
            output,
            spatial_key=args.spatial_key,
            section_key=args.section_key,
            expected_z=z,
        )
        print(f"Reusing validated prepared input: {output}")
        return

    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    source_hash = hashlib.sha256()
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as output_handle:
            while chunk := source_handle.read(16 * 1024 * 1024):
                source_hash.update(chunk)
                output_handle.write(chunk)
        shutil.copystat(source, temporary)

        with h5py.File(temporary, "r+") as handle:
            obs = handle["obs"]
            column_order = decode_strings(obs.attrs["column-order"])
            if args.section_key in obs:
                raise ValueError(f"source already contains obs[{args.section_key!r}]")

            section_group = obs.create_group(args.section_key)
            section_group.attrs["encoding-type"] = "categorical"
            section_group.attrs["encoding-version"] = "0.2.0"
            section_group.attrs["ordered"] = True

            text_dtype = h5py.string_dtype(encoding="utf-8")
            categories = section_group.create_dataset(
                "categories",
                data=np.asarray([format(value, ".17g") for value in section_z], dtype=object),
                dtype=text_dtype,
                compression="gzip",
            )
            categories.attrs["encoding-type"] = "string-array"
            categories.attrs["encoding-version"] = "0.2.0"

            codes = section_group.create_dataset(
                "codes",
                data=section_codes.astype(np.int16),
                compression="gzip",
            )
            codes.attrs["encoding-type"] = "array"
            codes.attrs["encoding-version"] = "0.2.0"
            obs.attrs["column-order"] = np.asarray(column_order + [args.section_key], dtype=text_dtype)

        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    validate_output(
        output,
        spatial_key=args.spatial_key,
        section_key=args.section_key,
        expected_z=z,
    )
    section_counts = np.bincount(section_codes)
    manifest = {
        "source": str(source),
        "source_sha256": source_hash.hexdigest(),
        "output": str(output),
        "spatial_key": args.spatial_key,
        "section_key": args.section_key,
        "n_spots": int(z.size),
        "n_sections": int(section_z.size),
        "section_z": section_z.tolist(),
        "section_counts": section_counts.tolist(),
        "successive_z_gaps": np.diff(section_z).tolist(),
    }
    manifest_path = output.with_suffix(".provenance.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared input: {output}")
    print(f"Provenance: {manifest_path}")


if __name__ == "__main__":
    main()

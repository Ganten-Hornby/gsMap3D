#!/usr/bin/env python3
"""Prepare the z-axis section view of the full Mouse E11.5 embryo."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def set_slice_id_from_z_axis(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Define physical sections from unique registered z coordinates."""

    coordinates = np.asarray(adata.obsm["z_correction"], dtype=np.float64)
    if coordinates.shape != (adata.n_obs, 3):
        raise ValueError(f"Expected N×3 corrected coordinates, got {coordinates.shape}")
    if not np.isfinite(coordinates).all():
        raise ValueError("Corrected coordinates contain non-finite values")

    z = coordinates[:, 2]
    unique_z, section_codes = np.unique(z, return_inverse=True)
    adata.obs["source_slice_id"] = adata.obs["slice_id"].astype(str).to_numpy()
    adata.obs["slice_id"] = pd.Categorical.from_codes(
        section_codes,
        categories=unique_z,
        ordered=True,
    )
    return unique_z, section_codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        existing = ad.read_h5ad(args.output, backed="r")
        try:
            required = {"aligned_spatial_2D", "z_correction"}
            z = np.asarray(existing.obsm["z_correction"][:, 2])
            if (
                existing.n_obs == 6_993_667
                and required.issubset(existing.obsm)
                and {"slice_id", "source_slice_id", "mapped_celltype"}.issubset(
                    existing.obs.columns
                )
                and existing.obs["slice_id"].nunique() == np.unique(z).size
                and np.array_equal(
                    existing.obs["slice_id"].astype(float).to_numpy(),
                    z,
                )
                and args.audit.is_file()
            ):
                print(f"validated_existing={args.output}")
                return
        finally:
            existing.file.close()

    adata = ad.read_h5ad(args.source)
    required_obs = {"slice_id", "mapped_celltype"}
    missing_obs = required_obs.difference(adata.obs.columns)
    if missing_obs:
        raise ValueError(f"Missing required obs fields: {sorted(missing_obs)}")
    if "z_correction" not in adata.obsm:
        raise ValueError("Missing obsm['z_correction']")
    if not adata.obs_names.is_unique:
        raise ValueError("Observation IDs must be unique")

    coordinates = np.asarray(adata.obsm["z_correction"], dtype=np.float64)
    adata.obsm["aligned_spatial_2D"] = coordinates[:, :2].copy()

    original_sections = adata.obs["slice_id"].astype(str).to_numpy()
    unique_z, _ = set_slice_id_from_z_axis(adata)
    z = coordinates[:, 2]

    mappings = []
    for section in np.unique(original_sections):
        section_z = np.unique(z[original_sections == section])
        if section_z.size != 1:
            raise ValueError(
                f"Original section {section!r} spans {section_z.size} corrected z values"
            )
        z_value = float(section_z[0])
        mappings.append(
            {
                "slice_id": section,
                "z_correction": z_value,
                "z_axis_slice_id": z_value,
                "n_obs": int(np.sum(original_sections == section)),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".h5ad.tmp")
    adata.write_h5ad(temporary)
    temporary.replace(args.output)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "output": str(args.output.resolve()),
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "original_slice_labels": int(np.unique(original_sections).size),
                "physical_sections": int(unique_z.size),
                "physical_z_values": unique_z.tolist(),
                "slice_to_physical_section": mappings,
                "no_observation_downsampling": True,
                "coordinate_source": "obsm['z_correction']",
                "duplicate_z_policy": "merge labels occupying the same measured z plane",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"prepared={args.output} observations={adata.n_obs} genes={adata.n_vars} "
        f"original_sections={np.unique(original_sections).size} "
        f"physical_sections={unique_z.size}"
    )


if __name__ == "__main__":
    main()

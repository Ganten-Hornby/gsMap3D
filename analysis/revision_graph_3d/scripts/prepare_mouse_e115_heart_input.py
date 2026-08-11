#!/usr/bin/env python3
"""Prepare the 2D coordinate view for the Mouse E11.5 heart comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.source)
    required_obs = {"ctype_user", "heart_regions", "slices"}
    missing_obs = required_obs.difference(adata.obs.columns)
    if missing_obs:
        raise ValueError(f"Missing required obs fields: {sorted(missing_obs)}")
    if "aligned_spatial_3D_mm" not in adata.obsm:
        raise ValueError("Missing obsm['aligned_spatial_3D_mm']")
    if not adata.obs_names.is_unique:
        raise ValueError("Observation IDs must be unique")
    if adata.obs[list(required_obs)].isna().any().any():
        raise ValueError("Required annotation and section fields must not contain missing values")

    coordinates = np.asarray(adata.obsm["aligned_spatial_3D_mm"], dtype=np.float64)
    if coordinates.shape != (adata.n_obs, 3):
        raise ValueError(f"Expected N×3 physical coordinates, got {coordinates.shape}")
    if not np.isfinite(coordinates).all():
        raise ValueError("Physical coordinates contain non-finite values")
    adata.obsm["aligned_spatial_2D_mm"] = coordinates[:, :2].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    print(
        f"prepared={args.output} observations={adata.n_obs} genes={adata.n_vars} "
        f"sections={adata.obs['slices'].nunique()}"
    )


if __name__ == "__main__":
    main()

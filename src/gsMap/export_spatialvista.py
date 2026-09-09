"""Export completed spatial reports without reading expression or marker matrices."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def export_spatialvista(
    project_dir: str | Path,
    output: str | Path | None = None,
    *,
    traits: list[str] | None = None,
    annotations: list[str] | None = None,
    max_spots: int | None = None,
    overwrite: bool = False,
    metadata: pd.DataFrame | None = None,
) -> Path:
    """Write a lightweight H5AD with validated uns['gsmap'] schema v1.

    Report trait columns already contain -log10(P). Preserve them unchanged;
    never transform them a second time or replace missing associations by zero.
    All spots are exported by default, independently of HTML display sampling.
    """
    project_dir = Path(project_dir).resolve()
    report_dir = project_dir / "report_data"
    meta_path = project_dir / "gsmap_web_report" / "report_meta.json"
    meta = json.loads(meta_path.read_text())
    available_traits = list(meta["traits"])
    available_annotations = list(meta["annotations"])
    traits = available_traits if traits is None else list(traits)
    annotations = available_annotations if annotations is None else list(annotations)
    for kind, selected, available in (
        ("traits", traits, available_traits),
        ("annotations", annotations, available_annotations),
    ):
        if not selected or len(set(selected)) != len(selected):
            raise ValueError(f"Select a nonempty, unique list of {kind}")
        if missing := set(selected).difference(available):
            raise ValueError(f"Unknown {kind}: {sorted(missing)}")
    if set(traits).intersection(annotations + ["sample_name", "spot", "3d_x", "3d_y", "3d_z"]):
        raise ValueError("Trait names must not collide with metadata columns")
    output = (
        Path(output)
        if output is not None
        else project_dir / f"{project_dir.name}.spatialvista.h5ad"
    )
    if output.suffix.lower() != ".h5ad":
        raise ValueError("SpatialVista export must have a .h5ad extension")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Export exists: {output}; use overwrite=True / --overwrite")
    columns = list(
        dict.fromkeys(["spot", "3d_x", "3d_y", "3d_z", "sample_name", *annotations, *traits])
    )
    if metadata is None:
        # Keep numeric annotation labels and literal strings such as 'NA' intact.
        metadata = pd.read_csv(
            report_dir / "spot_metadata.csv",
            usecols=columns,
            converters=dict.fromkeys(["spot", "sample_name", *annotations], str),
        )
    if missing := set(columns).difference(metadata.columns):
        raise ValueError(f"Missing 3D report metadata columns: {sorted(missing)}")
    if not len(metadata):
        raise ValueError("No spots available for SpatialVista export")
    total = len(metadata)
    frame = metadata.loc[:, columns]
    if (
        frame["spot"].isna().any()
        or frame["spot"].astype(str).eq("").any()
        or frame["spot"].astype(str).duplicated().any()
    ):
        raise ValueError("Spot IDs must be nonempty and unique")
    if max_spots is not None:
        if isinstance(max_spots, bool) or not isinstance(max_spots, int) or max_spots <= 0:
            raise ValueError("max_spots must be a positive integer or None")
        if len(frame) > max_spots:
            selected = np.sort(
                np.random.default_rng(2024).choice(len(frame), size=max_spots, replace=False)
            )
            frame = frame.iloc[selected]
    frame = frame.set_index("spot", verify_integrity=True)
    frame.index = frame.index.astype(str)
    coords = frame[["3d_x", "3d_y", "3d_z"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates must be finite")
    sample_order = list(meta["samples"])
    if len(set(sample_order)) != len(sample_order):
        raise ValueError("Report section order contains duplicates")
    if frame["sample_name"].isna().any() or not set(frame["sample_name"].astype(str)).issubset(
        sample_order
    ):
        raise ValueError("Every spot must map to the recorded report section order")
    obs = frame[list(dict.fromkeys(["sample_name", *annotations]))].copy()
    for key in annotations:
        obs[key] = (
            obs[key].astype("string").fillna("Unknown").replace("", "Unknown").astype("category")
        )
    obs["sample_name"] = pd.Categorical(
        frame["sample_name"].astype(str), categories=sample_order, ordered=True
    )
    mapping = {}
    for trait in traits:
        slug = re.sub(r"[^A-Za-z0-9_]", "_", trait)[:80]
        digest = hashlib.sha256(trait.encode()).hexdigest()[:10]
        key = f"gsMap_logp_{slug}_{digest}"
        if key in obs:
            raise ValueError(f"Export column collides with annotation: {key}")
        column = frame[trait]
        if (
            not pd.api.types.is_numeric_dtype(column.dtype)
            or pd.api.types.is_bool_dtype(column.dtype)
            or pd.api.types.is_complex_dtype(column.dtype)
        ):
            raise ValueError(f"Trait {trait!r} must be a real numeric column")
        values = column.to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(values).any() or (values < 0).any() or not np.isfinite(values).any():
            raise ValueError(
                f"Trait {trait!r} requires nonnegative -log10(P) values or NaN, with at least one finite value"
            )
        if np.nanmax(values) > np.finfo(np.float32).max:
            raise ValueError(f"Trait {trait!r} exceeds float32 display range")
        obs[key] = values.astype(np.float32)
        mapping[key] = trait
    coordinates_path = report_dir / "spatial_3d_coordinate_info.json"
    coordinate_info = (
        json.loads(coordinates_path.read_text()) if coordinates_path.is_file() else {}
    )
    result = ad.AnnData(obs=obs, obsm={"spatial": coords})
    result.uns["gsmap"] = {
        "schema_version": 1,
        "value_scale": "neg_log10_p",
        "spatial_key": "spatial",
        "section_key": "sample_name",
        "annotation_keys": annotations,
        "default_annotation": annotations[0],
        "traits": mapping,
        "coordinate_provenance_json": json.dumps(coordinate_info, allow_nan=False),
        "source_n_spots": total,
        "export_n_spots": result.n_obs,
        "sampling": "all" if result.n_obs == total else "uniform_without_replacement_seed_2024",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".h5ad", dir=output.parent
    )
    os.close(descriptor)
    try:
        result.write_h5ad(temporary, compression="gzip")
        if overwrite:
            os.replace(temporary, output)
        else:
            # Exclusive publication, including when a competing process wrote it.
            os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output

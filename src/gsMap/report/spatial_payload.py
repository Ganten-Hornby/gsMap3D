"""Compact, browser-oriented payloads for spatial report visualizations."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SPATIAL_PAYLOAD_VERSION = 3
P_VALUE_COLORS = [
    "#313695",
    "#4575b4",
    "#74add1",
    "#fee090",
    "#fdae61",
    "#f46d43",
    "#d73027",
    "#a50026",
]


def safe_name(value: str) -> str:
    """Return the filename-safe convention used throughout the report."""

    return "".join(character if character.isalnum() else "_" for character in str(value))


def encode_typed_array(values, dtype: str = "float32") -> dict:
    """Encode a one-dimensional NumPy-compatible array for compact local JS loading."""

    dtype_map = {
        "float32": np.dtype("<f4"),
        "uint16": np.dtype("<u2"),
        "uint32": np.dtype("<u4"),
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported browser payload dtype: {dtype}")

    array = np.ascontiguousarray(values, dtype=dtype_map[dtype]).reshape(-1)
    return {
        "encoding": "base64",
        "dtype": dtype,
        "length": int(array.size),
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def decode_typed_array(payload: dict) -> np.ndarray:
    """Decode a payload in Python; primarily useful for validation and tests."""

    dtype_map = {
        "float32": np.dtype("<f4"),
        "uint16": np.dtype("<u2"),
        "uint32": np.dtype("<u4"),
    }
    raw = base64.b64decode(payload["data"])
    return np.frombuffer(raw, dtype=dtype_map[payload["dtype"]], count=payload["length"])


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_to_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_keyed_payload(path: Path, root_name: str, key: str, payload: dict) -> None:
    serialized_key = json.dumps(str(key), ensure_ascii=False)
    serialized_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    _atomic_write_text(
        path,
        f"window.{root_name}=window.{root_name}||{{}};"
        f"window.{root_name}[{serialized_key}]={serialized_payload};",
    )


def _categorical_schema(values: pd.Series | np.ndarray) -> tuple[list[str], list[str]]:
    series = pd.Series(values, copy=False).astype("string").fillna("NaN").astype(str)
    categories = sorted(series.unique().tolist(), key=str)

    from gsMap.report.visualize import _create_color_map

    color_map = _create_color_map(categories, hex=True, rng=42)
    colors = [color_map[category] for category in categories]
    return categories, colors


def _categorical_payload(
    values: pd.Series | np.ndarray,
    categories: list[str] | None = None,
    colors: list[str] | None = None,
) -> dict:
    series = pd.Series(values, copy=False).astype("string").fillna("NaN").astype(str)
    if categories is None or colors is None:
        categories, colors = _categorical_schema(series)
    categorical = pd.Categorical(series, categories=categories)
    if (categorical.codes < 0).any():
        raise ValueError("Categorical schema does not include every observed value")
    dtype = "uint16" if len(categories) <= np.iinfo(np.uint16).max else "uint32"
    return {
        "codes": encode_typed_array(categorical.codes, dtype=dtype),
        "categories": categories,
        "colors": colors,
    }


def _stratified_positions(
    metadata: pd.DataFrame,
    target_size: int,
    annotation_columns: list[str],
    random_state: int = 42,
) -> np.ndarray:
    """Select an exact-size display sample while retaining samples and rare annotations."""

    n_rows = len(metadata)
    if target_size >= n_rows:
        return np.arange(n_rows, dtype=np.int64)

    strata = (
        metadata["sample_name"].astype("string").fillna("NaN")
        if "sample_name" in metadata
        else pd.Series("all", index=metadata.index, dtype="string")
    )
    if annotation_columns:
        primary = annotation_columns[0]
        if primary in metadata:
            strata = strata.str.cat(metadata[primary].astype("string").fillna("NaN"), sep="\x1f")

    codes, _ = pd.factorize(strata, sort=False)
    counts = np.bincount(codes)
    exact = target_size * counts / n_rows
    allocations = np.floor(exact).astype(np.int64)
    preserve_each_stratum = len(counts) <= target_size
    if preserve_each_stratum:
        allocations[(counts > 0) & (allocations == 0)] = 1

    while allocations.sum() > target_size:
        candidates = np.flatnonzero(allocations > int(preserve_each_stratum))
        if candidates.size == 0:
            break
        candidate = candidates[np.argmax(allocations[candidates] - exact[candidates])]
        allocations[candidate] -= 1

    remainder_order = np.argsort(-(exact - np.floor(exact)))
    remaining = target_size - int(allocations.sum())
    while remaining > 0:
        progressed = False
        for group in remainder_order:
            if allocations[group] < counts[group]:
                allocations[group] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    rng = np.random.default_rng(random_state)
    selected = []
    for group, allocation in enumerate(allocations):
        if allocation <= 0:
            continue
        members = np.flatnonzero(codes == group)
        selected.append(rng.choice(members, size=int(allocation), replace=False))

    positions = np.sort(np.concatenate(selected).astype(np.int64, copy=False))
    if positions.size != target_size:
        raise RuntimeError(
            f"Spatial display sampling selected {positions.size:,} rows; expected {target_size:,}."
        )
    return positions


def _top_trait_positions(
    values: np.ndarray,
    selected_mask: np.ndarray,
    n_extra: int,
) -> np.ndarray:
    """Return top finite trait positions not already present in the display sample."""

    if n_extra <= 0:
        return np.empty(0, dtype=np.int64)

    ranked_values = np.where(np.isfinite(values), values, -np.inf)
    finite_count = int(np.isfinite(values).sum())
    if finite_count == 0:
        return np.empty(0, dtype=np.int64)

    candidate_count = min(finite_count, max(n_extra * 2, n_extra + 64))
    while True:
        if candidate_count >= len(values):
            candidates = np.arange(len(values), dtype=np.int64)
        else:
            split = len(values) - candidate_count
            candidates = np.argpartition(ranked_values, split)[split:]
        candidates = candidates[np.argsort(ranked_values[candidates])[::-1]]
        candidates = candidates[~selected_mask[candidates]]
        if len(candidates) >= n_extra or candidate_count >= finite_count:
            return candidates[:n_extra]
        candidate_count = min(finite_count, candidate_count * 2)


def _robust_trait_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    n_top = min(20, finite.size)
    top = np.partition(finite, finite.size - n_top)[-n_top:]
    upper = float(np.median(top))
    return upper if upper > 0 else max(1.0, float(np.max(finite)))


def _point_size_3d(n_points: int) -> float:
    return round(float(np.clip(520.0 / np.sqrt(max(1, n_points)), 1.0, 4.0)), 2)


def _outputs_exist(root: Path, outputs: list[str]) -> bool:
    return bool(outputs) and all(
        (root / relative_path).is_file() and (root / relative_path).stat().st_size > 0
        for relative_path in outputs
    )


def export_spatial_3d_payloads(
    metadata: pd.DataFrame,
    traits: list[str],
    annotations: list[str],
    report_data_dir: Path,
    web_report_dir: Path,
    max_points: int,
    force: bool = False,
    extreme_points_per_trait: int = 500,
    coordinate_info: dict | None = None,
) -> dict | None:
    """Create shared 3D coordinates and lazy feature payloads for Plotly rendering."""

    coordinate_columns = ["3d_x", "3d_y", "3d_z"]
    if any(column not in metadata for column in coordinate_columns):
        logger.warning("3D coordinates are incomplete; skipping the 3D report viewer.")
        return None

    all_coordinates = np.column_stack(
        [
            pd.to_numeric(metadata[column], errors="coerce").to_numpy(dtype=np.float32)
            for column in coordinate_columns
        ]
    )
    if not np.isfinite(all_coordinates).any(axis=0).all():
        logger.warning("At least one 3D coordinate axis has no finite values; skipping 3D.")
        return None

    coordinate_bounds = {
        column: [
            float(np.nanmin(all_coordinates[:, index])),
            float(np.nanmax(all_coordinates[:, index])),
        ]
        for index, column in enumerate(coordinate_columns)
    }
    coordinate_ranges = [bounds[1] - bounds[0] for bounds in coordinate_bounds.values()]

    available_traits = [trait for trait in traits if trait in metadata]
    available_annotations = [annotation for annotation in annotations if annotation in metadata]
    target_size = min(len(metadata), max(1, int(max_points)))
    metadata_file = report_data_dir / "spot_metadata.csv"
    source_signature = None
    if metadata_file.is_file():
        source_stat = metadata_file.stat()
        source_signature = {
            "inode": source_stat.st_ino,
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        }
    signature = {
        "version": SPATIAL_PAYLOAD_VERSION,
        "n_total": len(metadata),
        "n_display": target_size,
        "extreme_points_per_trait": extreme_points_per_trait,
        "traits": available_traits,
        "annotations": available_annotations,
        "coordinate_info": coordinate_info,
        "source": source_signature,
        "coordinate_bounds": coordinate_bounds,
    }

    data_dir = report_data_dir / "spatial_3d"
    provenance_path = data_dir / "spatial_3d_display_spots.parquet"
    manifest_path = data_dir / "spatial_3d_manifest.json"
    if not force and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        if (
            manifest.get("signature") == signature
            and _outputs_exist(web_report_dir, manifest.get("outputs", []))
            and provenance_path.is_file()
        ):
            logger.info("Compact 3D viewer payloads are current. Skipping export.")
            return manifest.get("viewer_info")

    logger.info(
        f"Exporting compact 3D viewer payloads for {len(metadata):,} spots; "
        f"browser display sample={target_size:,}"
    )
    started = time.perf_counter()
    positions = _stratified_positions(metadata, target_size, available_annotations)
    selected = metadata.iloc[positions]
    coordinates = all_coordinates[positions]

    sample_payload = None
    if "sample_name" in selected:
        sample_payload = _categorical_payload(selected["sample_name"])
    annotation_schemas = {
        annotation: _categorical_schema(metadata[annotation])
        for annotation in available_annotations
    }

    base_payload = {
        "version": SPATIAL_PAYLOAD_VERSION,
        "n_points": target_size,
        "n_points_original": len(metadata),
        "x": encode_typed_array(coordinates[:, 0]),
        "y": encode_typed_array(coordinates[:, 1]),
        "z": encode_typed_array(coordinates[:, 2]),
        "coordinate_ranges": coordinate_ranges,
        "sample": sample_payload,
        "point_size": _point_size_3d(target_size),
    }
    base_relative = "js_data/spatial_3d/spatial_3d_base.js"
    _atomic_write_text(
        web_report_dir / base_relative,
        "window.GSMAP_SPATIAL_3D_BASE="
        + json.dumps(base_payload, ensure_ascii=False, separators=(",", ":"))
        + ";",
    )

    selected_mask = np.zeros(len(metadata), dtype=bool)
    selected_mask[positions] = True
    trait_index = {}
    outputs = [base_relative]
    for trait in available_traits:
        values = pd.to_numeric(metadata[trait], errors="coerce").to_numpy(dtype=np.float32)
        extra_positions = _top_trait_positions(
            values, selected_mask, min(extreme_points_per_trait, len(metadata) - target_size)
        )
        extra = {
            "x": encode_typed_array(all_coordinates[extra_positions, 0]),
            "y": encode_typed_array(all_coordinates[extra_positions, 1]),
            "z": encode_typed_array(all_coordinates[extra_positions, 2]),
            "values": encode_typed_array(values[extra_positions]),
        }
        payload = {
            "values": encode_typed_array(values[positions]),
            "extra": extra,
            "n_extra": int(len(extra_positions)),
            "cmin": 0.0,
            "cmax": _robust_trait_max(values),
        }
        filename = f"trait_{safe_name(trait)}.js"
        relative = f"js_data/spatial_3d/{filename}"
        _write_keyed_payload(web_report_dir / relative, "GSMAP_SPATIAL_3D_TRAITS", trait, payload)
        trait_index[trait] = {"file": relative}
        outputs.append(relative)

    annotation_index = {}
    for annotation in available_annotations:
        categories, colors = annotation_schemas[annotation]
        payload = _categorical_payload(selected[annotation], categories, colors)
        filename = f"annotation_{safe_name(annotation)}.js"
        relative = f"js_data/spatial_3d/{filename}"
        _write_keyed_payload(
            web_report_dir / relative,
            "GSMAP_SPATIAL_3D_ANNOTATIONS",
            annotation,
            payload,
        )
        annotation_index[annotation] = {"file": relative}
        outputs.append(relative)

    index_payload = {
        "version": SPATIAL_PAYLOAD_VERSION,
        "base_file": base_relative,
        "n_points": target_size,
        "n_points_original": len(metadata),
        "downsampled": target_size < len(metadata),
        "traits": trait_index,
        "annotations": annotation_index,
        "colorscale": P_VALUE_COLORS,
        "extreme_points_per_trait": extreme_points_per_trait,
        "coordinate_info": coordinate_info,
    }
    index_relative = "js_data/spatial_3d/spatial_3d_index.js"
    _atomic_write_text(
        web_report_dir / index_relative,
        "window.GSMAP_SPATIAL_3D_INDEX="
        + json.dumps(index_payload, ensure_ascii=False, separators=(",", ":"))
        + ";",
    )
    outputs.append(index_relative)

    provenance = pd.DataFrame(
        {
            "display_index": np.arange(target_size, dtype=np.int64),
            "source_position": positions,
            "spot": selected["spot"].astype(str).to_numpy()
            if "spot" in selected
            else selected.index.astype(str).to_numpy(),
            "sample_name": selected["sample_name"].astype(str).to_numpy()
            if "sample_name" in selected
            else "all",
        }
    )
    _atomic_to_parquet(provenance, provenance_path)

    viewer_info = {
        "schema_version": SPATIAL_PAYLOAD_VERSION,
        "index_file": index_relative,
        "n_points": target_size,
        "n_points_original": len(metadata),
        "downsampled": target_size < len(metadata),
        "point_size": base_payload["point_size"],
        "coordinate_info": coordinate_info,
    }
    manifest = {
        "version": SPATIAL_PAYLOAD_VERSION,
        "signature": signature,
        "outputs": sorted(outputs),
        "viewer_info": viewer_info,
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )
    elapsed = time.perf_counter() - started
    logger.info(
        f"Compact 3D viewer export complete in {elapsed:.2f}s: "
        f"{len(outputs)} payload files, {target_size:,}/{len(metadata):,} display spots"
    )
    return viewer_info


def export_sample_spatial_payloads(
    metadata: pd.DataFrame,
    samples: list[str],
    traits: list[str],
    annotations: list[str],
    js_data_dir: Path,
    max_points: int,
) -> tuple[dict, list[str]]:
    """Write compact per-sample bases and independently loadable trait arrays."""

    payload_dir = js_data_dir / "spatial_2d"
    payload_dir.mkdir(parents=True, exist_ok=True)
    sample_index = {}
    outputs = []

    sample_values = metadata["sample_name"].astype(str).to_numpy()
    has_spatial_coordinates = "sx" in metadata and "sy" in metadata
    sx_all = (
        pd.to_numeric(metadata["sx"], errors="coerce").to_numpy(dtype=np.float32)
        if has_spatial_coordinates
        else None
    )
    sy_all = (
        pd.to_numeric(metadata["sy"], errors="coerce").to_numpy(dtype=np.float32)
        if has_spatial_coordinates
        else None
    )
    sample_positions = {}
    annotation_schemas = {
        annotation: _categorical_schema(metadata[annotation])
        for annotation in annotations
        if annotation in metadata
    }

    for sample_name in samples:
        all_positions = np.flatnonzero(sample_values == str(sample_name))
        if all_positions.size == 0:
            raise ValueError(f"No spot metadata found for configured sample: {sample_name}")
        n_original = int(all_positions.size)
        if has_spatial_coordinates and n_original > max_points:
            local_positions = _stratified_positions(
                metadata.iloc[all_positions],
                max_points,
                annotations,
                random_state=42,
            )
            positions = all_positions[local_positions]
        else:
            positions = all_positions
        sample_positions[sample_name] = positions

        if has_spatial_coordinates:
            sx = sx_all[positions]
            sy = sy_all[positions]
            from gsMap.report.visualize import estimate_plotly_point_size

            _, point_size = estimate_plotly_point_size(np.column_stack((sx, sy)), 560)
        else:
            sx = sy = None
            point_size = 5.0

        annotation_payloads = {}
        for annotation, (categories, colors) in annotation_schemas.items():
            annotation_payloads[annotation] = _categorical_payload(
                metadata[annotation].iloc[positions], categories, colors
            )
        include_spot_labels = len(positions) <= 50_000
        base_payload = {
            "version": SPATIAL_PAYLOAD_VERSION,
            "n_points": int(len(positions)),
            "n_points_original": n_original,
            "point_size": round(float(point_size), 2),
            "sx": encode_typed_array(sx) if sx is not None else None,
            "sy": encode_typed_array(sy) if sy is not None else None,
            "spot": metadata["spot"].iloc[positions].astype(str).tolist()
            if include_spot_labels and "spot" in metadata
            else None,
            "annotations": annotation_payloads,
        }

        safe_sample = safe_name(sample_name)
        base_filename = f"sample_{safe_sample}_base.js"
        base_relative = f"spatial_2d/{base_filename}"
        _write_keyed_payload(
            js_data_dir / base_relative, "GSMAP_SAMPLE_BASES", sample_name, base_payload
        )
        outputs.append(base_relative)

        sample_index[sample_name] = {
            "version": SPATIAL_PAYLOAD_VERSION,
            "base_file": base_relative,
            "trait_files": {},
            "n_spots": int(len(positions)),
            "n_spots_original": n_original,
            "point_size": round(float(point_size), 2),
            "has_spot_labels": include_spot_labels,
        }

    # Convert each full trait column once, then export the selected rows for every sample.
    for trait in traits:
        if trait not in metadata:
            continue
        all_values = pd.to_numeric(metadata[trait], errors="coerce").to_numpy(dtype=np.float32)
        for sample_name in samples:
            positions = sample_positions[sample_name]
            values = np.round(all_values[positions], decimals=1)
            payload = {"values": encode_typed_array(values)}
            filename = f"sample_{safe_name(sample_name)}_trait_{safe_name(trait)}.js"
            relative = f"spatial_2d/{filename}"
            compound_key = f"{sample_name}\x1f{trait}"
            _write_keyed_payload(
                js_data_dir / relative, "GSMAP_SAMPLE_TRAITS", compound_key, payload
            )
            sample_index[sample_name]["trait_files"][trait] = {"file": relative}
            outputs.append(relative)

    for sample_name in samples:
        info = sample_index[sample_name]
        logger.info(
            f"Exported compact sample payload: {sample_name}, "
            f"{info['n_spots']:,}/{info['n_spots_original']:,} spots, "
            f"{len(info['trait_files'])} lazy traits"
        )

    return sample_index, outputs

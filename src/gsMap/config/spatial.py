"""Shared spatial semantics and bounded, expression-free coordinate validation."""

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger("gsMap.config")
_validation_scope = ContextVar("gsmap_spatial_validation", default=None)
PROBE_ROWS = 4096


class DatasetType(StrEnum):
    SCRNA_SEQ = "scRNA"
    SPATIAL_2D = "spatial2D"
    SPATIAL_3D = "spatial3D"


def spatial_coordinates(values, dataset_type, *, n_obs=None):
    """Select the computational dimensions and fully validate them at runtime."""
    coords = np.asarray(values)
    mode = DatasetType(dataset_type)
    widths = (3,) if mode == DatasetType.SPATIAL_3D else (2, 3)
    if coords.ndim != 2 or coords.shape[1] not in widths:
        raise ValueError(
            f"{mode} requires an N×{'3 XYZ' if len(widths) == 1 else '2 XY or N×3 XYZ'} array; got {coords.shape}"
        )
    if n_obs is not None and coords.shape[0] != n_obs:
        raise ValueError(f"Spatial coordinate rows {coords.shape[0]} do not match n_obs={n_obs}")
    if coords.dtype.kind not in "iuf":
        raise ValueError("Spatial coordinates must have a real numeric dtype")
    used = coords[:, :2] if mode == DatasetType.SPATIAL_2D else coords
    if not np.isfinite(used).all():
        raise ValueError(
            "Spatial coordinates contain NaN or infinite values in the used dimensions"
        )
    return used


def coordinate_columns(node):
    """AnnData DataFrame encoding: preserve declared column order, no data reads."""
    if not isinstance(node, h5py.Group) or node.attrs.get("encoding-type") != "dataframe":
        raise ValueError("Coordinates must be a dense numeric array or numeric DataFrame")
    names = [
        v.decode() if isinstance(v, bytes) else str(v) for v in node.attrs.get("column-order", [])
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError("Coordinate DataFrame has empty or duplicate columns")
    columns = [node[name] for name in names]
    if any(
        not isinstance(col, h5py.Dataset) or col.ndim != 1 or col.dtype.kind not in "iuf"
        for col in columns
    ):
        raise ValueError(
            "Coordinate DataFrame columns must be one-dimensional real numeric arrays"
        )
    if len({col.shape[0] for col in columns}) != 1:
        raise ValueError("Coordinate DataFrame column lengths differ")
    return columns


def probe_coordinates(path, key, dataset_type):
    """Read metadata and at most 4096 deterministic rows; never load X or AnnData."""
    with h5py.File(path, "r") as handle:
        if not key or f"obsm/{key}" not in handle:
            raise ValueError(f"Spatial key {key!r} is missing in {path}")
        dataset = handle[f"obsm/{key}"]
        columns = None if isinstance(dataset, h5py.Dataset) else coordinate_columns(dataset)
        shape = dataset.shape if columns is None else (columns[0].shape[0], len(columns))
        if len(shape) != 2 or shape[0] == 0 or shape[1] not in (2, 3):
            raise ValueError(
                f"Invalid spatial coordinate shape {shape} in {path}; expected nonempty N×2/N×3"
            )
        if DatasetType(dataset_type) == DatasetType.SPATIAL_3D and shape[1] != 3:
            raise ValueError(
                f"spatial3D requires XYZ coordinates (N×3), got {shape} in {path}; no artificial Z is added"
            )
        index_key = handle["obs"].attrs.get("_index", "_index")
        index = handle["obs"][index_key]
        n_obs = index.shape[0] if isinstance(index, h5py.Dataset) else index["codes"].shape[0]
        if shape[0] != n_obs:
            raise ValueError(
                f"Spatial coordinate rows {shape[0]} do not match n_obs={n_obs} in {path}"
            )
        if columns is None and dataset.dtype.kind not in "iuf":
            raise ValueError(f"Spatial coordinates must have a real numeric dtype in {path}")
        rows = np.linspace(0, shape[0] - 1, min(PROBE_ROWS, shape[0]), dtype=np.int64)
        # Column slicing also avoids treating an unused third coordinate as 2D data.
        width = 2 if DatasetType(dataset_type) == DatasetType.SPATIAL_2D else 3
        values = (
            dataset[rows, :width]
            if columns is None
            else np.column_stack([col[rows] for col in columns[:width]])
        )
        sample = spatial_coordinates(values, dataset_type)
    return {
        "shape": shape,
        "median": np.median(sample, axis=0),
        "span": np.diff(np.quantile(sample, [0.01, 0.99], axis=0), axis=0)[0],
    }


@contextmanager
def reuse_spatial_validation(config):
    """Share probes/diagnostics within one quick-mode run, not between runs."""
    token = _validation_scope.set(getattr(config, "_spatial_validation_cache", {}))
    try:
        yield
    finally:
        _validation_scope.reset(token)


def is_thick_tissue(config):
    if getattr(config, "dataset_type", None) != DatasetType.SPATIAL_3D:
        return False
    layout = getattr(config, "_spatial_layout", None)
    if layout is not None:
        return layout == "volume"
    samples = getattr(config, "sample_h5ad_dict", None)
    return samples is not None and len(samples) == 1


def spatial_layout(config):
    if getattr(config, "dataset_type", None) == DatasetType.SCRNA_SEQ:
        return "nonspatial"
    return "volume" if is_thick_tissue(config) else "sections"


def report_geometry(config):
    """Small report-cache key, independent of display sampling limits."""
    mode = getattr(config, "dataset_type", DatasetType.SPATIAL_2D)
    physical = mode == DatasetType.SPATIAL_3D
    matching = (
        physical and not is_thick_tissue(config) and getattr(config, "n_adjacent_slices", 2) > 0
    )
    return {
        "version": 2,
        "dataset_type": str(mode),
        "layout": spatial_layout(config),
        "spatial_key": getattr(config, "spatial_key", "spatial"),
        "section_axis": 2 if physical else None,
        "section_scale": getattr(config, "cross_slice_z_scale", 1.0) if matching else 1.0,
    }


def validate_spatial_config(config):
    """Resolve section/volume semantics before expensive computation starts."""
    config.dataset_type = DatasetType(config.dataset_type)
    if config.dataset_type == DatasetType.SCRNA_SEQ:
        config._spatial_layout = "nonspatial"
        return
    adjacent = getattr(config, "n_adjacent_slices", 0)
    if isinstance(adjacent, bool) or not isinstance(adjacent, int | np.integer) or adjacent < 0:
        raise ValueError("n_adjacent_slices must be a nonnegative integer")
    samples = config.sample_h5ad_dict
    if not samples:
        raise ValueError("Spatial validation requires the original per-sample H5AD inputs")
    mode = config.dataset_type

    config._spatial_layout = (
        "volume" if mode == DatasetType.SPATIAL_3D and len(samples) == 1 else "sections"
    )
    cache = _validation_scope.get()
    if cache is None:
        cache = getattr(config, "_spatial_validation_cache", {})
    config._spatial_validation_cache = cache
    probes = []
    for name, path in samples.items():
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, config.spatial_key, str(mode))
        if key not in cache:
            cache[key] = probe_coordinates(path, config.spatial_key, mode)
        probes.append(cache[key])

    def warn_once(message, *args):
        key = ("diagnostic", message, args)
        if key not in cache:
            logger.warning(message, *args)
            cache[key] = True

    if mode == DatasetType.SPATIAL_2D:
        if any(p["shape"][1] == 3 for p in probes):
            warn_once(
                "spatial2D received XYZ coordinates: only columns 0 and 1 are used for graphs, neighbors and 2D reports. Use spatial3D for volume distances."
            )
        if getattr(config, "graph_mode", None) == "section_aware_3d":
            raise ValueError("graph_mode='section_aware_3d' requires dataset_type='spatial3D'")
    else:
        if is_thick_tissue(config):
            logger.info(
                "Single spatial3D input: thick tissue; full XYZ neighbors, no cross-section matching, 3D spatial report."
            )
            ignored = {
                "n_adjacent_slices": (0, 2),
                "adjacent_slice_spatial_neighbors": (0, 200),
                "cross_slice_selection_mode": ("ordinal",),
                "cross_slice_max_distance": (None,),
                "cross_slice_marker_score_strategy": ("hierarchical_pool",),
                "cross_slice_z_scale": (1.0,),
            }
            changed = tuple(
                k
                for k, defaults in ignored.items()
                if hasattr(config, k) and getattr(config, k) not in defaults
            )
            if changed:
                warn_once(
                    "Thick tissue ignores cross-section settings: %s. Original XYZ geometry is retained.",
                    ", ".join(changed),
                )
            if np.min(probes[0]["span"]) <= np.max(probes[0]["span"]) * 1e-6:
                warn_once(
                    "Single spatial3D input has a degenerate coordinate extent; verify that XYZ represents a volume. The mode is not changed automatically."
                )
        else:
            if getattr(config, "n_adjacent_slices", 2) == 0:
                warn_once(
                    "Multiple spatial3D sections with n_adjacent_slices=0: cross-section homogeneous matching is disabled, but full XYZ distances remain. An explicitly selected section_aware_3d embedding graph can still connect sections."
                )
            for name, probe in zip(samples, probes, strict=True):
                spans = probe["span"]
                if not (spans[2] < spans[0] and spans[2] < spans[1]):
                    raise ValueError(
                        f"Sample {name!r} ({samples[name]}): invalid XYZ section geometry. "
                        f"Sampled 1st–99th percentile spans are "
                        f"X={spans[0]:.6g}, Y={spans[1]:.6g}, Z={spans[2]:.6g}; "
                        "Z span must be strictly smaller than both X and Y spans. "
                        "Multi-section spatial3D inputs must use columns [X, Y, Z] "
                        "with Z along the section-stacking direction and comparable units. "
                        "Reorient/register the inputs consistently before running gsMap."
                    )
            positions = np.array([p["median"][2] for p in probes])
            differences = np.diff(positions)
            if np.any(np.isclose(positions[1:], positions[:-1], atol=0, rtol=1e-6)):
                warn_once(
                    "Some consecutive sections have indistinguishable median Z positions; check registration and input order.",
                )
            elif not (np.all(differences > 0) or np.all(differences < 0)):
                warn_once(
                    "Section Z positions are not monotonic in input order. Supply anatomical slice order; gsMap does not reorder the input list.",
                )
    if mode == DatasetType.SPATIAL_2D or is_thick_tissue(config):
        if hasattr(config, "n_adjacent_slices"):
            config.n_adjacent_slices = 0
            config.adjacent_slice_spatial_neighbors = 0
            config.fix_cross_slice_homogenous_neighbors = False


def spatial_contract(config):
    """Computation semantics only; report display limits do not invalidate analyses."""
    mode = str(config.dataset_type)
    volume = is_thick_tissue(config)
    contract = {
        "version": 2,
        "dataset_type": mode,
        "layout": spatial_layout(config),
        "spatial_key": getattr(config, "spatial_key", None),
        "samples": list(config.sample_h5ad_dict or {}),
        "dimensions": 3 if mode == "spatial3D" else 2,
    }
    if hasattr(config, "graph_mode"):
        contract["graph"] = {k: getattr(config, k) for k in ("graph_mode", "graph_z_scale")}

    if hasattr(config, "n_adjacent_slices"):
        active = mode == "spatial3D" and not volume and config.n_adjacent_slices > 0
        contract["matching"] = {"n_adjacent_slices": config.n_adjacent_slices if active else 0}
        if active:
            for k in (
                "adjacent_slice_spatial_neighbors",
                "cross_slice_selection_mode",
                "cross_slice_max_distance",
                "cross_slice_z_scale",
                "cross_slice_marker_score_strategy",
            ):
                contract["matching"][k] = getattr(config, k)

    return contract


def guard_spatial_contract(config, *, write=False):
    """Refuse silent mixing of outputs computed with different spatial semantics."""
    path = config.project_dir / "spatial_contract.json"
    expected = spatial_contract(config)
    if path.exists():
        recorded = json.loads(path.read_text())
        for key in expected.keys() & recorded.keys():
            if recorded[key] != expected[key]:
                raise ValueError(
                    f"Spatial computation settings changed ({key}). Use a new project directory; existing embeddings, marker scores and LDSC results will not be overwritten or reused."
                )
        expected = {**recorded, **expected}
    else:
        existing = any(
            p.exists()
            for p in (
                config.find_latent_metadata_path,
                config.latent2gene_metadata_path,
                config.rank_memmap_path,
                config.marker_scores_memmap_path,
                config.model_path,
            )
        )
        if existing:
            raise ValueError(
                "Existing analysis has no spatial computation contract. Use a new project directory to recompute with verified spatial semantics; existing results are preserved."
            )
    if write:
        import os
        import tempfile

        # A killed process must not leave a truncated contract.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".spatial-contract-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(expected, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

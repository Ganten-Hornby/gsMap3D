"""Sequential marker-score row layout and observation alignment metadata."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np

MARKER_SCORE_LAYOUT_VERSION = 2
MARKER_ROW_MAPPING_DTYPE = np.dtype(np.int64)
MARKER_ROW_MAPPING_DIRECTION = "marker_row_to_adata_obs"
MARKER_ROW_LAYOUT_KIND = "sequential_permutation"


def marker_data_path(marker_path: str | Path) -> Path:
    """Resolve a marker memmap base or data path to its ``.dat`` file."""
    path = Path(marker_path)
    return path if path.suffix == ".dat" else path.with_suffix(".dat")


def marker_obs_indices_path(marker_path: str | Path) -> Path:
    """Return the companion marker-row to AnnData-row mapping path."""
    data_path = marker_data_path(marker_path)
    return data_path.with_name(f"{data_path.stem}.obs_indices.dat")


class SequentialMarkerRowWriter:
    """Build an atomic output-row to original-observation permutation.

    Scored cells are appended in the exact order in which marker batches are
    calculated. Cells skipped because of missing annotations or minimum-size
    filtering are appended in original observation order during finalization;
    their marker rows remain zero in the newly created sparse memmap.
    """

    def __init__(self, marker_path: str | Path, n_obs: int) -> None:
        if n_obs < 1:
            raise ValueError("Marker row layout requires at least one observation")
        self.n_obs = int(n_obs)
        self.final_path = marker_obs_indices_path(marker_path)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.final_path.with_name(
            f".{self.final_path.name}.incomplete.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        self.mapping = np.memmap(
            self.temp_path,
            dtype=MARKER_ROW_MAPPING_DTYPE,
            mode="w+",
            shape=(self.n_obs,),
        )
        self.cursor = 0
        self.finalized = False

    def append(self, original_obs_indices: np.ndarray) -> slice:
        """Append original rows and return their contiguous marker-row slice."""
        if self.finalized:
            raise RuntimeError("Marker row layout was already finalized")
        rows = np.asarray(original_obs_indices)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise TypeError("Original observation indices must be a one-dimensional integer array")
        end = self.cursor + rows.size
        if end > self.n_obs:
            raise ValueError(f"Marker row layout exceeds {self.n_obs} observations at row {end}")
        if rows.size and (rows.min() < 0 or rows.max() >= self.n_obs):
            raise IndexError("Original observation indices are outside the AnnData row range")
        start = self.cursor
        self.mapping[start:end] = rows
        self.cursor = end
        return slice(start, end)

    def finalize(self) -> dict[str, object]:
        """Validate the permutation, append skipped rows, and atomically publish it."""
        if self.finalized:
            raise RuntimeError("Marker row layout was already finalized")

        seen = np.zeros(self.n_obs, dtype=bool)
        chunk_rows = 1_000_000
        for start in range(0, self.cursor, chunk_rows):
            end = min(start + chunk_rows, self.cursor)
            rows = np.asarray(self.mapping[start:end])
            if rows.size and (rows.min() < 0 or rows.max() >= self.n_obs):
                raise IndexError("Marker row layout contains an out-of-range observation")
            if np.unique(rows).size != rows.size or np.any(seen[rows]):
                raise ValueError("Marker row layout contains duplicate observations")
            seen[rows] = True

        skipped_rows = np.flatnonzero(~seen)
        if skipped_rows.size:
            end = self.cursor + skipped_rows.size
            if end != self.n_obs:
                raise RuntimeError("Marker row permutation accounting is inconsistent")
            self.mapping[self.cursor : end] = skipped_rows
            self.cursor = end

        if self.cursor != self.n_obs:
            raise RuntimeError(
                f"Marker row permutation has {self.cursor}/{self.n_obs} observations"
            )

        self.mapping.flush()
        self.mapping._mmap.close()
        del self.mapping
        os.replace(self.temp_path, self.final_path)
        self.finalized = True
        return {
            "version": MARKER_SCORE_LAYOUT_VERSION,
            "kind": MARKER_ROW_LAYOUT_KIND,
            "direction": MARKER_ROW_MAPPING_DIRECTION,
            "mapping_file": self.final_path.name,
            "dtype": MARKER_ROW_MAPPING_DTYPE.name,
            "shape": [self.n_obs],
            "validated_permutation": True,
            "scored_rows": self.n_obs - int(skipped_rows.size),
            "unscored_rows": int(skipped_rows.size),
        }

    def abort(self) -> None:
        """Close and remove only this writer's unpublished temporary mapping."""
        mapping = getattr(self, "mapping", None)
        if mapping is not None:
            try:
                mapping._mmap.close()
            except (AttributeError, OSError):
                pass
            del self.mapping
        if not self.finalized:
            self.temp_path.unlink(missing_ok=True)


def open_marker_obs_indices(
    marker_path: str | Path,
    metadata: dict[str, object],
    expected_obs: int,
) -> np.memmap | None:
    """Open a validated v2 row mapping, or return ``None`` for legacy identity layout."""
    layout = metadata.get("row_layout")
    if layout is None:
        return None
    if not isinstance(layout, dict):
        raise ValueError("Marker row-layout metadata must be an object")
    expected_fields = {
        "version": MARKER_SCORE_LAYOUT_VERSION,
        "kind": MARKER_ROW_LAYOUT_KIND,
        "direction": MARKER_ROW_MAPPING_DIRECTION,
        "dtype": MARKER_ROW_MAPPING_DTYPE.name,
        "shape": [expected_obs],
        "validated_permutation": True,
    }
    for key, expected in expected_fields.items():
        if layout.get(key) != expected:
            raise ValueError(
                f"Invalid marker row-layout field {key!r}: "
                f"expected {expected!r}, observed {layout.get(key)!r}"
            )

    mapping_name = layout.get("mapping_file")
    if not isinstance(mapping_name, str) or Path(mapping_name).name != mapping_name:
        raise ValueError("Marker row-layout mapping must be a local companion filename")
    mapping_path = marker_data_path(marker_path).parent / mapping_name
    expected_bytes = expected_obs * MARKER_ROW_MAPPING_DTYPE.itemsize
    if not mapping_path.is_file() or mapping_path.stat().st_size != expected_bytes:
        raise ValueError(f"Marker row mapping is missing or has the wrong size: {mapping_path}")
    return np.memmap(
        mapping_path,
        dtype=MARKER_ROW_MAPPING_DTYPE,
        mode="r",
        shape=(expected_obs,),
    )

"""Bounded rank-window planning and remapping for large cell types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class RankWindow:
    """A maximal contiguous range of focal rows that fits the rank cache."""

    start: int
    end: int
    unique_rows: int


@dataclass(frozen=True)
class RankWindowMapping:
    """Rows and device slots needed to materialize one rank window."""

    unique_rows: np.ndarray
    rows_to_read: np.ndarray
    update_slots: np.ndarray
    local_neighbor_indices: np.ndarray
    retained_rows: int


@njit
def _maximal_rank_window_arrays(
    neighbor_indices: np.ndarray,
    max_cache_rows: int,
    global_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close a window exactly when adding its next focal row exceeds the cap."""
    seen = np.zeros(global_rows, dtype=np.int32)
    newly_seen = np.empty(neighbor_indices.shape[1], dtype=np.int32)
    starts = np.empty(neighbor_indices.shape[0], dtype=np.int64)
    ends = np.empty(neighbor_indices.shape[0], dtype=np.int64)
    unique_counts = np.empty(neighbor_indices.shape[0], dtype=np.int64)

    stamp = 0
    start = 0
    window_count = 0
    while start < neighbor_indices.shape[0]:
        stamp += 1
        unique_count = 0
        end = start
        while end < neighbor_indices.shape[0]:
            row = neighbor_indices[end]
            added_count = 0
            for column in range(row.size):
                value = row[column]
                if seen[value] != stamp:
                    seen[value] = stamp
                    newly_seen[added_count] = value
                    added_count += 1
            if unique_count + added_count > max_cache_rows:
                # The rejected focal row must be reconsidered in the next window.
                for column in range(added_count):
                    seen[newly_seen[column]] = 0
                break
            unique_count += added_count
            end += 1

        if end == start:
            raise ValueError("One focal row exceeds the rank-window capacity")
        starts[window_count] = start
        ends[window_count] = end
        unique_counts[window_count] = unique_count
        window_count += 1
        start = end

    return (
        starts[:window_count],
        ends[:window_count],
        unique_counts[:window_count],
    )


def maximal_rank_windows(
    neighbor_indices: np.ndarray,
    max_cache_rows: int,
    global_rows: int,
) -> list[RankWindow]:
    """Partition an ordered neighbor trace into capacity-maximal windows."""
    neighbors = np.asarray(neighbor_indices)
    if neighbors.ndim != 2 or neighbors.size == 0:
        raise ValueError(f"Neighbor indices must be a non-empty matrix, got {neighbors.shape}")
    if max_cache_rows < 1 or global_rows < 1:
        raise ValueError("Cache and global row counts must be positive")
    if not np.issubdtype(neighbors.dtype, np.integer):
        raise TypeError("Neighbor indices must be integers")
    if neighbors.min() < 0 or neighbors.max() >= global_rows:
        raise IndexError("Neighbor indices are outside the rank matrix")

    starts, ends, unique_counts = _maximal_rank_window_arrays(
        neighbors,
        int(max_cache_rows),
        int(global_rows),
    )
    return [
        RankWindow(int(start), int(end), int(unique_rows))
        for start, end, unique_rows in zip(starts, ends, unique_counts, strict=True)
    ]


def plan_rank_windows(
    neighbor_indices: np.ndarray,
    max_cache_rows: int,
    global_rows: int,
) -> list[RankWindow]:
    """Plan one compact cache or several capacity-maximal windows.

    Homogeneous neighbors are restricted to the focal cell type. Therefore, a
    type with no more focal cells than cache rows is known to fit without the
    global-row stamp array used by the large-trace planner. The exact distinct
    count is still recorded for a right-sized one-window allocation.
    """
    neighbors = np.asarray(neighbor_indices)
    if neighbors.ndim != 2 or neighbors.size == 0:
        raise ValueError(f"Neighbor indices must be a non-empty matrix, got {neighbors.shape}")
    if max_cache_rows < 1 or global_rows < 1:
        raise ValueError("Cache and global row counts must be positive")
    if not np.issubdtype(neighbors.dtype, np.integer):
        raise TypeError("Neighbor indices must be integers")
    if neighbors.min() < 0 or neighbors.max() >= global_rows:
        raise IndexError("Neighbor indices are outside the rank matrix")

    if neighbors.shape[0] <= max_cache_rows:
        unique_rows = int(np.unique(neighbors).size)
        if unique_rows <= max_cache_rows:
            return [RankWindow(0, neighbors.shape[0], unique_rows)]

    return maximal_rank_windows(neighbors, max_cache_rows, global_rows)


def independent_rank_window_mapping(
    neighbor_indices: np.ndarray,
) -> RankWindowMapping:
    """Map one independent window to a compact prefix of the device cache."""
    neighbors = np.asarray(neighbor_indices)
    unique_rows, inverse = np.unique(neighbors.reshape(-1), return_inverse=True)
    update_slots = np.arange(unique_rows.size, dtype=np.int32)
    return RankWindowMapping(
        unique_rows=unique_rows,
        rows_to_read=unique_rows,
        update_slots=update_slots,
        local_neighbor_indices=inverse.reshape(neighbors.shape).astype(np.int32, copy=False),
        retained_rows=0,
    )


class AdjacentRankWindowIndex:
    """Maintain stable device slots for rows shared by adjacent windows."""

    def __init__(self, capacity_rows: int, global_rows: int) -> None:
        if capacity_rows < 1 or global_rows < 1:
            raise ValueError("Cache and global row counts must be positive")
        self.capacity_rows = int(capacity_rows)
        self.slot_of_rank = np.full(global_rows, -1, dtype=np.int32)
        self.free_slots = np.arange(capacity_rows, dtype=np.int32)
        self.free_count = int(capacity_rows)
        self.previous_rows: np.ndarray | None = None

    def map_window(self, neighbor_indices: np.ndarray) -> RankWindowMapping:
        """Retain intersection rows and assign free slots to newly needed rows."""
        neighbors = np.asarray(neighbor_indices)
        unique_rows = np.unique(neighbors.reshape(-1))
        if unique_rows.size > self.capacity_rows:
            raise ValueError(
                f"Window needs {unique_rows.size} rows, cache holds {self.capacity_rows}"
            )
        if unique_rows.size and (unique_rows[0] < 0 or unique_rows[-1] >= self.slot_of_rank.size):
            raise IndexError("Neighbor indices are outside the rank matrix")

        if self.previous_rows is not None:
            leaving_rows = np.setdiff1d(
                self.previous_rows,
                unique_rows,
                assume_unique=True,
            )
            if leaving_rows.size:
                leaving_slots = self.slot_of_rank[leaving_rows]
                end = self.free_count + leaving_slots.size
                self.free_slots[self.free_count : end] = leaving_slots
                self.free_count = end
                self.slot_of_rank[leaving_rows] = -1

        retained_mask = self.slot_of_rank[unique_rows] >= 0
        retained_rows = int(retained_mask.sum())
        rows_to_read = unique_rows[~retained_mask]
        if rows_to_read.size > self.free_count:
            raise RuntimeError("Insufficient free device-cache slots")

        self.free_count -= int(rows_to_read.size)
        update_slots = self.free_slots[
            self.free_count : self.free_count + rows_to_read.size
        ].copy()
        self.slot_of_rank[rows_to_read] = update_slots
        local_neighbors = self.slot_of_rank[neighbors]
        if np.any(local_neighbors < 0):
            raise RuntimeError("Rank-window remapping left an unassigned neighbor")
        self.previous_rows = unique_rows

        return RankWindowMapping(
            unique_rows=unique_rows,
            rows_to_read=rows_to_read,
            update_slots=update_slots,
            local_neighbor_indices=local_neighbors,
            retained_rows=retained_rows,
        )

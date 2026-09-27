"""JAX-accelerated row ordering optimization using jax.lax.scan"""

import logging

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def _morton_order(coords: np.ndarray, *, preserve_axis_scale: bool) -> np.ndarray:
    """Return a stable Morton order for one coordinate matrix."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 1:
        raise ValueError(f"Spatial coordinates must be a non-empty 2D matrix, got {coords.shape}")

    n_rows = coords.shape[0]
    if n_rows == 0:
        return np.empty(0, dtype=np.int32)

    # Morton encoding supports three axes. Two-dimensional inputs receive a
    # constant z axis.
    working = np.zeros((n_rows, 3), dtype=np.float64)
    n_dimensions = min(coords.shape[1], 3)
    common_span = None
    if preserve_axis_scale:
        spans = []
        for dimension in range(n_dimensions):
            finite_values = coords[np.isfinite(coords[:, dimension]), dimension]
            spans.append(float(np.ptp(finite_values)) if finite_values.size else 0.0)
        common_span = max(spans, default=0.0)

    for dimension in range(n_dimensions):
        values = coords[:, dimension]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        lower = values[finite].min()
        upper = values[finite].max()
        span = upper - lower
        denominator = common_span if preserve_axis_scale else span
        if denominator is not None and denominator > 0:
            working[finite, dimension] = (values[finite] - lower) / denominator
        # Put missing values at the centre of the observed range. This keeps
        # the order defined without allowing a NaN-to-integer cast to leak in.
        working[~finite, dimension] = (
            0.5 * span / denominator if denominator is not None and denominator > 0 else 0.0
        )

    integer = np.rint(np.clip(working, 0.0, 1.0) * ((1 << 21) - 1)).astype(np.uint64)

    def spread_21(value: np.ndarray) -> np.ndarray:
        value = value & np.uint64(0x1FFFFF)
        value = (value | (value << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        value = (value | (value << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        value = (value | (value << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
        value = (value | (value << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
        value = (value | (value << np.uint64(2))) & np.uint64(0x1249249249249249)
        return value

    signature = spread_21(integer[:, 0])
    signature |= spread_21(integer[:, 1]) << np.uint64(1)
    signature |= spread_21(integer[:, 2]) << np.uint64(2)
    return np.argsort(signature, kind="stable").astype(np.int32, copy=False)


def optimize_spatial_row_order(
    coords: np.ndarray,
    *,
    slice_ids: np.ndarray | None = None,
    dataset_type: str | None = None,
    z_scale: float = 1.0,
) -> np.ndarray:
    """Choose a spatial row order that matches the dataset geometry.

    Generic calls retain the original independently normalized 2D/3D Morton
    order. Dataset-aware calls distinguish three cases:

    - Spatial2D: order each section independently because neighbors never cross
      sections.
    - Spatial3D: retain all measured XYZ axes, including constant axes; apply
      the configured scale to Z (column 2), without manufacturing geometry.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if dataset_type is None:
        return _morton_order(coords, preserve_axis_scale=False)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Spatial datasets require at least x/y coordinates, got {coords.shape}")

    normalized_type = str(getattr(dataset_type, "value", dataset_type)).casefold()
    if normalized_type not in {"spatial2d", "spatial3d"}:
        raise ValueError(f"Unsupported spatial dataset type: {dataset_type!r}")
    if slice_ids is None:
        raise ValueError(f"{dataset_type} row ordering requires slice IDs")
    slice_ids = np.asarray(slice_ids)
    if slice_ids.ndim != 1 or slice_ids.shape[0] != coords.shape[0]:
        raise ValueError(f"Slice IDs must have shape ({coords.shape[0]},), got {slice_ids.shape}")
    if coords.shape[0] == 0:
        return np.empty(0, dtype=np.int32)

    if normalized_type == "spatial2d":
        sections = []
        for slice_id in np.unique(slice_ids):
            section_rows = np.flatnonzero(slice_ids == slice_id)
            local_order = _morton_order(
                coords[section_rows, :2],
                preserve_axis_scale=False,
            )
            sections.append(section_rows[local_order])
        logger.info(f"Using section-separated 2D Morton order for {len(sections)} slices")
        return np.concatenate(sections).astype(np.int32, copy=False)

    if not np.isfinite(z_scale) or z_scale <= 0:
        raise ValueError("Spatial3D z scale must be positive")
    from gsMap.config.spatial import spatial_coordinates

    physical_coords = spatial_coordinates(coords, "spatial3D").copy()
    physical_coords[:, 2] *= z_scale
    logger.info("Using full XYZ Morton order (Z scale=%s)", z_scale)
    return _morton_order(physical_coords, preserve_axis_scale=True)


# @partial(jit, static_argnames=['k'])
def _optimize_weighted_scan(
    neighbor_indices: jnp.ndarray, neighbor_weights: jnp.ndarray, cell_indices: jnp.ndarray, k: int
) -> jnp.ndarray:
    """
    JIT-compiled weighted row ordering using jax.lax.scan for optimal performance.

    Args:
        neighbor_indices: (n_cells, k) neighbor indices (global)
        neighbor_weights: (n_cells, k) neighbor weights
        cell_indices: (n_cells,) global cell indices
        k: Number of neighbors per cell

    Returns:
        Reordered row indices (local 0..n-1)
    """
    n_cells = len(neighbor_indices)

    # Pre-compute global to local mapping
    # Handle -1 values in neighbor_indices (invalid neighbors)
    valid_mask = neighbor_indices >= 0
    max_global_idx = jnp.max(jnp.where(valid_mask, neighbor_indices, 0))

    # Create inverse mapping from global to local indices
    inverse_map = jnp.full(max_global_idx + 1, -1, dtype=jnp.int32)
    inverse_map = inverse_map.at[cell_indices].set(jnp.arange(n_cells))

    # Convert to local indices, preserving -1 for invalid neighbors
    neighbor_indices_flat = neighbor_indices.ravel()
    # For invalid indices (-1), keep them as -1
    # For valid indices, map them through inverse_map
    local_indices_flat = jnp.where(
        neighbor_indices_flat >= 0,
        inverse_map[jnp.where(neighbor_indices_flat >= 0, neighbor_indices_flat, 0)],
        -1,
    )
    local_neighbor_indices = local_indices_flat.reshape(n_cells, k)

    # Initialize with highest weight cell
    # Only consider valid neighbors when computing max weights
    weights_masked = jnp.where(valid_mask, neighbor_weights, 0)
    max_weights = jnp.max(weights_masked, axis=1)
    start_node = jnp.argmax(max_weights)

    # Initial state for scan
    initial_state = {
        "current": start_node,
        "visited": jnp.zeros(n_cells, dtype=jnp.bool_).at[start_node].set(True),
        "ordered": jnp.full(n_cells, -1, dtype=jnp.int32).at[0].set(start_node),
        "max_weights": max_weights,
        "local_neighbor_indices": local_neighbor_indices,
        "neighbor_weights": neighbor_weights,
    }

    def scan_step(state, t):
        """Single step of the greedy ordering algorithm"""
        current = state["current"]
        visited = state["visited"]

        # Find best unvisited neighbor
        neighbors = state["local_neighbor_indices"][current]
        weights = state["neighbor_weights"][current]

        # Handle -1 values: create a mask for valid neighbors
        is_valid = neighbors != -1
        # For invalid neighbors, use False for is_unvisited to avoid indexing with -1
        is_unvisited = jnp.where(is_valid, ~visited[jnp.where(is_valid, neighbors, 0)], False)
        mask = is_valid & is_unvisited

        neighbor_scores = jnp.where(mask, weights, -jnp.inf)
        best_idx = jnp.argmax(neighbor_scores)

        # If neighbor found, use it; otherwise pick highest weight unvisited
        has_neighbor = neighbor_scores[best_idx] > -jnp.inf
        next_neighbor = neighbors[best_idx]

        # Fallback to highest weight unvisited cell
        unvisited_weights = jnp.where(visited, -jnp.inf, state["max_weights"])
        next_maxweight = jnp.argmax(unvisited_weights)

        next_cell = jnp.where(has_neighbor, next_neighbor, next_maxweight)

        # Update state
        new_state = {
            "current": next_cell,
            "visited": visited.at[next_cell].set(True),
            "ordered": state["ordered"].at[t].set(next_cell),
            "max_weights": state["max_weights"],
            "local_neighbor_indices": state["local_neighbor_indices"],
            "neighbor_weights": state["neighbor_weights"],
        }

        return new_state, None

    # Run scan for n_cells-1 iterations (first cell already placed)
    final_state, _ = jax.lax.scan(scan_step, initial_state, jnp.arange(1, n_cells))

    return final_state["ordered"]


def optimize_row_order_jax(
    neighbor_indices: np.ndarray,
    cell_indices: np.ndarray,
    neighbor_weights: np.ndarray | None,
    device: jax.Device | None = None,
) -> np.ndarray:
    """
    High-performance JAX-based row ordering for cache efficiency.

    Args:
        neighbor_indices: (n_cells, k) neighbor indices (global)
        cell_indices: (n_cells,) global cell indices
        neighbor_weights: (n_cells, k) neighbor weights
        device: JAX device or None (uses default JAX device)

    Returns:
        Reordered row indices (local 0..n-1) as NumPy array
    """
    # Use default device if not provided (should be already configured by configure_jax_platform)
    if device is None:
        device = jax.devices()[0]

    # Skip if on CPU - scanning is extremely slow on CPU
    if device.platform == "cpu":
        n_cells = len(neighbor_indices)
        logger.info(
            f"Skipping JAX-based row ordering optimization on CPU for {n_cells} cells (too slow)."
        )
        return np.arange(n_cells)

    n_cells, k = neighbor_indices.shape
    logger.debug(
        f"Running JAX scan-based weighted ordering for {n_cells} cells on {device.platform}"
    )

    with jax.default_device(device):
        neighbor_indices_jax = jnp.asarray(neighbor_indices)
        neighbor_weights_jax = jnp.asarray(neighbor_weights)
        cell_indices_jax = jnp.asarray(cell_indices)

        # Run optimized weighted ordering
        ordered_jax = _optimize_weighted_scan(
            neighbor_indices_jax, neighbor_weights_jax, cell_indices_jax, k
        )

    return np.array(ordered_jax)

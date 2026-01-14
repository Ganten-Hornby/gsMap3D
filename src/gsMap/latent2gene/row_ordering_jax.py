"""JAX-accelerated row ordering optimization using jax.lax.scan"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Optional
import logging
from functools import partial

logger = logging.getLogger(__name__)


# @partial(jit, static_argnames=['k'])
def _optimize_weighted_scan(
        neighbor_indices: jnp.ndarray,
        neighbor_weights: jnp.ndarray,
        cell_indices: jnp.ndarray,
        k: int
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
        -1
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
        "neighbor_weights": neighbor_weights
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
            "neighbor_weights": state["neighbor_weights"]
        }
        
        return new_state, None

    # Run scan for n_cells-1 iterations (first cell already placed)
    final_state, _ = jax.lax.scan(scan_step, initial_state, jnp.arange(1, n_cells))
    
    return final_state["ordered"]


def optimize_row_order_jax(
        neighbor_indices: np.ndarray,
        cell_indices: np.ndarray,
        neighbor_weights: Optional[np.ndarray],
        device: Optional[jax.Device] = None
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
    if device.platform == 'cpu':
        n_cells = len(neighbor_indices)
        logger.info(f"Skipping JAX-based row ordering optimization on CPU for {n_cells} cells (too slow).")
        return np.arange(n_cells)

    n_cells, k = neighbor_indices.shape
    logger.debug(f"Running JAX scan-based weighted ordering for {n_cells} cells on {device.platform}")
    
    with jax.default_device(device):
        neighbor_indices_jax = jnp.asarray(neighbor_indices)
        neighbor_weights_jax = jnp.asarray(neighbor_weights)
        cell_indices_jax = jnp.asarray(cell_indices)
    
        # Run optimized weighted ordering
        ordered_jax = _optimize_weighted_scan(
            neighbor_indices_jax,
            neighbor_weights_jax,
            cell_indices_jax,
            k
        )
    
    return np.array(ordered_jax)
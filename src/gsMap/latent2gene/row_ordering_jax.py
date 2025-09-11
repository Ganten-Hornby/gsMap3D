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
    max_global_idx = jnp.max(cell_indices)
    inverse_map = jnp.full(max_global_idx + 1, -1, dtype=jnp.int32)
    inverse_map = inverse_map.at[cell_indices].set(jnp.arange(n_cells))

    # Convert to local indices
    local_neighbor_indices = inverse_map[neighbor_indices.ravel()].reshape(n_cells, k)

    # Initialize with highest weight cell
    max_weights = jnp.max(neighbor_weights, axis=1)
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
        
        is_valid = neighbors != -1
        is_unvisited = ~visited[neighbors]
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
        neighbor_weights: Optional[np.ndarray] = None,
        method: Optional[str] = None,
        device: Optional[str] = None
) -> np.ndarray:
    """
    High-performance JAX-based row ordering for cache efficiency.
    
    Args:
        neighbor_indices: (n_cells, k) neighbor indices (global)
        cell_indices: (n_cells,) global cell indices
        neighbor_weights: (n_cells, k) neighbor weights
        method: Ignored, always uses scan-based weighted method
        device: 'gpu', 'cpu', or None (auto-detect)
    
    Returns:
        Reordered row indices (local 0..n-1) as NumPy array
    """
    n_cells, k = neighbor_indices.shape
    
    # Validate inputs
    if neighbor_weights is None:
        # Generate uniform weights if not provided
        neighbor_weights = np.ones_like(neighbor_indices, dtype=np.float32)
        logger.info("No weights provided, using uniform weights")
    
    # Auto-detect device if not specified
    if device is None:
        try:
            _ = jax.devices('gpu')[0]
            device = 'gpu'
            logger.debug(f"Using GPU for {n_cells} cells")
        except:
            device = 'cpu'
            logger.debug(f"Using CPU with JAX for {n_cells} cells")
    
    # Convert to JAX arrays
    neighbor_indices_jax = jnp.asarray(neighbor_indices, dtype=jnp.int32)
    neighbor_weights_jax = jnp.asarray(neighbor_weights, dtype=jnp.float32)
    cell_indices_jax = jnp.asarray(cell_indices, dtype=jnp.int32)
    
    # Run optimized weighted ordering
    logger.debug(f"Running JAX scan-based weighted ordering for {n_cells} cells")
    ordered_jax = _optimize_weighted_scan(
        neighbor_indices_jax,
        neighbor_weights_jax,
        cell_indices_jax,
        k
    ).block_until_ready()
    
    return np.array(ordered_jax)
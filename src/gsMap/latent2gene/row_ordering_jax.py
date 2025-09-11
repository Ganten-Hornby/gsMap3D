"""
JAX-accelerated row ordering optimization for GPU execution
Optimized implementation using jax.lax.scan and sparse BCOO matrices
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental.sparse import BCOO
import numpy as np
from typing import Optional, Tuple
import logging
from functools import partial

logger = logging.getLogger(__name__)


# ==============================================================================
# Core JIT-Compiled Functions
# ==============================================================================

@partial(jit, static_argnames=['k', 'fallback_k'])
def _optimize_weighted_scan(
        neighbor_indices: jnp.ndarray,
        neighbor_weights: jnp.ndarray,
        cell_indices: jnp.ndarray,
        k: int,
        fallback_k: int = 10
) -> jnp.ndarray:
    """
    JIT-compiled core using jax.lax.scan for optimal GPU performance.
    Best for datasets < 500k cells where scan overhead is acceptable.
    """
    n_cells = len(neighbor_indices)

    # Pre-compute global to local index mapping
    max_global_idx = jnp.max(cell_indices)
    inverse_map = jnp.full(max_global_idx + 1, -1, dtype=jnp.int32)
    inverse_map = inverse_map.at[cell_indices].set(jnp.arange(n_cells))

    # Convert neighbor indices to local at once
    local_neighbor_indices = inverse_map[neighbor_indices.ravel()].reshape(n_cells, k)

    # Build sparse adjacency matrices for efficient lookup
    row_indices = jnp.repeat(jnp.arange(n_cells), k)
    valid_mask = local_neighbor_indices.ravel() != -1

    # W[i,j] = weight from i to j, W_T[j,i] = weight from i to j
    W_T = BCOO(
        (local_neighbor_indices.ravel()[valid_mask], row_indices[valid_mask]),
        neighbor_weights.ravel()[valid_mask],
        shape=(n_cells, n_cells)
    )
    W = W_T.T

    # Initialize with highest weight cell
    max_weights = jnp.max(neighbor_weights, axis=1)
    start_node = jnp.argmax(max_weights)

    initial_carry = {
        "current": start_node,
        "visited": jnp.zeros(n_cells, dtype=jnp.bool_).at[start_node].set(True),
        "ordered": jnp.full(n_cells, -1, dtype=jnp.int32).at[0].set(start_node),
        "max_weights": max_weights,
        "W": W,
        "W_T": W_T
    }

    def scan_body(carry, t):
        """Single step: select next cell in ordering"""
        current = carry["current"]
        visited = carry["visited"]

        # Check direct neighbors first (most common case)
        neighbors = local_neighbor_indices[current]
        weights = neighbor_weights[current]

        # Vectorized neighbor scoring
        is_valid = neighbors != -1
        is_unvisited = ~visited[neighbors]
        mask = is_valid & is_unvisited

        neighbor_scores = jnp.where(mask, weights, -jnp.inf)
        best_neighbor_idx = jnp.argmax(neighbor_scores)
        best_neighbor_score = neighbor_scores[best_neighbor_idx]

        # Direct neighbor found
        next_direct = neighbors[best_neighbor_idx]
        found_direct = best_neighbor_score > -jnp.inf

        # Fallback: check connections from recent cells (only if needed)
        # Use conditional to avoid computation when direct neighbor exists
        def compute_fallback():
            start_idx = jnp.maximum(0, t - fallback_k)
            recent_indices = jax.lax.dynamic_slice(
                carry["ordered"], (start_idx,), (fallback_k,)
            )
            recent_valid = recent_indices >= 0
            recent_indices = jnp.where(recent_valid, recent_indices, 0)

            # Efficient sparse matrix operations
            forward = carry["W"][recent_indices, :].sum(axis=0).todense().ravel()
            reverse = carry["W_T"][recent_indices, :].sum(axis=0).todense().ravel()
            scores = forward + reverse

            scores = jnp.where(visited, -jnp.inf, scores)
            return jnp.argmax(scores), jnp.max(scores) > 0

        # Only compute fallback if no direct neighbor
        next_fallback, found_fallback = jax.lax.cond(
            found_direct,
            lambda: (0, False),  # Dummy values, won't be used
            compute_fallback
        )

        # Final fallback: highest unvisited weight
        unvisited_weights = jnp.where(visited, -jnp.inf, carry["max_weights"])
        next_maxweight = jnp.argmax(unvisited_weights)

        # Select next cell
        next_cell = jax.lax.select(
            found_direct,
            next_direct,
            jax.lax.select(found_fallback, next_fallback, next_maxweight)
        )

        # Update state
        new_carry = {
            "current": next_cell,
            "visited": visited.at[next_cell].set(True),
            "ordered": carry["ordered"].at[t].set(next_cell),
            "max_weights": carry["max_weights"],
            "W": carry["W"],
            "W_T": carry["W_T"]
        }

        return new_carry, None

    final_carry, _ = jax.lax.scan(scan_body, initial_carry, jnp.arange(1, n_cells))
    return final_carry["ordered"]


@partial(jit, static_argnames=['k'])
def _optimize_weighted_loop(
        neighbor_indices: jnp.ndarray,
        neighbor_weights: jnp.ndarray,
        cell_indices: jnp.ndarray,
        k: int
) -> jnp.ndarray:
    """
    JIT-compiled core using simple loop for very large datasets.
    Avoids scan overhead and unnecessary fallback computations.
    """
    n_cells = len(neighbor_indices)

    # Pre-compute global to local mapping
    max_global_idx = jnp.max(cell_indices)
    inverse_map = jnp.full(max_global_idx + 1, -1, dtype=jnp.int32)
    inverse_map = inverse_map.at[cell_indices].set(jnp.arange(n_cells))

    # Convert to local indices
    local_neighbor_indices = inverse_map[neighbor_indices.ravel()].reshape(n_cells, k)

    # Initialize
    max_weights = jnp.max(neighbor_weights, axis=1)
    current = jnp.argmax(max_weights)

    visited = jnp.zeros(n_cells, dtype=jnp.bool_)
    ordered = jnp.zeros(n_cells, dtype=jnp.int32)

    # Simple greedy loop without fallback computation
    for t in range(n_cells):
        ordered = ordered.at[t].set(current)
        visited = visited.at[current].set(True)

        if t < n_cells - 1:
            # Find best unvisited neighbor
            neighbors = local_neighbor_indices[current]
            weights = neighbor_weights[current]

            is_valid = neighbors != -1
            is_unvisited = ~visited[neighbors]
            mask = is_valid & is_unvisited

            neighbor_scores = jnp.where(mask, weights, -jnp.inf)
            best_idx = jnp.argmax(neighbor_scores)

            # If neighbor found, use it; otherwise pick highest weight unvisited
            has_neighbor = neighbor_scores[best_idx] > -jnp.inf
            next_neighbor = neighbors[best_idx]

            # Fallback to highest weight
            unvisited_weights = jnp.where(visited, -jnp.inf, max_weights)
            next_maxweight = jnp.argmax(unvisited_weights)

            current = jnp.where(has_neighbor, next_neighbor, next_maxweight)

    return ordered


@partial(jit, static_argnames=['chunk_size'])
def _process_chunk(
        chunk_neighbors: jnp.ndarray,
        chunk_weights: jnp.ndarray,
        chunk_cells: jnp.ndarray,
        chunk_size: int
) -> jnp.ndarray:
    """Process a single chunk for very large datasets"""
    k = chunk_neighbors.shape[1]

    # Use simple loop for chunks
    return _optimize_weighted_loop(
        chunk_neighbors,
        chunk_weights,
        chunk_cells,
        k
    )


# ==============================================================================
# Main User-Facing Function
# ==============================================================================

def optimize_row_order_jax(
        neighbor_indices: np.ndarray,
        cell_indices: np.ndarray,
        neighbor_weights: Optional[np.ndarray] = None,
        method: Optional[str] = None,
        fallback_k: int = 10,
        chunk_threshold: int = 500000,
        device: Optional[str] = None
) -> np.ndarray:
    """
    High-performance JAX-based row ordering with automatic optimization selection.

    Args:
        neighbor_indices: (n_cells, k) neighbor indices (global)
        cell_indices: (n_cells,) global cell indices
        neighbor_weights: (n_cells, k) neighbor weights (required for weighted method)
        method: 'scan' (jax.lax.scan), 'loop' (simple loop), 'chunked', or None (auto)
        fallback_k: Number of recent cells for fallback scoring (scan method only)
        chunk_threshold: Threshold for automatic chunking (default 500k)
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

    # Auto-detect device
    if device is None:
        try:
            _ = jax.devices('gpu')[0]
            device = 'gpu'
            logger.info(f"Using GPU for {n_cells} cells")
        except:
            device = 'cpu'
            logger.info(f"Using CPU with JAX for {n_cells} cells")

    # Auto-select method based on dataset size
    if method is None:
        if n_cells > chunk_threshold:
            method = 'chunked'
            logger.info(f"Auto-selected chunked method for {n_cells} cells")
        elif n_cells > 100000:
            method = 'loop'
            logger.info(f"Auto-selected loop method for {n_cells} cells")
        else:
            method = 'scan'
            logger.info(f"Auto-selected scan method for {n_cells} cells")

    # Handle different methods
    if method == 'chunked' and n_cells > chunk_threshold:
        # Process in chunks for very large datasets
        chunk_size = min(100000, chunk_threshold // 5)
        n_chunks = (n_cells + chunk_size - 1) // chunk_size

        logger.info(f"Processing {n_cells} cells in {n_chunks} chunks of ~{chunk_size}")

        all_ordered = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_cells)

            chunk_neighbors = neighbor_indices[start:end]
            chunk_weights = neighbor_weights[start:end]
            chunk_cells = cell_indices[start:end]

            # Convert chunk to JAX
            chunk_neighbors_jax = jnp.asarray(chunk_neighbors, dtype=jnp.int32)
            chunk_weights_jax = jnp.asarray(chunk_weights, dtype=jnp.float32)
            chunk_cells_jax = jnp.asarray(chunk_cells, dtype=jnp.int32)

            # Process chunk
            chunk_ordered = _process_chunk(
                chunk_neighbors_jax,
                chunk_weights_jax,
                chunk_cells_jax,
                end - start
            ).block_until_ready()

            # Adjust indices to global
            all_ordered.extend([int(idx) + start for idx in chunk_ordered])

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{n_chunks} chunks")

        return np.array(all_ordered)

    else:
        # Convert to JAX arrays
        neighbor_indices_jax = jnp.asarray(neighbor_indices, dtype=jnp.int32)
        neighbor_weights_jax = jnp.asarray(neighbor_weights, dtype=jnp.float32)
        cell_indices_jax = jnp.asarray(cell_indices, dtype=jnp.int32)

        # Select implementation based on method
        if method == 'loop' or (method == 'scan' and n_cells > 100000):
            if n_cells > 100000:
                logger.info(f"Using loop method to avoid scan overhead for {n_cells} cells")

            ordered_jax = _optimize_weighted_loop(
                neighbor_indices_jax,
                neighbor_weights_jax,
                cell_indices_jax,
                k
            ).block_until_ready()

        else:  # method == 'scan'
            logger.info(f"Using scan method with fallback_k={fallback_k}")

            ordered_jax = _optimize_weighted_scan(
                neighbor_indices_jax,
                neighbor_weights_jax,
                cell_indices_jax,
                k,
                fallback_k
            ).block_until_ready()

        return np.array(ordered_jax)


# ==============================================================================
# Additional Optimized Variants
# ==============================================================================

@partial(jit, static_argnames=['batch_size'])
def compute_similarity_matrix_sparse(
        neighbor_indices: jnp.ndarray,
        batch_size: int = 1000
) -> BCOO:
    """
    Compute sparse similarity matrix for Jaccard-based ordering.
    Uses batched processing to handle large matrices efficiently.
    """
    n_cells, k = neighbor_indices.shape

    # Build sparse binary matrix
    row_indices = jnp.repeat(jnp.arange(n_cells), k)
    col_indices = neighbor_indices.ravel()
    valid_mask = col_indices >= 0

    # Create sparse neighbor matrix
    neighbor_matrix = BCOO(
        (row_indices[valid_mask], col_indices[valid_mask]),
        jnp.ones(jnp.sum(valid_mask), dtype=jnp.float32),
        shape=(n_cells, jnp.max(col_indices) + 1)
    )

    # Compute intersection via matrix multiplication
    intersection = neighbor_matrix @ neighbor_matrix.T

    # Compute set sizes
    set_sizes = neighbor_matrix.sum(axis=1).todense()

    # Build Jaccard similarity sparse matrix
    # Only keeping non-zero similarities to save memory
    intersection_data = intersection.data
    intersection_indices = intersection.indices

    # Compute union for non-zero intersections
    i_idx = intersection_indices[:, 0]
    j_idx = intersection_indices[:, 1]
    union_data = set_sizes[i_idx] + set_sizes[j_idx] - intersection_data

    # Jaccard similarity
    jaccard_data = jnp.where(union_data > 0, intersection_data / union_data, 0.0)

    # Filter out zeros and diagonal
    keep_mask = (jaccard_data > 0) & (i_idx != j_idx)

    jaccard_matrix = BCOO(
        (intersection_indices[keep_mask]),
        jaccard_data[keep_mask],
        shape=(n_cells, n_cells)
    )

    return jaccard_matrix


@jit
def greedy_order_from_similarity(similarity_matrix: BCOO, n_cells: int) -> jnp.ndarray:
    """
    Greedy ordering using precomputed sparse similarity matrix.
    More memory efficient than dense matrix version.
    """
    visited = jnp.zeros(n_cells, dtype=jnp.bool_)
    ordered = jnp.zeros(n_cells, dtype=jnp.int32)

    # Start with random cell
    current = 0

    for t in range(n_cells):
        ordered = ordered.at[t].set(current)
        visited = visited.at[current].set(True)

        if t < n_cells - 1:
            # Get similarities from current cell (sparse row)
            current_row = similarity_matrix[current].todense()

            # Mask visited cells
            scores = jnp.where(visited, -1.0, current_row)

            # Find most similar unvisited
            next_cell = jnp.argmax(scores)

            # If no similar cells, pick any unvisited
            has_similar = scores[next_cell] > 0
            any_unvisited = jnp.argmax(~visited)

            current = jnp.where(has_similar, next_cell, any_unvisited)

    return ordered


def optimize_row_order_similarity(
        neighbor_indices: np.ndarray,
        cell_indices: np.ndarray,
        batch_size: int = 1000
) -> np.ndarray:
    """
    Alternative ordering based on Jaccard similarity.
    Good for datasets where weights are not available.
    """
    n_cells = len(cell_indices)

    # Convert global to local indices
    global_to_local = {g: l for l, g in enumerate(cell_indices)}
    local_neighbors = np.full_like(neighbor_indices, -1)

    for i in range(n_cells):
        for j in range(neighbor_indices.shape[1]):
            global_idx = neighbor_indices[i, j]
            if global_idx in global_to_local:
                local_neighbors[i, j] = global_to_local[global_idx]

    # Convert to JAX
    local_neighbors_jax = jnp.asarray(local_neighbors, dtype=jnp.int32)

    # Compute similarity matrix
    logger.info(f"Computing similarity matrix for {n_cells} cells")
    similarity_matrix = compute_similarity_matrix_sparse(
        local_neighbors_jax,
        batch_size
    )

    # Greedy ordering
    logger.info("Computing greedy ordering from similarity matrix")
    ordered_jax = greedy_order_from_similarity(similarity_matrix, n_cells)

    return np.array(ordered_jax)


# ==============================================================================
# Utility Functions
# ==============================================================================

def benchmark_methods(
        neighbor_indices: np.ndarray,
        cell_indices: np.ndarray,
        neighbor_weights: np.ndarray
) -> dict:
    """
    Benchmark different ordering methods for comparison.
    """
    import time

    n_cells = len(cell_indices)
    results = {}

    # Test scan method (if not too large)
    if n_cells <= 100000:
        start = time.time()
        _ = optimize_row_order_jax(
            neighbor_indices, cell_indices, neighbor_weights,
            method='scan'
        )
        results['scan'] = time.time() - start
        logger.info(f"Scan method: {results['scan']:.3f}s")

    # Test loop method
    start = time.time()
    _ = optimize_row_order_jax(
        neighbor_indices, cell_indices, neighbor_weights,
        method='loop'
    )
    results['loop'] = time.time() - start
    logger.info(f"Loop method: {results['loop']:.3f}s")

    # Test chunked method (if large enough)
    if n_cells > 10000:
        start = time.time()
        _ = optimize_row_order_jax(
            neighbor_indices, cell_indices, neighbor_weights,
            method='chunked', chunk_threshold=10000
        )
        results['chunked'] = time.time() - start
        logger.info(f"Chunked method: {results['chunked']:.3f}s")

    return results
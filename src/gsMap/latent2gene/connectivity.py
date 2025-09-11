"""
Connectivity matrix building for homogeneous spot identification
Implements the spatial → anchor → homogeneous neighbor finding algorithm
"""

import logging
from typing import Optional, Tuple, Union
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from rich.progress import track
import scanpy as sc
import anndata as ad

from gsMap.config import LatentToGeneConfig
from gsMap.config.dataclasses import MarkerScoreCrossSliceStrategy, DatasetType

logger = logging.getLogger(__name__)

# Configure JAX
jax.config.update("jax_enable_x64", False)  # Use float32 for speed


def find_spatial_neighbors_with_slices(
    coords: np.ndarray,
    slice_ids: Optional[np.ndarray] = None,
    cell_mask: Optional[np.ndarray] = None,
    k_central: int = 101,
    k_adjacent: int = 50,
    n_adjacent_slices: int = 1
) -> np.ndarray:
    """
    Find spatial neighbors with slice-aware search for 3D data.
    
    For 2D data (slice_ids is None), performs standard KNN.
    For 3D data, implements slice-aware neighbor search:
    - Finds k_central neighbors on the same slice
    - Finds k_adjacent neighbors on each of n_adjacent_slices above and below
    
    Assumes k_central > k_adjacent and slices have sufficient points.
    
    Args:
        coords: Spatial coordinates (n_cells, 2) - only x,y coordinates
        slice_ids: Slice/z-coordinate indices (n_cells,) - sequential integers
        cell_mask: Boolean mask for cells to process
        k_central: Number of neighbors to find on the central slice
        k_adjacent: Number of neighbors to find on each adjacent slice
        n_adjacent_slices: Number of slices to search above and below
    
    Returns:
        spatial_neighbors: Array of neighbor indices (n_masked, k_central + 2*n_adjacent_slices*k_adjacent)
    """
    n_cells = len(coords)
    if cell_mask is None:
        cell_mask = np.ones(n_cells, dtype=bool)
    
    cell_indices = np.where(cell_mask)[0]
    n_masked = len(cell_indices)
    
    # If no slice_ids provided, perform standard 2D KNN
    if slice_ids is None:
        logger.info(f"No slice IDs provided, performing standard 2D KNN with k={k_central}")
        nbrs = NearestNeighbors(
            n_neighbors=min(k_central, n_masked),
            metric='euclidean',
            algorithm='kd_tree'
        )
        nbrs.fit(coords[cell_mask])
        _, spatial_neighbors = nbrs.kneighbors(coords[cell_mask])
        return cell_indices[spatial_neighbors]
    
    # Slice-aware neighbor search with fixed-size arrays
    logger.info(f"Performing slice-aware neighbor search: k_central={k_central}, "
                f"k_adjacent={k_adjacent}, n_adjacent_slices={n_adjacent_slices}")
    
    masked_slice_ids = slice_ids[cell_mask]
    masked_coords = coords[cell_mask]
    
    # Pre-allocate output with fixed size, initialized with -1 (invalid)
    total_k = k_central + 2 * n_adjacent_slices * k_adjacent
    # Always use int32 for spatial neighbors since they contain global cell indices
    # which can easily exceed int16 range in large spatial datasets
    spatial_neighbors = np.full((n_masked, total_k), -1, dtype=np.int32)
    
    # Get unique slices and create mapping
    unique_slices = np.unique(masked_slice_ids)
    slice_to_indices = {s: np.where(masked_slice_ids == s)[0] for s in unique_slices}
    
    # Pre-compute KNN for each slice
    slice_knn_models = {}
    for slice_id, slice_local_indices in slice_to_indices.items():
        if len(slice_local_indices) >= max(k_central, k_adjacent):
            slice_coords = masked_coords[slice_local_indices]
            nbrs = NearestNeighbors(
                n_neighbors=max(k_central, k_adjacent),
                metric='euclidean',
                algorithm='kd_tree'
            )
            nbrs.fit(slice_coords)
            slice_knn_models[slice_id] = (nbrs, slice_local_indices)
    
    # Batch process all cells
    for slice_id in unique_slices:
        if slice_id in slice_knn_models:

            # Get all cells on this slice
            cells_on_slice = slice_to_indices[slice_id]
            query_coords = masked_coords[cells_on_slice]

            # Central slice neighbors (fixed k_central)
            nbrs, slice_local_indices = slice_knn_models[slice_id]
            _, local_neighbors = nbrs.kneighbors(query_coords, n_neighbors=k_central)
            global_neighbors = cell_indices[slice_local_indices[local_neighbors]]
            spatial_neighbors[cells_on_slice, :k_central] = global_neighbors

            # Adjacent slices (fixed k_adjacent for each)
            col_offset = k_central
            for offset in range(1, n_adjacent_slices + 1):
                for direction in [1, -1]:
                    adjacent_slice = slice_id + direction * offset
                    if adjacent_slice in slice_knn_models:
                        nbrs_adj, adj_local_indices = slice_knn_models[adjacent_slice]
                        _, adj_local_neighbors = nbrs_adj.kneighbors(query_coords, n_neighbors=k_adjacent)
                        adj_global_neighbors = cell_indices[adj_local_indices[adj_local_neighbors]]
                        spatial_neighbors[cells_on_slice, col_offset:col_offset+k_adjacent] = adj_global_neighbors
                    # If adjacent slice doesn't exist, already filled with -1
                    col_offset += k_adjacent

    # Handle edge cases: cells on slices with too few points
    for slice_id, slice_local_indices in slice_to_indices.items():
        if slice_id not in slice_knn_models and len(slice_local_indices) > 0:
            # These cells have too few neighbors on their slice
            cells_on_slice = slice_local_indices
            n_cells_on_slice = len(slice_local_indices)

            if n_cells_on_slice > 1:
                # Few cells on slice - use all available as neighbors, rest already -1
                slice_coords = masked_coords[slice_local_indices]
                nbrs = NearestNeighbors(
                    n_neighbors=min(n_cells_on_slice, k_central),
                    metric='euclidean',
                    algorithm='kd_tree'
                )
                nbrs.fit(slice_coords)
                _, local_neighbors = nbrs.kneighbors(slice_coords)
                global_neighbors = cell_indices[slice_local_indices[local_neighbors]]
                
                # Fill what we can, rest remains -1
                actual_k = min(n_cells_on_slice, k_central)
                spatial_neighbors[cells_on_slice, :actual_k] = global_neighbors
            # If only 1 cell on slice, it remains all -1

    return spatial_neighbors


@partial(jit, static_argnums=(5, 6))
def _find_anchors_and_homogeneous_batch_jit(
    emb_gcn_batch_norm: jnp.ndarray,      # (batch_size, d1) - pre-normalized
    emb_indv_batch_norm: jnp.ndarray,      # (batch_size, d2) - pre-normalized
    spatial_neighbors: jnp.ndarray,   # (batch_size, k1)
    all_emb_gcn_norm: jnp.ndarray,         # (n_all, d1) - pre-normalized
    all_emb_indv_norm: jnp.ndarray,        # (n_all, d2) - pre-normalized
    num_homogeneous: int,
    similarity_threshold: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled function to find anchors and homogeneous neighbors.
    Processes a batch of cells to manage GPU memory.
    Expects pre-normalized embeddings for efficiency.
    
    Args:
        similarity_threshold: Minimum similarity threshold. Weights for similarities 
                            below this threshold will be set to 0 after softmax.
    """
    batch_size = emb_gcn_batch_norm.shape[0]
    
    # Step 1: Extract spatial neighbors' embeddings (already normalized)
    # Use a safe index (0) for invalid neighbors, will mask them later
    safe_neighbors = jnp.where(spatial_neighbors >= 0, spatial_neighbors, 0)
    spatial_emb_gcn_norm = all_emb_gcn_norm[safe_neighbors]  # (batch_size, k1, d1)
    spatial_emb_indv_norm = all_emb_indv_norm[safe_neighbors]  # (batch_size, k1, d2)
    
    # Step 2: Compute both GCN and individual similarities for spatial neighbors
    # Compute GCN similarities (embeddings are already normalized)
    anchor_sims = jnp.einsum('bd,bkd->bk', emb_gcn_batch_norm, spatial_emb_gcn_norm)
    
    # Compute individual/cell similarities (embeddings are already normalized)
    cell_sims = jnp.einsum('bd,bkd->bk', emb_indv_batch_norm, spatial_emb_indv_norm)
    
    # Apply threshold to both similarities (set to 0 if below threshold)
    anchor_sims_thresholded = jnp.where(anchor_sims >= similarity_threshold, anchor_sims, 0.0)
    cell_sims_thresholded = jnp.where(cell_sims >= similarity_threshold, cell_sims, 0.0)
    
    # Multiply the thresholded similarities
    combined_sims = anchor_sims_thresholded * cell_sims_thresholded
    
    # Mask out invalid neighbors by setting their similarities to -0
    combined_sims = jnp.where(spatial_neighbors >= 0, combined_sims, 0)
    
    # Select top homogeneous neighbors based on combined similarity
    top_homo_idx = jnp.argsort(-combined_sims, axis=1)[:, :num_homogeneous]
    batch_idx = jnp.arange(batch_size)[:, None]
    homogeneous_neighbors = spatial_neighbors[batch_idx, top_homo_idx]  # (batch_size, num_homogeneous)
    homogeneous_weights = combined_sims[batch_idx, top_homo_idx]
    
    return homogeneous_neighbors, homogeneous_weights


@partial(jit, static_argnums=(5, 6, 7, 8, 9))
def _find_anchors_and_homogeneous_batch_3d_jit(
    emb_gcn_batch_norm: jnp.ndarray,      # (batch_size, d1) - pre-normalized
    emb_indv_batch_norm: jnp.ndarray,      # (batch_size, d2) - pre-normalized
    spatial_neighbors: jnp.ndarray,        # (batch_size, k_total)
    all_emb_gcn_norm: jnp.ndarray,         # (n_all, d1) - pre-normalized
    all_emb_indv_norm: jnp.ndarray,        # (n_all, d2) - pre-normalized
    num_homogeneous_per_slice: int,
    k_central: int,
    k_adjacent: int,
    n_adjacent_slices: int,
    similarity_threshold: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled function for 3D spatial data with per-slice homogeneous neighbor constraints.
    Ensures each slice (central and adjacent) contributes exactly num_homogeneous_per_slice neighbors.
    
    Args:
        spatial_neighbors: All spatial neighbors with structure:
                          [central_slice_neighbors | adjacent_slice_1 | adjacent_slice_2 | ...]
        num_homogeneous_per_slice: Number of homogeneous neighbors to select from each slice
        k_central: Number of neighbors in central slice
        k_adjacent: Number of neighbors per adjacent slice
        n_adjacent_slices: Number of adjacent slices above and below
        similarity_threshold: Minimum similarity threshold
    
    Returns:
        homogeneous_neighbors: (batch_size, total_homogeneous) where 
                               total_homogeneous = num_homogeneous_per_slice * (1 + 2*n_adjacent_slices)
        homogeneous_weights: Corresponding weights
    """
    batch_size = emb_gcn_batch_norm.shape[0]
    n_slices = 1 + 2 * n_adjacent_slices
    total_homogeneous = num_homogeneous_per_slice * n_slices
    
    # Process central slice
    central_neighbors = spatial_neighbors[:, :k_central]
    safe_neighbors = jnp.where(central_neighbors >= 0, central_neighbors, 0)
    central_emb_gcn = all_emb_gcn_norm[safe_neighbors]
    central_emb_indv = all_emb_indv_norm[safe_neighbors]
    
    # Compute similarities for central slice
    anchor_sims = jnp.einsum('bd,bkd->bk', emb_gcn_batch_norm, central_emb_gcn)
    cell_sims = jnp.einsum('bd,bkd->bk', emb_indv_batch_norm, central_emb_indv)
    anchor_sims = jnp.where(anchor_sims >= similarity_threshold, anchor_sims, 0.0)
    cell_sims = jnp.where(cell_sims >= similarity_threshold, cell_sims, 0.0)
    combined_sims = anchor_sims * cell_sims
    combined_sims = jnp.where(central_neighbors >= 0, combined_sims, -jnp.inf)
    
    # Select top k from central slice
    top_k_idx = jnp.argsort(-combined_sims, axis=1)[:, :num_homogeneous_per_slice]
    batch_idx = jnp.arange(batch_size)[:, None]
    central_homo_neighbors = central_neighbors[batch_idx, top_k_idx]
    central_homo_weights = combined_sims[batch_idx, top_k_idx]
    central_homo_weights = jnp.where(central_homo_weights == -jnp.inf, 0.0, central_homo_weights)
    
    # Process adjacent slices if n_adjacent_slices > 0
    # Generalized implementation using matrix operations for any n_adjacent_slices
    if n_adjacent_slices > 0:
        # Reshape adjacent neighbors into (batch, 2*n_adjacent_slices, k_adjacent)
        offset = k_central
        total_adjacent = 2 * n_adjacent_slices

        # Extract all adjacent slice neighbors at once
        adjacent_neighbors = spatial_neighbors[:, offset:offset + total_adjacent * k_adjacent]
        # Reshape to (batch_size, total_adjacent, k_adjacent)
        adjacent_neighbors = adjacent_neighbors.reshape(batch_size, total_adjacent, k_adjacent)

        # Handle invalid neighbors
        safe_neighbors = jnp.where(adjacent_neighbors >= 0, adjacent_neighbors, 0)

        # Get embeddings for all adjacent slices: (batch, total_adjacent, k_adjacent, d)
        adj_emb_gcn = all_emb_gcn_norm[safe_neighbors]
        adj_emb_indv = all_emb_indv_norm[safe_neighbors]

        # Compute similarities for all adjacent slices at once using einsum
        # adj_emb_gcn: (batch, total_adjacent, k_adjacent, d)
        # emb_gcn_batch_norm: (batch, d)
        # Result: (batch, total_adjacent, k_adjacent)
        anchor_sims = jnp.einsum('bd,bskd->bsk', emb_gcn_batch_norm, adj_emb_gcn)
        cell_sims = jnp.einsum('bd,bskd->bsk', emb_indv_batch_norm, adj_emb_indv)

        # Apply threshold
        anchor_sims = jnp.where(anchor_sims >= similarity_threshold, anchor_sims, 0.0)
        cell_sims = jnp.where(cell_sims >= similarity_threshold, cell_sims, 0.0)
        combined_sims = anchor_sims * cell_sims

        # Mask invalid neighbors
        combined_sims = jnp.where(adjacent_neighbors >= 0, combined_sims, -jnp.inf)

        # Vectorized selection of top k neighbors for all slices at once
        # Sort similarities for each slice: (batch, total_adjacent, k_adjacent)
        sorted_indices = jnp.argsort(-combined_sims, axis=2)

        # Select top k indices for each slice: (batch, total_adjacent, num_homogeneous_per_slice)
        top_k_indices = sorted_indices[:, :, :num_homogeneous_per_slice]

        # Create batch and slice indices for gathering
        batch_idx = jnp.arange(batch_size)[:, None, None]  # (batch, 1, 1)
        slice_idx = jnp.arange(total_adjacent)[None, :, None]  # (1, total_adjacent, 1)

        # Gather the selected neighbors and weights
        # Shape: (batch, total_adjacent, num_homogeneous_per_slice)
        selected_neighbors = adjacent_neighbors[batch_idx, slice_idx, top_k_indices]
        selected_weights = combined_sims[batch_idx, slice_idx, top_k_indices]

        # Replace -inf weights with 0
        selected_weights = jnp.where(selected_weights == -jnp.inf, 0.0, selected_weights)

        # Reshape to concatenate with central slice results
        # From (batch, total_adjacent, num_homogeneous_per_slice) to (batch, total_adjacent * num_homogeneous_per_slice)
        adj_homo_neighbors = selected_neighbors.reshape(batch_size, total_adjacent * num_homogeneous_per_slice)
        adj_homo_weights = selected_weights.reshape(batch_size, total_adjacent * num_homogeneous_per_slice)

        # Concatenate central and adjacent results
        homogeneous_neighbors = jnp.concatenate([central_homo_neighbors, adj_homo_neighbors], axis=1)
        homogeneous_weights = jnp.concatenate([central_homo_weights, adj_homo_weights], axis=1)
    else:
        # No adjacent slices, just use central results
        homogeneous_neighbors = central_homo_neighbors
        homogeneous_weights = central_homo_weights
    
    # # Normalize weights using mean pooling across slices
    # # Each slice gets equal weight (1/n_slices)
    # weights_reshaped = homogeneous_weights.reshape(batch_size, n_slices, num_homogeneous_per_slice)
    #
    # # Normalize within each slice
    # slice_sums = weights_reshaped.sum(axis=2, keepdims=True)
    # slice_sums = jnp.where(slice_sums > 0, slice_sums, 1.0)
    # weights_normalized = weights_reshaped / slice_sums
    #
    # # Apply mean pooling weight (1/n_slices per slice)
    # weights_normalized = weights_normalized / n_slices
    #
    # # Reshape back
    # homogeneous_weights = weights_normalized.reshape(batch_size, total_homogeneous)
    
    return homogeneous_neighbors, homogeneous_weights


def build_scrna_connectivity(
    emb_cell: np.ndarray,
    cell_mask: Optional[np.ndarray] = None,
    n_neighbors: int = 21,
    metric: str = 'euclidean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build connectivity for scRNA-seq data using KNN on cell embeddings.
    
    Args:
        emb_cell: Cell embeddings (n_cells, d)
        cell_mask: Boolean mask for cells to process
        n_neighbors: Number of nearest neighbors
        metric: Distance metric for KNN
    
    Returns:
        neighbor_indices: (n_masked, n_neighbors) array of neighbor indices
        neighbor_weights: (n_masked, n_neighbors) array of weights from KNN graph
    """
    n_cells = len(emb_cell)
    if cell_mask is None:
        cell_mask = np.ones(n_cells, dtype=bool)
    
    cell_indices = np.where(cell_mask)[0]
    n_masked = len(cell_indices)
    
    logger.info(f"Building scRNA-seq connectivity using KNN with k={n_neighbors}")
    
    # Create temporary AnnData for using scanpy's neighbors function
    adata_temp = ad.AnnData(X=emb_cell[cell_mask])
    adata_temp.obsm['X_emb'] = emb_cell[cell_mask]
    
    # Compute neighbors using scanpy
    sc.pp.neighbors(
        adata_temp,
        n_neighbors=n_neighbors,
        use_rep='X_emb',
        metric=metric,
        method='umap'
    )
    
    # Extract connectivity matrix
    connectivities = adata_temp.obsp['connectivities'].tocsr()
    
    # Convert to dense format for consistency with spatial methods
    # Check the actual max cell index value to determine dtype
    max_cell_idx = cell_indices.max() if len(cell_indices) > 0 else 0
    idx_dtype = np.int16 if max_cell_idx < 32768 else np.int32
    neighbor_indices = np.zeros((n_masked, n_neighbors), dtype=idx_dtype)
    neighbor_weights = np.zeros((n_masked, n_neighbors), dtype=np.float16)
    
    for i in range(n_masked):
        row = connectivities.getrow(i)
        neighbors = row.indices
        weights = row.data
        
        # Sort by weight (descending)
        sorted_idx = np.argsort(-weights)[:n_neighbors]
        
        if len(sorted_idx) < n_neighbors:
            # Pad with self-index if needed
            n_found = len(sorted_idx)
            neighbor_indices[i, :n_found] = cell_indices[neighbors[sorted_idx]]
            neighbor_indices[i, n_found:] = cell_indices[i]
            neighbor_weights[i, :n_found] = weights[sorted_idx]
            neighbor_weights[i, n_found:] = 0.0
        else:
            neighbor_indices[i] = cell_indices[neighbors[sorted_idx]]
            neighbor_weights[i] = weights[sorted_idx]
    
    # Normalize weights to sum to 1
    weight_sums = neighbor_weights.sum(axis=1, keepdims=True)
    weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
    neighbor_weights = neighbor_weights / weight_sums
    
    logger.info(f"scRNA-seq connectivity built: {n_masked} cells × {n_neighbors} neighbors")
    
    return neighbor_indices, neighbor_weights


class ConnectivityMatrixBuilder:
    """Build connectivity matrix using JAX-accelerated computation with GPU memory optimization"""
    
    def __init__(self, config: LatentToGeneConfig):
        """
        Initialize with configuration
        
        Args:
            config: LatentToGeneConfig object
        """
        self.config = config
        # Use configured batch size for GPU processing
        self.mkscore_batch_size = config.mkscore_batch_size
        self.dataset_type = config.dataset_type
    
    def build_connectivity_matrix(
        self,
        coords: Optional[np.ndarray] = None,
        emb_gcn: Optional[np.ndarray] = None,
        emb_indv: Optional[np.ndarray] = None,
        cell_mask: Optional[np.ndarray] = None,
        slice_ids: Optional[np.ndarray] = None,
        return_dense: bool = True,
        k_central: Optional[int] = None,
        k_adjacent: Optional[int] = None,
        n_adjacent_slices: Optional[int] = None
    ) -> Union[csr_matrix, Tuple[np.ndarray, np.ndarray]]:
        """
        Build connectivity matrix for a group of cells based on dataset type.
        
        For scRNA-seq: Uses KNN on cell embeddings (emb_indv)
        For spatial2D: Uses spatial anchors and homogeneous neighbors
        For spatial3D: Uses slice-aware spatial anchors and homogeneous neighbors
        
        Args:
            coords: Spatial coordinates (n_cells, 2) - required for spatial datasets
            emb_gcn: Spatial niche embeddings (n_cells, d1) - required for spatial datasets
            emb_indv: Cell identity embeddings (n_cells, d2) - required for all datasets
            cell_mask: Boolean mask for cells to process
            slice_ids: Optional slice/z-coordinate indices (n_cells,) for spatial3D
            return_dense: If True, return dense (n_cells, k) array
            k_central: Number of neighbors on central slice (defaults to config settings)
            k_adjacent: Number of neighbors on adjacent slices for spatial3D
            n_adjacent_slices: Number of slices to search above/below for spatial3D
        
        Returns:
            Connectivity matrix (sparse or dense format)
        """
        # Check dataset type and call appropriate method
        if self.dataset_type == DatasetType.SCRNA_SEQ:
            logger.info("Building connectivity for scRNA-seq dataset")
            if emb_indv is None:
                raise ValueError("emb_indv (cell embeddings) required for scRNA-seq dataset")
            
            return build_scrna_connectivity(
                emb_cell=emb_indv,
                cell_mask=cell_mask,
                n_neighbors=self.config.num_homogeneous,
                metric='euclidean'
            )
        
        elif self.dataset_type in ['spatial2D', 'spatial3D']:
            logger.info(f"Building connectivity for {self.dataset_type} dataset")
            
            # Validate required inputs for spatial datasets
            if coords is None or emb_gcn is None or emb_indv is None:
                raise ValueError("coords, emb_gcn, and emb_indv required for spatial datasets")
            
            # Use config defaults if not provided
            if k_central is None:
                k_central = self.config.num_neighbour_spatial
            if k_adjacent is None:
                k_adjacent = self.config.k_adjacent
            if n_adjacent_slices is None:
                if self.dataset_type == DatasetType.SPATIAL_2D:
                    n_adjacent_slices = 0
                else:  # spatial3D
                    n_adjacent_slices = self.config.n_adjacent_slices
            
            # For spatial2D, ensure no cross-slice search (but can still have slice_ids)
            if self.dataset_type == DatasetType.SPATIAL_2D:
                n_adjacent_slices = 0  # No cross-slice search, but keep slice_ids if provided
            
            return self._build_spatial_connectivity(
                coords=coords,
                emb_gcn=emb_gcn,
                emb_indv=emb_indv,
                cell_mask=cell_mask,
                slice_ids=slice_ids,
                return_dense=return_dense,
                k_central=k_central,
                k_adjacent=k_adjacent,
                n_adjacent_slices=n_adjacent_slices
            )
        
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _build_spatial_connectivity(
        self,
        coords: np.ndarray,
        emb_gcn: np.ndarray,
        emb_indv: np.ndarray,
        cell_mask: Optional[np.ndarray] = None,
        slice_ids: Optional[np.ndarray] = None,
        return_dense: bool = True,
        k_central: int = 101,
        k_adjacent: int = 50,
        n_adjacent_slices: int = 1
    ) -> Union[csr_matrix, Tuple[np.ndarray, np.ndarray]]:
        """
        Internal method for building spatial connectivity matrix.
        
        Args:
            coords: Spatial coordinates (n_cells, 2) - only x,y coordinates
            emb_gcn: Spatial niche embeddings (n_cells, d1)
            emb_indv: Cell identity embeddings (n_cells, d2)
            cell_mask: Boolean mask for cells to process
            slice_ids: Optional slice/z-coordinate indices (n_cells,) for 3D data
            return_dense: If True, return dense (n_cells, k) array
            k_central: Number of neighbors on central slice
            k_adjacent: Number of neighbors on adjacent slices for 3D data
            n_adjacent_slices: Number of slices to search above/below for 3D data
        
        Returns:
            Connectivity matrix (sparse or dense format)
        """
        
        n_cells = len(coords)
        if cell_mask is None:
            cell_mask = np.ones(n_cells, dtype=bool)
        
        cell_indices = np.where(cell_mask)[0]
        n_masked = len(cell_indices)
        
        # Step 1: Find spatial neighbors (slice-aware if slice_ids provided)
        spatial_neighbors = find_spatial_neighbors_with_slices(
            coords=coords,
            slice_ids=slice_ids,
            cell_mask=cell_mask,
            k_central=k_central,
            k_adjacent=k_adjacent,
            n_adjacent_slices=n_adjacent_slices
        )
        
        # Step 2 & 3: Find anchors and homogeneous neighbors in batches
        logger.info(f"Finding anchors and homogeneous neighbors (batch size: {self.mkscore_batch_size})...")

        # Convert to JAX arrays once with float16 for memory efficiency
        # Note: float16 provides sufficient precision for normalized embeddings
        all_emb_gcn_norm_jax = jnp.array(emb_gcn, dtype=jnp.float16)
        all_emb_indv_norm_jax = jnp.array(emb_indv, dtype=jnp.float16)
        spatial_neighbors_jax = jnp.array(spatial_neighbors, dtype=jnp.int32)

        # Move masked embeddings to GPU once
        masked_cell_indices = np.where(cell_mask)[0]
        emb_gcn_masked_jax = all_emb_gcn_norm_jax[masked_cell_indices]
        emb_indv_masked_jax = all_emb_indv_norm_jax[masked_cell_indices]
        
        if self.config.fix_cross_slice_homogenous_neighbors:
            logger.info(f"Using 3D constrained selection (ensuring {self.config.num_homogeneous} neighbors per slice)")
            
            # Process in batches to avoid GPU OOM
            homogeneous_neighbors_list = []
            homogeneous_weights_list = []
            
            for batch_start in track(range(0, n_masked, self.mkscore_batch_size), description="Finding homogeneous neighbors (3D constrained)", transient=True):
                batch_end = min(batch_start + self.mkscore_batch_size, n_masked)
                batch_indices = slice(batch_start, batch_end)
                
                # Get batch data directly from JAX arrays (no GPU movement)
                emb_gcn_batch_norm = emb_gcn_masked_jax[batch_indices]
                emb_indv_batch_norm = emb_indv_masked_jax[batch_indices]
                spatial_neighbors_batch = spatial_neighbors_jax[batch_indices]
                
                # Process batch with 3D-specific JIT-compiled function
                homo_neighbors_batch, homo_weights_batch = _find_anchors_and_homogeneous_batch_3d_jit(
                    emb_gcn_batch_norm,
                    emb_indv_batch_norm,
                    spatial_neighbors_batch,
                    all_emb_gcn_norm_jax,
                    all_emb_indv_norm_jax,
                    self.config.num_homogeneous,  # This is per-slice
                    k_central,
                    k_adjacent,
                    n_adjacent_slices,
                    self.config.similarity_threshold
                )
                
                # Convert back to numpy and append
                homogeneous_neighbors_list.append(np.array(homo_neighbors_batch))
                homogeneous_weights_list.append(np.array(homo_weights_batch))
        else:
            # Use the standard function (2D or 3D without fix_cross_slice_homogenous_neighbors)
            # Process in batches to avoid GPU OOM
            homogeneous_neighbors_list = []
            homogeneous_weights_list = []
            
            for batch_start in track(range(0, n_masked, self.mkscore_batch_size), description="Finding homogeneous neighbors", transient=True):
                batch_end = min(batch_start + self.mkscore_batch_size, n_masked)
                batch_indices = slice(batch_start, batch_end)
                
                # Get batch data directly from JAX arrays (no GPU movement)
                emb_gcn_batch_norm = emb_gcn_masked_jax[batch_indices]
                emb_indv_batch_norm = emb_indv_masked_jax[batch_indices]
                spatial_neighbors_batch = spatial_neighbors_jax[batch_indices]
                
                # Process batch with single JIT-compiled function
                homo_neighbors_batch, homo_weights_batch = _find_anchors_and_homogeneous_batch_jit(
                    emb_gcn_batch_norm,
                    emb_indv_batch_norm,
                    spatial_neighbors_batch,
                    all_emb_gcn_norm_jax,
                    all_emb_indv_norm_jax,
                    self.config.total_homogeneous_neighbor_per_cell,
                    self.config.similarity_threshold
                )
                
                # Convert back to numpy and append
                homogeneous_neighbors_list.append(np.array(homo_neighbors_batch))
                homogeneous_weights_list.append(np.array(homo_weights_batch))
        
        # Concatenate all batches
        homogeneous_neighbors = np.vstack(homogeneous_neighbors_list)
        homogeneous_weights = np.vstack(homogeneous_weights_list)
        
        if return_dense:
            # Return dense format: (n_masked, num_homogeneous) arrays
            return homogeneous_neighbors, homogeneous_weights
        else:
            # Build sparse matrix
            rows = np.repeat(cell_indices, self.config.num_homogeneous)
            cols = homogeneous_neighbors.flatten()
            data = homogeneous_weights.flatten()
            
            connectivity = csr_matrix(
                (data, (rows, cols)),
                shape=(n_cells, n_cells)
            )
            return connectivity
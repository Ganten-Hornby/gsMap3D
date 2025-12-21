"""
Connectivity matrix building for homogeneous spot identification
Implements the spatial → anchor → homogeneous neighbor finding algorithm
"""

import logging
from typing import Optional, Tuple, Union, Dict
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, \
    TimeRemainingColumn, MofNCompleteColumn, TimeElapsedColumn
import scanpy as sc
import anndata as ad

from gsMap.config import LatentToGeneConfig, MarkerScoreCrossSliceStrategy, DatasetType

logger = logging.getLogger(__name__)

# Configure JAX
jax.config.update("jax_enable_x64", False)  # Use float32 for speed


def find_spatial_neighbors_with_slices(
    coords: np.ndarray,
    slice_ids: Optional[np.ndarray] = None,
    query_cell_mask: Optional[np.ndarray] = None,
    high_quality_cell_mask: np.ndarray = None,
    k_central: int = 101,
    k_adjacent: int = 50,
    n_adjacent_slices: int = 1
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Find spatial neighbors with slice-aware search for 3D data.

    For 2D data (slice_ids is None), performs standard KNN.
    For 3D data, implements slice-aware neighbor search:
    - Finds k_central neighbors on the same slice
    - Finds k_adjacent neighbors on each of n_adjacent_slices above and below

    KDTrees are built using only high quality cells, and neighbors are searched
    within high quality cells only.

    Args:
        coords: Spatial coordinates (n_cells, 2) - only x,y coordinates
        slice_ids: Slice/z-coordinate indices (n_cells,) - sequential integers
        query_cell_mask: Boolean mask for cells to find neighbors for
        high_quality_cell_mask: Boolean mask for high quality cells (defines the neighbor search pool)
        k_central: Number of neighbors to find on the central slice
        k_adjacent: Number of neighbors to find on each adjacent slice
        n_adjacent_slices: Number of slices to search above and below

    Returns:
        Tuple of:
        - spatial_neighbors: Array of neighbor indices (n_query_cells, k_central + 2*n_adjacent_slices*k_adjacent)
        - neighbor_pool_per_slice: Dict mapping slice_id to local indices of neighbor pool cells on that slice
    """
    n_cells = len(coords)
    if query_cell_mask is None:
        query_cell_mask = np.ones(n_cells, dtype=bool)

    if high_quality_cell_mask is None:
        high_quality_cell_mask = np.ones(n_cells, dtype=bool)

    logger.debug(f"Finding neighbors: {high_quality_cell_mask.sum()}/{n_cells} cells are high quality")

    query_cell_indices = np.where(query_cell_mask)[0]
    n_query_cells = len(query_cell_indices)

    # Neighbor pool: high quality cells that will be used to build KDTrees
    # (intersection of query_cell_mask and high_quality_cell_mask)
    neighbor_pool_mask = query_cell_mask & high_quality_cell_mask
    neighbor_pool_indices = np.where(neighbor_pool_mask)[0]

    # If no slice_ids provided, perform standard 2D KNN
    if slice_ids is None:
        logger.info(f"No slice IDs provided, performing standard 2D KNN with k={k_central}")
        # Build tree with neighbor pool cells only
        kdtree = cKDTree(coords[neighbor_pool_mask])
        # Query for all query cells (including non-HQ ones)
        _, neighbor_local_indices = kdtree.query(
            coords[query_cell_mask],
            k=min(k_central, len(neighbor_pool_indices)),
            workers=-1  # Use all available cores
        )
        # Convert local indices to global indices
        spatial_neighbors = neighbor_pool_indices[neighbor_local_indices]
        return spatial_neighbors, {}
    
    # Slice-aware neighbor search with fixed-size arrays
    logger.info(f"Performing slice-aware neighbor search: k_central={k_central}, "
                f"k_adjacent={k_adjacent}, n_adjacent_slices={n_adjacent_slices}")

    query_cell_slice_ids = slice_ids[query_cell_mask]
    query_cell_coords = coords[query_cell_mask]

    # Pre-allocate output with fixed size, initialized with -1 (invalid)
    total_neighbors_per_cell = k_central + 2 * n_adjacent_slices * k_adjacent
    # Always use int32 for spatial neighbors since they contain global cell indices
    # which can easily exceed int16 range in large spatial datasets
    spatial_neighbors = np.full((n_query_cells, total_neighbors_per_cell), -1, dtype=np.int32)

    # Get unique slices and create mapping for all query cells
    unique_slice_ids = np.unique(query_cell_slice_ids)
    slice_to_query_cell_indices = {s: np.where(query_cell_slice_ids == s)[0] for s in unique_slice_ids}

    # Create mapping for neighbor pool cells per slice (for building KDTrees)
    neighbor_pool_per_slice = {}

    # Get slice IDs and coordinates for neighbor pool cells
    neighbor_pool_slice_ids = slice_ids[neighbor_pool_mask]
    neighbor_pool_coords = coords[neighbor_pool_mask]

    for slice_id in unique_slice_ids:
        # Find neighbor pool cells on this slice (in the neighbor_pool array)
        slice_neighbor_pool_mask = neighbor_pool_slice_ids == slice_id
        slice_neighbor_pool_local_indices = np.where(slice_neighbor_pool_mask)[0]
        neighbor_pool_per_slice[slice_id] = slice_neighbor_pool_local_indices
        logger.debug(f"Slice {slice_id}: {len(slice_neighbor_pool_local_indices)} high quality cells")

    # Pre-compute KDTree for each slice using neighbor pool cells only
    # Note: KDTree can handle cases where k > number of points in tree
    # It will return valid neighbors first, then fill remaining slots with invalid indices
    slice_kdtrees = {}
    for slice_id in unique_slice_ids:
        # Use neighbor pool cells to build tree
        slice_neighbor_pool_local = neighbor_pool_per_slice.get(slice_id, np.array([]))
        if len(slice_neighbor_pool_local) > 0:  # Build tree as long as there's at least 1 hq cell
            # Get coordinates of neighbor pool cells on this slice
            slice_neighbor_pool_coords = neighbor_pool_coords[slice_neighbor_pool_local]
            kdtree = cKDTree(slice_neighbor_pool_coords)
            # Store tree with global indices of neighbor pool cells
            slice_kdtrees[slice_id] = (kdtree, neighbor_pool_indices[slice_neighbor_pool_local])

            if len(slice_neighbor_pool_local) < max(k_central, k_adjacent):
                logger.warning(f"Slice {slice_id} has only {len(slice_neighbor_pool_local)} high quality cells, "
                             f"which is less than required k={max(k_central, k_adjacent)}. "
                             # f"Some neighbors will be invalid (-1)."
                               )

    # Build a mapping of slice_id -> list of adjacent slice_ids to search
    # This ensures all slices search the same total number of adjacent slices
    slice_adjacent_mapping = {}
    min_slice_id = min(unique_slice_ids)
    max_slice_id = max(unique_slice_ids)

    for slice_id in unique_slice_ids:
        adjacent_slices = []

        # Collect slices in both directions up to n_adjacent_slices
        for offset in range(1, n_adjacent_slices + 1):
            neg_slice = slice_id - offset
            pos_slice = slice_id + offset
            if neg_slice in slice_kdtrees:
                adjacent_slices.append((neg_slice, offset, -1))  # (slice_id, offset, direction)
            if pos_slice in slice_kdtrees:
                adjacent_slices.append((pos_slice, offset, 1))

        # If we don't have enough adjacent slices (2*n_adjacent_slices total), compensate
        target_count = 2 * n_adjacent_slices
        if len(adjacent_slices) < target_count:
            # Search deeper in available directions
            extra_offset = n_adjacent_slices + 1
            while len(adjacent_slices) < target_count and extra_offset <= max(max_slice_id - min_slice_id, 10):
                neg_slice = slice_id - extra_offset
                pos_slice = slice_id + extra_offset

                if neg_slice >= min_slice_id and neg_slice in slice_kdtrees:
                    if not any(s[0] == neg_slice for s in adjacent_slices):
                        adjacent_slices.append((neg_slice, extra_offset, -1))
                        if len(adjacent_slices) >= target_count:
                            break

                if pos_slice <= max_slice_id and pos_slice in slice_kdtrees:
                    if not any(s[0] == pos_slice for s in adjacent_slices):
                        adjacent_slices.append((pos_slice, extra_offset, 1))
                        if len(adjacent_slices) >= target_count:
                            break

                extra_offset += 1

        # Sort adjacent slices: first by offset (closer slices first), then by direction
        # This maintains consistency: [offset=1,dir=-1], [offset=1,dir=1], [offset=2,dir=-1], [offset=2,dir=1], ...
        adjacent_slices.sort(key=lambda x: (x[1], -x[2]))  # Sort by offset, then direction (-1 before 1)

        slice_adjacent_mapping[slice_id] = adjacent_slices

        if len(adjacent_slices) < target_count:
            logger.warning(f"Slice {slice_id} only has {len(adjacent_slices)} adjacent slices available "
                         f"(target: {target_count}). Some neighbor slots will remain empty.")

    # Log the adjacent slice mapping for verification
    for slice_id in sorted(slice_adjacent_mapping.keys()):
        adj_slice_ids = [s[0] for s in slice_adjacent_mapping[slice_id]]
        logger.debug(f"Slice {slice_id} will search in adjacent slices: {adj_slice_ids}")

    # Batch process all query cells (including edge cases with few neighbor pool cells)
    for slice_id in unique_slice_ids:
        if slice_id in slice_kdtrees:
            # Get all query cells on this slice
            query_cells_on_slice = slice_to_query_cell_indices[slice_id]
            query_coords = query_cell_coords[query_cells_on_slice]

            # Central slice neighbors (fixed k_central)
            central_kdtree, central_neighbor_pool_global_indices = slice_kdtrees[slice_id]
            _, central_neighbor_local_indices = central_kdtree.query(query_coords, k=k_central, workers=-1)

            # Handle invalid indices returned by KDTree when k > number of points in tree
            # KDTree returns index >= tree_size for invalid slots
            n_central_pool_cells = len(central_neighbor_pool_global_indices)
            # Initialize with -1 (invalid)
            central_neighbor_global_indices = np.full_like(central_neighbor_local_indices, -1, dtype=np.int32)
            # Create mask for valid indices (< tree size)
            central_valid_mask = central_neighbor_local_indices < n_central_pool_cells
            # Map valid local indices to global indices
            central_neighbor_global_indices[central_valid_mask] = central_neighbor_pool_global_indices[
                central_neighbor_local_indices[central_valid_mask]
            ]
            spatial_neighbors[query_cells_on_slice, :k_central] = central_neighbor_global_indices

            # Adjacent slices - use the pre-computed mapping
            # Get the list of adjacent slices to search for this slice_id
            adjacent_slices_to_search = slice_adjacent_mapping.get(slice_id, [])

            neighbor_column_offset = k_central

            # We need to fill exactly 2*n_adjacent_slices slots in the output array
            # The array structure is: [offset=1,dir=-1], [offset=1,dir=+1], [offset=2,dir=-1], [offset=2,dir=+1], ...
            for slot_idx in range(2 * n_adjacent_slices):
                if slot_idx < len(adjacent_slices_to_search):
                    # We have a slice to search for this slot
                    adjacent_slice_id, _, _ = adjacent_slices_to_search[slot_idx]

                    adjacent_kdtree, adjacent_neighbor_pool_global_indices = slice_kdtrees[adjacent_slice_id]
                    _, adjacent_neighbor_local_indices = adjacent_kdtree.query(query_coords, k=k_adjacent, workers=-1)

                    # Handle invalid indices for adjacent slices
                    n_adjacent_pool_cells = len(adjacent_neighbor_pool_global_indices)
                    adjacent_neighbor_global_indices = np.full_like(adjacent_neighbor_local_indices, -1, dtype=np.int32)
                    adjacent_valid_mask = adjacent_neighbor_local_indices < n_adjacent_pool_cells
                    adjacent_neighbor_global_indices[adjacent_valid_mask] = adjacent_neighbor_pool_global_indices[
                        adjacent_neighbor_local_indices[adjacent_valid_mask]
                    ]
                    spatial_neighbors[query_cells_on_slice, neighbor_column_offset:neighbor_column_offset+k_adjacent] = adjacent_neighbor_global_indices
                # else: slot remains as -1 (already initialized) if no slice available

                neighbor_column_offset += k_adjacent

    # Note: Slices with 0 neighbor pool cells will not be in slice_kdtrees
    # Query cells on such slices will have all neighbors remain as -1 (already initialized)

    return spatial_neighbors, neighbor_pool_per_slice


@partial(jit, static_argnums=(5, 6))
def _find_anchors_and_homogeneous_batch_jit(
    emb_niche_batch_norm: jnp.ndarray,      # (batch_size, d1) - pre-normalized (always exists, may be dummy ones)
    emb_indv_batch_norm: jnp.ndarray,      # (batch_size, d2) - pre-normalized
    spatial_neighbors: jnp.ndarray,   # (batch_size, k1)
    all_emb_niche_norm: jnp.ndarray,         # (n_all, d1) - pre-normalized (always exists, may be dummy ones)
    all_emb_indv_norm: jnp.ndarray,        # (n_all, d2) - pre-normalized
    num_homogeneous: int,
    cell_embedding_similarity_threshold: float = 0.0,
    spatial_domain_similarity_threshold: float = 0.5
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled function to find anchors and homogeneous neighbors.
    Processes a batch of cells to manage GPU memory.
    Expects pre-normalized embeddings for efficiency.

    Args:
        cell_embedding_similarity_threshold: Minimum similarity threshold for cell embedding.
        spatial_domain_similarity_threshold: Minimum similarity threshold for spatial domain embedding.
    """
    batch_size = emb_indv_batch_norm.shape[0]

    # Step 1: Extract spatial neighbors' embeddings (already normalized)
    # Use a safe index (0) for invalid neighbors, will mask them later
    safe_neighbors = jnp.where(spatial_neighbors >= 0, spatial_neighbors, 0)
    spatial_emb_indv_norm = all_emb_indv_norm[safe_neighbors]  # (batch_size, k1, d2)

    # Step 2: Compute both niche and individual similarities for spatial neighbors
    # Compute individual/cell similarities (embeddings are already normalized)
    cell_sims = jnp.einsum('bd,bkd->bk', emb_indv_batch_norm, spatial_emb_indv_norm)
    cell_sims = jnp.where(cell_sims >= cell_embedding_similarity_threshold, cell_sims, 0.0)

    # Compute niche/spatial domain similarities (always exists now, may be dummy ones)
    spatial_emb_niche_norm = all_emb_niche_norm[safe_neighbors]  # (batch_size, k1, d1)
    anchor_sims = jnp.einsum('bd,bkd->bk', emb_niche_batch_norm, spatial_emb_niche_norm)
    # If dummy ones, anchor_sims is 1.0. Since 1.0 >= threshold, we always select cell_sims.
    combined_sims = jnp.where(anchor_sims >= spatial_domain_similarity_threshold, cell_sims, 0.0)

    # Mask out invalid neighbors by setting their similarities to 0
    combined_sims = jnp.where(spatial_neighbors >= 0, combined_sims, 0)

    # Select top homogeneous neighbors based on combined similarity
    top_homo_idx = jnp.argsort(-combined_sims, axis=1)[:, :num_homogeneous]
    batch_idx = jnp.arange(batch_size)[:, None]
    homogeneous_neighbors = spatial_neighbors[batch_idx, top_homo_idx]  # (batch_size, num_homogeneous)
    homogeneous_weights = combined_sims[batch_idx, top_homo_idx]

    return homogeneous_neighbors, homogeneous_weights


def _find_homogeneous_3d_memory_efficient(
    emb_niche_masked_jax: jnp.ndarray,
    emb_indv_masked_jax: jnp.ndarray,
    spatial_neighbors: np.ndarray,
    all_emb_niche_norm_jax: jnp.ndarray,
    all_emb_indv_norm_jax: jnp.ndarray,
    num_homogeneous_per_slice: int,
    k_central: int,
    k_adjacent: int,
    n_adjacent_slices: int,
    cell_embedding_similarity_threshold: float,
    spatial_domain_similarity_threshold: float,
    find_homogeneous_batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient version of 3D homogeneous neighbor finding.
    Processes slices separately to avoid large memory allocations.

    Args:
        emb_niche_masked_jax: Niche embeddings for masked cells (JAX array, float16, always exists, may be dummy ones)
        emb_indv_masked_jax: Individual embeddings for masked cells (JAX array, float16)
        spatial_neighbors: Spatial neighbors array with structure [central | adj1 | adj2 | ...] (numpy array)
        all_emb_niche_norm_jax: All normalized niche embeddings (JAX array, float16, always exists, may be dummy ones)
        all_emb_indv_norm_jax: All normalized individual embeddings (JAX array, float16)
        num_homogeneous_per_slice: Number of neighbors to select per slice
        k_central: Number of neighbors in central slice
        k_adjacent: Number of neighbors per adjacent slice
        n_adjacent_slices: Number of adjacent slices above and below
        cell_embedding_similarity_threshold: Minimum similarity threshold for cell embedding
        spatial_domain_similarity_threshold: Minimum similarity threshold for spatial domain embedding
        find_homogeneous_batch_size: Batch size for processing

    Returns:
        homogeneous_neighbors: Selected neighbors
        homogeneous_weights: Corresponding weights
    """
    n_masked = emb_indv_masked_jax.shape[0]
    n_slices = 1 + 2 * n_adjacent_slices
    
    homogeneous_neighbors_all_slices = []
    homogeneous_weights_all_slices = []
    
    # Process all slices (central + adjacent) in a single loop
    total_slices = 1 + 2 * n_adjacent_slices
    
    # Create overall slice progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        refresh_per_second=1
    ) as slice_progress:
        # Overall slice progress task
        slice_task = slice_progress.add_task(
            "Finding homogeneous neighbors (3D cross-slice)...",
            total=total_slices
        )
        
        for slice_num in range(total_slices):
            # Determine slice name and parameters
            if slice_num == 0:
                slice_name = "central slice"
                slice_start = 0
                slice_end = k_central
                k_slice = k_central
            else:
                # Adjacent slices
                adj_idx = slice_num - 1
                if adj_idx < n_adjacent_slices:
                    slice_name = f"adjacent slice -{n_adjacent_slices - adj_idx}"
                else:
                    slice_name = f"adjacent slice +{adj_idx - n_adjacent_slices + 1}"
                
                slice_start = k_central + adj_idx * k_adjacent
                slice_end = slice_start + k_adjacent
                k_slice = k_adjacent
            
            # Convert slice neighbors to JAX array once per slice
            spatial_neighbors_slice = jnp.asarray(spatial_neighbors[:, slice_start:slice_end])
            
            homogeneous_neighbors_slice_list = []
            homogeneous_weights_slice_list = []
            
            # Process batches for this slice using simple transient track
            for batch_start in track(range(0, n_masked, find_homogeneous_batch_size),
                                    description=f"Finding homogeneous neighbors ({slice_name})",
                                    transient=True):
                batch_end = min(batch_start + find_homogeneous_batch_size, n_masked)
                batch_indices = slice(batch_start, batch_end)

                # Get batch data (emb_niche always exists now, may be dummy ones)
                emb_niche_batch_norm = emb_niche_masked_jax[batch_indices]
                emb_indv_batch_norm = emb_indv_masked_jax[batch_indices]
                
                # Extract batch of neighbors for this slice
                spatial_neighbors_slice_batch = spatial_neighbors_slice[batch_indices, :]
                
                # Process with 2D function
                homo_neighbors_batch, homo_weights_batch = _find_anchors_and_homogeneous_batch_jit(
                    emb_niche_batch_norm=emb_niche_batch_norm,
                    emb_indv_batch_norm=emb_indv_batch_norm,
                    spatial_neighbors=spatial_neighbors_slice_batch,
                    all_emb_niche_norm=all_emb_niche_norm_jax,
                    all_emb_indv_norm=all_emb_indv_norm_jax,
                    num_homogeneous=num_homogeneous_per_slice,
                    cell_embedding_similarity_threshold=cell_embedding_similarity_threshold,
                    spatial_domain_similarity_threshold=spatial_domain_similarity_threshold
                )
                
                homogeneous_neighbors_slice_list.append(np.array(homo_neighbors_batch))
                homogeneous_weights_slice_list.append(np.array(homo_weights_batch))
            
            # Concatenate this slice's results
            homogeneous_neighbors_slice = np.vstack(homogeneous_neighbors_slice_list)
            homogeneous_weights_slice = np.vstack(homogeneous_weights_slice_list)
            homogeneous_neighbors_all_slices.append(homogeneous_neighbors_slice)
            homogeneous_weights_all_slices.append(homogeneous_weights_slice)
            
            # Update slice progress
            slice_progress.update(slice_task, advance=1)
    
    # Concatenate all slices along axis 1
    homogeneous_neighbors = np.concatenate(homogeneous_neighbors_all_slices, axis=1)
    homogeneous_weights = np.concatenate(homogeneous_weights_all_slices, axis=1)
    
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
        self.find_homogeneous_batch_size = config.find_homogeneous_batch_size
        self.dataset_type = config.dataset_type
    
    def build_connectivity_matrix(
        self,
        coords: Optional[np.ndarray] = None,
        emb_niche: np.ndarray = None,
        emb_indv: Optional[np.ndarray] = None,
        cell_mask: Optional[np.ndarray] = None,
        high_quality_mask: np.ndarray = None,
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
            emb_niche: Niche embeddings (n_cells, d1) - always provided (may be dummy ones for scRNA-seq)
            emb_indv: Cell identity embeddings (n_cells, d2) - required for all datasets
            cell_mask: Boolean mask for cells to process
            high_quality_mask: Boolean mask for high quality cells (used for neighbor search in spatial data)
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
                n_neighbors=self.config.homogeneous_neighbors,
                metric='euclidean'
            )
        
        elif self.dataset_type in ['spatial2D', 'spatial3D']:
            logger.info(f"Building connectivity for {self.dataset_type} dataset")
            
            # Validate required inputs for spatial datasets
            if coords is None or emb_indv is None:
                raise ValueError("coords and emb_indv required for spatial datasets")
            
            # Use config defaults if not provided
            if k_central is None:
                k_central = self.config.spatial_neighbors
            if k_adjacent is None:
                k_adjacent = self.config.adjacent_slice_spatial_neighbors
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
                emb_niche=emb_niche,
                emb_indv=emb_indv,
                cell_mask=cell_mask,
                high_quality_mask=high_quality_mask,
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
        emb_indv: np.ndarray,
        emb_niche: np.ndarray = None,
        cell_mask: Optional[np.ndarray] = None,
        high_quality_mask: np.ndarray = None,
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
            emb_niche: Niche embeddings (n_cells, d1) - always provided (may be dummy ones)
            emb_indv: Cell identity embeddings (n_cells, d2)
            cell_mask: Boolean mask for cells to process
            high_quality_mask: Boolean mask for high quality cells (used for neighbor search)
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
        spatial_neighbors, neighbor_pool_per_slice = find_spatial_neighbors_with_slices(
            coords=coords,
            slice_ids=slice_ids,
            query_cell_mask=cell_mask,
            high_quality_cell_mask=high_quality_mask,
            k_central=k_central,
            k_adjacent=k_adjacent,
            n_adjacent_slices=n_adjacent_slices
        )

        # Log statistics about neighbor pool cells per slice if available
        if neighbor_pool_per_slice:
            for slice_id, neighbor_pool_indices in neighbor_pool_per_slice.items():
                logger.debug(f"Slice {slice_id}: {len(neighbor_pool_indices)} high quality cells available for neighbor search")
        
        # Step 2 & 3: Find anchors and homogeneous neighbors in batches
        logger.info(f"Finding anchors and homogeneous neighbors (batch size: {self.find_homogeneous_batch_size})...")

        # Convert embeddings to JAX arrays once (shared for both paths)
        # Note: float16 provides sufficient precision for normalized embeddings
        # emb_niche is guaranteed to exist now (may be dummy ones)
        all_emb_niche_norm_jax = jnp.array(emb_niche, dtype=jnp.float16)
        all_emb_indv_norm_jax = jnp.array(emb_indv, dtype=jnp.float16)

        # Get masked embeddings
        masked_cell_indices = np.where(cell_mask)[0]
        emb_niche_masked_jax = all_emb_niche_norm_jax[masked_cell_indices]
        emb_indv_masked_jax = all_emb_indv_norm_jax[masked_cell_indices]


        if self.config.fix_cross_slice_homogenous_neighbors:
            logger.info(f"Using 3D constrained selection (ensuring {self.config.num_homogeneous} neighbors per slice)")

            # Use memory-efficient version that processes slices separately
            homogeneous_neighbors, homogeneous_weights = _find_homogeneous_3d_memory_efficient(
                emb_niche_masked_jax=emb_niche_masked_jax,
                emb_indv_masked_jax=emb_indv_masked_jax,
                spatial_neighbors=spatial_neighbors,
                all_emb_niche_norm_jax=all_emb_niche_norm_jax,
                all_emb_indv_norm_jax=all_emb_indv_norm_jax,
                num_homogeneous_per_slice=self.config.homogeneous_neighbors,
                k_central=k_central,
                k_adjacent=k_adjacent,
                n_adjacent_slices=n_adjacent_slices,
                cell_embedding_similarity_threshold=self.config.cell_embedding_similarity_threshold,
                spatial_domain_similarity_threshold=self.config.spatial_domain_similarity_threshold,
                find_homogeneous_batch_size=self.find_homogeneous_batch_size
            )
            
        else:
            # Convert spatial_neighbors to JAX array for regular processing
            spatial_neighbors_jax = jnp.array(spatial_neighbors, dtype=jnp.int32)
            homogeneous_neighbors_list = []
            homogeneous_weights_list = []


            # Use the standard function (2D or 3D without fix_cross_slice_homogenous_neighbors)
            for batch_start in track(range(0, n_masked, self.find_homogeneous_batch_size), description="Finding homogeneous neighbors", transient=True):
                batch_end = min(batch_start + self.find_homogeneous_batch_size, n_masked)
                batch_indices = slice(batch_start, batch_end)

                # Get batch data directly from JAX arrays (no GPU movement)
                emb_niche_batch_norm = emb_niche_masked_jax[batch_indices]
                emb_indv_batch_norm = emb_indv_masked_jax[batch_indices]
                spatial_neighbors_batch = spatial_neighbors_jax[batch_indices]

                # Process batch with single JIT-compiled function
                homo_neighbors_batch, homo_weights_batch = _find_anchors_and_homogeneous_batch_jit(
                    emb_niche_batch_norm=emb_niche_batch_norm,
                    emb_indv_batch_norm=emb_indv_batch_norm,
                    spatial_neighbors=spatial_neighbors_batch,
                    all_emb_niche_norm=all_emb_niche_norm_jax,
                    all_emb_indv_norm=all_emb_indv_norm_jax,
                    num_homogeneous=self.config.total_homogeneous_neighbor_per_cell,
                    cell_embedding_similarity_threshold=self.config.cell_embedding_similarity_threshold,
                    spatial_domain_similarity_threshold=self.config.spatial_domain_similarity_threshold
            )

                # Convert back to numpy and append
                homogeneous_neighbors_list.append(np.array(homo_neighbors_batch))
                homogeneous_weights_list.append(np.array(homo_weights_batch))

            # Regular batched processing
            homogeneous_neighbors = np.vstack(homogeneous_neighbors_list)
            homogeneous_weights = np.vstack(homogeneous_weights_list)
        
        if return_dense:
            # Return dense format: (n_masked, num_homogeneous) arrays
            return homogeneous_neighbors, homogeneous_weights
        else:
            # Build sparse matrix
            rows = np.repeat(cell_indices, self.config.homogeneous_neighbors)
            cols = homogeneous_neighbors.flatten()
            data = homogeneous_weights.flatten()
            
            connectivity = csr_matrix(
                (data, (rows, cols)),
                shape=(n_cells, n_cells)
            )
            return connectivity
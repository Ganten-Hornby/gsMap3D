"""
Rank calculation from latent representations
Extracts and processes the rank calculation logic from find_latent_representation.py
"""

import gc
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import h5py
import pandas as pd
from anndata._io.specs import read_elem

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from jax import jit
from scipy.sparse import csr_matrix
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from rich.table import Table
import jax.scipy
import anndata as ad
from gsMap.config import LatentToGeneConfig
from .memmap_io import MemMapDense

logger = logging.getLogger(__name__)


@jit
def jax_process_chunk(dense_matrix, n_genes):

    nonzero_mask = dense_matrix != 0

    ranks = jax.scipy.stats.rankdata(dense_matrix, method='average', axis=1)
    log_ranks = jnp.log(ranks)
    # Sum log ranks (with fill_zero)
    sum_log_ranks = log_ranks.sum(axis=0)
    # Sum fraction (count of non-zeros)
    sum_frac = nonzero_mask.sum(axis=0)

    return log_ranks, sum_log_ranks, sum_frac


def rank_data_jax(X: csr_matrix, n_genes,
                  memmap_dense=None,
                  metadata: Optional[Dict[str, Any]] = None,
                  chunk_size: int = 1000,
                  write_interval: int = 10,
                  current_row_offset: int = 0,
                  progress=None,
                  progress_task=None
                  ):
    """JAX-optimized rank calculation with batched writing to memory-mapped storage.

    Args:
        X: Input sparse matrix
        n_genes: Total number of genes
        memmap_dense: Optional MemMapDense instance for writing
        metadata: Optional metadata dictionary
        chunk_size: Size of chunks for processing
        write_interval: How often to write chunks to memory map
        current_row_offset: Offset for writing to memory map (for multiple sections)
        progress: Progress instance for updates
        progress_task: Task ID for progress updates

    Returns:
        Tuple of (sum_log_ranks, sum_frac) as numpy arrays
    """
    assert X.nnz != 0, "Input matrix must not be empty"

    n_rows, n_cols = X.shape

    # Initialize accumulators (use float32 for accumulators to avoid precision loss)
    sum_log_ranks = jnp.zeros(n_genes, dtype=jnp.float32)
    sum_frac = jnp.zeros(n_genes, dtype=jnp.float32)

    # Process in chunks to manage memory
    chunk_size = min(chunk_size, n_rows)
    pending_chunks = []  # Buffer for batching writes
    pending_indices = []  # Track global indices for writing
    chunks_processed = 0
    
    # Track speed at chunk level
    chunk_start_time = time.time()
    processed_cells = 0

    for start_idx in range(0, n_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, n_rows)

        # Convert chunk to dense - use JAX asarray for zero-copy when possible
        chunk_X = X[start_idx:end_idx]
        chunk_dense = chunk_X.toarray().astype(np.float32)
        chunk_jax = jnp.asarray(chunk_dense)  # Use asarray for zero-copy conversion

        # Process chunk with JIT-compiled function (ranking + accumulators)
        chunk_log_ranks, chunk_sum_log_ranks, chunk_sum_frac = jax_process_chunk(chunk_jax, n_genes)

        # Update global accumulators
        sum_log_ranks += chunk_sum_log_ranks
        sum_frac += chunk_sum_frac

        # Convert JAX array to numpy float16 for storage efficiency
        # This reduces memory usage by 50% compared to float32
        chunk_log_ranks_np = np.array(chunk_log_ranks, dtype=np.float16)
        pending_chunks.append(chunk_log_ranks_np)
        # Calculate global indices for this chunk
        global_start = current_row_offset + start_idx
        global_end = current_row_offset + end_idx
        pending_indices.append((global_start, global_end))
        chunks_processed += 1

        # Write to memory map periodically
        if memmap_dense and chunks_processed % write_interval == 0:
            # Combine pending chunks for batch write
            combined_data = np.vstack(pending_chunks)
            # Calculate row indices for batch write
            start_row = pending_indices[0][0]
            end_row = pending_indices[-1][1]
            # Write as a contiguous block
            memmap_dense.write_batch(combined_data, row_indices=slice(start_row, end_row))
            pending_chunks.clear()
            pending_indices.clear()

        # Update progress bar with speed calculation
        if progress and progress_task is not None:
            chunk_cells = end_idx - start_idx
            processed_cells += chunk_cells
            elapsed_time = time.time() - chunk_start_time
            speed = processed_cells / elapsed_time if elapsed_time > 0 else 0
            progress.update(progress_task, advance=chunk_cells, speed=f"{speed:.0f}")

    # Write any remaining chunks
    if memmap_dense and pending_chunks:
        combined_data = np.vstack(pending_chunks)
        start_row = pending_indices[0][0]
        end_row = pending_indices[-1][1]
        memmap_dense.write_batch(combined_data, row_indices=slice(start_row, end_row))

    return np.array(sum_log_ranks), np.array(sum_frac)


class RankCalculator:
    """Calculate gene expression ranks and create concatenated latent representations"""
    
    def __init__(self, config: LatentToGeneConfig):
        """
        Initialize RankCalculator with configuration
        
        Args:
            config: LatentToGeneConfig object with all necessary parameters
        """
        self.config = config
        self.latent_dir = Path(config.latent_dir)
        self.output_dir = Path(config.latent2gene_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        
    def calculate_ranks_and_concatenate(
        self,
        sample_h5ad_dict: Optional[Dict[str, Path]] = None,
        annotation_key: Optional[str] = None,
        data_layer: str = "counts"
    ) -> Dict[str, Any]:
        """
        Calculate expression ranks and create concatenated latent representation
        
        This combines the rank calculation and concatenation logic from find_latent_representation.py
        
        Args:
            sample_h5ad_dict: Optional dict of sample_name -> h5ad path. If None, uses config.sample_h5ad_dict
            annotation_key: Optional annotation to filter cells. If None, uses config.annotation
            data_layer: Data layer to use for expression
            
        Returns:
            Dictionary with paths to:
                - concatenated_latent_adata: Path to concatenated latent representations
                - rank_zarr: Path to rank zarr file
                - mean_frac: Path to mean expression fraction parquet
        """
        
        # Use provided sample_h5ad_dict or get from config
        if sample_h5ad_dict is None:
            sample_h5ad_dict = self.config.sample_h5ad_dict
        
        # Use provided annotation_key or get from config
        if annotation_key is None:
            annotation_key = self.config.annotation
            
        # Output paths from config
        concat_adata_path = Path(self.config.concatenated_latent_adata_path)
        rank_memmap_path = Path(self.config.rank_memmap_path)
        mean_frac_path = Path(self.config.mean_frac_path)
        
        # Check if outputs already exist
        if concat_adata_path.exists() and mean_frac_path.exists() and MemMapDense.check_complete(rank_memmap_path)[0]:
            logger.info(f"Rank outputs already exist in {self.output_dir}")
            return {
                "concatenated_latent_adata": str(concat_adata_path),
                "rank_memmap": str(rank_memmap_path),
                "mean_frac": str(mean_frac_path)
            }

        logger.info("Starting rank calculation and concatenation...")
        logger.info(f"Processing {len(sample_h5ad_dict)} samples")
        
        # Process each section
        adata_list = []
        n_total_cells = 0
        gene_list = None
        
        # Initialize global accumulators
        sum_log_ranks = None
        sum_frac = None
        total_cells = 0
        rank_memmap = None
        current_row_offset = 0  # Track current position in rank memory map

        # First pass: count total cells to initialize memory map
        logger.info("Counting total cells across all sections...")
        total_cells_expected = 0
        filtering_stats = []

        for sample_name, h5ad_path in sample_h5ad_dict.items():
            # Apply same filtering logic as in main loop
            with h5py.File(h5ad_path, 'r') as f:
                adata_temp_obs = read_elem(f['obs'])
                n_cells_before = adata_temp_obs.shape[0]
                nan_count = 0
                small_group_removed = 0
                
                if annotation_key is not None:
                    assert annotation_key and annotation_key in adata_temp_obs.columns, \
                        f"Annotation key '{annotation_key}' not found in the obs of {sample_name}"

                    # Check for and handle NaN values
                    nan_count = adata_temp_obs[annotation_key].isna().sum()
                    if nan_count > 0:
                        adata_temp_obs = adata_temp_obs[adata_temp_obs[annotation_key].notna()].copy()

                    # Filter cells based on annotation group size
                    min_cells_per_type = self.config.homogeneous_neighbors
                    annotation_counts = adata_temp_obs[annotation_key].value_counts()
                    valid_annotations = annotation_counts[annotation_counts >= min_cells_per_type].index

                    if len(valid_annotations) < len(annotation_counts):
                        # Filter to valid annotations
                        mask = adata_temp_obs[annotation_key].isin(valid_annotations)
                        adata_temp_obs = adata_temp_obs[mask].copy()
                        small_group_removed = n_cells_before - nan_count - adata_temp_obs.shape[0]

                n_cells_after = adata_temp_obs.shape[0]
                total_cells_expected += n_cells_after
                
                filtering_stats.append({
                    "Sample": sample_name,
                    "Total": n_cells_before,
                    "NaN": nan_count,
                    "Small Group": small_group_removed,
                    "Final": n_cells_after
                })

        # Display filtering summary table
        table = Table(title="[bold]Cell Filtering Summary[/bold]", show_header=True, header_style="bold magenta")
        table.add_column("Sample", style="dim")
        table.add_column("Input Cells", justify="right")
        table.add_column("NaN Removed", justify="right", style="red")
        table.add_column("Small Group Removed", justify="right", style="red")
        table.add_column("Final Cells", justify="right", style="green")

        for stat in filtering_stats:
            table.add_row(
                stat["Sample"],
                str(stat["Total"]),
                str(stat["NaN"]),
                str(stat["Small Group"]),
                str(stat["Final"])
            )
        
        # Add a total row
        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            str(sum(s["Total"] for s in filtering_stats)),
            str(sum(s["NaN"] for s in filtering_stats)),
            str(sum(s["Small Group"] for s in filtering_stats)),
            f"[bold green]{total_cells_expected}[/bold green]"
        )
        
        self.console.print(table)
        logger.info(f"Expected total cells after filtering: {total_cells_expected}")
        
        # Create overall section progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            refresh_per_second=1
        ) as section_progress:
            # Overall section progress task
            section_task = section_progress.add_task(
                "Processing sections",
                total=len(sample_h5ad_dict)
            )
            
            processed_cells_total = 0
            
            for st_id, (sample_name, h5ad_path) in enumerate(sample_h5ad_dict.items()):
                
                section_progress.console.log(f"Loading {sample_name} ({st_id + 1}/{len(sample_h5ad_dict)})...")
                
                # Load the h5ad file (which should already contain latent representations)
                adata = sc.read_h5ad(h5ad_path)
                
                # Add slice information
                adata.obs['slice_id'] = st_id
                adata.obs['slice_name'] = sample_name

                # make unique index
                adata.obs_names_make_unique()
                adata.obs_names = adata.obs_names.astype(str) +'|'+ adata.obs['slice_name'].astype(str)

                # Apply the same filtering logic as in counting phase
                # This must be done BEFORE adding to rank zarr to maintain index consistency
                if annotation_key is not None:
                    assert annotation_key and annotation_key in adata.obs.columns
                    # Check for and handle NaN values
                    nan_count = adata.obs[annotation_key].isna().sum()
                    if nan_count > 0:
                        adata = adata[adata.obs[annotation_key].notna()].copy()

                    # Filter cells based on annotation group size
                    min_cells_per_type = self.config.homogeneous_neighbors
                    annotation_counts = adata.obs[annotation_key].value_counts()
                    valid_annotations = annotation_counts[annotation_counts >= min_cells_per_type].index

                    if len(valid_annotations) < len(annotation_counts):
                        adata = adata[adata.obs[annotation_key].isin(valid_annotations)].copy()

                # Get gene list (should be consistent across sections)
                if gene_list is None:
                    gene_list = adata.var_names.tolist()
                    n_genes = len(gene_list)
                    # Initialize rank memory map as dense matrix with float16 for 50% space savings
                    # Log ranks typically have sufficient precision with float16
                    rank_memmap = MemMapDense(
                        str(rank_memmap_path),
                        shape=(total_cells_expected, n_genes),
                        dtype=np.float16,  # Use float16 to save space
                        mode='w',
                        num_write_workers=self.config.mkscore_write_workers
                    )
                    # Initialize global accumulators
                    sum_log_ranks = np.zeros(n_genes, dtype=np.float64)
                    sum_frac = np.zeros(n_genes, dtype=np.float64)
                else:
                    # Verify gene list consistency
                    assert adata.var_names.tolist() == gene_list, \
                        f"Gene list mismatch in section {st_id}"

                # Get expression data for ranking
                if data_layer in adata.layers:
                    X = adata.layers[data_layer]
                else:
                    X = adata.X

                # Efficient sparse matrix conversion
                if not hasattr(X, 'tocsr'):
                    X = csr_matrix(X, dtype=np.float32)
                else:
                    X = X.tocsr()
                    if X.dtype != np.float32:
                        X = X.astype(np.float32)

                # Pre-allocate output arrays for efficiency
                X.sort_indices()  # Sort indices for better cache performance

                # Get number of cells after filtering
                n_cells = X.shape[0]
                
                # Use nested progress bar for detailed chunk processing
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[bold blue]Ranking {sample_name} ({{task.fields[cells]}} cells)"),
                    BarColumn(bar_width=None),
                    MofNCompleteColumn(),
                    TaskProgressColumn(),
                    TextColumn("[bold green]{task.fields[speed]} cells/s"),
                    TimeRemainingColumn(),
                    refresh_per_second=2,
                    transient=True
                ) as chunk_progress:
                    # Detailed chunk progress task
                    chunk_task = chunk_progress.add_task(
                        f"Processing chunks",
                        total=n_cells,
                        cells=n_cells,
                        speed="0"
                    )
                    
                    # Use JAX rank calculation with nested progress
                    metadata = {'name': sample_name, 'cells': n_cells, 'study_id': st_id}

                    batch_sum_log_ranks, batch_frac = rank_data_jax(
                        X,
                        n_genes,
                        memmap_dense=rank_memmap,
                        metadata=metadata,
                        chunk_size=self.config.rank_batch_size,
                        write_interval=self.config.rank_write_interval,
                        current_row_offset=current_row_offset,
                        progress=chunk_progress,
                        progress_task=chunk_task
                    )

                # Update global sums
                sum_log_ranks += batch_sum_log_ranks
                sum_frac += batch_frac
                total_cells += n_cells
                current_row_offset += n_cells  # Update offset for next section
                processed_cells_total += n_cells
                
                # Update section progress
                section_progress.update(section_task, advance=1)

                # Create minimal AnnData with empty X matrix but keep obs and obsm
                minimal_adata = ad.AnnData(
                    X=csr_matrix((adata.n_obs, n_genes), dtype=np.float32),
                    obs=adata.obs.copy(),
                    var=pd.DataFrame(index=gene_list),
                    obsm=adata.obsm.copy()  # Keep all latent representations
                )

                adata_list.append(minimal_adata)
                n_total_cells += n_cells

                # Clean up memory
                del adata, X, minimal_adata
                gc.collect()

            # Close rank memory map
            if rank_memmap is not None:
                rank_memmap.close()
                logger.info(f"Saved rank matrix to {rank_memmap_path}")
        
        # Calculate mean log ranks and mean fraction
        mean_log_ranks = sum_log_ranks / total_cells
        mean_frac = sum_frac / total_cells
        
        # Save mean and fraction to parquet file
        mean_frac_df = pd.DataFrame(
            data=dict(
                G_Mean=mean_log_ranks,
                frac=mean_frac,
                gene_name=gene_list,
            ),
            index=gene_list,
        )
        # Save outside the progress context
        mean_frac_df.to_parquet(
            mean_frac_path,
            index=True,
            compression="gzip",
        )
        logger.info(f"Mean fraction data saved to {mean_frac_path}")
        
        # Concatenate all sections
        if adata_list:
            with self.console.status("[bold blue]Concatenating and saving latent representations..."):
                concatenated_adata = ad.concat(adata_list, axis=0, join='outer', merge='same')

                # Ensure the var_names are the common genes
                concatenated_adata.var_names = gene_list

                # Save concatenated adata
                concatenated_adata.write_h5ad(concat_adata_path)
                logger.info(f"Saved concatenated latent representations to {concat_adata_path}")
                logger.info(f"  - Total cells: {concatenated_adata.n_obs}")
                logger.info(f"  - Total genes: {concatenated_adata.n_vars}")
                logger.info(f"  - Latent representations in obsm: {list(concatenated_adata.obsm.keys())}")
                if 'slice_id' in concatenated_adata.obs.columns:
                    logger.info(f"  - Number of slices: {concatenated_adata.obs['slice_id'].nunique()}")

                # Clean up
                del adata_list, concatenated_adata
                gc.collect()

        # Final completion message
        logger.info("Rank calculation and concatenation completed successfully")
        
        return {
            "concatenated_latent_adata": str(concat_adata_path),
            "rank_memmap": str(rank_memmap_path),
            "mean_frac": str(mean_frac_path)
        }
"""
Report Data Preparation Module - Modern Version

This module prepares data for the Alpine.js + Tailwind CSS + Plotly.js report.
All data is exported as JS files with window global variables to bypass CORS restrictions
when opening index.html via file:// protocol.
"""

import gc
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import plotly
import scanpy as sc
import scipy.sparse as sp

from gsMap.config import QuickModeConfig
from gsMap.find_latent.st_process import normalize_for_analysis, setup_data_layer
from gsMap.report.diagnosis import filter_snps, load_gwas_data
from gsMap.spatial_ldsc.io import load_marker_scores_memmap_format

logger = logging.getLogger(__name__)


# =============================================================================
# Parallel Task Functions (must be at module level for ProcessPoolExecutor)
# =============================================================================

def _render_gene_plot_task(task_data: dict):
    """Parallel task to render Expression or GSS plots using draw_scatter."""
    try:
        from pathlib import Path
        import pandas as pd
        from gsMap.report.visualize import draw_scatter

        gene = task_data['gene']
        coords = task_data['coords']
        vals = task_data['values']
        output_path = Path(task_data['output_path'])
        title = task_data['title']
        point_size = task_data.get('point_size')
        width = task_data.get('width', 800)
        height = task_data.get('height', 800)

        df = pd.DataFrame({
            'sx': coords[:, 0],
            'sy': coords[:, 1],
            'val': vals
        })

        fig = draw_scatter(
            df,
            title=title,
            point_size=point_size,
            width=width,
            height=height,
            color_by='val',
        )

        fig.write_image(str(output_path))
        return True
    except Exception as e:
        return str(e)


def _render_multi_sample_gene_plot_task(task_data: dict):
    """
    Parallel task to render multi-sample Expression or GSS plots.
    Creates a grid of subplots showing gene values across all samples.
    """
    try:
        import gc
        from pathlib import Path

        import matplotlib
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.spatial import KDTree

        matplotlib.use('Agg')

        gene = task_data['gene']
        plot_type = task_data['plot_type']
        sample_data_list = task_data['sample_data_list']
        output_path = Path(task_data['output_path'])
        n_rows = task_data.get('n_rows', 2)
        n_cols = task_data.get('n_cols', 4)
        subplot_width = task_data.get('subplot_width', 4.0)
        dpi = task_data.get('dpi', 150)

        # Custom colormap
        custom_colors = [
            '#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
            '#fee090', '#fdae61', '#f46d43', '#d73027'
        ]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', custom_colors)

        fig_width = n_cols * subplot_width
        fig_height = n_rows * subplot_width

        fig = plt.figure(figsize=(fig_width, fig_height))
        title_text = f"{gene} - {'Expression' if plot_type == 'exp' else 'GSS'}"
        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)

        grid_specs = fig.add_gridspec(nrows=n_rows, ncols=n_cols, wspace=0.1, hspace=0.15)

        # Calculate global min/max for consistent color scale
        all_values = np.concatenate([
            data[2] for data in sample_data_list
            if data[2] is not None and len(data[2]) > 0
        ])
        vmin = 0
        vmax = np.percentile(all_values, 99.5) if len(all_values) > 0 else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

        scatter = None
        for idx, (sample_name, coords, values) in enumerate(sample_data_list[:n_rows * n_cols]):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(grid_specs[row, col])

            if coords is not None and values is not None and len(coords) > 0:
                tree = KDTree(coords)
                distances, _ = tree.query(coords, k=min(2, len(coords)))
                avg_dist = np.mean(distances[:, 1]) if len(coords) > 1 else 1.0

                x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
                y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
                data_range = max(x_range, y_range)
                if data_range > 0:
                    point_size = ((avg_dist / data_range) * subplot_width * 72) ** 2 * 1.2
                else:
                    point_size = 10
                point_size = min(max(point_size, 1), 200)

                scatter = ax.scatter(
                    coords[:, 0], coords[:, 1],
                    c=values, cmap=custom_cmap,
                    s=point_size, vmin=vmin, vmax=vmax,
                    marker='o', edgecolors='none', rasterized=True
                )

            ax.axis('off')
            ax.set_title(sample_name, fontsize=10, pad=2)

        # Hide unused subplots
        for idx in range(len(sample_data_list), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(grid_specs[row, col])
            ax.axis('off')

        # Add colorbar
        if scatter is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(scatter, cax=cbar_ax)
            cbar.set_label('Expression' if plot_type == 'exp' else 'GSS', fontsize=10)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        gc.collect()

        return True
    except Exception as e:
        import traceback
        return f"{str(e)}\n{traceback.format_exc()}"


# =============================================================================
# UMAP Calculation Functions
# =============================================================================

def _detect_z_axis(coords_3d: np.ndarray, sample_names: pd.Series) -> int:
    """
    Detect which dimension is the Z axis (stacking dimension) in 3D coordinates.

    The Z axis is identified as the dimension with the least within-sample variance,
    since samples are stacked along Z and should have minimal variation in that dimension.

    Args:
        coords_3d: 3D coordinates array (n_spots, 3)
        sample_names: Series of sample names for each spot

    Returns:
        Index of the Z axis (0, 1, or 2)
    """
    # Calculate within-sample variance for each dimension
    within_sample_vars = []

    for dim in range(3):
        dim_values = coords_3d[:, dim]

        # Calculate variance within each sample
        sample_vars = []
        for sample in sample_names.unique():
            mask = sample_names == sample
            sample_values = dim_values[mask]
            if len(sample_values) > 1:
                sample_vars.append(np.var(sample_values))

        # Average within-sample variance for this dimension
        avg_within_var = np.mean(sample_vars) if sample_vars else 0
        within_sample_vars.append(avg_within_var)

    # Z axis has the least within-sample variance
    z_axis = int(np.argmin(within_sample_vars))
    logger.info(f"Detected Z axis: dimension {z_axis} (within-sample variances: {within_sample_vars})")

    return z_axis


def _reorder_coords_for_3d(coords_3d: np.ndarray, z_axis: int) -> Tuple[np.ndarray, List[str]]:
    """
    Reorder 3D coordinates so that Z axis is the last dimension.

    Returns:
        Tuple of (reordered coords, column names)
    """
    # Create axis order: put z_axis last
    axis_order = [i for i in range(3) if i != z_axis] + [z_axis]
    reordered = coords_3d[:, axis_order]

    # Column names reflect the reordering
    col_names = ['3d_x', '3d_y', '3d_z']

    return reordered, col_names

def _stratified_subsample(
    spot_names: np.ndarray,
    sample_names: pd.Series,
    n_samples: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Stratified subsampling that ensures representation from all samples.

    Args:
        spot_names: Array of spot identifiers
        sample_names: Series mapping spot names to sample names
        n_samples: Target number of samples
        random_state: Random seed for reproducibility

    Returns:
        Array of selected spot names
    """
    np.random.seed(random_state)

    # Get sample counts
    sample_counts = sample_names.value_counts()
    n_samples_total = len(spot_names)

    if n_samples >= n_samples_total:
        return spot_names

    # Calculate proportional samples per group
    selected_spots = []
    for sample_name, count in sample_counts.items():
        sample_spots = spot_names[sample_names == sample_name]
        # Proportional allocation
        n_select = max(1, int(np.ceil(n_samples * count / n_samples_total)))
        n_select = min(n_select, len(sample_spots))

        selected = np.random.choice(sample_spots, n_select, replace=False)
        selected_spots.extend(selected)

    # If we have too many, randomly remove some
    selected_spots = np.array(selected_spots)
    if len(selected_spots) > n_samples:
        selected_spots = np.random.choice(selected_spots, n_samples, replace=False)

    return selected_spots


def _calculate_umap_from_embeddings(
    adata: ad.AnnData,
    embedding_key: str,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    random_state: int = 42
) -> np.ndarray:
    """
    Calculate UMAP coordinates from embeddings.

    Args:
        adata: AnnData object with embeddings in obsm
        embedding_key: Key for embeddings in obsm
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random seed

    Returns:
        UMAP coordinates array (n_cells, 2)
    """
    import umap

    embeddings = adata.obsm[embedding_key]
    if hasattr(embeddings, 'toarray'):
        embeddings = embeddings.toarray()

    # L2 normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-8)

    logger.info(f"Calculating UMAP for {embedding_key} with {embeddings_norm.shape[0]} spots...")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='cosine',
        random_state=random_state,
        n_jobs=-1
    )

    umap_coords = reducer.fit_transform(embeddings_norm)
    return umap_coords


def _prepare_umap_data(
    config: QuickModeConfig,
    metadata: pd.DataFrame,
    report_dir: Path
) -> Optional[Dict]:
    """
    Prepare UMAP data from cell and niche embeddings.

    Returns dict with umap_cell, umap_niche (optional), and metadata for visualization.
    """
    logger.info("Preparing UMAP data from embeddings...")

    # Load concatenated adata
    adata_path = config.concatenated_latent_adata_path
    if not adata_path.exists():
        logger.warning(f"Concatenated adata not found at {adata_path}")
        return None

    adata = ad.read_h5ad(adata_path)

    # Get embedding keys from config
    cell_emb_key = getattr(config, 'latent_representation_cell', 'emb_cell')
    niche_emb_key = getattr(config, 'latent_representation_niche', None)

    # Check if embeddings exist
    if cell_emb_key not in adata.obsm:
        logger.warning(f"Cell embedding '{cell_emb_key}' not found in adata.obsm")
        return None

    has_niche = niche_emb_key is not None and niche_emb_key in adata.obsm

    # Stratified subsampling
    n_subsample = getattr(config, 'downsampling_n_spots', 10000)
    spot_names = adata.obs_names.values
    sample_names = adata.obs['sample_name']

    if len(spot_names) > n_subsample:
        logger.info(f"Stratified subsampling from {len(spot_names)} to {n_subsample} spots...")
        selected_spots = _stratified_subsample(spot_names, sample_names, n_subsample)
        adata_sub = adata[selected_spots].copy()
    else:
        adata_sub = adata.copy()
        selected_spots = spot_names

    logger.info(f"Using {len(selected_spots)} spots for UMAP calculation")

    # Calculate UMAPs
    umap_cell = _calculate_umap_from_embeddings(adata_sub, cell_emb_key)

    umap_niche = None
    if has_niche:
        umap_niche = _calculate_umap_from_embeddings(adata_sub, niche_emb_key)

    # Prepare metadata for the subsampled spots
    umap_metadata = pd.DataFrame({
        'spot': adata_sub.obs_names,
        'sample_name': adata_sub.obs['sample_name'].values,
        'umap_cell_x': umap_cell[:, 0],
        'umap_cell_y': umap_cell[:, 1],
    })

    if umap_niche is not None:
        umap_metadata['umap_niche_x'] = umap_niche[:, 0]
        umap_metadata['umap_niche_y'] = umap_niche[:, 1]

    # Add all annotation columns
    for anno in config.annotation_list:
        if anno in adata_sub.obs.columns:
            umap_metadata[anno] = adata_sub.obs[anno].values

    # Save to CSV
    umap_metadata.to_csv(report_dir / "umap_data.csv", index=False)
    logger.info(f"UMAP data saved with {len(umap_metadata)} points")

    return {
        'has_niche': has_niche,
        'cell_emb_key': cell_emb_key,
        'niche_emb_key': niche_emb_key,
        'n_points': len(umap_metadata)
    }


def _prepare_3d_visualization(
    config: QuickModeConfig,
    metadata: pd.DataFrame,
    traits: List[str],
    report_dir: Path
) -> Optional[str]:
    """
    Create 3D visualization widget using spatialvista and save as HTML.

    Args:
        config: QuickModeConfig with annotation_list and other settings
        metadata: DataFrame with 3d_x, 3d_y, 3d_z coordinates and annotations
        traits: List of trait names (for continuous values)
        report_dir: Output directory

    Returns:
        Path to saved HTML file, or None if failed
    """
    try:
        import spatialvista as spv
        from ipywidgets import embed
        from ipywidgets.embed import embed_minimal_html
    except ImportError:
        logger.warning("spatialvista or ipywidgets not installed. Skipping 3D visualization.")
        return None

    # Check if we have 3D coordinates
    if '3d_x' not in metadata.columns or '3d_y' not in metadata.columns or '3d_z' not in metadata.columns:
        logger.warning("3D coordinates not found in metadata. Skipping 3D visualization.")
        return None

    logger.info("Creating 3D visualization widget...")

    # Stratified subsample for visualization (limit to reasonable size)
    n_max_points = getattr(config, 'downsampling_n_spots', 10000)
    if len(metadata) > n_max_points:
        sample_names = metadata['sample_name']
        selected_idx = _stratified_subsample(
            metadata.index.values, sample_names, n_max_points
        )
        metadata_sub = metadata.loc[selected_idx].copy()
        logger.info(f"Subsampled to {len(metadata_sub)} spots for 3D visualization")
    else:
        metadata_sub = metadata.copy()

    # Create AnnData for spatialvista
    # X matrix can be empty since we're not showing genes
    n_spots = len(metadata_sub)
    adata_vis = ad.AnnData(
        X=sp.csr_matrix((n_spots, 1), dtype=np.float32),
        obs=metadata_sub.set_index('spot')
    )
    # Set spatial coordinates
    adata_vis.obsm['spatial_3d'] = metadata_sub[['3d_x', '3d_y', '3d_z']].values
    adata_vis.obsm['spatial_2d'] = metadata_sub[['sx', 'sy']].values

    # let trait col float32
    adata_vis.obs = adata_vis.obs.astype({trait: np.float32 for trait in traits if trait in adata_vis.obs.columns})
    # Add traits as continuous values
    continuous_cols = traits

    # Add annotations (categorical)
    annotation_cols = []
    for anno in config.annotation_list:
        if anno in metadata_sub.columns:
            annotation_cols.append(anno)

    # Use first annotation as color, rest as additional annotations
    color_col = annotation_cols[0] if annotation_cols else 'sample_name'
    other_annotations = annotation_cols[1:] if len(annotation_cols) > 1 else []

    logger.info(f"3D visualization: color={color_col}, annotations={other_annotations}, continuous={continuous_cols}")

    # Create visualization widget
    try:
        widget = spv.vis(
            adata_vis,
            position='spatial',
            color=color_col,
            annotations=other_annotations if other_annotations else None,
            continuous=continuous_cols if continuous_cols else None,
            genes=None,
            section='sample_name',
            mode='3D',
        )

        # Set widget size
        widget.sizing_mode = 'stretch_both'
        widget.layout.width = '100%'
        widget.layout.height = '800px'

        # Save to HTML
        embed.DEFAULT_EMBED_REQUIREJS_URL = 'https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js'
        html_path = report_dir / "spv_spatial_3d_widget.html"
        embed_minimal_html(
            str(html_path),
            views=[widget],
            title="gsMap 3D Visualization",
            drop_defaults=False
        )

        logger.info(f"3D visualization saved to {html_path}")
        return str(html_path)

    except Exception as e:
        logger.warning(f"Failed to create 3D visualization widget: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Data Loading Functions
# =============================================================================

def _load_ldsc_results(config: QuickModeConfig) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load combined LDSC results and extract traits/samples."""
    logger.info(f"Loading combined LDSC results from {config.ldsc_combined_parquet_path}")

    if not config.ldsc_combined_parquet_path.exists():
        raise FileNotFoundError(
            f"Combined LDSC parquet not found at {config.ldsc_combined_parquet_path}. "
            "Please run Cauchy combination first."
        )

    ldsc_df = pd.read_parquet(config.ldsc_combined_parquet_path)
    if 'spot' in ldsc_df.columns:
        ldsc_df.set_index('spot', inplace=True)

    assert 'sample_name' in ldsc_df.columns, "LDSC combined results must have 'sample_name' column."

    traits = config.trait_name_list
    sample_names = sorted(ldsc_df['sample_name'].unique().tolist())

    return ldsc_df, traits, sample_names


def _load_coordinates(config: QuickModeConfig) -> Tuple[pd.DataFrame, bool, Optional[int]]:
    """
    Load spatial coordinates from concatenated adata.

    Returns:
        Tuple of (coords DataFrame, is_3d flag, z_axis index if 3D)
    """
    from gsMap.config import DatasetType

    logger.info(f"Loading coordinates from {config.concatenated_latent_adata_path}")

    adata_concat = ad.read_h5ad(config.concatenated_latent_adata_path, backed='r')

    assert config.spatial_key in adata_concat.obsm
    coords_data = adata_concat.obsm[config.spatial_key]
    sample_info = adata_concat.obs[['sample_name']]

    # Check if dataset type is 3D
    is_3d_type = (
        hasattr(config, 'dataset_type') and
        config.dataset_type == DatasetType.SPATIAL_3D
    )
    has_3d_coords = coords_data.shape[1] >= 3

    z_axis = None
    if is_3d_type and has_3d_coords:
        # True 3D coordinates
        logger.info("Detected 3D spatial coordinates")
        coords_3d = coords_data[:, :3]

        # Detect Z axis
        z_axis = _detect_z_axis(coords_3d, adata_concat.obs['sample_name'])

        # Reorder coordinates so Z is last
        reordered_coords, col_names = _reorder_coords_for_3d(coords_3d, z_axis)
        coords = pd.DataFrame(reordered_coords, columns=col_names, index=adata_concat.obs_names)

        # Also keep 2D coords for compatibility (use x and y from reordered)
        coords['sx'] = coords['3d_x']
        coords['sy'] = coords['3d_y']

    elif is_3d_type and not has_3d_coords:
        # 3D dataset type but only 2D coordinates - create pseudo Z axis
        logger.info("3D dataset type with 2D coordinates - creating pseudo Z axis based on sample_name")
        coords_2d = coords_data[:, :2]
        coords = pd.DataFrame(coords_2d, columns=['sx', 'sy'], index=adata_concat.obs_names)

        # Calculate the span for pseudo Z axis
        sx_span = coords['sx'].max() - coords['sx'].min()
        sy_span = coords['sy'].max() - coords['sy'].min()
        z_span = max(sx_span, sy_span)

        # Assign Z values based on sample order
        sample_names = adata_concat.obs['sample_name']
        unique_samples = sample_names.unique()
        n_samples = len(unique_samples)

        # Create evenly spaced Z values for each sample
        if n_samples > 1:
            z_values_per_sample = {
                sample: z_span * i / (n_samples - 1)
                for i, sample in enumerate(unique_samples)
            }
        else:
            z_values_per_sample = {unique_samples[0]: 0.0}

        # Assign Z values to each spot
        pseudo_z = sample_names.map(z_values_per_sample).values

        coords['3d_x'] = coords['sx']
        coords['3d_y'] = coords['sy']
        coords['3d_z'] = pseudo_z
        z_axis = 2  # Pseudo Z is always the last dimension

        logger.info(f"Created pseudo Z axis with span {z_span:.2f} for {n_samples} samples")

    else:
        # 2D coordinates (non-3D dataset type)
        coords_2d = coords_data[:, :2]
        coords = pd.DataFrame(coords_2d, columns=['sx', 'sy'], index=adata_concat.obs_names)

    coords = pd.concat([coords, sample_info], axis=1)

    # Return is_3d as True if dataset type is 3D (regardless of coord dimensions)
    return coords, is_3d_type, z_axis


def _load_gss_and_calculate_stats(
    config: QuickModeConfig,
    ldsc_df: pd.DataFrame,
    coords: pd.DataFrame,
    report_dir: Path,
    traits: List[str]
) -> Tuple[np.ndarray, ad.AnnData, pd.DataFrame]:
    """Load GSS data, calculate PCC, and return analysis results."""
    logger.info("Loading GSS data...")

    adata_gss = load_marker_scores_memmap_format(config)
    common_spots = np.intersect1d(adata_gss.obs_names, ldsc_df.index)
    logger.info(f"Common spots (gss & ldsc): {len(common_spots)}")
    assert len(common_spots) > 0, "No common spots found between GSS and LDSC results."

    # Stratified subsample if requested
    analysis_spots = common_spots
    if config.downsampling_n_spots and len(common_spots) > config.downsampling_n_spots:
        # Get sample names for stratified sampling
        sample_names = ldsc_df.loc[common_spots, 'sample_name']
        analysis_spots = _stratified_subsample(
            common_spots, sample_names, config.downsampling_n_spots
        )
        logger.info(f"Stratified subsampled to {len(analysis_spots)} spots for PCC calculation.")

    # Filter to high expression genes
    exp_frac = pd.read_parquet(config.mean_frac_path)
    high_expr_genes = exp_frac[exp_frac['frac'] > 0.01].index.tolist()
    logger.info(f"Using {len(high_expr_genes)} high expression genes for PCC calculation.")

    adata_gss_sub = adata_gss[analysis_spots, high_expr_genes]
    gss_matrix = adata_gss_sub.X

    # Save gene list
    gene_names = adata_gss_sub.var_names.tolist()
    pd.DataFrame({'gene': gene_names}).to_csv(report_dir / "gene_list.csv", index=False)

    # Pre-calculate GSS statistics
    logger.info("Pre-calculating GSS statistics...")
    if hasattr(gss_matrix, 'toarray'):
        gss_matrix = gss_matrix.toarray()
    gss_mean = gss_matrix.mean(axis=0).astype(np.float32)
    gss_centered = (gss_matrix - gss_mean).astype(np.float32)
    gss_ssq = np.sum(gss_centered ** 2, axis=0)

    # Calculate gene annotation stats
    adata_concat = ad.read_h5ad(config.concatenated_latent_adata_path, backed='r')
    anno_col = config.annotation_list[0]
    annotations = adata_concat.obs.loc[analysis_spots, anno_col]

    gss_df_temp = pd.DataFrame(gss_matrix, index=analysis_spots, columns=gene_names)
    grouped_gss = gss_df_temp.groupby(annotations).median()

    gene_stats_df = pd.DataFrame({
        'gene': grouped_gss.idxmax().index,
        'Annotation': grouped_gss.idxmax().values,
        'Median_GSS': grouped_gss.max().values
    })
    gene_stats_df.dropna(subset=['Median_GSS'], inplace=True)

    # Calculate PCC for each trait
    all_pcc = _calculate_pcc_for_traits(
        traits, ldsc_df, analysis_spots, gene_names,
        gss_centered, gss_ssq, gene_stats_df, config, report_dir
    )

    if all_pcc:
        pd.concat(all_pcc).to_csv(report_dir / "gene_trait_correlation.csv", index=False)

    return common_spots, adata_gss, gene_stats_df


def _calculate_pcc_for_traits(
    traits: List[str],
    ldsc_df: pd.DataFrame,
    analysis_spots: np.ndarray,
    gene_names: List[str],
    gss_centered: np.ndarray,
    gss_ssq: np.ndarray,
    gene_stats_df: pd.DataFrame,
    config: QuickModeConfig,
    report_dir: Path
) -> List[pd.DataFrame]:
    """Calculate PCC (Pearson Correlation Coefficient) for each trait."""

    def fast_corr(centered_matrix, ssq_matrix, vector):
        v_centered = vector - vector.mean()
        numerator = np.dot(v_centered, centered_matrix)
        denominator = np.sqrt(np.sum(v_centered ** 2) * ssq_matrix)
        return numerator / (denominator + 1e-12)

    all_pcc = []
    gene_plot_dir = report_dir / "gene_diagnostic_plots"
    gene_plot_dir.mkdir(exist_ok=True)

    for trait in traits:
        if trait not in ldsc_df.columns:
            logger.warning(f"Trait {trait} not found in LDSC combined results. Skipping PCC calculation.")
            continue

        logger.info(f"Processing PCC for trait: {trait}")
        logp_vec = ldsc_df.loc[analysis_spots, trait].values.astype(np.float32)

        assert not np.any(np.isnan(logp_vec)), f"NaN values found in LDSC results for trait {trait}."
        pccs = fast_corr(gss_centered, gss_ssq, logp_vec)

        trait_pcc = pd.DataFrame({
            'gene': gene_names,
            'PCC': pccs,
            'trait': trait
        })

        if gene_stats_df is not None:
            trait_pcc = trait_pcc.merge(gene_stats_df, on='gene', how='left')

        trait_pcc_sorted = trait_pcc.sort_values('PCC', ascending=False)

        diag_info_path = config.get_gene_diagnostic_info_save_path(trait)
        diag_info_path.parent.mkdir(parents=True, exist_ok=True)
        trait_pcc_sorted.to_csv(diag_info_path, index=False)

        all_pcc.append(trait_pcc_sorted)

    return all_pcc


# =============================================================================
# Data Preparation Functions
# =============================================================================

def _save_metadata(
    ldsc_df: pd.DataFrame,
    coords: pd.DataFrame,
    report_dir: Path
) -> pd.DataFrame:
    """Save metadata and coordinates to CSV."""
    logger.info("Saving metadata and coordinates...")

    common_indices = ldsc_df.index.intersection(coords.index)
    ldsc_subset = ldsc_df.loc[common_indices]
    cols_to_use = ldsc_subset.columns.difference(coords.columns)
    metadata = pd.concat([coords.loc[common_indices], ldsc_subset[cols_to_use]], axis=1)

    metadata.index.name = 'spot'
    metadata = metadata.reset_index()
    metadata = metadata.loc[:, ~metadata.columns.duplicated()]
    metadata.to_csv(report_dir / "spot_metadata.csv", index=False)

    return metadata


def _prepare_manhattan_data(
    config: QuickModeConfig,
    traits: List[str],
    report_dir: Path
) -> Dict:
    """Prepare Manhattan plot data for all traits."""
    logger.info("Preparing Manhattan data with filtering and gene mapping...")
    manhattan_dir = report_dir / "manhattan_data"
    manhattan_dir.mkdir(exist_ok=True)

    logger.info(f"Loading weights from {config.snp_gene_weight_adata_path}")
    weight_adata = ad.read_h5ad(config.snp_gene_weight_adata_path)

    pcc_file = report_dir / "gene_trait_correlation.csv"
    all_top_pcc = pd.read_csv(pcc_file) if pcc_file.exists() else None

    genes_file = report_dir / "gene_list.csv"
    gene_names_ref = pd.read_csv(genes_file)['gene'].tolist() if genes_file.exists() else []

    chrom_tick_positions = {}

    for trait in traits:
        try:
            chrom_tick_positions[trait] = _prepare_manhattan_for_trait(
                config, trait, weight_adata, all_top_pcc, gene_names_ref, manhattan_dir
            )
        except Exception as e:
            logger.warning(f"Failed to prepare Manhattan data for {trait}: {e}")

    return chrom_tick_positions


def _prepare_manhattan_for_trait(
    config: QuickModeConfig,
    trait: str,
    weight_adata: ad.AnnData,
    all_top_pcc: Optional[pd.DataFrame],
    gene_names_ref: List[str],
    manhattan_dir: Path
) -> Dict:
    """Prepare Manhattan data for a single trait."""
    from gsMap.utils.manhattan_plot import _ManhattanPlot

    sumstats_file = config.sumstats_config_dict.get(trait)
    if not sumstats_file or not Path(sumstats_file).exists():
        return {}

    logger.info(f"Processing Manhattan for {trait}...")
    gwas_data = load_gwas_data(sumstats_file)

    common_snps = weight_adata.obs_names[weight_adata.obs_names.isin(gwas_data["SNP"])]
    gwas_subset = gwas_data.set_index("SNP").loc[common_snps].reset_index()

    gwas_subset = gwas_subset.drop(columns=[c for c in ["CHR", "BP"] if c in gwas_subset.columns])
    gwas_subset = gwas_subset.set_index("SNP").join(weight_adata.obs[["CHR", "BP"]]).reset_index()

    snps2plot_ids = filter_snps(gwas_subset.sort_values("P"), SUBSAMPLE_SNP_NUMBER=50000)
    gwas_plot_data = gwas_subset[gwas_subset["SNP"].isin(snps2plot_ids)].copy()

    gwas_plot_data["CHR"] = pd.to_numeric(gwas_plot_data["CHR"], errors='coerce')
    gwas_plot_data["BP"] = pd.to_numeric(gwas_plot_data["BP"], errors='coerce')
    gwas_plot_data = gwas_plot_data.dropna(subset=["CHR", "BP"])

    # Gene assignment
    target_genes = [g for g in weight_adata.var_names if g in gene_names_ref and g != "unmapped"]
    if target_genes:
        sub_weight = weight_adata[gwas_plot_data["SNP"], target_genes].to_memory()
        weights_matrix = sub_weight.X

        if sp.issparse(weights_matrix):
            max_idx = np.array(weights_matrix.argmax(axis=1)).ravel()
            max_val = np.array(weights_matrix.max(axis=1).toarray()).ravel()
        else:
            max_idx = np.argmax(weights_matrix, axis=1)
            max_val = np.max(weights_matrix, axis=1)

        gene_map = np.where(max_val > 1, np.array(target_genes)[max_idx], "None")
        gwas_plot_data["GENE"] = gene_map

        if all_top_pcc is not None:
            top_n = getattr(config, 'top_corr_genes', 50)
            trait_pcc_df = all_top_pcc[all_top_pcc['trait'] == trait]
            trait_top_genes = trait_pcc_df.sort_values('PCC', ascending=False).head(top_n)['gene'].tolist()
            gwas_plot_data["is_top_pcc"] = gwas_plot_data["GENE"].isin(trait_top_genes)
        else:
            gwas_plot_data["is_top_pcc"] = False

    if "GENE" not in gwas_plot_data.columns:
        gwas_plot_data["GENE"] = "None"
    if "is_top_pcc" not in gwas_plot_data.columns:
        gwas_plot_data["is_top_pcc"] = False

    # Calculate cumulative positions
    chrom_ticks = {}
    try:
        mp_helper = _ManhattanPlot(gwas_plot_data)
        gwas_plot_data["BP_cum"] = mp_helper.data["POSITION"].values
        gwas_plot_data["CHR_INDEX"] = mp_helper.data["INDEX"].values

        chrom_groups = gwas_plot_data.groupby("CHR")["BP_cum"]
        chrom_ticks = {int(chrom): float(positions.median()) for chrom, positions in chrom_groups}
    except Exception as e:
        logger.warning(f"Failed to calculate Manhattan coordinates: {e}")
        gwas_plot_data["BP_cum"] = np.arange(len(gwas_plot_data))
        gwas_plot_data["CHR_INDEX"] = gwas_plot_data["CHR"] % 2

    gwas_plot_data.to_csv(manhattan_dir / f"{trait}_manhattan.csv", index=False)
    return chrom_ticks


# =============================================================================
# Plot Rendering Functions
# =============================================================================

def _render_static_plots(
    config: QuickModeConfig,
    metadata: pd.DataFrame,
    common_spots: np.ndarray,
    adata_gss: ad.AnnData,
    traits: List[str],
    sample_names: List[str],
    report_dir: Path
):
    """Pre-render static plots for LDSC, annotations, and gene diagnostics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    logger.info("Pre-rendering static plots for Traits and Annotations...")

    from .visualize import VisualizeRunner
    visualizer = VisualizeRunner(config)

    n_samples = len(sample_names)
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    obs_data = metadata.copy()
    if 'sample' not in obs_data.columns:
        obs_data['sample'] = obs_data['sample_name']

    # Render LDSC plots
    _render_ldsc_plots(visualizer, obs_data, traits, n_rows, n_cols, report_dir)

    # Render annotation plots
    _render_annotation_plots(visualizer, obs_data, config.annotation_list, sample_names, n_rows, n_cols, report_dir)

    # Render gene diagnostic plots
    _render_gene_diagnostic_plots(config, metadata, common_spots, adata_gss, n_rows, n_cols, report_dir)


def _render_ldsc_plots(
    visualizer,
    obs_data: pd.DataFrame,
    traits: List[str],
    n_rows: int,
    n_cols: int,
    report_dir: Path
):
    """Render LDSC spatial plots for all traits."""
    spatial_plot_dir = report_dir / "spatial_plots"
    spatial_plot_dir.mkdir(exist_ok=True)

    for trait in traits:
        logger.info(f"Pre-rendering LDSC plot for {trait}...")
        trait_plot_path = spatial_plot_dir / f"ldsc_{trait}.png"
        visualizer._create_single_trait_multi_sample_matplotlib_plot(
            obs_ldsc_merged=obs_data,
            trait_abbreviation=trait,
            output_png_path=trait_plot_path,
            n_rows=n_rows, n_cols=n_cols,
            subplot_width_inches=5.0
        )


def _render_annotation_plots(
    visualizer,
    obs_data: pd.DataFrame,
    annotations: List[str],
    sample_names: List[str],
    n_rows: int,
    n_cols: int,
    report_dir: Path
):
    """Render annotation spatial plots."""
    import matplotlib.pyplot as plt

    anno_dir = report_dir / "annotation_plots"
    anno_dir.mkdir(exist_ok=True)

    for anno in annotations:
        logger.info(f"Pre-rendering plot for annotation: {anno}...")
        anno_plot_path = anno_dir / f"anno_{anno}.png"
        fig = visualizer._create_multi_sample_annotation_plot(
            obs_ldsc_merged=obs_data,
            annotation=anno,
            sample_names_list=sample_names,
            output_dir=None,
            n_rows=n_rows, n_cols=n_cols,
            fig_width=5 * n_cols, fig_height=5 * n_rows
        )
        fig.savefig(anno_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def _render_gene_diagnostic_plots(
    config: QuickModeConfig,
    metadata: pd.DataFrame,
    common_spots: np.ndarray,
    adata_gss: ad.AnnData,
    n_rows: int,
    n_cols: int,
    report_dir: Path
):
    """Render gene expression and GSS diagnostic plots."""
    trait_pcc_file = report_dir / "gene_trait_correlation.csv"
    gene_plot_dir = report_dir / "gene_diagnostic_plots"
    gene_plot_dir.mkdir(exist_ok=True)

    if config.sample_h5ad_dict is None:
        config._process_h5ad_inputs()

    if not trait_pcc_file.exists() or not config.sample_h5ad_dict:
        logger.warning("Skipping gene diagnostic plots: missing PCC file or h5ad dict")
        return

    top_genes_df = pd.read_csv(trait_pcc_file)
    top_n = config.top_corr_genes
    all_top_genes = top_genes_df.groupby('trait').head(top_n)['gene'].unique().tolist()
    sample_names_sorted = sorted(config.sample_h5ad_dict.keys())

    logger.info(f"Preparing expression data for {len(sample_names_sorted)} samples...")

    # Load expression data for each sample
    exp_chunks = []
    for sample_name in sample_names_sorted:
        h5ad_path = config.sample_h5ad_dict[sample_name]
        logger.info(f"Preparing sample {sample_name} for diagnostic plots...")

        adata_rep = ad.read_h5ad(h5ad_path)
        suffix = f"|{sample_name}"
        if not str(adata_rep.obs_names[0]).endswith(suffix):
            adata_rep.obs_names = adata_rep.obs_names.astype(str) + suffix

        is_count, _ = setup_data_layer(adata_rep, config.data_layer, verbose=False)
        if is_count:
            sc.pp.normalize_total(adata_rep, target_sum=1e4)
            sc.pp.log1p(adata_rep)

        sample_metadata = metadata[metadata['sample_name'] == sample_name]
        sample_spots = sample_metadata[sample_metadata['spot'].isin(common_spots)]['spot'].values

        adata_rep_sub = adata_rep[sample_spots, all_top_genes].copy()
        adata_rep_sub.obs['sample_name'] = sample_name
        exp_chunks.append(adata_rep_sub)

        del adata_rep
        gc.collect()

    if not exp_chunks:
        logger.warning("No expression data found for gene diagnostic plots")
        return

    adata_exp_trait = ad.concat(exp_chunks, axis=0)
    adata_exp_trait = adata_exp_trait[common_spots, :].copy()
    adata_gss_trait = adata_gss[common_spots, all_top_genes].to_memory()

    # Build sample data cache
    sample_data_cache = {}
    for sample_name, group in metadata[metadata['spot'].isin(common_spots)].groupby('sample_name'):
        sample_data_cache[sample_name] = {
            'coords': group[['sx', 'sy']].values,
            'spots': group['spot'].values
        }

    # Build and execute render tasks
    tasks = _build_gene_plot_tasks(
        top_genes_df, top_n, all_top_genes, sample_names_sorted,
        sample_data_cache, adata_exp_trait, adata_gss_trait,
        n_rows, n_cols, gene_plot_dir
    )

    if tasks:
        logger.info(f"Rendering {len(tasks)} multi-sample diagnostic plots in parallel...")
        with ProcessPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
            results = list(executor.map(_render_multi_sample_gene_plot_task, tasks))

        errors = [r for r in results if r is not True]
        if errors:
            logger.warning(f"{len(errors)} plots failed to render")
            for err in errors[:3]:
                logger.warning(f"  Error: {err}")


def _build_gene_plot_tasks(
    top_genes_df: pd.DataFrame,
    top_n: int,
    all_top_genes: List[str],
    sample_names_sorted: List[str],
    sample_data_cache: Dict,
    adata_exp_trait: ad.AnnData,
    adata_gss_trait: ad.AnnData,
    n_rows: int,
    n_cols: int,
    gss_plot_dir: Path
) -> List[dict]:
    """Build list of gene plot rendering tasks."""
    tasks = []

    for trait, group in top_genes_df.groupby('trait'):
        top_group = group.sort_values('PCC', ascending=False).head(top_n)

        for _, row in top_group.iterrows():
            gene = row['gene']
            if gene not in all_top_genes:
                continue

            exp_sample_data = []
            gss_sample_data = []

            for sample_name in sample_names_sorted:
                cache = sample_data_cache.get(sample_name)
                if cache is None:
                    exp_sample_data.append((sample_name, None, None))
                    gss_sample_data.append((sample_name, None, None))
                    continue

                coords = cache['coords']
                spots = cache['spots']

                # Expression data
                exp_vals = adata_exp_trait[spots, gene].X
                if sp.issparse(exp_vals):
                    exp_vals = exp_vals.toarray()
                exp_vals = np.ravel(exp_vals)
                exp_sample_data.append((sample_name, coords, exp_vals))

                # GSS data
                gss_vals = adata_gss_trait[spots, gene].X
                if sp.issparse(gss_vals):
                    gss_vals = gss_vals.toarray()
                gss_vals = np.ravel(gss_vals)
                gss_sample_data.append((sample_name, coords, gss_vals))

            # Expression plot task
            if any(data[2] is not None for data in exp_sample_data):
                tasks.append({
                    'gene': gene, 'trait': trait, 'plot_type': 'exp',
                    'sample_data_list': exp_sample_data,
                    'output_path': gss_plot_dir / f"gene_{trait}_{gene}_exp.png",
                    'n_rows': n_rows, 'n_cols': n_cols,
                    'subplot_width': 4.0, 'dpi': 150
                })

            # GSS plot task
            if any(data[2] is not None for data in gss_sample_data):
                tasks.append({
                    'gene': gene, 'trait': trait, 'plot_type': 'gss',
                    'sample_data_list': gss_sample_data,
                    'output_path': gss_plot_dir / f"gene_{trait}_{gene}_gss.png",
                    'n_rows': n_rows, 'n_cols': n_cols,
                    'subplot_width': 4.0, 'dpi': 150
                })

    return tasks


# =============================================================================
# Results Collection Functions
# =============================================================================

def _collect_cauchy_results(
    config: QuickModeConfig,
    traits: List[str],
    report_dir: Path
):
    """Collect and save Cauchy combination results."""
    logger.info("Collecting Cauchy combination results...")
    all_cauchy = []

    for trait in traits:
        for annotation in config.annotation_list:
            # Aggregated results
            cauchy_file_all = config.get_cauchy_result_file(trait, annotation=annotation, all_samples=True)
            if cauchy_file_all.exists():
                try:
                    df = pd.read_csv(cauchy_file_all)
                    df['trait'] = trait
                    df['annotation_name'] = annotation
                    df['type'] = 'aggregated'
                    if 'sample_name' not in df.columns:
                        df['sample_name'] = 'All Samples'
                    all_cauchy.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load aggregated Cauchy result {cauchy_file_all}: {e}")

            # Per-sample results
            cauchy_file = config.get_cauchy_result_file(trait, annotation=annotation, all_samples=False)
            if cauchy_file.exists():
                try:
                    df = pd.read_csv(cauchy_file)
                    df['trait'] = trait
                    df['annotation_name'] = annotation
                    df['type'] = 'sample'
                    all_cauchy.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load sample Cauchy result {cauchy_file}: {e}")

    if all_cauchy:
        combined_cauchy = pd.concat(all_cauchy, ignore_index=True)
        if 'sample' in combined_cauchy.columns and 'sample_name' not in combined_cauchy.columns:
            combined_cauchy = combined_cauchy.rename(columns={'sample': 'sample_name'})

        cauchy_save_path = report_dir / "cauchy_results.csv"
        combined_cauchy.to_csv(cauchy_save_path, index=False)
        logger.info(f"Saved {len(combined_cauchy)} Cauchy results to {cauchy_save_path}")
    else:
        pd.DataFrame(columns=['trait', 'annotation_name', 'p_cauchy', 'type', 'sample_name']).to_csv(
            report_dir / "cauchy_results.csv", index=False
        )
        logger.warning("No Cauchy results found to save.")


def _save_report_metadata(
    config: QuickModeConfig,
    traits: List[str],
    sample_names: List[str],
    report_dir: Path,
    chrom_tick_positions: Optional[Dict] = None,
    umap_info: Optional[Dict] = None,
    is_3d: bool = False,
    spatial_3d_html: bool = False
):
    """Save report configuration metadata as JSON."""
    logger.info("Saving report configuration metadata...")
    report_meta = config.to_dict_with_paths_as_strings()

    report_meta['traits'] = traits
    report_meta['samples'] = sample_names
    report_meta['annotations'] = config.annotation_list
    report_meta['top_corr_genes'] = getattr(config, 'top_corr_genes', 50)

    # Add chromosome tick positions for Manhattan plot
    if chrom_tick_positions:
        report_meta['chrom_tick_positions'] = chrom_tick_positions

    # Add UMAP info
    if umap_info:
        report_meta['umap_info'] = umap_info

    # Add 3D visualization info
    report_meta['is_3d'] = is_3d
    report_meta['has_3d_widget'] = spatial_3d_html

    with open(report_dir / "report_meta.json", "w") as f:
        json.dump(report_meta, f)


def _copy_js_assets(report_dir: Path):
    """Copy bundled JS assets for local usage (no network required)."""
    logger.info("Copying JS assets for local usage...")
    js_lib_dir = report_dir / "js_lib"
    js_lib_dir.mkdir(exist_ok=True)

    # Copy bundled Alpine.js and Tailwind.js from static folder
    static_js_dir = Path(__file__).parent / "static" / "js_lib"
    if static_js_dir.exists():
        for js_file in static_js_dir.glob("*.js"):
            dest = js_lib_dir / js_file.name
            if not dest.exists():
                shutil.copy2(js_file, dest)
                logger.info(f"Copied {js_file.name}")

    # Copy plotly.min.js from installed plotly Python package
    plotly_js_src = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    plotly_dest = js_lib_dir / "plotly.min.js"
    if plotly_js_src.exists() and not plotly_dest.exists():
        shutil.copy2(plotly_js_src, plotly_dest)
        logger.info("Copied plotly.min.js from plotly package")

    # Copy anywidget resources
    import anywidget
    anywidget_src_dir = Path(anywidget.__file__).parent / "nbextension"
    anywidget_src_dir_js = anywidget_src_dir.glob("*.js")

    # should copy to report dir
    anywidget_index_file = anywidget_src_dir / "index.js"
    if anywidget_index_file.exists():
        shutil.copy2(anywidget_index_file, report_dir / "anywidget.js")
    # for js_file in anywidget_src_dir_js:
    #     dest = js_lib_dir / js_file.name
    #     if not dest.exists():
    #         shutil.copy2(js_file, dest)
    #         logger.info(f"Copied AnyWidget asset {js_file.name}")


# =============================================================================
# Main Entry Point
# =============================================================================

def prepare_report_data(config: QuickModeConfig) -> Path:
    """
    Prepare and aggregate data for the interactive report.
    Returns a directory containing the processed data.
    """
    import matplotlib
    matplotlib.use('Agg')

    report_dir = config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    is_3d = False
    try:
        # 1. Load LDSC results
        ldsc_df, traits, sample_names = _load_ldsc_results(config)

        # 2. Load coordinates (now returns is_3d flag and z_axis)
        coords, is_3d, z_axis = _load_coordinates(config)

        # 3. Load GSS data and calculate statistics
        common_spots, adata_gss, gene_stats_df = _load_gss_and_calculate_stats(
            config, ldsc_df, coords, report_dir, traits
        )

        # 4. Save metadata
        metadata = _save_metadata(ldsc_df, coords, report_dir)

        # 5. Prepare Manhattan data (returns chromosome tick positions)
        chrom_tick_positions = _prepare_manhattan_data(config, traits, report_dir)

        # 6. Pre-render static plots
        try:
            _render_static_plots(
                config, metadata, common_spots, adata_gss,
                traits, sample_names, report_dir
            )
        except Exception as e:
            logger.warning(f"Failed to pre-render static plots: {e}")
            import traceback
            traceback.print_exc()

        # 7. Collect Cauchy results
        _collect_cauchy_results(config, traits, report_dir)

        # 8. Prepare UMAP data from embeddings
        umap_info = None
        try:
            umap_info = _prepare_umap_data(config, metadata, report_dir)
        except Exception as e:
            logger.warning(f"Failed to prepare UMAP data: {e}")
            import traceback
            traceback.print_exc()

        # 9. Prepare 3D visualization if applicable
        spatial_3d_html = None
        if is_3d:
            try:
                spatial_3d_html = _prepare_3d_visualization(config, metadata, traits, report_dir)
            except Exception as e:
                logger.warning(f"Failed to prepare 3D visualization: {e}")
                import traceback
                traceback.print_exc()

        # 10. Save report metadata (including chromosome tick positions, UMAP info, and 3D info)
        _save_report_metadata(
            config, traits, sample_names, report_dir,
            chrom_tick_positions, umap_info,
            is_3d=is_3d, spatial_3d_html=spatial_3d_html is not None
        )

        # 11. Copy JS assets
        _copy_js_assets(report_dir)

        logger.info(f"Interactive report data prepared in {report_dir}")

    except FileNotFoundError as e:
        logger.error(str(e))

    return report_dir


# =============================================================================
# JS Export Functions
# =============================================================================

def export_data_as_js_modules(data_dir: Path):
    """
    Convert the CSV data in the report directory into JavaScript modules (.js files)
    that assign the data to window global variables.
    This allows loading data via <script> tags locally without CORS issues.
    """
    logger.info("Exporting data as JS modules...")
    js_data_dir = data_dir / "js_data"
    js_data_dir.mkdir(exist_ok=True)

    _export_metadata_js(data_dir, js_data_dir)
    _export_cauchy_js(data_dir, js_data_dir)
    _export_manhattan_js(data_dir, js_data_dir)
    _export_top_genes_pcc_js(data_dir, js_data_dir)
    _export_umap_js(data_dir, js_data_dir)
    _export_report_meta_js(data_dir, js_data_dir)

    logger.info(f"JS modules exported to {js_data_dir}")


def _export_metadata_js(data_dir: Path, js_data_dir: Path):
    """Export metadata CSV as JS module."""
    metadata_file = data_dir / "spot_metadata.csv"
    if metadata_file.exists():
        df = pd.read_csv(metadata_file)
        data_struct = {col: df[col].tolist() for col in df.columns}
        js_content = f"window.GSMAP_METADATA = {json.dumps(data_struct, separators=(',', ':'))};"
        with open(js_data_dir / "spot_metadata.js", "w", encoding='utf-8') as f:
            f.write(js_content)


def _export_cauchy_js(data_dir: Path, js_data_dir: Path):
    """Export Cauchy results as JS module."""
    cauchy_file = data_dir / "cauchy_results.csv"
    if cauchy_file.exists():
        df = pd.read_csv(cauchy_file)
        data_json = df.to_json(orient='records')
        js_content = f"window.GSMAP_CAUCHY = {data_json};"
        with open(js_data_dir / "cauchy_results.js", "w", encoding='utf-8') as f:
            f.write(js_content)


def _export_manhattan_js(data_dir: Path, js_data_dir: Path):
    """Export Manhattan data as JS modules (one per trait)."""
    manhattan_dir = data_dir / "manhattan_data"
    if not manhattan_dir.exists():
        return

    for csv_file in manhattan_dir.glob("*_manhattan.csv"):
        trait = csv_file.name.replace("_manhattan.csv", "")
        try:
            df = pd.read_csv(csv_file)
            if 'P' in df.columns and 'logp' not in df.columns:
                df['logp'] = -np.log10(df['P'] + 1e-300)

            data_struct = {
                'x': df['BP_cum'].tolist(),
                'y': df['logp'].tolist(),
                'gene': df['GENE'].fillna("").tolist(),
                'chr': df['CHR'].astype(int).tolist(),
                'snp': df['SNP'].tolist() if 'SNP' in df.columns else [],
                'is_top': df['is_top_pcc'].astype(int).tolist() if 'is_top_pcc' in df.columns else [],
                'bp': df['BP'].astype(int).tolist() if 'BP' in df.columns else []
            }

            json_str = json.dumps(data_struct, separators=(',', ':'))
            safe_trait = "".join(c if c.isalnum() else "_" for c in trait)
            js_content = f"window.GSMAP_MANHATTAN_{safe_trait} = {json_str};"

            with open(js_data_dir / f"manhattan_{trait}.js", "w", encoding='utf-8') as f:
                f.write(js_content)

        except Exception as e:
            logger.warning(f"Failed to export Manhattan JS for {trait}: {e}")


def _export_top_genes_pcc_js(data_dir: Path, js_data_dir: Path):
    """Export gene-trait correlation data as JS module."""
    pcc_file = data_dir / "gene_trait_correlation.csv"
    if pcc_file.exists():
        df = pd.read_csv(pcc_file)
        data_json = df.to_json(orient='records')
        js_content = f"window.GSMAP_GENE_TRAIT_CORRELATION = {data_json};"
        with open(js_data_dir / "gene_trait_correlation.js", "w", encoding='utf-8') as f:
            f.write(js_content)


def _export_report_meta_js(data_dir: Path, js_data_dir: Path):
    """Export report metadata as JS module."""
    meta_file = data_dir / "report_meta.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta = json.load(f)
        js_content = f"window.GSMAP_REPORT_META = {json.dumps(meta, separators=(',', ':'))};"
        with open(js_data_dir / "report_meta.js", "w", encoding='utf-8') as f:
            f.write(js_content)


def _export_umap_js(data_dir: Path, js_data_dir: Path):
    """Export UMAP data as JS module."""
    umap_file = data_dir / "umap_data.csv"
    if umap_file.exists():
        df = pd.read_csv(umap_file)

        # Build efficient columnar structure for ScatterGL
        data_struct = {
            'spot': df['spot'].tolist(),
            'sample_name': df['sample_name'].tolist(),
            'umap_cell_x': df['umap_cell_x'].tolist(),
            'umap_cell_y': df['umap_cell_y'].tolist(),
        }

        # Add niche UMAP if available
        if 'umap_niche_x' in df.columns:
            data_struct['umap_niche_x'] = df['umap_niche_x'].tolist()
            data_struct['umap_niche_y'] = df['umap_niche_y'].tolist()
            data_struct['has_niche'] = True
        else:
            data_struct['has_niche'] = False

        # Add all annotation columns (excluding known columns)
        known_cols = {'spot', 'sample_name', 'umap_cell_x', 'umap_cell_y', 'umap_niche_x', 'umap_niche_y'}
        annotation_cols = [c for c in df.columns if c not in known_cols]
        data_struct['annotation_columns'] = annotation_cols

        for col in annotation_cols:
            data_struct[col] = df[col].tolist()

        js_content = f"window.GSMAP_UMAP = {json.dumps(data_struct, separators=(',', ':'))};"
        with open(js_data_dir / "umap_data.js", "w", encoding='utf-8') as f:
            f.write(js_content)
        logger.info(f"Exported UMAP data with {len(df)} points")

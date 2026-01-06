"""
Report Data Preparation Module - Modern Version

This module prepares data for the Alpine.js + Tailwind CSS + Plotly.js report.
All data is exported as JS files with window global variables to bypass CORS restrictions
when opening index.html via file:// protocol.
"""

import logging
import json
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor

from gsMap.config import ReportConfig, QuickModeConfig
from gsMap.latent2gene.memmap_io import MemMapDense
from gsMap.report.visualize import load_ldsc, draw_scatter, estimate_point_size_for_plot
from gsMap.report.diagnosis import load_gwas_data, load_snp_gene_pairs, filter_snps, convert_z_to_p
from gsMap.spatial_ldsc.io import load_marker_scores_memmap_format
from gsMap.find_latent.st_process import setup_data_layer, normalize_for_analysis

logger = logging.getLogger(__name__)


def _render_gene_plot_task(task_data):
    """Parallel task to render Expression or GSS plots using draw_scatter."""
    try:
        import pandas as pd
        from pathlib import Path

        gene = task_data['gene']
        trait = task_data['trait']
        sample_name = task_data['sample_name']
        coords = task_data['coords']
        vals = task_data['values']
        output_path = Path(task_data['output_path'])
        title = task_data['title']
        point_size = task_data.get('point_size')
        width = task_data.get('width', 800)
        height = task_data.get('height', 800)

        # Construct DataFrame for draw_scatter
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

        # Save plot - Note: requires kaleido
        fig.write_image(str(output_path))
        return True
    except Exception as e:
        return str(e)


def _render_multi_sample_gene_plot_task(task_data):
    """
    Parallel task to render multi-sample Expression or GSS plots.
    Creates a grid of subplots showing gene values across all samples.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from scipy.spatial import KDTree
        import gc

        gene = task_data['gene']
        trait = task_data['trait']
        plot_type = task_data['plot_type']  # 'exp' or 'gss'
        sample_data_list = task_data['sample_data_list']  # List of (sample_name, coords, values)
        output_path = Path(task_data['output_path'])
        n_rows = task_data.get('n_rows', 2)
        n_cols = task_data.get('n_cols', 4)
        subplot_width = task_data.get('subplot_width', 4.0)
        dpi = task_data.get('dpi', 150)

        # Custom colormap (same as used in visualize.py)
        custom_colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
                        '#fee090', '#fdae61', '#f46d43', '#d73027']
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', custom_colors)

        # Calculate figure size
        fig_width = n_cols * subplot_width
        fig_height = n_rows * subplot_width

        fig = plt.figure(figsize=(fig_width, fig_height))
        title_text = f"{gene} - {'Expression' if plot_type == 'exp' else 'GSS'}"
        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)

        # Create grid
        grid_specs = fig.add_gridspec(nrows=n_rows, ncols=n_cols, wspace=0.1, hspace=0.15)

        # Calculate global min/max for consistent color scale
        all_values = np.concatenate([data[2] for data in sample_data_list if data[2] is not None and len(data[2]) > 0])
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
                # Estimate point size
                tree = KDTree(coords)
                distances, _ = tree.query(coords, k=min(2, len(coords)))
                if len(coords) > 1:
                    avg_dist = np.mean(distances[:, 1])
                else:
                    avg_dist = 1.0

                # Calculate point size based on subplot dimensions
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

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        gc.collect()

        return True
    except Exception as e:
        import traceback
        return f"{str(e)}\n{traceback.format_exc()}"


def prepare_report_data(config: QuickModeConfig):
    """
    Prepare and aggregate data for the interactive report.
    Returns a directory containing the processed data.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report_dir = config.get_report_dir("interactive")
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load combined LDSC results
    logger.info(f"Loading combined LDSC results from {config.ldsc_combined_parquet_path}")
    if not config.ldsc_combined_parquet_path.exists():
        logger.error(f"Combined LDSC parquet not found at {config.ldsc_combined_parquet_path}. Please run Cauchy combination first.")
        return report_dir

    ldsc_combined_df = pd.read_parquet(config.ldsc_combined_parquet_path)
    # Ensure 'spot' is the index
    if 'spot' in ldsc_combined_df.columns:
        ldsc_combined_df.set_index('spot', inplace=True)

    # Ensure ldsc_combined_df has a 'sample_name' column
    assert 'sample_name' in ldsc_combined_df.columns, "LDSC combined results must have 'sample_name' column."

    # Identify traits
    traits = config.trait_name_list
    sample_names = sorted(ldsc_combined_df['sample_name'].unique().tolist())

    # 2. Load Coordinates from concatenated_latent_adata_path
    logger.info(f"Loading coordinates from {config.concatenated_latent_adata_path}")

    # Load in backed mode to save memory, we only need obs and obsm
    adata_concat = ad.read_h5ad(config.concatenated_latent_adata_path, backed='r')

    # Check for spatial coordinates
    assert config.spatial_key in adata_concat.obsm
    coords_data = adata_concat.obsm[config.spatial_key][:, :2]
    coords = pd.DataFrame(coords_data, columns=['sx', 'sy'], index=adata_concat.obs_names)

    # Also get sample information if possible to ensure alignment
    sample_info = adata_concat.obs[['sample_name']]
    coords = pd.concat([coords, sample_info], axis=1)

    # 3. Load GSS data
    logger.info("Loading GSS data...")

    # Load GSS (Marker Scores)
    adata_gss = load_marker_scores_memmap_format(config)

    # We need spots that are present in both ldsc_combined_df and adata_gss
    common_spots = np.intersect1d(adata_gss.obs_names, ldsc_combined_df.index)
    logger.info(f"Common spots (gss & ldsc): {len(common_spots)}")
    assert len(common_spots) > 0, "No common spots found between GSS and LDSC results."

    all_pcc = []

    # Subsample if requested
    analysis_spots = common_spots
    if config.downsampling_n_spots and len(common_spots) > config.downsampling_n_spots:
        analysis_spots = np.random.choice(common_spots, config.downsampling_n_spots, replace=False)
        logger.info(f"Downsampled to {len(analysis_spots)} spots for PCC calculation.")

    exp_frac = pd.read_parquet(config.mean_frac_path,)
    high_expr_genes = exp_frac[exp_frac['frac'] > 0.01].index.tolist()
    logger.info(f"Using {len(high_expr_genes)} high expression genes for PCC calculation.")

    adata_gss_sub = adata_gss[analysis_spots, high_expr_genes]
    gss_matrix = adata_gss_sub.X

    # Save gene list for app regardless of PCC success
    gene_names = adata_gss_sub.var_names.tolist()
    genes_df = pd.DataFrame({'gene': gene_names})
    genes_df.to_csv(report_dir / "genes.csv", index=False)

    # Pre-calculate GSS statistics for speedup
    logger.info("Pre-calculating GSS statistics...")
    if hasattr(gss_matrix, 'toarray'):
        gss_matrix = gss_matrix.toarray()
    gss_mean = gss_matrix.mean(axis=0).astype(np.float32)
    gss_centered = (gss_matrix - gss_mean).astype(np.float32)
    gss_ssq = np.sum(gss_centered**2, axis=0)

    def fast_corr_with_centered_matrix(centered_matrix, ssq_matrix, vector):
        v_centered = vector - vector.mean()
        numerator = np.dot(v_centered, centered_matrix)
        denominator = np.sqrt(np.sum(v_centered**2) * ssq_matrix)
        return numerator / (denominator + 1e-12)

    # Calculate Annotation Stats (Annotation & Median_GSS)
    logger.info("Calculating gene annotation stats...")

    # Determine annotation column
    anno_col = config.annotation_list[0]
    annotations = adata_concat.obs.loc[analysis_spots, anno_col]

    gss_df_temp = pd.DataFrame(gss_matrix, index=analysis_spots, columns=gene_names)
    grouped_gss = gss_df_temp.groupby(annotations).median()

    max_annos = grouped_gss.idxmax()
    max_medians = grouped_gss.max()

    gene_stats_df = pd.DataFrame({
        'gene': max_annos.index,
        'Annotation': max_annos.values,
        'Median_GSS': max_medians.values
    })

    gene_stats_df.dropna(subset=['Median_GSS'], inplace=True)


    for trait in traits:
        if trait not in ldsc_combined_df.columns:
            logger.warning(f"Trait {trait} not found in LDSC combined results. Skipping PCC calculation.")
            continue

        logger.info(f"Processing PCC for trait: {trait}")
        logp_vec = ldsc_combined_df.loc[analysis_spots, trait].values.astype(np.float32)

        # Ensure no NaNs
        assert not np.any(np.isnan(logp_vec)), f"NaN values found in LDSC results for trait {trait}."
        pccs = fast_corr_with_centered_matrix(gss_centered, gss_ssq, logp_vec)

        trait_pcc = pd.DataFrame({
            'gene': gene_names,
            'PCC': pccs,
            'trait': trait
        })

        # Merge with gene stats if available
        if gene_stats_df is not None:
            trait_pcc = trait_pcc.merge(gene_stats_df, on='gene', how='left')

        # Save ALL gene diagnostic info (sorted by PCC)
        trait_pcc_sorted = trait_pcc.sort_values('PCC', ascending=False)

        # Organize into gss_plot folder
        gss_plot_dir = report_dir / "gss_plot"
        gss_plot_dir.mkdir(exist_ok=True)
        diag_info_path = config.get_gene_diagnostic_info_save_path(trait)
        # Ensure the path is relative to report_dir/gss_plot if we want it there
        # For now let's keep the config-specified path but ensures it exists
        diag_info_path.parent.mkdir(parents=True, exist_ok=True)
        trait_pcc_sorted.to_csv(diag_info_path, index=False)

        all_pcc.append(trait_pcc_sorted)

    if all_pcc:
        pd.concat(all_pcc).to_csv(report_dir / "top_genes_pcc.csv", index=False)


    # 5. Save metadata and coordinates
    logger.info("Saving metadata and coordinates...")
    common_indices = ldsc_combined_df.index.intersection(coords.index)
    # Avoid duplicate columns during concat
    ldsc_subset = ldsc_combined_df.loc[common_indices]
    cols_to_use = ldsc_subset.columns.difference(coords.columns)
    metadata = pd.concat([coords.loc[common_indices], ldsc_subset[cols_to_use]], axis=1)


    metadata.index.name = 'spot'
    metadata = metadata.reset_index()
    # Ensure no duplicate columns in metadata
    metadata = metadata.loc[:, ~metadata.columns.duplicated()]
    # Ensure 'sample' column exists for JS compatibility
    metadata.to_csv(report_dir / "metadata.csv", index=False)

    # 6. Prepare Manhattan Data
    logger.info("Preparing Manhattan data with filtering and gene mapping...")
    manhattan_dir = report_dir / "manhattan_plot"
    manhattan_dir.mkdir(exist_ok=True)

    # Assuming weight_adata must exist
    logger.info(f"Loading weights from {config.snp_gene_weight_adata_path}")
    weight_adata = ad.read_h5ad(config.snp_gene_weight_adata_path)

    # Get top genes df (saved in step 3)
    pcc_file = report_dir / "top_genes_pcc.csv"
    all_top_pcc = pd.read_csv(pcc_file) if pcc_file.exists() else None
    gene_names_ref = pd.read_csv(report_dir / "genes.csv")['gene'].tolist() if (report_dir / "genes.csv").exists() else []

    # Store chromosome tick positions for Manhattan plot
    chrom_tick_positions = {}

    try:
        for trait in traits:
            sumstats_file = config.sumstats_config_dict.get(trait)
            if sumstats_file and Path(sumstats_file).exists():
                logger.info(f"Processing Manhattan for {trait}...")
                gwas_data = load_gwas_data(sumstats_file)

                # Get common SNPs in the order of weight_adata
                common_snps = weight_adata.obs_names[weight_adata.obs_names.isin(gwas_data["SNP"])]

                # Filter gwas_data and reorder according to weight_adata
                gwas_subset = gwas_data.set_index("SNP").loc[common_snps].reset_index()

                # Join CHR and BP from weight_adata.obs
                # Drop existing CHR/BP in gwas_subset if any to avoid suffixes
                gwas_subset = gwas_subset.drop(columns=[c for c in ["CHR", "BP"] if c in gwas_subset.columns])
                gwas_subset = gwas_subset.set_index("SNP").join(weight_adata.obs[["CHR", "BP"]]).reset_index()

                # Subsample SNPs while prioritizing significant hits (filter_snps handles this)
                snps2plot_ids = filter_snps(gwas_subset.sort_values("P"), SUBSAMPLE_SNP_NUMBER=50000)

                # Filter gwas_subset by subsampled IDs but preserve the weight_adata order
                gwas_plot_data = gwas_subset[gwas_subset["SNP"].isin(snps2plot_ids)].copy()

                # Ensure CHR and BP are numeric
                gwas_plot_data["CHR"] = pd.to_numeric(gwas_plot_data["CHR"], errors='coerce')
                gwas_plot_data["BP"] = pd.to_numeric(gwas_plot_data["BP"], errors='coerce')

                gwas_plot_data = gwas_plot_data.dropna(subset=["CHR", "BP"])

                # Gene Assignment via Argmax Weight
                # Filter weight_adata to common genes and exclude unmapped
                target_genes = [g for g in weight_adata.var_names if g in gene_names_ref and g != "unmapped"]
                if target_genes:
                    sub_weight = weight_adata[gwas_plot_data["SNP"], target_genes].to_memory()
                    weights_matrix = sub_weight.X

                    import scipy.sparse as sp
                    if sp.issparse(weights_matrix):
                        max_idx = np.array(weights_matrix.argmax(axis=1)).ravel()
                        max_val = np.array(weights_matrix.max(axis=1).toarray()).ravel()
                    else:
                        max_idx = np.argmax(weights_matrix, axis=1)
                        max_val = np.max(weights_matrix, axis=1)

                    # Apply threshold > 1
                    gene_map = np.where(max_val > 1, np.array(target_genes)[max_idx], "None")
                    gwas_plot_data["GENE"] = gene_map

                    # Top PCC flag (using all_top_pcc)
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

                # Add Cumulative Position and Chromosome Index for Plotting
                from gsMap.utils.manhattan_plot import _ManhattanPlot
                try:
                    mp_helper = _ManhattanPlot(gwas_plot_data)
                    gwas_plot_data["BP_cum"] = mp_helper.data["POSITION"].values
                    gwas_plot_data["CHR_INDEX"] = mp_helper.data["INDEX"].values # For alternating colors

                    # Calculate chromosome tick positions (median position for each chrom)
                    chrom_groups = gwas_plot_data.groupby("CHR")["BP_cum"]
                    chrom_tick_positions[trait] = {
                        int(chrom): float(positions.median())
                        for chrom, positions in chrom_groups
                    }
                except Exception as e:
                    logger.warning(f"Failed to calculate Manhattan coordinates: {e}")
                    gwas_plot_data["BP_cum"] = np.arange(len(gwas_plot_data))
                    gwas_plot_data["CHR_INDEX"] = gwas_plot_data["CHR"] % 2

                gwas_plot_data.to_csv(manhattan_dir / f"{trait}_manhattan.csv", index=False)
    except Exception as e:
        logger.warning(f"Failed to prepare Manhattan data: {e}")
        import traceback
        traceback.print_exc()

    # 7. Pre-render Static Plots (LDSC results and Annotations)
    logger.info("Pre-rendering static plots for Traits and Annotations...")
    try:
        from .visualize import VisualizeRunner
        visualizer = VisualizeRunner(config)
        static_plots_dir = report_dir / "static_plots"
        static_plots_dir.mkdir(exist_ok=True)

        n_samples = len(sample_names)
        n_cols = min(4, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols

        # Ensure sample column exists if some code expects it
        obs_data = metadata.copy()
        if 'sample' not in obs_data.columns:
            obs_data['sample'] = obs_data['sample_name']

        # Pre-render Trait LDSC plots
        gsmap_plot_dir = report_dir / "gsmap_plot"
        gsmap_plot_dir.mkdir(exist_ok=True)
        for trait in traits:
            logger.info(f"Pre-rendering LDSC plot for {trait}...")
            trait_plot_path = gsmap_plot_dir / f"ldsc_{trait}.png"
            visualizer._create_single_trait_multi_sample_matplotlib_plot(
                obs_ldsc_merged=obs_data,
                trait_abbreviation=trait,
                output_png_path=trait_plot_path,
                n_rows=n_rows, n_cols=n_cols,
                subplot_width_inches=5.0
            )

        # Pre-render Annotation plots
        anno_dir = report_dir / "annotations" # Optional but following the spirit
        anno_dir.mkdir(exist_ok=True)
        for anno in config.annotation_list:
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
            import matplotlib.pyplot as plt
            plt.close(fig)

        # 7.3 Pre-render Top Gene Diagnostic plots (Multi-sample version)
        # Uses AnnCollection to efficiently handle multi-sample expression data
        # and caches top gene data in memory to speed up parallel rendering.
        trait_pcc_file = report_dir / "top_genes_pcc.csv"

        if config.sample_h5ad_dict is None:
            # load and validate h5ad inputs
            config._process_h5ad_inputs()

        if trait_pcc_file.exists() and config.sample_h5ad_dict:
            from anndata.experimental import AnnCollection
            import scipy.sparse as sp

            top_genes_df = pd.read_csv(trait_pcc_file)
            top_n = getattr(config, 'top_corr_genes', 50)
            
            # Identify unique top genes across all traits to cache in memory
            all_top_genes = top_genes_df.groupby('trait').head(top_n)['gene'].unique().tolist()
            sample_names_sorted = sorted(config.sample_h5ad_dict.keys())

            logger.info(f"Loading {len(sample_names_sorted)} samples using AnnCollection and caching top {len(all_top_genes)} genes...")

            # 1. Open all samples in backed mode
            backed_adatas = {}
            for sample_name in sample_names_sorted:
                h5ad_path = config.sample_h5ad_dict[sample_name]
                backed_adatas[sample_name] = ad.read_h5ad(h5ad_path, backed='r')

            # 2. Create AnnCollection for expression data
            collection = AnnCollection(backed_adatas, label='sample_name', join_vars='inner')
            # Only process genes that are present in both Expression and GSS data
            common_top_genes = [g for g in all_top_genes if g in collection.var_names and g in adata_gss.var_names]
            
            logger.info(f"Sub-selecting {len(common_top_genes)} common top genes into memory...")
            
            # Expression data aligned with common spots
            adata_exp_trait = collection[:, common_top_genes].to_memory()
            adata_exp_trait.obs_names = adata_exp_trait.obs_names.astype(str) + '|' + adata_exp_trait.obs['sample_name'].astype(str)
            # Filter to analysis spots and ensure order matches
            common_spots_in_exp = [s for s in common_spots if s in adata_exp_trait.obs_names]
            adata_exp_trait = adata_exp_trait[common_spots_in_exp, :].copy()

            # 3. Cache GSS data for the same top genes and spots
            adata_gss_trait = adata_gss[common_spots_in_exp, common_top_genes].to_memory()

            # 4. Fetch coordinates and spots once from metadata for all samples
            # This metadata mapping should match common_spots_in_exp exactly
            sample_data_cache = {}
            for sample_name, group in metadata[metadata['spot'].isin(common_spots_in_exp)].groupby('sample_name'):
                sample_data_cache[sample_name] = {
                    'coords': group[['sx', 'sy']].values,
                    'spots': group['spot'].values
                }

            # Calculate grid layout for report plots
            n_samples = len(sample_names_sorted)
            n_cols = min(4, n_samples)
            n_rows = (n_samples + n_cols - 1) // n_cols

            # 5. Collect tasks for multi-sample gene plots
            tasks = []
            for trait, group in top_genes_df.groupby('trait'):
                top_group = group.sort_values('PCC', ascending=False).head(top_n)

                for _, row in top_group.iterrows():
                    gene = row['gene']
                    if gene not in common_top_genes:
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

                        # Both are now perfectly aligned with spots in cache
                        # We just need to slice them efficiently
                        
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

            # 6. Cleanup and Execute
            # Close backed adatas before starting parallel tasks
            for ad_backed in backed_adatas.values():
                if hasattr(ad_backed, 'file'):
                    ad_backed.file.close()

            if tasks:
                logger.info(f"Rendering {len(tasks)} multi-sample diagnostic plots in parallel...")
                with ProcessPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
                    results = list(executor.map(_render_multi_sample_gene_plot_task, tasks))

                # Check for errors
                errors = [r for r in results if r is not True]
                if errors:
                    logger.warning(f"{len(errors)} plots failed to render")
                    for err in errors[:3]:
                        logger.warning(f"  Error: {err}")

    except Exception as e:
        logger.warning(f"Failed to pre-render static plots: {e}")
        import traceback
        traceback.print_exc()

    # 8. Collect Cauchy Results
    logger.info("Collecting Cauchy combination results...")
    all_cauchy = []
    for trait in traits:
        for annotation in config.annotation_list:
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

        # Save to report directory
        cauchy_save_path = report_dir / "all_cauchy.csv"
        combined_cauchy.to_csv(cauchy_save_path, index=False)
        logger.info(f"Saved {len(combined_cauchy)} Cauchy results to {cauchy_save_path}")
    else:
        # Create empty file with columns to avoid errors in JS
        pd.DataFrame(columns=['trait', 'annotation_name', 'p_cauchy', 'type', 'sample_name']).to_csv(report_dir / "all_cauchy.csv", index=False)
        logger.warning("No Cauchy results found to save.")

    # 9. Save additional metadata for JavaScript
    logger.info("Saving report configuration metadata...")
    report_meta = config.to_dict_with_paths_as_strings()

    # Explicitly ensure these are present and match the data used
    report_meta['traits'] = traits
    report_meta['samples'] = sample_names
    report_meta['annotations'] = config.annotation_list

    with open(report_dir / "report_meta.json", "w") as f:
        json.dump(report_meta, f)

    # 10. Download external JS assets for local usage (Fixes tracking/offline issues)
    logger.info("Downloading JS assets for local usage...")
    js_lib_dir = report_dir / "js_lib"
    js_lib_dir.mkdir(exist_ok=True)

    def _download_asset(url, filename):
        import urllib.request
        try:
            dest = js_lib_dir / filename
            if not dest.exists(): # Only download if not present to save time
                logger.info(f"Downloading {filename}...")
                with urllib.request.urlopen(url) as response, open(dest, 'wb') as out_file:
                    out_file.write(response.read())
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")

    _download_asset("https://cdn.jsdelivr.net/npm/alpinejs@3.13.3/dist/cdn.min.js", "alpine.min.js")
    _download_asset("https://cdn.tailwindcss.com", "tailwindcss.js")
    _download_asset("https://cdn.plot.ly/plotly-2.27.0.min.js", "plotly.min.js")

    logger.info(f"Interactive report data prepared in {report_dir}")
    return report_dir


def export_data_as_js_modules(data_dir: Path):
    """
    Convert the CSV data in the report directory into JavaScript modules (.js files)
    that assign the data to window global variables.
    This allows loading data via <script> tags locally without CORS issues.
    """
    logger.info("Exporting data as JS modules...")
    js_data_dir = data_dir / "js_data"
    js_data_dir.mkdir(exist_ok=True)

    # 1. Metadata (coordinates + trait values for gsMap plot)
    metadata_file = data_dir / "metadata.csv"
    if metadata_file.exists():
        df = pd.read_csv(metadata_file)
        # Column-oriented format for efficiency with large datasets
        data_struct = {col: df[col].tolist() for col in df.columns}
        js_content = f"window.GSMAP_METADATA = {json.dumps(data_struct, separators=(',', ':'))};"
        with open(js_data_dir / "metadata.js", "w", encoding='utf-8') as f:
            f.write(js_content)

    # 2. Cauchy Results
    cauchy_file = data_dir / "all_cauchy.csv"
    if cauchy_file.exists():
        df = pd.read_csv(cauchy_file)
        # Keep as records for table display (easier to work with in Alpine.js)
        data_json = df.to_json(orient='records')
        js_content = f"window.GSMAP_CAUCHY = {data_json};"
        with open(js_data_dir / "cauchy.js", "w", encoding='utf-8') as f:
            f.write(js_content)

    # 3. Manhattan Data - Column-oriented for Plotly efficiency
    manhattan_dir = data_dir / "manhattan_plot"
    if manhattan_dir.exists():
        for csv_file in manhattan_dir.glob("*_manhattan.csv"):
            trait = csv_file.name.replace("_manhattan.csv", "")
            try:
                df = pd.read_csv(csv_file)
                # Calculate -log10(P) if not already present
                if 'P' in df.columns and 'logp' not in df.columns:
                    df['logp'] = -np.log10(df['P'] + 1e-300) # prevent inf

                # Column-oriented data structure for ScatterGL
                data_struct = {
                    'x': df['BP_cum'].tolist(),
                    'y': df['logp'].tolist(),
                    'gene': df['GENE'].fillna("").tolist(),
                    'chr': df['CHR'].astype(int).tolist(),
                    'snp': df['SNP'].tolist() if 'SNP' in df.columns else [],
                    'is_top': df['is_top_pcc'].astype(int).tolist() if 'is_top_pcc' in df.columns else [],
                    'bp': df['BP'].astype(int).tolist() if 'BP' in df.columns else []
                }

                # Use separators to remove whitespace for smaller file size
                json_str = json.dumps(data_struct, separators=(',', ':'))

                # Safe variable name
                safe_trait = "".join(c if c.isalnum() else "_" for c in trait)
                js_content = f"window.GSMAP_MANHATTAN_{safe_trait} = {json_str};"

                with open(js_data_dir / f"manhattan_{trait}.js", "w", encoding='utf-8') as f:
                    f.write(js_content)

            except Exception as e:
                logger.warning(f"Failed to export Manhattan JS for {trait}: {e}")

    # 4. Gene List
    genes_file = data_dir / "genes.csv"
    if genes_file.exists():
        df = pd.read_csv(genes_file)
        genes = df['gene'].tolist()
        js_content = f"window.GSMAP_GENES = {json.dumps(genes)};"
        with open(js_data_dir / "genes.js", "w", encoding='utf-8') as f:
            f.write(js_content)

    # 5. Top Genes PCC (per-trait gene diagnostic table)
    pcc_file = data_dir / "top_genes_pcc.csv"
    if pcc_file.exists():
        df = pd.read_csv(pcc_file)
        # Keep as records for table display
        data_json = df.to_json(orient='records')
        js_content = f"window.GSMAP_TOP_GENES_PCC = {data_json};"
        with open(js_data_dir / "top_genes_pcc.js", "w", encoding='utf-8') as f:
            f.write(js_content)

    # 6. Report Metadata (traits, samples, annotations)
    meta_file = data_dir / "report_meta.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta = json.load(f)
        js_content = f"window.GSMAP_REPORT_META = {json.dumps(meta, separators=(',', ':'))};"
        with open(js_data_dir / "report_meta.js", "w", encoding='utf-8') as f:
            f.write(js_content)

    logger.info(f"JS modules exported to {js_data_dir}")

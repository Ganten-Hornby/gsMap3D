import logging
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from gsMap.config import ReportConfig, QuickModeConfig
from gsMap.latent2gene.memmap_io import MemMapDense
from gsMap.report.visualize import load_ldsc
from gsMap.report.diagnosis import load_gwas_data, load_snp_gene_pairs, filter_snps, convert_z_to_p
from gsMap.spatial_ldsc.io import load_marker_scores_memmap_format
from gsMap.find_latent.st_process import setup_data_layer, normalize_for_analysis

logger = logging.getLogger(__name__)

def _render_gene_plot_task(task_data):
    """Parallel task to render Expression or GSS plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path
        
        gene = task_data['gene']
        trait = task_data['trait']
        sample_name = task_data['sample_name']
        coords = task_data['coords']
        vals = task_data['values']
        output_path = Path(task_data['output_path'])
        title = task_data['title']
        cmap = task_data.get('cmap', 'viridis')
        
        fig, ax = plt.subplots(figsize=(5, 5))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, s=1.0, alpha=0.8)
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax)
        ax.axis('off')
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        return str(e)

def _extract_gene_expression_task(gene, adata_obs_names, adata_X, gene_idx):
    """Simple task to extract gene expression for a single gene (for parallelization)."""
    try:
        import numpy as np
        # Extract column from X
        # For sparse matrix, we need to be careful with indexing
        import scipy.sparse as sp
        if sp.issparse(adata_X):
            vals = adata_X[:, gene_idx].toarray().ravel()
        else:
            vals = np.ravel(adata_X[:, gene_idx])
            
        df = pd.DataFrame({
            'gene_val': vals,
            'spot': adata_obs_names,
            'gene': gene
        })
        return df
    except Exception:
        return None

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

    # Save gene list for app regardless of PCC success
    gene_names = adata_gss.var_names.tolist()
    genes_df = pd.DataFrame({'gene': gene_names})
    genes_df.to_csv(report_dir / "genes.csv", index=False)

    if len(common_spots) == 0:
        logger.error("No common spots found between GSS and LDSC results.")
    else:
        logger.info(f"Analyzing {len(common_spots)} common spots.")

        # Subsample if requested
        analysis_spots = common_spots
        if config.downsampling_n_spots and len(common_spots) > config.downsampling_n_spots:
            analysis_spots = np.random.choice(common_spots, config.downsampling_n_spots, replace=False)
            logger.info(f"Downsampled to {len(analysis_spots)} spots for PCC calculation.")

        adata_gss_sub = adata_gss[analysis_spots]
        gss_matrix = adata_gss_sub.X

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

        all_pcc = []
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

            # Save top 50 gene diagnostic info
            top_50 = trait_pcc.sort_values('PCC', ascending=False).head(50)

            # Organize into gss_plot folder
            gss_plot_dir = report_dir / "gss_plot"
            gss_plot_dir.mkdir(exist_ok=True)
            diag_info_path = config.get_gene_diagnostic_info_save_path(trait)
            # Ensure the path is relative to report_dir/gss_plot if we want it there
            # For now let's keep the config-specified path but ensures it exists
            diag_info_path.parent.mkdir(parents=True, exist_ok=True)
            top_50.to_csv(diag_info_path, index=False)

            all_pcc.append(top_50)

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
                snps2plot_ids = filter_snps(gwas_subset.sort_values("P"), SUBSAMPLE_SNP_NUMBER=100000)
                
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
                        trait_top_genes = all_top_pcc[all_top_pcc['trait'] == trait]['gene'].tolist()
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

        sample_names = sorted(metadata['sample_name'].unique().tolist())
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

        # 7.3 Pre-render Top Gene Diagnostic plots
        # Based on user request: Choose one representative sample and parallelize over genes
        trait_pcc_file = report_dir / "top_genes_pcc.csv"

        if config.sample_h5ad_dict is None:
            # load and validate h5ad inputs
            config._process_h5ad_inputs()

        if trait_pcc_file.exists() and config.sample_h5ad_dict:
            top_genes_df = pd.read_csv(trait_pcc_file)
            
            # Select first sample as representative
            rep_sample_name = list(config.sample_h5ad_dict.keys())[0]
            rep_h5ad_path = config.sample_h5ad_dict[rep_sample_name]
            
            logger.info(f"Preparing representative sample {rep_sample_name} for diagnostic plots...")
            adata_rep = ad.read_h5ad(rep_h5ad_path)
            # Align naming convention and slice directly (known subset)
            adata_rep.obs_names = adata_rep.obs_names.astype(str) + '|' + str(rep_sample_name)
            
            is_count, _ = setup_data_layer(adata_rep, config.data_layer, verbose=False)
            if is_count:
                adata_rep = normalize_for_analysis(adata_rep, is_count, preserve_raw=False)
            
            sample_metadata = metadata[metadata['sample_name'] == rep_sample_name]
            sample_spots = sample_metadata['spot'].values
            
            # Direct slicing using the prior that sample_spots is a subset
            adata_rep_sub = adata_rep[sample_spots].copy()
            coords_rep = adata_rep_sub.obsm[config.spatial_key or 'spatial'][:, :2]
            adata_gss_sample = adata_gss[sample_spots]
            
            tasks = []
            for _, row in top_genes_df.iterrows():
                trait = row['trait']
                gene = row['gene']
                
                # Expression task
                if gene in adata_rep_sub.var_names:
                    exp_vals = adata_rep_sub[:, gene].X
                    if hasattr(exp_vals, 'toarray'): exp_vals = exp_vals.toarray()
                    exp_vals = np.ravel(exp_vals)
                    
                    tasks.append({
                        'gene': gene, 'trait': trait, 'sample_name': rep_sample_name,
                        'coords': coords_rep, 'values': exp_vals,
                        'output_path': gss_plot_dir / f"gene_{trait}_{gene}_exp_{rep_sample_name}.png",
                        'title': f"{rep_sample_name} - {gene} Exp", 'cmap': 'viridis'
                    })
                
                # GSS task
                if gene in adata_gss_sample.var_names:
                    gss_vals = adata_gss_sample[:, gene].X
                    if hasattr(gss_vals, 'toarray'): gss_vals = gss_vals.toarray()
                    gss_vals = np.ravel(gss_vals)
                    
                    tasks.append({
                        'gene': gene, 'trait': trait, 'sample_name': rep_sample_name,
                        'coords': coords_rep, 'values': gss_vals,
                        'output_path': gss_plot_dir / f"gene_{trait}_{gene}_gss_{rep_sample_name}.png",
                        'title': f"{rep_sample_name} - {gene} GSS", 'cmap': 'plasma'
                    })

            if tasks:
                logger.info(f"Rendering {len(tasks)} diagnostic plots in parallel...")
                with ProcessPoolExecutor(max_workers=min(len(tasks), 8)) as executor:
                    executor.map(_render_gene_plot_task, tasks)

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
    logger.info(f"Interactive report data prepared in {report_dir}")
    return report_dir

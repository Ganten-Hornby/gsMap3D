import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import anndata as ad
from gsMap.config import ReportConfig
from gsMap.latent2gene.memmap_io import MemMapDense
from gsMap.report.visualize import load_ldsc
from gsMap.report.diagnosis import load_gwas_data, load_snp_gene_pairs, filter_snps, convert_z_to_p
from gsMap.spatial_ldsc.io import load_marker_scores_memmap_format

logger = logging.getLogger(__name__)

def prepare_report_data(config: ReportConfig):
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

    all_ldsc = pd.read_parquet(config.ldsc_combined_parquet_path)
    # Ensure 'spot' is the index
    if 'spot' in all_ldsc.columns:
        all_ldsc.set_index('spot', inplace=True)

    # Ensure all_ldsc has a 'sample_name' column
    if 'sample' in all_ldsc.columns and 'sample_name' not in all_ldsc.columns:
        all_ldsc = all_ldsc.rename(columns={'sample': 'sample_name'})
    
    # Identify traits
    traits = config.trait_name_list
    if not traits:
        # Fallback: identify numeric columns that are not annotations or metadata
        exclude_cols = config.annotation_list + ['sample_name', 'x', 'y', 'sx', 'sy']
        traits = [c for c in all_ldsc.columns if pd.api.types.is_numeric_dtype(all_ldsc[c]) and c not in exclude_cols]
    
    # 2. Load Coordinates from concatenated_latent_adata_path
    logger.info(f"Loading coordinates from {config.concatenated_latent_adata_path}")
    coords = None
    if config.concatenated_latent_adata_path.exists():
        # Load in backed mode to save memory, we only need obs and obsm
        adata_concat = ad.read_h5ad(config.concatenated_latent_adata_path, backed='r')
        
        # Check for spatial coordinates
        spatial_key = config.spatial_key or 'spatial'
        if spatial_key in adata_concat.obsm:
            coords_data = adata_concat.obsm[spatial_key][:, :2]
            coords = pd.DataFrame(coords_data, columns=['sx', 'sy'], index=adata_concat.obs_names)
        elif 'sx' in adata_concat.obs.columns and 'sy' in adata_concat.obs.columns:
            coords = adata_concat.obs[['sx', 'sy']]
        
        # Also get sample information if possible to ensure alignment
        if 'sample_name' in adata_concat.obs.columns:
            sample_info = adata_concat.obs[['sample_name']]
            coords = pd.concat([coords, sample_info], axis=1)

    if coords is None:
        logger.warning("Could not find spatial coordinates in concatenated latent adata.")

    # 3. Load Expression and GSS data
    logger.info("Loading Expression and GSS data...")
    adata_train_path = config.find_latent_metadata_path.parent / "training_adata.h5ad"
    adata_train = None
    if adata_train_path.exists():
        adata_train = ad.read_h5ad(adata_train_path)

    # Load GSS (Marker Scores)
    try:
        adata_gss = load_marker_scores_memmap_format(config)
        
        # We need spots that are present in both all_ldsc and adata_gss
        common_spots = np.intersect1d(adata_gss.obs_names, all_ldsc.index)
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
            
            def fast_corr_with_matrix(matrix, vector):
                m_mean = matrix.mean(axis=0)
                v_mean = vector.mean()
                m_centered = matrix - m_mean
                v_centered = vector - v_mean
                numerator = np.dot(v_centered, m_centered)
                denominator = np.sqrt(np.sum(v_centered**2) * np.sum(m_centered**2, axis=0))
                return numerator / (denominator + 1e-12)

            all_pcc = []
            for trait in traits:
                if trait not in all_ldsc.columns:
                    continue
                    
                logger.info(f"Processing PCC for trait: {trait}")
                logp_vec = all_ldsc.loc[analysis_spots, trait].values.astype(np.float32)
                
                # Ensure no NaNs
                valid_mask = ~np.isnan(logp_vec)
                if valid_mask.sum() < len(logp_vec):
                    logp_vec = logp_vec[valid_mask]
                    current_gss = gss_matrix[valid_mask]
                else:
                    current_gss = gss_matrix

                pccs = fast_corr_with_matrix(current_gss, logp_vec)
                
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
                
    except Exception as e:
        logger.error(f"Failed to process marker scores or calculate PCC: {e}")
        import traceback
        traceback.print_exc()

    # 5. Save metadata and coordinates
    logger.info("Saving metadata and coordinates...")
    if coords is not None:
        common_indices = all_ldsc.index.intersection(coords.index)
        # Avoid duplicate columns during concat
        ldsc_subset = all_ldsc.loc[common_indices]
        cols_to_use = ldsc_subset.columns.difference(coords.columns)
        metadata = pd.concat([coords.loc[common_indices], ldsc_subset[cols_to_use]], axis=1)
    else:
        metadata = all_ldsc.copy()
    
    metadata.index.name = 'spot'
    metadata = metadata.reset_index()
    # Ensure no duplicate columns in metadata
    metadata = metadata.loc[:, ~metadata.columns.duplicated()]
    # Ensure 'sample' column exists for JS compatibility
    if 'sample_name' in metadata.columns and 'sample' not in metadata.columns:
        metadata['sample'] = metadata['sample_name']
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
                from dataclasses import replace
                tmp_config = replace(config, sumstats_file=Path(sumstats_file), trait_name=trait)
                gwas_data = load_gwas_data(tmp_config)
                
                # Sort by P value for filter_snps
                gwas_data = gwas_data.sort_values("P")
                
                # Filter to common SNPs with weights
                common_snps_initial = weight_adata.obs_names.intersection(gwas_data["SNP"])
                gwas_subset = gwas_data[gwas_data["SNP"].isin(common_snps_initial)].copy()

                # Subsample SNPs
                snps2plot_ids = filter_snps(gwas_subset, SUBSAMPLE_SNP_NUMBER=100000)
                gwas_plot_data = gwas_subset[gwas_subset["SNP"].isin(snps2plot_ids)].copy()
                
                # Ensure CHR and BP are present and numeric
                if 'CHR' not in gwas_plot_data.columns and 'chrom' in gwas_plot_data.columns:
                    gwas_plot_data = gwas_plot_data.rename(columns={'chrom': 'CHR'})
                
                missing_cols = [c for c in ["CHR", "BP"] if c not in gwas_plot_data.columns]
                if missing_cols:
                    # Map 'chrom' from weight_adata to 'CHR' if needed
                    cols_to_pull = []
                    rename_map = {}
                    for mc in missing_cols:
                        if mc in weight_adata.obs.columns:
                            cols_to_pull.append(mc)
                        elif mc == 'CHR' and 'chrom' in weight_adata.obs.columns:
                            cols_to_pull.append('chrom')
                            rename_map['chrom'] = 'CHR'
                        elif mc == 'BP':
                            for alt in ['pos', 'position', 'basepair']:
                                if alt in weight_adata.obs.columns:
                                    cols_to_pull.append(alt)
                                    rename_map[alt] = 'BP'
                                    break
                    
                    if cols_to_pull:
                        shared_obs = weight_adata.obs.loc[gwas_plot_data["SNP"], cols_to_pull]
                        if rename_map:
                            shared_obs = shared_obs.rename(columns=rename_map)
                        gwas_plot_data = gwas_plot_data.set_index("SNP").join(shared_obs).reset_index()
                
                if "CHR" in gwas_plot_data.columns:
                    gwas_plot_data["CHR"] = pd.to_numeric(gwas_plot_data["CHR"], errors='coerce')
                else:
                    logger.warning(f"CHR column missing for {trait}. Using dummy values.")
                    gwas_plot_data["CHR"] = 1

                if "BP" in gwas_plot_data.columns:
                    gwas_plot_data["BP"] = pd.to_numeric(gwas_plot_data["BP"], errors='coerce')
                else:
                    logger.warning(f"BP column missing for {trait}. Using index as dummy BP.")
                    gwas_plot_data["BP"] = np.arange(len(gwas_plot_data))

                subset_to_drop = [c for c in ["CHR", "BP"] if c in gwas_plot_data.columns]
                gwas_plot_data = gwas_plot_data.dropna(subset=subset_to_drop)
                
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
                    gene_map = [target_genes[i] if v > 1 else "None" for i, v in zip(max_idx, max_val)]
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

        # 7.3 Pre-render Top Gene Diagnostic plots (Top 10 per trait)
        trait_pcc_file = report_dir / "top_genes_pcc.csv"
        if trait_pcc_file.exists() and (adata_train is not None or 'adata_gss' in locals()):
            top_genes_df = pd.read_csv(trait_pcc_file)
            
            for trait in traits:
                logger.info(f"Pre-rendering top gene plots for {trait}...")
                top_genes = top_genes_df[top_genes_df['trait'] == trait].head(10)['gene'].tolist()
                
                for gene in top_genes:
                    for sample in sample_names:
                        # Expression plot (from adata_train)
                        if adata_train is not None and gene in adata_train.var_names:
                            gene_sample_exp_path = gss_plot_dir / f"gene_{trait}_{gene}_exp_{sample}.png"
                            if not gene_sample_exp_path.exists():
                                try:
                                    sample_mask = adata_train.obs['sample_name'] == sample
                                    if sample_mask.any():
                                        sub = adata_train[sample_mask]
                                        fig, ax = plt.subplots(figsize=(5, 5))
                                        # Use manual scatter plot to avoid scanpy dependency/warnings
                                        spatial_coords = sub.obsm[config.spatial_key or 'spatial']
                                        
                                        # Correctly access gene expression
                                        gene_idx = sub.var_names.get_loc(gene)
                                        exp_vals = sub.X[:, gene_idx]
                                        if hasattr(exp_vals, 'toarray'): exp_vals = exp_vals.toarray()
                                        exp_vals = np.ravel(exp_vals)
                                        
                                        scatter = ax.scatter(
                                            spatial_coords[:, 0], spatial_coords[:, 1],
                                            c=exp_vals, cmap='viridis', s=1.0, alpha=0.8
                                        )
                                        ax.set_title(f"{sample} - {gene} Exp")
                                        plt.colorbar(scatter, ax=ax)
                                        ax.axis('off')
                                        fig.savefig(gene_sample_exp_path, dpi=100, bbox_inches='tight')
                                        plt.close(fig)
                                except Exception as e:
                                    logger.warning(f"Failed to plot gene exp {gene} for {sample}: {e}")

                        # GSS plot (from adata_gss)
                        if 'adata_gss' in locals() and gene in adata_gss.var_names:
                            gene_sample_gss_path = gss_plot_dir / f"gene_{trait}_{gene}_gss_{sample}.png"
                            if not gene_sample_gss_path.exists():
                                try:
                                    gene_idx = adata_gss.var_names.get_loc(gene)
                                    sample_spots = metadata[metadata['sample_name'] == sample]['spot'].values
                                    valid_spots = pd.Index(sample_spots).intersection(adata_gss.obs_names, sort=False).tolist()
                                    if valid_spots:
                                        row_indices = adata_gss.obs_names.get_indexer(valid_spots)
                                        # Handle both memmap and normal numpy/sparse X
                                        gss_vals = adata_gss.X[row_indices, gene_idx]
                                        if hasattr(gss_vals, 'toarray'): gss_vals = gss_vals.toarray()
                                        
                                        df_temp = metadata[metadata['spot'].isin(valid_spots)].copy()
                                        df_temp['GSS'] = np.ravel(gss_vals)
                                        
                                        fig, ax = plt.subplots(figsize=(5, 5))
                                        scatter = ax.scatter(df_temp['sx'], df_temp['sy'], 
                                                           c=df_temp['GSS'], cmap='plasma', s=5)
                                        ax.set_title(f"{sample} - {gene} GSS")
                                        plt.colorbar(scatter, ax=ax)
                                        ax.axis('off')
                                        fig.savefig(gene_sample_gss_path, dpi=100, bbox_inches='tight')
                                        plt.close(fig)
                                except Exception as e:
                                    logger.warning(f"Failed to plot gene GSS {gene} for {sample}: {e}")

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
        combined_cauchy.to_csv(report_dir / "all_cauchy.csv", index=False)

    logger.info(f"Interactive report data prepared in {report_dir}")
    return report_dir

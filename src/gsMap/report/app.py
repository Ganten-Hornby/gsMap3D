import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
import hvplot.pandas
from pathlib import Path
import yaml
import scanpy as sc
import plotly.graph_objects as go
import logging
from gsMap.config import ReportConfig
from gsMap.latent2gene.memmap_io import MemMapDense
from gsMap.report.visualize import load_ldsc, VisualizeRunner
from gsMap.utils.manhattan_plot import ManhattanPlot as PlotlyManhattanPlot

# Initialize panel
pn.extension('plotly', 'bokeh', 'tabulator')

logger = logging.getLogger(__name__)

class InteractiveReport:
    def __init__(self, data_dir: Path, config: ReportConfig = None, run_parameters: dict = None):
        self.data_dir = data_dir
        self.config = config
        self.run_parameters = run_parameters
        self.load_data()
        self.setup_widgets()
        
    def load_data(self):
        logger.info(f"Loading data from {self.data_dir}")
        self.all_cauchy = pd.read_feather(self.data_dir / "all_cauchy.feather")
        self.metadata = pd.read_feather(self.data_dir / "metadata.feather")
        pcc_path = self.data_dir / "top_genes_pcc.feather"
        if pcc_path.exists():
            self.top_genes_pcc = pd.read_feather(pcc_path)
        else:
            self.top_genes_pcc = pd.DataFrame(columns=['gene', 'PCC', 'trait'])
        self.genes = pd.read_feather(self.data_dir / "genes.feather")['gene'].tolist() if (self.data_dir / "genes.feather").exists() else []

        if self.run_parameters is None:
            summary_path = self.data_dir / "execution_summary.yaml"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    self.run_parameters = yaml.safe_load(f)
        
        self.mk_score_memmap = None
        self.adata_backed = None
        self.visualizer = None
        if self.config:
            if self.config.marker_scores_memmap_path.exists():
                from gsMap.spatial_ldsc.io import load_marker_scores_memmap_format
                self.adata_gss = load_marker_scores_memmap_format(self.config)
                self.mk_score_memmap = self.adata_gss.uns.get('memmap_manager')
            
            # Prefer training_adata for expression as it's more likely to contain full X
            adata_train_path = self.config.workdir / self.config.project_name / "find_latent_representations" / "training_adata.h5ad"
            adata_path = adata_train_path if adata_train_path.exists() else self.config.concatenated_latent_adata_path
            
            if adata_path.exists():
                # Load small files into memory to avoid fragile backed-mode indexing issues
                file_size_mb = adata_path.stat().st_size / (1024 * 1024)
                if file_size_mb < 1000: # 1GB threshold
                    logger.info(f"Loading AnnData from {adata_path} into memory ({file_size_mb:.1f} MB)")
                    self.adata_backed = sc.read_h5ad(adata_path)
                else:
                    logger.info(f"Loading AnnData from {adata_path} in backed mode ({file_size_mb:.1f} MB)")
                    self.adata_backed = sc.read_h5ad(adata_path, backed='r')
            
            self.visualizer = VisualizeRunner(self.config)
        
    def setup_widgets(self):
        traits = sorted(self.all_cauchy['trait'].unique().tolist())
        self.trait_select = pn.widgets.Select(name='Select Trait', options=traits)
        
        # Mapping trait selection to top 50 genes
        self.gene_select = pn.widgets.Select(name='Select Gene (Top 50)', options=[])
        
        @pn.depends(self.trait_select.param.value, watch=True)
        def _update_genes(trait):
            top_genes = self.top_genes_pcc[self.top_genes_pcc['trait'] == trait]['gene'].tolist()
            self.gene_select.options = top_genes
            if top_genes:
                self.gene_select.value = top_genes[0]
        
        _update_genes(self.trait_select.value)

        # Annotation selection
        annotations = self.config.annotation_list if self.config else sorted(self.all_cauchy['annotation_name'].unique().tolist())
        self.anno_select = pn.widgets.Select(name='Select Annotation', options=annotations)

        # Cauchy table toggle
        self.cauchy_type_toggle = pn.widgets.RadioButtonGroup(
            name='View Level', options=['Sample Level', 'Aggregated'], value='Aggregated'
        )

    def get_manhattan_view(self):
        @pn.depends(self.trait_select.param.value, self.gene_select.param.value)
        def manhattan_plot(trait, gene):
            manhattan_file = self.data_dir / f"{trait}_manhattan.feather"
            if not manhattan_file.exists():
                return pn.Column(
                    pn.pane.Markdown(f"### Manhattan Plot - {trait}"),
                    pn.pane.Alert(f"Manhattan data not found for trait: {trait}. This is expected in Quick Mode as SNP-gene mapping is not performed by default.", alert_type="warning"),
                    pn.pane.Markdown("To generate Manhattan plots, please run the full pipeline including `generate_ldscore` step.")
                )
            
            df = pd.read_feather(manhattan_file)
            highlight_list = [gene] if (gene and gene in df['GENE'].values) else None
            
            fig = PlotlyManhattanPlot(
                dataframe=df, title=f"Manhattan Plot - {trait}",
                point_size=3, highlight_gene_list=highlight_list
            )
            return pn.pane.Plotly(fig, sizing_mode='stretch_both', min_height=500)
        
        return pn.Column(manhattan_plot)

    def get_annotation_view(self):
        @pn.depends(self.anno_select.param.value)
        def anno_plot(annotation):
            # Try to load pre-rendered plot first
            static_path = self.data_dir / "static_plots" / f"anno_{annotation}.png"
            if static_path.exists():
                return pn.pane.PNG(str(static_path), sizing_mode='stretch_width')
            
            if not self.visualizer:
                return pn.pane.Markdown("Visualizer not initialized and pre-rendered plot not found.")
            
            samples = sorted(self.metadata['sample'].unique().tolist())
            n_samples = len(samples)
            n_cols = min(4, n_samples)
            n_rows = (n_samples + n_cols - 1) // n_cols

            # Ensure sample_name exists for visualizer
            obs_data = self.metadata.copy()
            if 'sample_name' not in obs_data.columns:
                obs_data['sample_name'] = obs_data['sample']

            fig = self.visualizer._create_multi_sample_annotation_plot(
                obs_ldsc_merged=obs_data,
                annotation=annotation,
                sample_names_list=samples,
                output_dir=None, n_rows=n_rows, n_cols=n_cols,
                fig_width=5 * n_cols, fig_height=5 * n_rows
            )
            return pn.pane.Matplotlib(fig, sizing_mode='stretch_width')
        
        return pn.Column(anno_plot)

    def get_ldsc_view(self):
        @pn.depends(self.trait_select.param.value)
        def ldsc_plot(trait):
            # Try to load pre-rendered plot first
            static_path = self.data_dir / "static_plots" / f"ldsc_{trait}.png"
            if static_path.exists():
                return pn.pane.PNG(str(static_path), sizing_mode='stretch_width')

            if not self.visualizer:
                return pn.pane.Markdown("Visualizer not initialized and pre-rendered plot not found.")
            
            samples = sorted(self.metadata['sample'].unique().tolist())
            n_samples = len(samples)
            n_cols = min(4, n_samples)
            n_rows = (n_samples + n_cols - 1) // n_cols

            obs_data = self.metadata.copy()
            if 'sample_name' not in obs_data.columns:
                obs_data['sample_name'] = obs_data['sample']

            fig = self.visualizer._create_single_trait_multi_sample_matplotlib_plot(
                obs_ldsc_merged=obs_data,
                trait_abbreviation=trait,
                n_rows=n_rows, n_cols=n_cols,
                subplot_width_inches=5.0
            )
            return pn.pane.Matplotlib(fig, sizing_mode='stretch_width')
        
        return pn.Column(ldsc_plot)

    def get_correlation_view(self):
        """View for trait-annotation correlations (Cauchy Heatmap)"""
        @pn.depends(self.anno_select.param.value)
        def heatmap_view(anno):
            heatmap_df = self.all_cauchy[self.all_cauchy['annotation_name'] == anno]
            if heatmap_df.empty:
                return pn.pane.Markdown(f"No Cauchy results found for annotation: {anno}")
            
            # Pivot: Annotation Value vs Trait
            plot_df = heatmap_df[heatmap_df['type'] == 'aggregated'].pivot_table(
                index='annotation', columns='trait', values='p_cauchy', aggfunc='mean'
            )
            if plot_df.empty:
                 plot_df = heatmap_df.pivot_table(index='annotation', columns='trait', values='p_cauchy', aggfunc='mean')

            if not self.visualizer:
                 return pn.pane.Markdown("Visualizer not initialized.")
                 
            fig = self.visualizer._plot_p_cauchy_heatmap(
                -np.log10(plot_df + 1e-300), title=f"Cauchy Heatmap: {anno}"
            )
            return pn.pane.Plotly(fig, sizing_mode='stretch_both', min_height=600)

        return pn.Column(
            "### Trait-Annotation Enrichment Heatmap",
            "This heatmap shows the $-\\log_{10}(\\text{Cauchy } P\\text{-value})$ for each annotation category across all traits.",
            heatmap_view
        )

    def get_summary_view(self):
        """Creates a summary view of the results."""
        # Top significant annotations across all traits
        top_cauchy = self.all_cauchy[self.all_cauchy['type'] == 'aggregated'].sort_values('p_cauchy', ascending=True).head(20)
        
        summary_md = f"## gsMap Analysis Summary: {self.config.project_name if self.config else ''}\n\n"
        if self.run_parameters:
            summary_md += "### Execution Information\n"
            summary_md += f"- **Project Name**: {self.run_parameters.get('Project Name', 'N/A')}\n"
            summary_md += f"- **Traits Analyzed**: {', '.join(self.run_parameters.get('Traits', []))}\n"
            summary_md += f"- **Total Processing Time**: {self.run_parameters.get('Spending Time', 'N/A')}\n"
            
            # Add worker info
            workers = self.run_parameters.get('Worker Configuration', {})
            if workers:
                summary_md += "- **Compute Resources**: " + ", ".join([f"{k}: {v}" for k, v in workers.items()]) + "\n"
            
            summary_md += "\n"
        
        summary_md += "### Top Significant Results (Aggregated)\n"
        summary_md += "The table below shows the most significant tissue/cell-type annotations identified across all analyzed traits.\n"
        
        table = pn.widgets.Tabulator(top_cauchy[['trait', 'annotation_name', 'annotation', 'p_median', 'p_cauchy']], 
                                    sizing_mode='stretch_width', disabled=True, 
                                    titles={'p_cauchy': 'Cauchy P-value', 'p_median': 'Median P-value'})
        
        return pn.Column(pn.pane.Markdown(summary_md, sizing_mode='stretch_width'), table)

    def get_gene_diagnostic_view(self):
        @pn.depends(self.trait_select.param.value)
        def gene_table(trait):
            trait_pcc = self.top_genes_pcc[self.top_genes_pcc['trait'] == trait].copy()
            trait_pcc = trait_pcc[['gene', 'PCC']].sort_values('PCC', ascending=False).head(50)
            table = pn.widgets.Tabulator(trait_pcc, name='Top 50 Gene Diagnostic Info', 
                                        sizing_mode='stretch_width', selection=[0])
            
            @pn.depends(table.param.selection, watch=True)
            def _update_gene_from_table(selection):
                if selection:
                    selected_gene = table.value.iloc[selection[0]]['gene']
                    self.gene_select.value = selected_gene
            
            return table

        @pn.depends(self.trait_select.param.value, self.gene_select.param.value)
        def gene_view(trait, gene):
            if not gene:
                return pn.pane.Markdown("Please select a gene from the dropdown or table.")
                
            samples = sorted(self.metadata['sample'].unique().tolist())
            plots = []
            
            # Spatial Plots
            for sample in samples:
                sample_metadata = self.metadata[self.metadata['sample'] == sample]
                sample_spots = sample_metadata['spot'].values
                
                row = []
                # Expression
                if self.adata_backed is not None and gene in self.adata_backed.var_names:
                    # Filter sample_spots to those present in adata_backed
                    valid_spots = [s for s in sample_spots if s in self.adata_backed.obs_names]
                    if valid_spots:
                        # Use obs_df for robust data retrieval from backed/sparse AnnData
                        # We request only the needed gene to be efficient
                        df_exp_gene = sc.get.obs_df(self.adata_backed, keys=[gene], use_raw=False)
                        exp_vals = df_exp_gene.loc[valid_spots, gene].values
                        
                        df_exp = sample_metadata[sample_metadata['spot'].isin(valid_spots)].copy()
                        df_exp['Expression'] = exp_vals.astype(np.float32)
                    
                        p1 = df_exp.hvplot.scatter(x='sx', y='sy', c='Expression', cmap='viridis', 
                                                  title=f"{sample} - {gene} Exp", width=350, height=300, colorbar=True)
                        row.append(p1.opts(xaxis=None, yaxis=None))

                # GSS
                if self.mk_score_memmap and gene in self.genes:
                    gene_idx = self.genes.index(gene)
                    # GSS memmap also needs original indices. 
                    # If mk_score_memmap matches concatenated_latent_adata, we need integer indices of those spots.
                    # Let's find integer indices of valid_spots in the full adata_backed
                    if 'valid_spots' in locals() and valid_spots:
                        # This matches the indices from adata_backed
                        full_indices = self.adata_backed.obs_names.get_indexer(valid_spots)
                        gss_vals = self.mk_score_memmap.data[full_indices, gene_idx].astype(np.float32)
                        
                        df_gss = sample_metadata[sample_metadata['spot'].isin(valid_spots)].copy()
                        df_gss['GSS'] = gss_vals
                        
                        p2 = df_gss.hvplot.scatter(x='sx', y='sy', c='GSS', cmap='plasma', 
                                                  title=f"{sample} - {gene} GSS", width=350, height=300, colorbar=True)
                        row.append(p2.opts(xaxis=None, yaxis=None))
                
                if row:
                    plots.append(pn.Row(*row))
            
            return pn.Column(*plots)
            
        return pn.Column(gene_table, "---", gene_view)

    def get_cauchy_view(self):
        self.cauchy_type_toggle = pn.widgets.RadioButtonGroup(
            name='View Level', options=['Aggregated Only', 'Sample Level Only', 'Show Both'], value='Show Both'
        )
        
        cauchy_table = pn.widgets.Tabulator(
            pagination='remote', page_size=15, sizing_mode='stretch_width',
            selectable=True, width=800
        )

        @pn.depends(self.trait_select.param.value, self.cauchy_type_toggle.param.value, watch=True)
        def _update_cauchy_table(trait, view_type):
            df = self.all_cauchy[self.all_cauchy['trait'] == trait].copy()
            
            # Helper to get sorting order from Aggregated p_median
            agg_order = df[df['type'] == 'aggregated'].sort_values('p_median', ascending=False)
            anno_order = agg_order['annotation'].tolist()
            
            # Apply filtering
            if view_type == 'Aggregated Only':
                df = df[df['type'] == 'aggregated'].sort_values('p_median', ascending=False)
            elif view_type == 'Sample Level Only':
                df = df[df['type'] == 'sample'].sort_values('p_median', ascending=False)
            else:
                # Hierarchical sort
                sorter = {anno: i for i, anno in enumerate(anno_order)}
                df['anno_sort'] = df['annotation'].map(sorter)
                df['type_sort'] = df['type'].map({'aggregated': 0, 'sample': 1})
                
                # Sort by annotation order (from agg), then type (agg first), then sample name
                df = df.sort_values(['anno_sort', 'type_sort', 'sample'])
                df = df.drop(columns=['anno_sort', 'type_sort'])
            
            cauchy_table.value = df
        
        _update_cauchy_table(self.trait_select.value, self.cauchy_type_toggle.value)

        @pn.depends(cauchy_table.param.selection, self.anno_select.param.value)
        def cauchy_heatmap_view(selection, anno):
            if not self.visualizer:
                return pn.pane.Markdown("Visualizer not initialized.")
            
            selected_anno = anno
            if selection:
                # The table might be filtered, so we get the value from the current table content
                selected_anno = cauchy_table.value.iloc[selection[0]]['annotation_name']
            
            heatmap_df = self.all_cauchy[self.all_cauchy['annotation_name'] == selected_anno]
            if heatmap_df.empty:
                return pn.pane.Markdown("No data for heatmap.")
            
            # Pivot: Annotation Value vs Trait
            # We use aggregated results for the heatmap to avoid cluttering if multiple samples exist
            plot_df = heatmap_df[heatmap_df['type'] == 'aggregated'].pivot_table(
                index='annotation', columns='trait', values='p_cauchy', aggfunc='mean'
            )
            if plot_df.empty:
                 # fallback to all data if no aggregated found
                 plot_df = heatmap_df.pivot_table(index='annotation', columns='trait', values='p_cauchy', aggfunc='mean')

            fig = self.visualizer._plot_p_cauchy_heatmap(
                -np.log10(plot_df + 1e-300), title=f"Cauchy Heatmap: {selected_anno}"
            )
            return pn.pane.Plotly(fig, sizing_mode='stretch_both', min_height=600)

        return pn.Column(
            pn.Row("View Style:", self.cauchy_type_toggle),
            cauchy_table,
            "---",
            cauchy_heatmap_view
        )

    def get_running_info(self):
        info_md = ""
        if self.run_parameters:
            info_md += "### Quick Mode Execution Summary\n"
            for k, v in self.run_parameters.items():
                if isinstance(v, dict):
                    info_md += f"- **{k}**:\n"
                    for sub_k, sub_v in v.items():
                        info_md += f"  - **{sub_k}**: {sub_v}\n"
                else:
                    info_md += f"- **{k}**: {v}\n"
        
        if self.config:
            info_md += "### Report Configuration\n"
            info_md += f"- **Project Name**: {self.config.project_name}\n"
            info_md += f"- **Annotations**: {self.config.annotation_list}\n"
        
        return pn.Card(pn.pane.Markdown(info_md if info_md else "No info"), 
                       title="Running Information", collapsed=True, sizing_mode='stretch_width')

    def create_layout(self, use_template=True):
        sidebar = pn.Column(
            "## gsMap Report Controls",
            self.trait_select,
            self.anno_select,
            self.gene_select,
            width=300
        )
        
        tabs = pn.Tabs(
            ("Summary", self.get_summary_view()),
            ("Manhattan", self.get_manhattan_view()),
            ("Annotations", self.get_annotation_view()),
            ("LDSC Results", self.get_ldsc_view()),
            ("Gene/GSS Diagnostic", self.get_gene_diagnostic_view()),
            ("Cauchy Combination", self.get_cauchy_view()),
            dynamic=True
        )
        
        if not use_template:
            return pn.Row(sidebar, tabs)
            
        return pn.template.FastListTemplate(
            title=f"gsMap Report - {self.config.project_name if self.config else ''}",
            sidebar=[sidebar, self.get_running_info()],
            main=[tabs]
        )

def launch_report(data_dir: Path, port=5006, show=True, config: ReportConfig = None, run_parameters: dict = None):
    report = InteractiveReport(data_dir, config=config, run_parameters=run_parameters)
    app = report.create_layout()
    if show:
        app.show(port=port)
    else:
        return app

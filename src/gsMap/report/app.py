import panel as pn
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews import opts
import hvplot.pandas
from pathlib import Path
import yaml
import plotly.graph_objects as go
import logging

# Handle optional dependencies for WASM/Pyodide compatibility
try:
    import scanpy as sc
except ImportError:
    sc = None

try:
    from gsMap.config import ReportConfig
except ImportError:
    ReportConfig = None

# Fallback for VisualizeRunner if gsMap is not available
try:
    from gsMap.report.visualize import VisualizeRunner
except ImportError:
    class VisualizeRunner:
        def __init__(self, config=None): pass
        def _plot_p_cauchy_heatmap(self, *args, **kwargs):
            return go.Figure()
        def _create_multi_sample_annotation_plot(self, *args, **kwargs):
            return None
        def _create_single_trait_multi_sample_matplotlib_plot(self, *args, **kwargs):
            return None

# Helper for data loading
def safe_read_csv(data_dir, filename):
    path = Path(data_dir) / filename
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

# Local ManhattanPlot implementation for Pyodide compatibility
def _get_hover_text(df, snpname=None, genename=None, annotationname=None):
    hover_text = ""
    if snpname is not None and snpname in df.columns:
        hover_text = "SNP: " + df[snpname].astype(str)
    if genename is not None and genename in df.columns:
        hover_text = hover_text + "<br>GENE: " + df[genename].astype(str)
    if annotationname is not None and annotationname in df.columns:
        hover_text = hover_text + "<br>" + df[annotationname].astype(str)
    return hover_text

class PlotlyManhattanPlot:
    def __init__(self, dataframe, title="Manhattan Plot", point_size=3, highlight_gene_list=None, **kwargs):
        self.df = dataframe.copy()
        self.title = title
        self.point_size = point_size
        self.highlight_gene_list = highlight_gene_list
        
    def _prepare_data(self):
        df = self.df
        # Basic requirements for Manhattan
        for col in ['CHR', 'BP', 'P']:
            if col not in df.columns:
                # Try to map common names if missing
                if col == 'CHR' and 'chrom' in df.columns: df['CHR'] = df['chrom']
                elif col == 'BP' and 'pos' in df.columns: df['BP'] = df['pos']
                elif col == 'P' and 'p_value' in df.columns: df['P'] = df['p_value']
        
        df = df.dropna(subset=['CHR', 'BP', 'P'])
        df['CHR'] = pd.to_numeric(df['CHR'], errors='coerce')
        df['BP'] = pd.to_numeric(df['BP'], errors='coerce')
        df['P'] = pd.to_numeric(df['P'], errors='coerce')
        df = df.dropna(subset=['CHR', 'BP', 'P']).sort_values(['CHR', 'BP'])
        
        # Calculate cumulative position
        df['pos'] = 0
        last_pos = 0
        ticks = []
        tick_labels = []
        for chrom in sorted(df['CHR'].unique()):
            mask = df['CHR'] == chrom
            df.loc[mask, 'pos'] = df.loc[mask, 'BP'] + last_pos
            ticks.append(last_pos + df.loc[mask, 'BP'].max() / 2)
            tick_labels.append(str(int(chrom)))
            last_pos += df.loc[mask, 'BP'].max()
        
        return df, ticks, tick_labels

    def create_figure(self):
        df, ticks, tick_labels = self._prepare_data()
        
        fig = go.Figure()
        
        # Plot by chromosome for colors
        colors = ['#636EFA', '#EF553B'] * (len(df['CHR'].unique()) // 2 + 1)
        for i, chrom in enumerate(sorted(df['CHR'].unique())):
            mask = df['CHR'] == chrom
            chrom_df = df[mask]
            
            # Subsample for performance if too many points
            if len(chrom_df) > 10000:
                chrom_df = chrom_df.sample(10000)
            
            fig.add_trace(go.Scattergl(
                x=chrom_df['pos'],
                y=-np.log10(chrom_df['P']),
                mode='markers',
                marker=dict(color=colors[i], size=self.point_size),
                name=f"Chr {int(chrom)}",
                text=_get_hover_text(chrom_df, snpname='SNP', genename='GENE'),
                hoverinfo='text',
                showlegend=False
            ))
            
        # Highlight genes
        if self.highlight_gene_list:
            highlight_df = df[df['GENE'].isin(self.highlight_gene_list)]
            if not highlight_df.empty:
                fig.add_trace(go.Scattergl(
                    x=highlight_df['pos'],
                    y=-np.log10(highlight_df['P']),
                    mode='markers',
                    marker=dict(color='red', size=self.point_size * 2),
                    name="Highlighted Genes",
                    text=_get_hover_text(highlight_df, snpname='SNP', genename='GENE'),
                    hoverinfo='text'
                ))
        
        fig.update_layout(
            title=self.title,
            xaxis=dict(title="Chromosome", tickvals=ticks, ticktext=tick_labels),
            yaxis=dict(title="-log10(P)"),
            template="plotly_white",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig

# Initialize panel with MathJax for LaTeX support
pn.extension('plotly', 'bokeh', 'tabulator', 'mathjax')

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
        
        self.all_cauchy = safe_read_csv(self.data_dir, "all_cauchy.csv")
        if not self.all_cauchy.empty:
            # Log transform P-values immediately for display
            self.all_cauchy['p_cauchy'] = -np.log10(self.all_cauchy['p_cauchy'].replace(0, 1e-300))
            self.all_cauchy['p_median'] = -np.log10(self.all_cauchy['p_median'].replace(0, 1e-300))
        
        self.metadata = safe_read_csv(self.data_dir, "metadata.csv")
        self.top_genes_pcc = safe_read_csv(self.data_dir, "top_genes_pcc.csv")
        self.genes = safe_read_csv(self.data_dir, "genes.csv")['gene'].tolist() if (self.data_dir / "genes.csv").exists() else []

        if self.run_parameters is None:
            summary_path = self.data_dir / "execution_summary.yaml"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    self.run_parameters = yaml.safe_load(f)
        
        self.mk_score_memmap = None
        self.adata_backed = None
        self.visualizer = None
        
        # Only try to load complex data if gsMap and scanpy are available (local interactive mode)
        if self.config and sc is not None:
            try:
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
            except Exception as e:
                logger.warning(f"Failed to load complex data (GSS/AnnData): {e}")
            
            self.visualizer = VisualizeRunner(self.config)
        
    def setup_widgets(self):
        traits = sorted(self.all_cauchy['trait'].unique().tolist())
        self.trait_select = pn.widgets.Select(name='Select Trait', options=traits)
        
        # Mapping trait selection to top genes (limited to 10 for standalone stability)
        self.gene_select = pn.widgets.Select(name='Select Gene (Top 10)', options=[])
        
        @pn.depends(self.trait_select.param.value, watch=True)
        def _update_genes(trait):
            # We only embed the top 10 genes to keep the standalone report functional and reasonably sized
            top_genes = self.top_genes_pcc[self.top_genes_pcc['trait'] == trait]['gene'].tolist()[:10]
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
            df = safe_read_csv(self.data_dir, f"{trait}_manhattan.csv")
            if df.empty:
                return pn.Column(
                    pn.pane.Markdown(f"### Manhattan Plot - {trait}"),
                    pn.pane.Alert(f"Manhattan data not found for trait: {trait}. This is expected in Quick Mode as SNP-gene mapping is not performed by default.", alert_type="warning"),
                    pn.pane.Markdown("To generate Manhattan plots, please run the full pipeline including `generate_ldscore` step.")
                )
            
            highlight_list = [gene] if (gene and gene in df['GENE'].values) else None
            
            plotter = PlotlyManhattanPlot(
                dataframe=df, title=f"Manhattan Plot - {trait}",
                point_size=3, highlight_gene_list=highlight_list
            )
            return pn.pane.Plotly(plotter.create_figure(), sizing_mode='stretch_both', min_height=600)
        
        return pn.Column(manhattan_plot)

    def _load_static_image(self, path):
        """Helper to load image as base64 for embedding."""
        if path.exists():
            import base64
            with open(path, "rb") as f:
                img_data = f.read()
            return pn.pane.PNG(img_data, sizing_mode='stretch_width')
        return None

    def get_annotation_view(self):
        @pn.depends(self.anno_select.param.value)
        def anno_plot(annotation):
            # Try to load pre-rendered plot first
            static_path = self.data_dir / "static_plots" / f"anno_{annotation}.png"
            pane = self._load_static_image(static_path)
            if pane: return pane
            
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
            pane = self._load_static_image(static_path)
            if pane: return pane

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
            # Use p_cauchy (which is now -log10(p))
            plot_df = heatmap_df[heatmap_df['type'] == 'aggregated'].pivot_table(
                index='annotation', columns='trait', values='p_cauchy', aggfunc='mean'
            )
            if plot_df.empty:
                 plot_df = heatmap_df.pivot_table(index='annotation', columns='trait', values='p_cauchy', aggfunc='mean')

            # We use a Plotly heatmap instead of Matplotlib for better WASM compatibility
            fig = go.Figure(data=go.Heatmap(
                z=plot_df.values,
                x=plot_df.columns,
                y=plot_df.index,
                colorscale='RdBu_r',
                reversescale=True
            ))
            fig.update_layout(
                title=f"Cauchy Heatmap (-log10 P): {anno}",
                xaxis_title="Trait",
                yaxis_title="Annotation Category",
                template="plotly_white",
                height=max(500, len(plot_df.index) * 25)
            )
            return pn.pane.Plotly(fig, sizing_mode='stretch_both', min_height=700)

        # Use a hidden LaTeX pane to ensure MathJax stays active
        hidden_math = pn.pane.LaTeX(" ", visible=False)

        return pn.Column(
            "### Trait-Annotation Enrichment Heatmap",
            hidden_math,
            pn.pane.Markdown(r"This heatmap shows the $-\log_{10}(\text{Cauchy } P\text{-value})$ for each annotation category across all traits."),
            heatmap_view
        )

    def get_summary_view(self):
        """Creates a summary view of the results."""
        # Top significant annotations across all traits
        df_agg = self.all_cauchy[self.all_cauchy['type'] == 'aggregated'].copy()
        # Sort by p_cauchy (smallest first)
        df_agg = df_agg.sort_values('p_cauchy', ascending=True)
        
        # Calculate -log10(p) for display in the table
        # We use clean column names internally and LaTeX in titles mapping
        df_agg['p_median_log'] = -np.log10(df_agg['p_median'].clip(lower=1e-300))
        df_agg['p_cauchy_log'] = -np.log10(df_agg['p_cauchy'].clip(lower=1e-300))
        
        top_cauchy = df_agg.head(20)
        
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
        
        display_cols = ['trait', 'annotation_name', 'annotation', 'p_median_log', 'p_cauchy_log']
        titles = {
            'trait': 'Trait',
            'annotation_name': 'Annotation Category',
            'annotation': 'Annotation Value',
            'p_median_log': r'$-\log_{10}(P_{\text{Median}})$',
            'p_cauchy_log': r'$-\log_{10}(P_{\text{Cauchy}})$'
        }
        
        table = pn.widgets.Tabulator(top_cauchy[display_cols], 
                                    sizing_mode='stretch_width', disabled=True,
                                    titles=titles)
        
        return pn.Column(
            pn.pane.Markdown(summary_md, sizing_mode='stretch_width'),
            pn.pane.LaTeX(" ", visible=False),
            table
        )

    def get_gene_diagnostic_view(self):
        @pn.depends(self.trait_select.param.value)
        def gene_table(trait):
            trait_pcc = self.top_genes_pcc[self.top_genes_pcc['trait'] == trait].copy()
            trait_pcc = trait_pcc[['gene', 'PCC']].sort_values('PCC', ascending=False).head(10)
            table = pn.widgets.Tabulator(trait_pcc, name='Top 10 Gene Diagnostic Info', 
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
                row = []
                
                # Check for pre-rendered images first
                exp_static = self.data_dir / "static_plots" / f"gene_{trait}_{gene}_exp_{sample}.png"
                gss_static = self.data_dir / "static_plots" / f"gene_{trait}_{gene}_gss_{sample}.png"
                
                exp_pane = self._load_static_image(exp_static)
                gss_pane = self._load_static_image(gss_static)
                
                if exp_pane and gss_pane:
                    plots.append(pn.Row(exp_pane, gss_pane))
                    continue

                # Fallback to dynamic generation
                sample_metadata = self.metadata[self.metadata['sample'] == sample]
                sample_spots = sample_metadata['spot'].values
                
                # Expression
                if self.adata_backed is not None and gene in self.adata_backed.var_names:
                    valid_spots = [s for s in sample_spots if s in self.adata_backed.obs_names]
                    if valid_spots:
                        df_exp_gene = sc.get.obs_df(self.adata_backed, keys=[gene], use_raw=False)
                        exp_vals = df_exp_gene.loc[valid_spots, gene].values
                        df_exp = sample_metadata[sample_metadata['spot'].isin(valid_spots)].copy()
                        df_exp['Expression'] = exp_vals.astype(np.float32)
                        p1 = df_exp.hvplot.scatter(x='sx', y='sy', c='Expression', cmap='viridis', 
                                                  title=f"{sample} - {gene} Exp", 
                                                  responsive=True, min_height=400, colorbar=True)
                        row.append(p1.opts(xaxis=None, yaxis=None, toolbar=None))

                # GSS
                if self.mk_score_memmap and gene in self.genes:
                    gene_idx = self.genes.index(gene)
                    valid_spots = [s for s in sample_spots if s in self.adata_backed.obs_names]
                    if valid_spots:
                        full_indices = self.adata_backed.obs_names.get_indexer(valid_spots)
                        gss_vals = self.mk_score_memmap.data[full_indices, gene_idx].astype(np.float32)
                        df_gss = sample_metadata[sample_metadata['spot'].isin(valid_spots)].copy()
                        df_gss['GSS'] = gss_vals
                        p2 = df_gss.hvplot.scatter(x='sx', y='sy', c='GSS', cmap='plasma', 
                                                  title=f"{sample} - {gene} GSS", 
                                                  responsive=True, min_height=400, colorbar=True)
                        row.append(p2.opts(xaxis=None, yaxis=None, toolbar=None))
                
                if row:
                    plots.append(pn.Row(*row, sizing_mode='stretch_width'))
            
            if not plots:
                return pn.pane.Markdown(f"No plots available for {gene}. Pre-rendered plots only available for Top 10 genes.")
                
            return pn.Column(*plots, sizing_mode='stretch_width')
            
        return pn.Column(gene_table, "---", gene_view)

    def get_cauchy_view(self):
        self.cauchy_type_toggle = pn.widgets.RadioButtonGroup(
            name='View Level', options=['Aggregated Only', 'Sample Level Only'], value='Aggregated Only'
        )
        
        cauchy_table = pn.widgets.Tabulator(
            pagination='remote', page_size=15, sizing_mode='stretch_width',
            selectable=True,
            titles={
                'p_cauchy': r'$-\log_{10}(P_{\text{Cauchy}})$',
                'p_median': r'$-\log_{10}(P_{\text{Median}})$'
            }
        )

        @pn.depends(self.trait_select.param.value, self.cauchy_type_toggle.param.value, watch=True)
        def _update_cauchy_table(trait, view_type):
            df_trait = self.all_cauchy[self.all_cauchy['trait'] == trait].copy()
            
            # Helper to get sorting order from Aggregated p_median
            agg_df = df_trait[df_trait['type'] == 'aggregated'].sort_values('p_median', ascending=False)
            anno_order = agg_df['annotation'].tolist()
            
            if view_type == 'Aggregated Only':
                df = agg_df
            else:
                # Sample Level Only
                df = df_trait[df_trait['type'] == 'sample'].copy()
                # Categorical sort for annotations based on aggregated p_median rank
                df['annotation'] = pd.Categorical(df['annotation'], categories=anno_order, ordered=True)
                df = df.sort_values(['annotation', 'sample'])
            
            # Drop unnecessary columns for display
            display_cols = ['trait', 'annotation_name', 'annotation', 'sample', 'p_median', 'p_cauchy']
            if view_type == 'Aggregated Only':
                display_cols = [c for c in display_cols if c != 'sample']
            
            cauchy_table.value = df[display_cols]
        
        _update_cauchy_table(self.trait_select.value, self.cauchy_type_toggle.value)

        return pn.Column(
            pn.Row("View Style:", self.cauchy_type_toggle),
            cauchy_table
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
        title = "gsMap Report"
        if self.config:
            title += f" - {self.config.project_name}"
        elif self.run_parameters:
            title += f" - {self.run_parameters.get('Project Name', '')}"

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
            ("Traits Correlation", self.get_correlation_view()),
            ("Gene/GSS Diagnostic", self.get_gene_diagnostic_view()),
            ("Cauchy Combination", self.get_cauchy_view()),
            dynamic=False
        )
        
        if not use_template:
            return pn.Row(sidebar, tabs)
            
        return pn.template.FastListTemplate(
            title=title,
            sidebar=[sidebar, self.get_running_info()],
            main=[tabs],
            header_background="#2c3e50",
            accent_base_color="#2c3e50",
        )

def launch_report(data_dir: Path, port=5006, show=True, config: ReportConfig = None, run_parameters: dict = None):
    report = InteractiveReport(data_dir, config=config, run_parameters=run_parameters)
    app = report.create_layout()
    if show:
        app.show(port=port)
    else:
        return app

# Check if running in a Panel environment (e.g., panel convert or panel serve)
if pn.state.curdoc:
    # Use current directory for data in standalone/WASM mode
    report = InteractiveReport(Path("."))
    report.create_layout().servable()

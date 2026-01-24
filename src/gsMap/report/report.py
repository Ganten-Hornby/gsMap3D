import logging
from datetime import datetime
from pathlib import Path

import gsMap
from gsMap.config import QuickModeConfig
from .report_data import ReportDataManager

logger = logging.getLogger(__name__)

def run_report(config: QuickModeConfig, run_parameters: dict = None):
    """
    Main entry point for report generation.
    Prepares data and saves the interactive report as a standalone Alpine+Tailwind HTML folder.

    Output structure:
        project_dir/
        ├── report_data/                    # Data files (CSV, h5ad)
        │   ├── spot_metadata.csv
        │   ├── cauchy_results.csv
        │   ├── umap_data.csv
        │   ├── gene_list.csv
        │   ├── gss_stats/
        │   │   └── gene_trait_correlation_{trait}.csv
        │   ├── manhattan_data/
        │   │   └── {trait}_manhattan.csv
        │   └── spatial_3d/
        │       └── spatial_3d.h5ad
        │
        └── gsmap_web_report/               # Web report (self-contained)
            ├── index.html
            ├── report_meta.json
            ├── execution_summary.yaml
            ├── spatial_plots/
            │   └── ldsc_{trait}.png
            ├── gene_diagnostic_plots/
            ├── annotation_plots/
            ├── spatial_3d/
            │   └── *.html
            ├── js_lib/
            └── js_data/
                ├── gss_stats/
                ├── sample_index.js
                ├── sample_{name}_spatial.js
                └── ... (other JS modules)
    """
    logger.info("Running gsMap Report Module (Alpine.js + Tailwind based)")

    # 1. Use ReportDataManager to prepare all data and JS assets
    manager = ReportDataManager(config)
    web_report_dir = manager.run()

    # 2. Save run_parameters for future reference
    if run_parameters:
        import yaml
        with open(web_report_dir / "execution_summary.yaml", "w") as f:
            yaml.dump(run_parameters, f)

    # 3. Render the Jinja2 template
    template_path = Path(__file__).parent / "static" / "template.html"
    if not template_path.exists():
        logger.error(f"Template file not found at {template_path}")
        return

    try:
        from jinja2 import Template
        with open(template_path, "r", encoding="utf-8") as f:
            template = Template(f.read())

        # Prepare context
        context = {
            "title": f"gsMap Report - {config.project_name}",
            "project_name": config.project_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gsmap_version": getattr(gsMap, "__version__", "unknown"),
        }

        rendered_html = template.render(**context)

        report_file = web_report_dir / "index.html"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(rendered_html)

        from rich import print as rprint
        rprint(f"\n[bold green]Report generated successfully![/bold green]")
        rprint(f"Web report directory: [cyan]{web_report_dir}[/cyan]")
        rprint(f"Data files directory: [cyan]{config.report_data_dir}[/cyan]\n")

        rprint("[bold]Ways to view the interactive report:[/bold]")
        rprint(f"1. [bold white]Remote Server:[/bold white] Run the command below to start a temporary web server:")
        rprint(f"   [bold cyan]gsmap report-view {web_report_dir}[/bold cyan]")
        rprint(f"2. [bold white]Local PC:[/bold white] Copy the [cyan]{web_report_dir.name}[/cyan] folder to your machine and open [cyan]index.html[/cyan].\n")

    except ImportError:
        logger.error("Jinja2 not found. Please install it with 'pip install jinja2'.")
    except Exception as e:
        logger.error(f"Failed to render report: {e}")
        import traceback
        traceback.print_exc()

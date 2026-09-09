import logging
import shlex
from datetime import datetime
from pathlib import Path

from jinja2 import Template

import gsMap
from gsMap.config import QuickModeConfig
from gsMap.config.latent2gene_config import DatasetType
from gsMap.logging_config import logged_stage

from .report_data import ReportDataManager, _atomic_write_text

logger = logging.getLogger(__name__)


def _render_report_html(project_name: str) -> str:
    template_path = Path(__file__).parent / "static" / "template.html"
    template = Template(template_path.read_text(encoding="utf-8"), autoescape=True)
    return template.render(
        title=f"gsMap Report - {project_name}",
        project_name=project_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        gsmap_version=getattr(gsMap, "__version__", "unknown"),
    )


@logged_stage("report")
def run_report(config: QuickModeConfig, run_parameters: dict = None):
    """
    Main entry point for report generation.
    Prepares data and saves the interactive report as a standalone Alpine+Tailwind HTML folder.

    Output structure:
        project_dir/
        ├── report_data/                    # Prepared data and export provenance
        │   ├── spot_metadata.csv
        │   ├── cauchy_results.csv
        │   ├── umap_data.csv
        │   ├── gene_list.csv
        │   ├── spatial_3d_coordinate_info.json
        │   ├── gss_stats/
        │   │   └── gene_trait_correlation_{trait}.csv
        │   ├── manhattan_data/
        │   │   └── {trait}_manhattan.csv
        │   └── spatial_3d/
        │       ├── spatial_3d_manifest.json
        │       └── spatial_3d_display_spots.parquet
        │
        └── gsmap_web_report/               # Web report (self-contained)
            ├── index.html
            ├── report_meta.json
            ├── execution_summary.yaml
            ├── spatial_plots/
            │   └── ldsc_{trait}.png
            ├── gene_diagnostic_plots/
            ├── annotation_plots/
            ├── js_lib/
            └── js_data/
                ├── gss_stats/
                ├── annotation_schemas.js     # Shared categorical colors
                ├── sample_index.js
                ├── spatial_2d/             # Sample bases + lazy trait arrays
                ├── spatial_3d/             # Shared geometry + lazy feature arrays
                └── ... (other JS modules)
    """
    logger.info("Running gsMap Report Module (Alpine.js + Tailwind based)")
    if (
        getattr(config, "export_spatialvista", False)
        and config.dataset_type != DatasetType.SPATIAL_3D
    ):
        raise ValueError("SpatialVista report export requires a spatial3D dataset")
    # Validate the template before doing expensive data preparation.
    rendered_html = _render_report_html(config.project_name)

    # 1. Use ReportDataManager to prepare all data and JS assets
    manager = ReportDataManager(config)
    web_report_dir = manager.run()

    if getattr(config, "export_spatialvista", False):
        from gsMap.export_spatialvista import export_spatialvista

        exported = export_spatialvista(
            config.project_dir, metadata=manager.metadata, overwrite=True
        )
        logger.info(f"SpatialVista export: {exported}")
        logger.info(
            f"Open with a gsMap-aware SpatialVista build: spatialvista --input {shlex.quote(str(exported))}"
        )

    # 2. Save run_parameters for future reference
    if run_parameters:
        import yaml

        with open(web_report_dir / "execution_summary.yaml", "w") as f:
            yaml.dump(run_parameters, f)

    # Publish the entry point atomically, only after its data files are ready.
    report_file = web_report_dir / "index.html"
    _atomic_write_text(report_file, rendered_html)

    from rich import print as rprint

    rprint("\n[bold green]Report generated successfully![/bold green]")
    rprint(f"Web report directory: [cyan]{web_report_dir}[/cyan]")
    rprint(f"Data files directory: [cyan]{config.report_data_dir}[/cyan]\n")

    rprint("[bold]Ways to view the interactive report:[/bold]")
    rprint(
        "1. [bold white]Remote Server:[/bold white] Run the command below to start a temporary web server:"
    )
    rprint(
        f"   [bold cyan]gsmap report-view {shlex.quote(str(web_report_dir))} --port 8080 --no-browser[/bold cyan]"
    )
    rprint(
        f"\n2. [bold white]Local PC:[/bold white] Copy the [cyan]{web_report_dir.name}[/cyan] folder to your machine and open [cyan]index.html[/cyan].\n"
    )

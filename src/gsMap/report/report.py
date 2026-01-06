import logging
from datetime import datetime
from pathlib import Path

import gsMap
from gsMap.config import QuickModeConfig
from .report_data import prepare_report_data, export_data_as_js_modules

logger = logging.getLogger(__name__)

def run_report(config: QuickModeConfig, run_parameters: dict = None):
    """
    Main entry point for report generation.
    Prepares data and saves the interactive report as a standalone Alpine+Tailwind HTML folder.

    Output structure:
        project_dir/gsMap_Report/
        ├── index.html
        ├── spot_metadata.csv
        ├── cauchy_results.csv
        ├── gene_list.csv
        ├── gene_trait_correlation.csv
        ├── {trait}_gene_diagnostic.csv
        ├── manhattan_data/
        ├── spatial_plots/
        ├── gene_diagnostic_plots/
        ├── annotation_plots/
        ├── js_lib/
        └── js_data/
    """
    logger.info("Running gsMap Report Module (Alpine.js + Tailwind based)")

    # 1. Prepare data (CSVs and PNGs) - writes directly to config.report_dir
    report_dir = prepare_report_data(config)

    # 2. Save run_parameters for future reference
    if run_parameters:
        import yaml
        with open(report_dir / "execution_summary.yaml", "w") as f:
            yaml.dump(run_parameters, f)

    # 3. Export Data as JS Modules
    export_data_as_js_modules(report_dir)

    # 4. Render the Jinja2 template
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

        report_file = report_dir / "index.html"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(rendered_html)

        logger.info(f"Report generated successfully! Saved at {report_file}")
        logger.info(f"You can view it by opening {report_file} in a browser or by running 'gsmap report-view'.")

    except ImportError:
        logger.error("Jinja2 not found. Please install it with 'pip install jinja2'.")
    except Exception as e:
        logger.error(f"Failed to render report: {e}")
        import traceback
        traceback.print_exc()

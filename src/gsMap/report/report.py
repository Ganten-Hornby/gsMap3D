import logging
import os
import shutil
from pathlib import Path

import gsMap
from gsMap.config import ReportConfig
from .report_data import prepare_report_data
from .app import InteractiveReport, launch_report

logger = logging.getLogger(__name__)

def run_report(config: ReportConfig, run_parameters: dict = None):
    """
    Main entry point for report generation.
    Prepares data and saves the interactive report as a standalone HTML folder.
    """
    logger.info("Running gsMap Report Module (Panel-based)")
    
    # 1. Prepare data
    data_dir = prepare_report_data(config)

    # Save run_parameters to data_dir for future use
    if run_parameters:
        import yaml
        with open(data_dir / "execution_summary.yaml", "w") as f:
            yaml.dump(run_parameters, f)
    
    # 2. Create the interactive report app
    report = InteractiveReport(data_dir, config=config, run_parameters=run_parameters)
    # Use basic layout for saving to avoid template-related embedding issues
    app = report.create_layout(use_template=False)
    
    # 3. Save as standalone HTML folder
    report_output_dir = config.get_report_dir("gsMap_Report")
    report_output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_output_dir / "index.html"
    logger.info(f"Exporting standalone report to {report_output_dir}...")
    
    try:
        # Save the panel app
        # For a truly standalone folder with static assets, we might need more effort, 
        # but Panel's save(embed=True) can embed state, or save() for a server-less view.
        # User wants a folder they can download.
        app.save(report_file, title=f"gsMap Report - {config.project_name}", embed=True)
        
        # Copy the interactive data directory into the report output dir for portability if not embedding everything
        # Actually, embed=True should be enough for basic interactivity if data is small, 
        # but for large data we might want to keep the feather files.
        # However, Bokeh/Panel embedding usually includes the data in the HTML.
        
        logger.info(f"Report generated successfully! Saved at {report_file}")
        logger.info(f"You can view it by opening {report_file} in a browser or by running 'gsmap report-view'.")
        
    except Exception as e:
        logger.error(f"Failed to export report: {e}")
        # Still try to provide the data_dir for report-view
        logger.info(f"Interactive data is available at {data_dir}")

def run_interactive_report(config: ReportConfig):
    """
    Prepare data for the interactive report and optionally launch it.
    """
    logger.info("Preparing data for the interactive report...")
    data_dir = prepare_report_data(config)
    
    if config.browser:
        logger.info(f"Launching interactive report viewer on port {config.port}...")
        run_report_viewer(config)
    else:
        logger.info(f"Interactive report data prepared in {data_dir}. Use 'gsmap report-view' to view it.")

def run_report_viewer(config: ReportConfig):
    """
    Launch the interactive report viewer.
    """
    data_dir = config.get_report_dir("interactive")
    if not data_dir.exists():
        logger.error(f"Interactive report data not found in {data_dir}. Please run 'gsmap report' first.")
        return
    
    logger.info(f"Starting gsMap Interactive Report viewer on http://localhost:{config.port}")
    launch_report(data_dir, port=config.port, show=True, config=config)

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

    app = report.create_layout(use_template=True)
    
    # 3. Export as Pyodide-based standalone HTML
    report_output_dir = config.get_report_dir("gsMap_Report")
    report_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy app.py and data to the report output dir for conversion
    app_source = Path(__file__).parent / "app.py"
    shutil.copy(app_source, report_output_dir / "app.py")
    
    # Copy all prepared data files (CSVs, YAMLs) to the report output dir
    for data_file in data_dir.glob("*"):
        if data_file.is_file():
            shutil.copy(data_file, report_output_dir / data_file.name)
        elif data_file.is_dir() and data_file.name == "static_plots":
             static_plots_dst = report_output_dir / "static_plots"
             if static_plots_dst.exists():
                 shutil.rmtree(static_plots_dst)
             shutil.copytree(data_file, static_plots_dst)

    report_file = report_output_dir / "index.html"
    logger.info(f"Converting interactive report to Pyodide (WASM) at {report_output_dir}...")
    
    try:
        import subprocess
        import sys
        
        # Run panel convert
        # We include all files in the directory as resources
        # The requirements are inferred but we can be explicit if needed
        cmd = [
            sys.executable, "-m", "panel", "convert",
            "app.py",
            "--to", "pyodide-worker",
            "--out", ".",
            "--requirements", "pandas", "numpy", "holoviews", "hvplot", "plotly", "pyyaml"
        ]
        
        # Add all data files as resources
        for f in report_output_dir.glob("*.csv"):
            cmd.extend(["--resources", f.name])
        for f in report_output_dir.glob("*.yaml"):
            cmd.extend(["--resources", f.name])
            
        # Include the static_plots images as resources
        static_plots_dir = report_output_dir / "static_plots"
        if static_plots_dir.exists():
            for f in static_plots_dir.glob("*.png"):
                # Resources should be relative to the app.py location
                cmd.extend(["--resources", f"static_plots/{f.name}"])
            
        logger.info(f"Running (in {report_output_dir}): {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(report_output_dir))
        
        if result.returncode != 0:
            logger.error(f"Panel convert failed: {result.stderr}")
            raise Exception(f"Panel convert failed with exit code {result.returncode}")

        # Rename app.html to index.html
        if (report_output_dir / "app.html").exists():
            if report_file.exists():
                report_file.unlink()
            (report_output_dir / "app.html").rename(report_file)
        
        # Clean up the temporary copy of app.py
        if (report_output_dir / "app.py").exists():
            (report_output_dir / "app.py").unlink()

        logger.info(f"Report generated successfully! Saved at {report_file}")
        logger.info(f"You can view it by opening {report_file} in a browser or by running 'gsmap report-view'.")
        
    except Exception as e:
        logger.error(f"Failed to export report: {e}")
        # Still try to provide the data_dir for report-view
        logger.info(f"Interactive data is available at {data_dir}")

def run_interactive_report(config: ReportConfig):
    """
    Prepare data for the interactive report and optionally launch it.
    Also exports the standalone version.
    """
    # Trigger full report generation (which includes data prep and standalone export)
    run_report(config)
    
    if config.browser:
        logger.info(f"Launching interactive report viewer on port {config.port}...")
        run_report_viewer(config)

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

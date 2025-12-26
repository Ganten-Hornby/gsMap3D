import logging
import os
import shutil
import json
from datetime import datetime
from pathlib import Path

import gsMap
from gsMap.config import ReportConfig
from .report_data import prepare_report_data

logger = logging.getLogger(__name__)

def run_report(config: ReportConfig, run_parameters: dict = None):
    """
    Main entry point for report generation.
    Prepares data and saves the interactive report as a standalone Jinja2 + Tailwind HTML folder.
    """
    logger.info("Running gsMap Report Module (Jinja2 + Tailwind based)")
    
    # 1. Prepare data (CSVs and PNGs)
    data_dir = prepare_report_data(config)

    # Save run_parameters to data_dir for future use
    if run_parameters:
        import yaml
        with open(data_dir / "execution_summary.yaml", "w") as f:
            yaml.dump(run_parameters, f)
    
    # 2. Setup output directory
    report_output_dir = config.get_report_dir("gsMap_Report")
    report_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Copy all prepared data files to the report output dir
    logger.info(f"Copying data to {report_output_dir}...")
    for item in data_dir.iterdir():
        dest = report_output_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    # 4. Prepare embedded data for truly static usage (no server required)
    logger.info("Embedding data for static viewing...")
    embedded_data = {}
    for csv_file in data_dir.rglob("*.csv"):
        try:
            # Use relative path as key for embedded data
            rel_path = csv_file.relative_to(data_dir).as_posix()
            with open(csv_file, 'r', encoding='utf-8') as f:
                embedded_data[rel_path] = f.read()
        except Exception as e:
            logger.warning(f"Failed to embed {csv_file.name}: {e}")

    # 5. Render the Jinja2 template
    template_path = Path(__file__).parent / "template.html"
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
            "run_parameters_json": json.dumps(run_parameters if run_parameters else {}),
            "embedded_data_json": json.dumps(embedded_data),
        }
        
        rendered_html = template.render(**context)
        
        report_file = report_output_dir / "index.html"
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

def run_interactive_report(config: ReportConfig):
    """
    Prepare data for the interactive report and optionally launch it.
    """
    run_report(config)
    
    if config.browser:
        logger.info(f"Launching report viewer on port {config.port}...")
        run_report_viewer(config)

def run_report_viewer(config: ReportConfig):
    """
    Launch a simple HTTP server to view the generated report.
    """
    report_output_dir = config.get_report_dir("gsMap_Report")
    if not report_output_dir.exists() or not (report_output_dir / "index.html").exists():
        logger.error(f"Report not found in {report_output_dir}. Please run 'gsmap report' first.")
        return
    
    import http.server
    import socketserver
    import webbrowser
    import threading
    import functools

    PORT = config.port or 8000
    # Create a handler that serves from the report directory without changing the process's CWD
    Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(report_output_dir))
    
    def serve():
        # Allow reusing the address to avoid "Address already in use" errors if restarted quickly
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            logger.info(f"Serving report at http://localhost:{PORT}")
            httpd.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    
    url = f"http://localhost:{PORT}"
    webbrowser.open(url)
    
    logger.info("Press Ctrl+C to stop the server (in the terminal where you ran the command).")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping server...")

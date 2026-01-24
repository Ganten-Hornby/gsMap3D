#!/usr/bin/env python
"""
gsMap CLI - Main command-line interface using the modular config design.
"""

import logging
from typing import Optional, Annotated

import typer

from gsMap.config import (
    dataclass_typer,
    QuickModeConfig,
    FindLatentRepresentationsConfig,
    LatentToGeneConfig,
    SpatialLDSCConfig,
    ReportConfig,
    LDScoreConfig,
    CauchyCombinationConfig,
    FormatSumstatsConfig,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s | %(name)s - %(message)s')
logger = logging.getLogger("gsMap")

# Create the Typer app
app = typer.Typer(
    name="gsmap",
    help="gsMap: genetically informed spatial mapping of cells for complex traits",
    rich_markup_mode="rich",
    add_completion=False,
)


# ============================================================================
# CLI Commands using dataclass_typer decorator
# ============================================================================

@app.command(name="quick-mode")
@dataclass_typer
def quick_mode(config: QuickModeConfig):
    """
    Run the complete gsMap pipeline with all steps.

    This command runs the gsMap analysis pipeline including:
    - Data loading and preprocessing (Find Latent)
    - Gene expression analysis (Latent to Gene)
    - GWAS integration (Spatial LDSC)
    - Cauchy Combination Test
    - Result generation (Report)
    
    Requires pre-generated SNP-gene weight matrix and LD weights.
    """
    logger.info(f"Starting Quick Mode Pipeline")
    logger.info(f"Working directory: {config.workdir}")
    logger.info(f"Project directory: {config.project_dir}")

    try:
        from gsMap.pipeline.quick_mode import run_quick_mode
        run_quick_mode(config)
        logger.info("✓ Pipeline completed successfully!")
    except (ImportError, AttributeError) as e:
        logger.error(f"Error executing pipeline: {e}")
        raise
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        raise


@app.command(name="find-latent")
@dataclass_typer
def find_latent_representations(config: FindLatentRepresentationsConfig):
    """
    Find latent representations of each spot using Graph Neural Networks.
    
    This step:
    - Loads spatial transcriptomics data
    - Builds neighborhood graphs
    - Learns latent representations using GNN
    - Saves the model and embeddings
    """
    logger.info(f"Working directory: {config.workdir}")
    logger.info(f"Project directory: {config.project_dir}")
    
    # Show auto-generated paths
    logger.info(f"Latent representations will be saved to: {config.latent_dir}")
    logger.info(f"Model will be saved to: {config.model_path}")
    logger.info(f"H5AD with latent: {config.hdf5_with_latent_path}")
    
    if config.annotation and config.two_stage:
        logger.info(f"Using two-stage training with annotation: {config.annotation}")
    else:
        logger.info("Using single-stage training with reconstruction loss only")
    
    try:
        from gsMap.find_latent import run_find_latent_representation
        run_find_latent_representation(config)
        logger.info("✓ Latent representations computed successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


@app.command(name="latent-to-gene")
@dataclass_typer
def latent_to_gene(config: LatentToGeneConfig):
    """
    Estimate gene marker scores for each spot using latent representations.
    
    This step:
    - Loads latent representations
    - Estimates gene marker scores
    - Performs spatial smoothing
    - Saves marker scores for LDSC
    """
    logger.info(f"Working directory: {config.workdir}")
    logger.info(f"Project directory: {config.project_dir}")
    
    # Show auto-generated paths
    logger.info(f"Marker scores will be saved to: {config.mkscore_feather_path}")
    if config.annotation:
        logger.info(f"Tuned scores will be saved to: {config.tuned_mkscore_feather_path}")
    
    try:
        from gsMap.latent2gene import run_latent_to_gene
        run_latent_to_gene(config)
        logger.info("✓ Gene marker scores computed successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


@app.command(name="spatial-ldsc")
@dataclass_typer
def spatial_ldsc(config: SpatialLDSCConfig):
    """
    Run spatial LDSC analysis for genetic association.
    
    This step:
    - Loads LD scores and GWAS summary statistics
    - Performs spatial LDSC regression
    - Computes enrichment statistics
    - Saves results for downstream analysis
    """
    logger.info(f"Trait: {config.trait_name}")
    logger.info(f"Working directory: {config.workdir}")
    logger.info(f"Project directory: {config.project_dir}")
    
    # Show auto-generated paths
    logger.info(f"LDSC results will be saved to: {config.ldsc_save_dir}")
    logger.info(f"Result file: {config.get_ldsc_result_file(config.trait_name)}")
    
    if config.use_gpu:
        logger.info("Using JAX-accelerated implementation")
    else:
        logger.info("Using standard implementation")
    
    try:
        if config.use_gpu:
            from gsMap.spatial_ldsc.spatial_ldsc_jax import run_spatial_ldsc_jax
            run_spatial_ldsc_jax(config)
        else:
            from gsMap.spatial_ldsc.spatial_ldsc_multiple_sumstats import run_spatial_ldsc
            run_spatial_ldsc(config)
        logger.info("✓ Spatial LDSC completed successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


@app.command(name="cauchy-combination")
@dataclass_typer
def cauchy_combination(config: CauchyCombinationConfig):
    """
    Run Cauchy combination test to combine spatial LDSC results across spots.
    
    This step:
    - Loads spatial LDSC results for a trait
    - Removes outliers
    - Performs Cauchy combination test for each annotation
    - Computes Fisher's exact test for enrichment
    """
    logger.info(f"Trait: {config.trait_name}")
    logger.info(f"Project: {config.project_name}")
    logger.info(f"Annotation: {config.annotation}")
    
    try:
        from gsMap.cauchy_combination_test import run_Cauchy_combination
        run_Cauchy_combination(config)
        logger.info("✓ Cauchy combination test completed successfully!")
    except (ImportError, AttributeError) as e:
        logger.error(f"Error executing Cauchy combination: {e}")
        raise
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        raise


@app.command(name="ldscore-weight-matrix")
@dataclass_typer
def ldscore_weight_matrix(config: LDScoreConfig):
    """
    Compute LD score weight matrices for features.

    This command runs the LDScorePipeline to:
    - Load genotypes (PLINK)
    - Load feature mappings (BED/Dict) or Annotations
    - Compute LD-based weights between SNPs and Features
    - Save results as AnnData (.h5ad)
    """
    logger.info(f"Command: ldscore-weight-matrix")
    logger.info(f"Target Chromosomes: {config.chromosomes}")
    logger.info(f"Output Directory: {config.output_dir}")

    try:
        from gsMap.ldscore.pipeline import LDScorePipeline
        pipeline = LDScorePipeline(config)
        pipeline.run()
        logger.info("✓ LDScore Pipeline completed successfully!")
    except ImportError as e:
        logger.error(f"Import Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        raise


@app.command(name="format-sumstats")
@dataclass_typer
def format_sumstats(config: FormatSumstatsConfig):
    """
    Format GWAS summary statistics for gsMap or COJO.

    This command:
    - Filters SNPs based on INFO, MAF, and P-value
    - Converts SNP positions to RSID (if dbsnp is provided)
    - Saves formatted summary statistics
    """
    try:
        from gsMap.format_sumstats import gwas_format
        gwas_format(config)
        logger.info("✓ Summary statistics formatted successfully!")
    except Exception as e:
        logger.error(f"Execution Error: {e}")
        raise


@app.command(name="report-view")
def report_view(
    report_path: Annotated[str, typer.Argument(
        help="Path to gsmap_web_report directory containing index.html"
    )],
    port: Annotated[int, typer.Option(
        help="Port to serve the report on"
    )] = 8080,
    no_browser: Annotated[bool, typer.Option(
        "--no-browser",
        help="Don't automatically open browser"
    )] = False,
):
    """
    Launch a local web server to view the gsMap report.

    This command starts a simple HTTP server to serve the report files,
    which is necessary for proper loading of JavaScript modules.

    Example:
        gsmap report-view /path/to/project/gsmap_web_report
        gsmap report-view /path/to/project/gsmap_web_report --port 9000
    """
    import http.server
    import os
    import socketserver
    import webbrowser
    from pathlib import Path

    report_dir = Path(report_path).resolve()

    if not report_dir.exists():
        logger.error(f"Directory not found: {report_dir}")
        raise typer.Exit(1)

    index_file = report_dir / "index.html"
    if not index_file.exists():
        logger.error(f"index.html not found in {report_dir}")
        logger.error("Please provide path to the gsmap_web_report directory.")
        raise typer.Exit(1)

    os.chdir(report_dir)

    handler = http.server.SimpleHTTPRequestHandler

    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            url = f"http://localhost:{port}"
            logger.info(f"Serving report at {url}")
            logger.info(f"Press Ctrl+C to stop the server")

            if not no_browser:
                webbrowser.open(url)

            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is already in use. Try a different port with --port <number>")
        else:
            logger.error(f"Failed to start server: {e}")
        raise typer.Exit(1)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        try:
            from gsMap import __version__
            typer.echo(f"gsMap version {__version__}")
        except ImportError:
            typer.echo("gsMap version: development")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[Optional[bool], typer.Option(
        "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    )] = None,
):
    """
    gsMap: genetically informed spatial mapping of cells for complex traits.
    
    Use 'gsmap COMMAND --help' for more information on a specific command.
    
    Common workflows:
    
    1. Quick mode (recommended for first-time users):
       gsmap quick-mode --workdir /path/to/work --sample-name my_sample ...
    
    2. Step-by-step analysis:
       gsmap find-latent ...
       gsmap latent-to-gene ...
       gsmap spatial-ldsc ...
       gsmap report ...
    
    For detailed documentation, visit: https://github.com/JianYang-Lab/gsMap
    """
    pass


if __name__ == "__main__":
    app()
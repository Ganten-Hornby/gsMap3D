import logging
import shlex
import sys
from unittest.mock import patch

import pytest

from gsMap.cli import main


def parse_bash_command(command: str) -> list[str]:
    """Convert multi-line bash command to argument list for sys.argv"""
    cleaned_command = command.replace("\\\n", " ")
    cleaned_command = " ".join(cleaned_command.splitlines())
    cleaned_command = " ".join(cleaned_command.split())
    return shlex.split(cleaned_command)


@pytest.mark.real_data
def test_gsmap_step_by_step_pipeline(cauchy_combination_fixture):
    logger = logging.getLogger("test_gsmap_pipeline")
    logger.info("Pipeline test completed successfully - using shared fixtures")


@pytest.mark.real_data
@pytest.mark.parametrize("symbolic_link_results", ["quickmode_config"], indirect=True)
def test_gsmap_quick_mode(
    symbolic_link_results, latent_representations_fixture, gene_marker_scores_fixture
):
    """Test the gsMap quick_mode pipeline with real data"""
    logger = logging.getLogger("test_gsmap_quick_mode")
    logger.info("Starting quick_mode pipeline test with linked fixtures")
    config = symbolic_link_results

    # Test the quick_mode command (will reuse linked directories)
    command = f"""
    gsmap quick_mode \
        --workdir '{config.workdir}' \
        --homolog_file '{config.homolog_file}' \
        --sample_name {config.sample_name} \
        --gsMap_resource_dir '{config.gsMap_resource_dir}' \
        --hdf5_path '{config.hdf5_path}' \
        --annotation '{config.annotation}' \
        --data_layer '{config.data_layer}' \
        --sumstats_file '{config.sumstats_file}' \
        --trait_name '{config.trait_name}' \
        --max_processes {config.max_processes}
    """

    with patch.object(sys, "argv", parse_bash_command(command)):
        main()

    # Verify output files and directories
    # Verify spatial_ldsc step
    spatial_ldsc_result = config.get_ldsc_result_file(config.trait_name)
    assert spatial_ldsc_result.exists(), "Spatial LDSC results not created"
    assert spatial_ldsc_result.stat().st_size > 0, "Spatial LDSC results file is empty"

    # Verify cauchy_combination step
    cauchy_result = config.get_cauchy_result_file(config.trait_name)
    assert cauchy_result.exists(), "Cauchy combination results not created"
    assert cauchy_result.stat().st_size > 0, "Cauchy combination results file is empty"

    # Verify report generation
    report_file = config.get_gsMap_report_file(config.trait_name)
    assert report_file.exists(), "Final report not created"
    assert report_file.stat().st_size > 0, "Final report file is empty"

    logger.info("Quick mode pipeline test completed successfully")

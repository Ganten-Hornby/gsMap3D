import copy
import logging
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from gsMap.cli import main
from gsMap.config import QuickModeConfig


def parse_bash_command(command: str) -> list[str]:
    """Convert multi-line bash command to argument list for sys.argv"""
    cleaned_command = command.replace("\\\n", " ")
    cleaned_command = " ".join(cleaned_command.splitlines())
    cleaned_command = " ".join(cleaned_command.split())
    return shlex.split(cleaned_command)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-real-data",
        action="store_true",
        default=False,
        help="run tests that require real data",
    )
    parser.addoption(
        "--test-data",
        action="store",
        default=None,
        help="Path to test data directory for gsMap tests",
    )
    parser.addoption(
        "--work-dir",
        action="store",
        default=None,
        help="Path to working directory for test outputs (defaults to a temporary directory)",
    )


@pytest.fixture(scope="session")
def test_data_dir(request):
    """Get test data directory"""
    test_dir = request.config.getoption("--test-data")
    if not test_dir:
        pytest.skip("--test-data not provided")

    test_dir = Path(test_dir)
    if not test_dir.exists():
        pytest.skip(f"Test data directory does not exist: {test_dir}")

    return test_dir


@pytest.fixture(scope="session")
def example_data_dir(test_data_dir):
    """Get path to example data directory"""
    example_dir = test_data_dir / "gsMap_example_data"
    if not example_dir.exists():
        pytest.skip(f"Example data directory not found: {example_dir}")

    return example_dir


@pytest.fixture(scope="session")
def resource_dir(test_data_dir):
    """Get path to resource directory"""
    resource_dir = test_data_dir / "gsMap_resource"
    if not resource_dir.exists():
        pytest.skip(f"Resource directory not found: {resource_dir}")

    return resource_dir


@pytest.fixture(scope="session")
def work_dir(request, tmp_path_factory):
    """
    If --work-dir is provided, use that directory instead of a temporary one.
    """
    custom_dir = request.config.getoption("--work-dir")

    if custom_dir:
        work_dir = Path(custom_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    # Otherwise, use a temporary directory
    return tmp_path_factory.mktemp("Mouse_Embryo")


@pytest.fixture(scope="session")
def homolog_file(resource_dir):
    """Get path to mouse to human homolog file"""
    homolog_path = resource_dir / "homologs/mouse_human_homologs.txt"
    if not homolog_path.exists():
        pytest.skip(f"Homolog file not found: {homolog_path}")
    return homolog_path


@pytest.fixture(scope="session")
def subsampled_h5ad_file1(example_data_dir):
    """Get path to first subsampled h5ad file"""
    h5ad_path = example_data_dir / "ST/E16.5_E1S1.MOSTA_subsampled.h5ad"
    if not h5ad_path.exists():
        pytest.skip(f"Subsampled h5ad file not found: {h5ad_path}")
    return h5ad_path


@pytest.fixture(scope="session")
def subsampled_h5ad_file2(example_data_dir):
    """Get path to second subsampled h5ad file"""
    h5ad_path = example_data_dir / "ST/E16.5_E2S11.MOSTA_subsampled.h5ad"
    if not h5ad_path.exists():
        pytest.skip(f"Subsampled h5ad file not found: {h5ad_path}")
    return h5ad_path


@pytest.fixture(scope="session")
def iq_sumstats_file(example_data_dir):
    """Get path to IQ GWAS summary statistics file"""
    sumstats_path = example_data_dir / "GWAS/filtered_IQ_NG_2018.sumstats.gz"
    if not sumstats_path.exists():
        pytest.skip(f"Summary statistics file not found: {sumstats_path}")
    return sumstats_path


@pytest.fixture(scope="session")
def reference_panel(resource_dir):
    """Get path to reference panel file"""
    ref_panel_path_prefix = (
        resource_dir / "LD_Reference_Panel_subset/1000G.EUR.QC.subset"
    ).as_posix()
    for chromosome in range(1, 23):
        bim_file = Path(f"{ref_panel_path_prefix}.{chromosome}.bim")
        assert bim_file.exists(), f"Reference panel file not found: {bim_file}"
    return ref_panel_path_prefix


@pytest.fixture(scope="session")
def base_config(
    work_dir,
    resource_dir,
    subsampled_h5ad_file1,
    homolog_file,
    iq_sumstats_file,
    reference_panel,
):
    """Create a base RunAllModeConfig fixture"""
    basic_config = QuickModeConfig(
        workdir=work_dir,
        sample_name="test_sample",  # This will be overridden in specific test fixtures
        annotation="annotation",
        data_layer="count",
        homolog_file=str(homolog_file),
        hdf5_path=str(subsampled_h5ad_file1),
        trait_name="IQ",
        sumstats_file=str(iq_sumstats_file),
        max_processes=4,
        gsMap_resource_dir=str(resource_dir),
        n_comps=100,
    )
    basic_config.bfile_root = reference_panel
    return basic_config


@pytest.fixture(scope="session")
def stepbystep_config(base_config):
    """Create a config for step-by-step pipeline tests"""
    config = copy.deepcopy(base_config)
    config.sample_name = "step_by_step_test"

    return config


@pytest.fixture
def quickmode_config(base_config):
    """Create a config for quick mode tests"""
    config = copy.deepcopy(base_config)
    config.sample_name = "quick_mode_test"
    return config


@pytest.fixture
def conditional_config(base_config):
    """Create a config for conditional analysis tests"""
    config = copy.deepcopy(base_config)
    config.sample_name = "conditional_analysis_test"
    return config


@pytest.fixture
def customlatent_config(base_config):
    """Create a config for customized latent tests"""
    config = copy.deepcopy(base_config)
    config.sample_name = "custom_latent_test"
    return config


@pytest.fixture
def biorep_config1(base_config, subsampled_h5ad_file1):
    """Create a config for first biological replicate"""
    config = copy.deepcopy(base_config)
    config.sample_name = "bio_rep_test_1"
    config.hdf5_path = str(subsampled_h5ad_file1)
    return config


@pytest.fixture
def biorep_config2(base_config, subsampled_h5ad_file2):
    """Create a config for second biological replicate"""
    config = copy.deepcopy(base_config)
    config.sample_name = "bio_rep_test_2"
    config.hdf5_path = str(subsampled_h5ad_file2)
    return config


@pytest.fixture
def additional_baseline_dir(test_data_dir):
    """Get path to additional baseline annotations directory"""
    baseline_dir = test_data_dir / "gsMap_additional_annotation"
    if not baseline_dir.exists():
        pytest.skip(f"Additional baseline annotation directory not found: {baseline_dir}")
    return baseline_dir


# conftest.py - add these new fixtures


@pytest.fixture(scope="session")
def latent_representations_fixture(stepbystep_config):
    """Run find_latent_representations step once and share results"""
    config = stepbystep_config
    logger = logging.getLogger("latent_representations_fixture")

    if not config.hdf5_with_latent_path.exists():
        logger.info("Running find_latent_representations...")
        command = f"""
        gsmap run_find_latent_representations \
            --workdir '{config.workdir}' \
            --sample_name {config.sample_name} \
            --input_hdf5_path '{config.hdf5_path}' \
            --annotation '{config.annotation}' \
            --data_layer '{config.data_layer}' \
            --n_comps '{config.n_comps}'
        """
        with patch.object(sys, "argv", parse_bash_command(command)):
            main()
    else:
        logger.info("Using existing latent representations...")

    # Verify the results exist
    assert config.hdf5_with_latent_path.exists(), "Latent representation h5ad file not created"
    assert config.hdf5_with_latent_path.stat().st_size > 0, (
        "Latent representation h5ad file is empty"
    )

    return config.hdf5_with_latent_path


@pytest.fixture(scope="session")
def gene_marker_scores_fixture(latent_representations_fixture, stepbystep_config):
    """Run latent_to_gene step once and share results"""
    config = stepbystep_config
    logger = logging.getLogger("gene_marker_scores_fixture")

    if not config.mkscore_feather_path.exists():
        logger.info("Running latent_to_gene...")
        command = f"""
        gsmap run_latent_to_gene \
            --workdir '{config.workdir}' \
            --sample_name {config.sample_name} \
            --annotation '{config.annotation}' \
            --latent_representation 'latent_GVAE' \
            --num_neighbour {config.num_neighbour} \
            --num_neighbour_spatial {config.num_neighbour_spatial} \
            --homolog_file '{config.homolog_file}'
        """
        with patch.object(sys, "argv", parse_bash_command(command)):
            main()
    else:
        logger.info("Using existing gene marker scores...")

    # Verify the results exist
    assert config.mkscore_feather_path.exists(), "Mkscore feather file not created"
    assert config.mkscore_feather_path.stat().st_size > 0, "Mkscore feather file is empty"

    return config.mkscore_feather_path


@pytest.fixture(scope="session")
def ldscores_fixture(gene_marker_scores_fixture, stepbystep_config):
    """Run generate_ldscore step once and share results"""
    config = stepbystep_config
    logger = logging.getLogger("ldscores_fixture")

    # Check if the generate_ldscore step has been completed
    ldscore_done_file = (
        Path(config.ldscore_save_dir) / f"{config.sample_name}_generate_ldscore.done"
    )
    ldscore_chunk1_file = (
        config.ldscore_save_dir
        / f"{config.sample_name}_chunk1"
        / f"{config.sample_name}.22.l2.ldscore.feather"
    )

    if not ldscore_done_file.exists() or not ldscore_chunk1_file.exists():
        logger.info("Running generate_ldscore...")
        command = f"""
        gsmap run_generate_ldscore \
            --workdir '{config.workdir}' \
            --sample_name {config.sample_name} \
            --chrom all \
            --bfile_root '{config.bfile_root}' \
            --keep_snp_root '{config.keep_snp_root}' \
            --gtf_annotation_file '{config.gtffile}' \
            --gene_window_size 50000
        """
        with patch.object(sys, "argv", parse_bash_command(command)):
            main()

        # Create a done file if it doesn't exist
        if not ldscore_done_file.exists():
            ldscore_done_file.touch()
    else:
        logger.info("Using existing LD scores...")

    # Verify the results exist
    assert ldscore_chunk1_file.exists(), "LDScore chunk1 file not created"
    assert ldscore_chunk1_file.stat().st_size > 0, "LDScore chunk1 file is empty"

    return config.ldscore_save_dir


@pytest.fixture(scope="session")
def spatial_ldsc_fixture(ldscores_fixture, stepbystep_config):
    """Run spatial_ldsc step once and share results"""
    config = stepbystep_config
    logger = logging.getLogger("spatial_ldsc_fixture")

    spatial_ldsc_result = config.get_ldsc_result_file(config.trait_name)

    if not spatial_ldsc_result.exists():
        logger.info("Running spatial_ldsc...")
        command = f"""
        gsmap run_spatial_ldsc \
            --workdir '{config.workdir}' \
            --sample_name {config.sample_name} \
            --trait_name '{config.trait_name}' \
            --sumstats_file '{config.sumstats_file}' \
            --w_file '{config.w_file}' \
            --num_processes {config.max_processes}
        """
        with patch.object(sys, "argv", parse_bash_command(command)):
            main()
    else:
        logger.info("Using existing spatial LDSC results...")

    # Verify the results exist
    assert spatial_ldsc_result.exists(), "Spatial LDSC results not created"
    assert spatial_ldsc_result.stat().st_size > 0, "Spatial LDSC results file is empty"

    return spatial_ldsc_result


@pytest.fixture(scope="session")
def cauchy_combination_fixture(spatial_ldsc_fixture, stepbystep_config):
    """Run cauchy_combination step once and share results"""
    config = stepbystep_config
    logger = logging.getLogger("cauchy_combination_fixture")

    cauchy_result = config.get_cauchy_result_file(config.trait_name)

    if not cauchy_result.exists():
        logger.info("Running cauchy_combination...")
        command = f"""
        gsmap run_cauchy_combination \
            --workdir '{config.workdir}' \
            --sample_name {config.sample_name} \
            --trait_name '{config.trait_name}' \
            --annotation '{config.annotation}'
        """
        with patch.object(sys, "argv", parse_bash_command(command)):
            main()
    else:
        logger.info("Using existing Cauchy combination results...")

    # Verify the results exist
    assert cauchy_result.exists(), "Cauchy combination results not created"
    assert cauchy_result.stat().st_size > 0, "Cauchy combination results file is empty"

    return cauchy_result


@pytest.fixture
def symbolic_link_results(request, base_config):
    """Create symbolic links for test-specific directories to shared fixtures"""
    # Get the test-specific config
    test_config_name = request.param
    test_config = request.getfixturevalue(test_config_name)

    # Create source config with fixture sample name
    source_config = copy.deepcopy(base_config)
    source_config.sample_name = "step_by_step_test"  # The sample name used in fixtures

    # Create parent directories
    Path(test_config.workdir).mkdir(parents=True, exist_ok=True)
    Path(test_config.workdir / test_config.sample_name).mkdir(parents=True, exist_ok=True)

    # Logger for debugging
    logger = logging.getLogger("symbolic_link_results")

    # Define link mappings based on test type
    mappings = []

    if test_config_name in ["quickmode_config", "conditional_config"]:
        # For quick mode and bioreps, link latent representations and marker scores
        mappings = [
            (source_config.hdf5_with_latent_path, test_config.hdf5_with_latent_path),
            (source_config.mkscore_feather_path, test_config.mkscore_feather_path),
        ]
    elif test_config_name == "customlatent_config":
        # For custom latent test, only link latent representations
        mappings = [(source_config.hdf5_with_latent_path, test_config.hdf5_with_latent_path)]
    elif test_config_name == "biorep_config1":
        mappings = [
            (source_config.hdf5_with_latent_path, test_config.hdf5_with_latent_path),
            (
                source_config.get_ldsc_result_file(test_config.trait_name),
                test_config.get_ldsc_result_file(test_config.trait_name),
            ),
        ]

        # Create symbolic links
    for source_path, target_path in mappings:
        if source_path.exists() and not target_path.exists():
            # Create parent directory if needed
            target_path.parent.mkdir(exist_ok=True, parents=True)

            # Create symbolic link
            target_path.symlink_to(source_path)
            logger.info(f"Created symlink from {target_path} to {source_path}")
        elif not source_path.exists():
            logger.warning(f"Source path does not exist: {source_path}")

    return test_config


def pytest_configure(config):
    """Configure pytest with custom markers and default options"""
    # Add marker descriptions
    config.addinivalue_line(
        "markers", "real_data: mark test that requires real data (disabled by default)"
    )

    # Set default options (equivalent to addopts in pytest.ini)
    if not config.option.verbose:
        config.option.verbose = True


def pytest_collection_modifyitems(config, items):
    """Skip real_data tests by default unless --run-real-data is specified"""
    if not config.getoption("--run-real-data"):
        skip_real_data = pytest.mark.skip(reason="need --run-real-data option to run")
        for item in items:
            if "real_data" in item.keywords:
                item.add_marker(skip_real_data)

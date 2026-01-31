import logging
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from gsMap.cli import main


def parse_bash_command(command: str) -> list[str]:
    """Convert multi-line bash command to argument list for sys.argv"""
    cleaned_command = command.replace("\\\n", " ")
    cleaned_command = " ".join(cleaned_command.splitlines())
    cleaned_command = " ".join(cleaned_command.split())
    return shlex.split(cleaned_command)


@pytest.mark.real_data
@pytest.mark.parametrize("symbolic_link_results", ["conditional_config"], indirect=True)
def test_conditional_analysis(
    symbolic_link_results,
    gene_marker_scores_fixture,
    additional_baseline_dir,
    spatial_ldsc_fixture,
):
    """Test the conditional analysis functionality by providing additional baseline annotations"""
    logger = logging.getLogger("test_conditional_analysis")
    config = symbolic_link_results  # This will have links to latent_to_gene data

    logger.info("Using linked fixtures for latent representations and gene marker scores")

    # Step 3: Generate LDScores with additional baseline annotation
    logger.info("Step 3: Generating LDScores with additional baseline annotation")
    command = f"""
    gsmap run_generate_ldscore \
        --workdir '{config.workdir}' \
        --sample_name {config.sample_name} \
        --chrom all \
        --bfile_root '{config.bfile_root}' \
        --keep_snp_root '{config.keep_snp_root}' \
        --gtf_annotation_file '{config.gtffile}' \
        --gene_window_size 50000 \
        --additional_baseline_annotation '{additional_baseline_dir}'
    """
    with patch.object(sys, "argv", parse_bash_command(command)):
        main()

    # Verify additional baseline annotation directory was created
    additional_baseline_dir_output = (
        Path(config.workdir) / config.sample_name / "generate_ldscore" / "additional_baseline"
    )
    assert additional_baseline_dir_output.exists(), "Additional baseline directory was not created"

    # Step 4: Run spatial LDSC using the additional baseline annotation
    logger.info("Step 4: Running spatial LDSC with additional baseline annotation")
    command = f"""
    gsmap run_spatial_ldsc \
        --workdir '{config.workdir}' \
        --sample_name {config.sample_name} \
        --trait_name '{config.trait_name}' \
        --sumstats_file '{config.sumstats_file}' \
        --w_file '{config.w_file}' \
        --num_processes {config.max_processes} \
        --use_additional_baseline_annotation True
    """
    with patch.object(sys, "argv", parse_bash_command(command)):
        main()

    # Verify LDSC results
    ldsc_result_file = config.get_ldsc_result_file(config.trait_name)
    assert ldsc_result_file.exists(), "LDSC result file was not created"

    logger.info("Conditional analysis test completed successfully")


@pytest.mark.real_data
@pytest.mark.parametrize("symbolic_link_results", ["biorep_config1"], indirect=True)
def test_biological_replicates(symbolic_link_results, biorep_config2, work_dir):
    """Test gsMap on biological replicates using the slice mean functionality"""
    logger = logging.getLogger("test_biological_replicates")
    config1 = symbolic_link_results
    config2 = biorep_config2  # Just use the config directly without linking

    slice_mean_file = work_dir / "slice_mean_test.parquet"

    # Step 1: Create the slice mean from multiple samples
    logger.info("Step 1: Creating slice mean from multiple samples")
    command = f"""
    gsmap create_slice_mean \
        --sample_name_list {config1.sample_name} {config2.sample_name} \
        --h5ad_list {config1.hdf5_path} {config2.hdf5_path} \
        --slice_mean_output_file {slice_mean_file} \
        --data_layer '{config1.data_layer}' \
        --homolog_file '{config1.homolog_file}'
    """
    with patch.object(sys, "argv", parse_bash_command(command)):
        main()

    # Verify slice mean file was created
    assert slice_mean_file.exists(), "Slice mean file was not created"

    # Rest of the test continues as before...
    # Verify slice mean file contains expected data
    slice_mean_df = pd.read_parquet(slice_mean_file)
    assert "G_Mean" in slice_mean_df.columns, "G_Mean column not found in slice mean file"
    assert "frac" in slice_mean_df.columns, "frac column not found in slice mean file"
    assert len(slice_mean_df) > 0, "Slice mean file is empty"

    # Update config with slice mean file
    config1.gM_slices = str(slice_mean_file)

    # Step 2: Test using the slice mean with latent_to_gene
    logger.info("Step 2: Using slice mean with quick_mode")
    command = f"""
    gsmap run_latent_to_gene \
        --workdir '{config1.workdir}' \
        --sample_name {config1.sample_name} \
        --annotation '{config1.annotation}' \
        --latent_representation 'latent_GVAE' \
        --num_neighbour {config1.num_neighbour} \
        --num_neighbour_spatial {config1.num_neighbour_spatial} \
        --homolog_file '{config1.homolog_file}'\
        --gM_slices '{config1.gM_slices}'
    """
    with patch.object(sys, "argv", parse_bash_command(command)):
        main()

    # Verify quick_mode results
    mkscore_file = config1.mkscore_feather_path
    assert mkscore_file.exists(), "Marker score file was not created in quick_mode"

    # Step 3: Run Cauchy combination across multiple samples
    combined_result_file = work_dir / "combined_IQ_cauchy.csv.gz"
    command = f"""
    gsmap run_cauchy_combination \
        --workdir '{config1.workdir}' \
        --sample_name_list {config1.sample_name} {config1.sample_name} \
        --trait_name '{config1.trait_name}' \
        --annotation '{config1.annotation}' \
        --output_file '{combined_result_file}'
    """
    with patch.object(sys, "argv", parse_bash_command(command)):
        main()

    # Verify combined result file was created
    assert combined_result_file.exists(), "Combined Cauchy result file was not created"

    logger.info("Biological replicates test completed successfully")


@pytest.mark.real_data
@pytest.mark.parametrize("symbolic_link_results", ["customlatent_config"], indirect=True)
def test_customized_latent_representations(symbolic_link_results, latent_representations_fixture):
    """Test using customized latent representations in gsMap"""
    logger = logging.getLogger("test_customized_latent")
    config = symbolic_link_results  # This will have links to latent_representation data

    logger.info("Using linked fixtures for initial latent representations")

    # Step 2: Use the PCA latent representation instead of the default GVAE
    custom_latent = "latent_PCA"
    logger.info(f"Step 2: Using custom latent representation: {custom_latent}")
    command = f"""
    gsmap run_latent_to_gene \
        --workdir '{config.workdir}' \
        --sample_name {config.sample_name} \
        --annotation '{config.annotation}' \
        --latent_representation '{custom_latent}' \
        --num_neighbour {config.num_neighbour} \
        --num_neighbour_spatial {config.num_neighbour_spatial} \
        --homolog_file '{config.homolog_file}'
    """
    with patch.object(sys, "argv", parse_bash_command(command)):
        main()

    # Verify mkscore file was created with the custom latent
    mkscore_file = config.mkscore_feather_path
    assert mkscore_file.exists(), f"Marker score file was not created with {custom_latent}"

    logger.info("Customized latent representations test completed successfully")

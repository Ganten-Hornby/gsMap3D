"""Keep the shared spatial-LDSC batch default and explicit overrides consistent."""

from dataclasses import fields

import click
import pytest
from typer.main import get_command

from gsMap.cli import app
from gsMap.config.quick_mode_config import QuickModeConfig
from gsMap.config.spatial_ldsc_config import SpatialLDSCComputeConfig, SpatialLDSCConfig


@pytest.mark.parametrize(
    "config_type", [SpatialLDSCComputeConfig, SpatialLDSCConfig, QuickModeConfig]
)
def test_spatial_ldsc_batch_default(config_type):
    config_fields = {field.name: field for field in fields(config_type)}
    assert config_fields["spots_per_chunk_quick_mode"].default == 100


def test_python_batch_override():
    assert SpatialLDSCComputeConfig().spots_per_chunk_quick_mode == 100
    assert SpatialLDSCComputeConfig(spots_per_chunk_quick_mode=50).spots_per_chunk_quick_mode == 50


@pytest.mark.parametrize("command_name", ["spatial-ldsc", "quick-mode"])
def test_cli_batch_default_and_override(command_name):
    command = get_command(app).commands[command_name]
    parameter = next(p for p in command.params if p.name == "spots_per_chunk_quick_mode")
    with click.Context(command) as context:
        assert parameter.get_default(context) == 100
        assert parameter.process_value(context, "50") == 50
        assert parameter.process_value(context, "1") == 1
        with pytest.raises(click.BadParameter):
            parameter.process_value(context, "0")


@pytest.mark.parametrize("command_name", ["spatial-ldsc", "quick-mode"])
def test_cli_multi_gpu_controls(command_name):
    command = get_command(app).commands[command_name]
    parameters = {parameter.name: parameter for parameter in command.params}
    with click.Context(command) as context:
        assert parameters["ldsc_num_gpus"].get_default(context) == 1
        assert parameters["ldsc_num_gpus"].process_value(context, "8") == 8
        assert parameters["ldsc_share_ld"].get_default(context) is True
        assert parameters["ldsc_trait_batch_size"].get_default(context) == 4
        with pytest.raises(click.BadParameter):
            parameters["ldsc_num_gpus"].process_value(context, "0")


@pytest.mark.parametrize("config_type", [SpatialLDSCConfig, QuickModeConfig])
def test_multi_gpu_config_defaults(config_type):
    config_fields = {field.name: field for field in fields(config_type)}
    assert config_fields["ldsc_num_gpus"].default == 1
    assert config_fields["ldsc_share_ld"].default is True


@pytest.mark.parametrize("option", [{"use_gpu": False}, {"ldsc_share_ld": False}])
def test_multi_gpu_rejects_incompatible_python_configuration(tmp_path, option):
    gwas = tmp_path / "gwas"
    gwas.touch()
    with pytest.raises(ValueError, match="Multiple LDSC GPUs require"):
        SpatialLDSCConfig(
            workdir=tmp_path,
            project_name="p",
            sumstats_config_dict={"trait": gwas},
            ldsc_num_gpus=2,
            **option,
        )


def test_multi_gpu_configuration_does_not_initialize_parent_backend(tmp_path, monkeypatch):
    from gsMap.config import utils

    monkeypatch.setattr(
        utils, "configure_jax_platform", lambda *args: pytest.fail("Parent initialized JAX")
    )
    gwas = tmp_path / "gwas"
    gwas.touch()
    config = SpatialLDSCConfig(
        workdir=tmp_path,
        project_name="p",
        sumstats_config_dict={"trait": gwas},
        ldsc_num_gpus=2,
        snp_gene_weight_adata_path=tmp_path / "weight.h5ad",
        w_ld_dir=tmp_path,
    )
    assert config.ldsc_num_gpus == 2

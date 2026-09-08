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

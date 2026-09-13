"""Automatic CPU/GPU LDSC batches, public help and explicit overrides."""

from dataclasses import fields
from types import SimpleNamespace

import click
import pytest
from typer.main import get_command
from typer.testing import CliRunner

from gsMap.cli import app
from gsMap.config.quick_mode_config import QuickModeConfig
from gsMap.config.spatial_ldsc_config import (
    SpatialLDSCComputeConfig,
    SpatialLDSCConfig,
    resolve_spatial_ldsc_batch_size,
)


@pytest.mark.parametrize(
    "config_type", [SpatialLDSCComputeConfig, SpatialLDSCConfig, QuickModeConfig]
)
def test_spatial_ldsc_batch_default(config_type):
    config_fields = {field.name: field for field in fields(config_type)}
    assert config_fields["spots_per_chunk_quick_mode"].default is None


def test_python_batch_override():
    assert SpatialLDSCComputeConfig().spots_per_chunk_quick_mode is None
    assert SpatialLDSCComputeConfig(spots_per_chunk_quick_mode=50).spots_per_chunk_quick_mode == 50


@pytest.mark.parametrize("command_name", ["spatial-ldsc", "quick-mode"])
def test_cli_batch_default_and_override(command_name):
    command = get_command(app).commands[command_name]
    parameter = next(p for p in command.params if p.name == "spots_per_chunk_quick_mode")
    with click.Context(command) as context:
        assert parameter.get_default(context) is None
        assert parameter.show_default == "auto: CPU=400, GPU=100"
        assert parameter.process_value(context, "50") == 50
        assert parameter.process_value(context, "100") == 100
        assert parameter.process_value(context, "400") == 400
        assert parameter.process_value(context, "1") == 1
        with pytest.raises(click.BadParameter):
            parameter.process_value(context, "0")


@pytest.mark.parametrize("command_name", ["spatial-ldsc", "quick-mode"])
def test_cli_help_displays_device_dependent_default(command_name):
    result = CliRunner().invoke(app, [command_name, "--help"], env={"COLUMNS": "240"})
    assert result.exit_code == 0
    assert "--spots-per-chunk-quick-mode" in result.output
    assert "auto: CPU=400, GPU=100" in result.output


@pytest.mark.parametrize(("platform", "expected"), [("cpu", 400), ("gpu", 100), ("tpu", 100)])
def test_resolve_automatic_batch(platform, expected):
    assert resolve_spatial_ldsc_batch_size(None, platform) == expected


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
@pytest.mark.parametrize("batch", [1, 50, 100, 400, 1600])
def test_explicit_batch_always_wins(platform, batch):
    assert resolve_spatial_ldsc_batch_size(batch, platform) == batch


@pytest.mark.parametrize("batch", [0, -1, True, 1.5, "400"])
def test_invalid_python_batch_rejected(batch):
    with pytest.raises(ValueError, match="positive integer or None"):
        resolve_spatial_ldsc_batch_size(batch, "cpu")


@pytest.mark.parametrize("mode", ["weight_matrix", "direct_ldscore"])
@pytest.mark.parametrize(
    ("platform", "gpus", "requested", "expected"),
    [
        ("cpu", 1, None, 400),
        ("gpu", 1, None, 100),
        ("gpu", 4, None, 100),
        ("cpu", 1, 100, 100),
        ("gpu", 1, 400, 400),
        ("gpu", 4, 50, 50),
    ],
)
def test_engine_resolves_batch_and_keeps_tail(
    tmp_path, monkeypatch, caplog, mode, platform, gpus, requested, expected
):
    import anndata as ad
    import jax
    import numpy as np
    import pandas as pd

    from gsMap.spatial_ldsc.execution.engine import SpatialLDSCEngine

    def execution_platform():
        assert gpus == 1, "Unnecessary coordinator device initialization"
        assert requested is None, "An explicit batch needs no device initialization"
        return platform

    monkeypatch.setattr(jax, "default_backend", execution_platform)
    config = SimpleNamespace(
        sample_filter=None,
        cell_indices_range=None,
        use_gpu=True,
        ldsc_num_gpus=gpus,
        spots_per_chunk_quick_mode=requested,
    )  # use_gpu=True with platform=cpu exercises actual fallback sizing.
    cells, genes = 811, 201
    if mode == "direct_ldscore":
        marker, weight = None, None
        direct = ad.AnnData(np.ones((2, cells), np.float32))
    else:
        marker = ad.AnnData(np.ones((cells, genes), np.float32))
        weight = ad.AnnData(np.ones((2, genes), np.float32))
        direct = None
    caplog.set_level("INFO", logger="gsMap.spatial_ldsc.engine")
    engine = SpatialLDSCEngine(
        config, tmp_path, marker, weight, pd.DataFrame(), pd.DataFrame(), ldscore_adata=direct
    )
    assert engine.chunk_size == expected
    assert engine.chunk_starts == list(range(0, cells, expected))
    assert engine._cell_end == cells
    assert config.spots_per_chunk_quick_mode is requested  # Keep the user's auto policy.
    source = f"auto: {platform.upper()}" if requested is None else "explicit override"
    assert f"batch size: {expected} spots/chunk ({source})" in caplog.text


@pytest.mark.parametrize("requested", [None, 100, 400])
def test_quick_mode_passes_batch_policy_to_spatial_ldsc(tmp_path, monkeypatch, requested):
    from gsMap.config import utils

    monkeypatch.setattr(utils, "configure_jax_platform", lambda *args: None)
    sumstats = tmp_path / "trait.sumstats.gz"
    sumstats.touch()
    config = QuickModeConfig(
        workdir=tmp_path,
        project_name="batch",
        annotation="annotation",
        sumstats_config_dict={"trait": sumstats},
        start_step="spatial_ldsc",
        stop_step="spatial_ldsc",
        use_gpu=False,
        snp_gene_weight_adata_path=tmp_path / "weights.h5ad",
        w_ld_dir=tmp_path,
        spots_per_chunk_quick_mode=requested,
    )
    assert config.spatial_ldsc_config.spots_per_chunk_quick_mode == requested


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

"""Public CLI entry points and graph controls."""

import pytest

from gsMap.cli import app


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["quick-mode"],
        ["find-latent"],
        ["latent-to-gene"],
        ["spatial-ldsc"],
        ["cauchy-combination"],
        ["ldscore-weight-matrix"],
        ["format-sumstats"],
        ["report-view"],
    ],
)
def test_cli_help(cli_runner, command):
    result = cli_runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output


def test_cli_version(cli_runner):
    result = cli_runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower() or "gsmap" in result.output.lower()


@pytest.mark.parametrize("command", ["find-latent", "quick-mode"])
def test_section_graph_options(cli_runner, command):
    result = cli_runner.invoke(app, [command, "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for option in ("--graph-mode", "--graph-z-scale"):
        assert option in result.output
    assert "--graph-section-key" not in result.output
    if command == "quick-mode":
        assert "--inference-chunk-size" in result.output

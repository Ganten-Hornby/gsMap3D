import re
import shlex
from pathlib import Path

import pytest

from gsMap.cli import app

# ---------------------------------------------------------------------------
# Helpers for extracting CLI commands from markdown documentation
# ---------------------------------------------------------------------------


def extract_bash_blocks(markdown_text: str) -> list[str]:
    """Extract fenced bash/shell code blocks from markdown text."""
    pattern = r"(?:```|~~~)(?:bash|shell|sh)\s*\n(.*?)(?:```|~~~)"
    return [m.group(1).strip() for m in re.finditer(pattern, markdown_text, re.DOTALL)]


def parse_gsmap_commands(bash_script: str) -> list[dict]:
    """Find gsmap invocations in a bash script and split into argument lists."""

    def join_continuation_lines(script: str) -> str:
        lines = script.split("\n")
        joined, current = [], ""
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if current:
                    joined.append(current)
                    current = ""
                continue
            if stripped.endswith("\\"):
                current += stripped[:-1].strip() + " "
            else:
                current += stripped
                joined.append(current)
                current = ""
        if current:
            joined.append(current)
        return "\n".join(joined)

    processed = join_continuation_lines(bash_script)
    gsmap_re = re.compile(r"\b(?:/?\w+/)*gsmap(?:\.(?:exe|sh))?\s+(.*)", re.VERBOSE)
    commands = []
    for m in gsmap_re.finditer(processed):
        full = m.group(0).strip()
        args_str = m.group(1).strip()
        args = shlex.split(args_str)
        commands.append({"full_command": full, "arguments": args})
    return commands


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner_local():
    from typer.testing import CliRunner

    return CliRunner()


SUBCOMMANDS = [
    "quick-mode",
    "find-latent",
    "latent-to-gene",
    "spatial-ldsc",
    "cauchy-combination",
    "ldscore-weight-matrix",
    "format-sumstats",
]


def test_cli_help(cli_runner_local):
    """Verify the top-level --help exits cleanly."""
    result = cli_runner_local.invoke(app, ["--help"])
    assert result.exit_code == 0, f"--help failed:\n{result.output}"


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_cli_subcommand_help(cli_runner_local, subcommand):
    """Verify each subcommand's --help exits cleanly."""
    result = cli_runner_local.invoke(app, [subcommand, "--help"])
    assert result.exit_code == 0, f"{subcommand} --help failed:\n{result.output}"


def test_cli_version(cli_runner_local):
    """Verify --version prints the version string."""
    result = cli_runner_local.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower() or "gsmap" in result.output.lower()


@pytest.fixture
def tutorial_files():
    """Markdown documentation files that may contain gsmap CLI examples."""
    docs_dir = Path("docs/source")
    if not docs_dir.exists():
        return []
    return [str(p) for p in docs_dir.rglob("*.md")]


def test_docs_commands_parseable(cli_runner_local, tutorial_files):
    """Extract gsmap commands from docs and verify they are parseable by typer."""
    if not tutorial_files:
        pytest.skip("No documentation files found")

    # Error patterns that indicate real parsing failures (unrecognised options/args).
    # File-not-found, directory-not-found, and missing required options are expected
    # because docs use placeholder paths and partial command examples.
    parsing_error_patterns = re.compile(
        r"No such option|Got unexpected extra argument",
        re.IGNORECASE,
    )

    for file_path in tutorial_files:
        markdown_text = Path(file_path).read_text(encoding="utf-8")
        bash_blocks = extract_bash_blocks(markdown_text)
        for block in bash_blocks:
            commands = parse_gsmap_commands(block)
            for cmd in commands:
                result = cli_runner_local.invoke(app, cmd["arguments"], catch_exceptions=True)
                # exit_code 2 can mean either a real parsing error (unknown option)
                # or a validation error (file does not exist).  Only fail on the former.
                if result.exit_code == 2 and parsing_error_patterns.search(result.output):
                    pytest.fail(
                        f"Typer could not parse command from docs: {cmd['full_command']}\n"
                        f"Output: {result.output}"
                    )

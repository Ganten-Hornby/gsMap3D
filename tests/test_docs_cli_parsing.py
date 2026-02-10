import re
import shlex
from pathlib import Path

import pytest

from gsMap.cli import (
    create_parser,  # type: ignore[attr-defined]  # TODO: update test for typer CLI
)


# Import your original functions
def extract_bash_blocks(markdown_text):
    bash_block_pattern = r"""
        (?:
            (?:```|~~~)(?:bash|shell|sh){1,}\s*\n
            (.*?)
            (?:```|~~~)
        )
    """
    blocks = re.finditer(bash_block_pattern, markdown_text, re.VERBOSE | re.DOTALL)
    return [block.group(1).strip() for block in blocks]


def parse_gsmap_commands(bash_script):
    def join_multiline_commands(script):
        lines = script.split("\n")
        joined_lines = []
        current_line = ""

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                if current_line:
                    joined_lines.append(current_line)
                    current_line = ""
                continue

            if line.endswith("\\"):
                current_line += line[:-1].strip() + " "
            else:
                current_line += line
                joined_lines.append(current_line)
                current_line = ""

        if current_line:
            joined_lines.append(current_line)

        return "\n".join(joined_lines)

    gsmap_pattern = r"""
        \b(?:\/?\w+\/)*gsmap
        (?:\.(?:exe|sh))?
        \s+
        (.*)
    """

    processed_script = join_multiline_commands(bash_script)
    matches = re.finditer(gsmap_pattern, processed_script, re.VERBOSE)

    gsmap_commands = []
    for match in matches:
        full_command = match.group(0).strip()
        args_str = match.group(1).strip()
        args = parse_bash_command(args_str)
        gsmap_commands.append({"full_command": full_command, "arguments": args})

    return gsmap_commands


def parse_markdown_gsmap_commands(markdown_text):
    all_commands = []
    bash_blocks = extract_bash_blocks(markdown_text)

    for i, block in enumerate(bash_blocks, 1):
        commands = parse_gsmap_commands(block)
        if commands:
            all_commands.append({"block_number": i, "commands": commands})

    return all_commands


def parse_bash_command(command: str) -> list[str]:
    cleaned_command = command.replace("\\\n", " ")
    cleaned_command = " ".join(cleaned_command.splitlines())
    cleaned_command = " ".join(cleaned_command.split())
    return shlex.split(cleaned_command)


# Test fixtures
@pytest.fixture
def tutorial_files():
    return [
        "docs/source/advanced_usage.md",
        "docs/source/data_format.md",
        "docs/source/quick_mode.md",
        "docs/source/step_by_step.md",
    ]


@pytest.fixture
def gsmap_parser():
    return create_parser()


# Test functions
def test_markdown_files_exist(tutorial_files):
    """Test if all documentation files exist"""
    for file_path in tutorial_files:
        assert Path(file_path).exists(), f"File {file_path} does not exist"


def test_parse_commands_from_all_docs(tutorial_files, gsmap_parser):
    """Test if all gsmap commands in documentation can be parsed"""
    for file_path in tutorial_files:
        markdown_content = Path(file_path).read_text()
        parsed_commands = parse_markdown_gsmap_commands(markdown_content)
        # Test each command block
        for command_block in parsed_commands:
            for cmd in command_block["commands"]:
                try:
                    args = gsmap_parser.parse_args(cmd["arguments"])
                    assert hasattr(args, "func"), (
                        f"Command missing function handler: {cmd['full_command']}"
                    )
                except SystemExit as e:
                    pytest.fail(f"Failed to parse command: {cmd['full_command']}\n{e}")

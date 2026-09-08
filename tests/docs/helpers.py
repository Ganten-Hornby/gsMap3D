"""Inspect documented CLI options without executing analysis callbacks."""

import ast
import shlex

from click import Context, UsageError
from typer.main import get_command

from gsMap.cli import app


def _literal(value):
    if isinstance(value, tuple | list):
        return [_literal(item) for item in value]
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def cli_settings(code):
    """Parse syntax only; leave data-path validation and callbacks to integration tests.

    Click's parser handles positional arguments, repeated options, and flag pairs
    without calling parameter converters (which may check local files). Partial
    examples may omit required options, but must not contain invalid syntax.
    """
    steps = []
    root = get_command(app)
    for line in code.replace("\\\n", " ").splitlines():
        if not line.lstrip().startswith("gsmap "):
            continue
        argv = shlex.split(line, comments=True)
        if argv[1].startswith("-"):
            command, name, index = root, "gsmap", 1
        else:
            name, index = argv[1], 2
            if name not in root.commands:
                raise ValueError(f"Unknown gsmap command: {name}")
            command = root.commands[name]
        with Context(command) as context:
            try:
                values, extra, _ = command.make_parser(context).parse_args(argv[index:])
            except UsageError as error:
                raise ValueError(f"Invalid arguments for {name}: {error}") from error
        if extra:
            raise ValueError(f"Unexpected arguments for {name}: {extra}")
        steps.append((name, {key: _literal(value) for key, value in values.items()}))
    return steps

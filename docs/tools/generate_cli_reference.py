"""Generate CLI reference from Click parameters without importing compute backends."""

from pathlib import Path

from click import Context
from typer.main import get_command

from gsMap.cli import app


def render():
    command = get_command(app)
    lines = [
        "# CLI reference",
        "",
        "Generated from the current `gsMap.cli` application; do not edit by hand.",
        "",
    ]
    for name, cmd in [("gsmap", command)] + [
        (f"gsmap {name}", cmd) for name, cmd in command.commands.items()
    ]:
        ctx = Context(cmd, info_name=name)
        lines.extend(
            [f"## `{name}`", "", cmd.help or "", "", "| Option | Description |", "|---|---|"]
        )
        for parameter in cmd.get_params(ctx):
            record = parameter.get_help_record(ctx)
            if record:
                option, description = record
                option = option.replace("|", "\\|")
                description = description.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{option}` | {description} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    Path("docs/source/cli_reference.md").write_text(render())

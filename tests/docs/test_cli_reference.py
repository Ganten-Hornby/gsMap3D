"""Keep the public CLI reference in sync with the application."""

from docs.tools.generate_cli_reference import render

from tests.paths import REPO_ROOT


def test_cli_reference_matches_current_application():
    assert (REPO_ROOT / "docs/source/cli_reference.md").read_text() == render()

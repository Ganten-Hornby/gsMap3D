"""Check every documented CLI option without running commands on local data."""

import pytest
from scripts.docs.audit_commands import blocks

from tests.docs.helpers import cli_settings
from tests.paths import REPO_ROOT


def test_documented_cli_flags_exist():
    checked = []
    for path in sorted((REPO_ROOT / "docs/source").rglob("*")):
        if path.suffix not in {".md", ".rst"}:
            continue
        for line, language, code in blocks(path):
            if language not in {"bash", "shell"}:
                continue
            try:
                checked.extend(cli_settings(code))
            except ValueError as error:
                pytest.fail(f"{path}:{line}: {error}")
    assert len(checked) >= 20


@pytest.mark.parametrize(
    "code",
    [
        "gsmap quick-mode --h5ad-path missing.h5ad --unknown-flag",
        "gsmap removed-command --help",
        "gsmap spatial-ldsc --spots-per-chunk-quick-mode",
    ],
)
def test_documented_flag_audit_rejects_bad_commands_even_with_missing_inputs(code):
    with pytest.raises(ValueError, match="Unknown|Invalid"):
        cli_settings(code)


def test_documented_flag_audit_accepts_positional_paths_without_opening_them():
    assert cli_settings("gsmap report-view missing/report --port 8080 --no-browser") == [
        ("report-view", {"report_path": "missing/report", "port": 8080, "no_browser": True})
    ]

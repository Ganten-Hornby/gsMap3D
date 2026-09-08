"""Keep both custom LD examples executable and consistent with the public catalog."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.docs.test_ld_tutorial import ld_tutorial_blocks
from tests.test_direct_tutorials import cli_settings

import gsMap.config
import gsMap.ldscore.pipeline

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("number", "strategy", "window"), [(0, "tss", 50000), (1, "center", 0)])
def test_ld_cli_python_settings_match(number, strategy, window, monkeypatch):
    recipes = ld_tutorial_blocks(REPO / "docs/source/ldscore_weight_matrix.md")
    assert len(recipes["cli"]) == len(recipes["python"]) == 2
    recorded = []

    def capture(**kwargs):
        recorded.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(gsMap.config, "LDScoreConfig", capture)
    monkeypatch.setattr(
        gsMap.ldscore.pipeline.LDScorePipeline, "__init__", lambda self, config: None
    )
    monkeypatch.setattr(gsMap.ldscore.pipeline.LDScorePipeline, "run", lambda self: None)
    exec(compile(recipes["python"][number], "LD tutorial", "exec"), {})
    _, cli = cli_settings(recipes["cli"][number])[0]
    python = {
        key: str(value) if isinstance(value, Path) else value for key, value in recorded[0].items()
    }
    cli["chromosomes"] = str(cli["chromosomes"])
    assert cli == python
    assert cli["strategy"] == strategy
    assert cli["feature_window_size"] == window
    assert cli["calculate_w_ld"] is True
    assert cli["mac_min"] == 5
    assert cli["ld_unit"] == "CM"
    assert cli["mhc_start_bp"] == 25000000
    assert cli["mhc_end_bp"] == 34000000


def test_ld_tutorial_keeps_enhancer_inputs_and_simple_setup():
    text = (REPO / "docs/source/ldscore_weight_matrix.md").read_text()
    assert text.index("## Files needed for this tutorial") < text.index("```bash")
    assert "BRN/ABC_roadmap_merged.bed" in text
    for word in ("score", "tss", "center", "allow_repeat"):
        assert f"--strategy {word}" in text
    assert "kidney folder has no separate ABC annotation" in text
    assert "GSMAP_DATA_URL" not in text
    assert "staged" not in text.lower()
    assert "## 1. Download" in text
    assert "## 3. View and use the outputs" in text


def test_reference_settings_document_current_qc_defaults():
    from dataclasses import fields

    defaults = {field.name: field.default for field in fields(gsMap.config.LDScoreConfig)}
    assert defaults["mac_min"] == 5
    assert defaults["maf_min"] is None
    assert defaults["snps_only"] is True
    assert defaults["exclude_mhc_from_regression"] is True
    assert defaults["mhc_start_bp"] == 28510120
    assert defaults["mhc_end_bp"] == 33480577
    assert defaults["ld_unit"] == "CM"
    assert defaults["ld_wind"] == 1
    assert defaults["calculate_w_ld"] is False

    text = (REPO / "docs/source/ldscore_weight_matrix.md").read_text()
    reference = text.split("### Reference settings", 1)[1]
    for name in (
        "snps_only",
        "mac_min",
        "maf_min",
        "ld_unit",
        "ld_wind",
        "exclude_mhc_from_regression",
        "mhc_start_bp",
        "mhc_end_bp",
        "calculate_w_ld",
        "w_ld_dir",
    ):
        assert f"`--{name.replace('_', '-')}`" in reference
    assert "MAC > 5" in reference
    assert "MAF ≥ 0.01" in reference
    assert "as well as passing the MAC filter" in reference
    assert "`28510120` / `33480577`" in reference
    assert "`25000000` / `34000000`" in reference

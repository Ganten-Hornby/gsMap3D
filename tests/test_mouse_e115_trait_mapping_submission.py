from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
SBATCH = (
    PROJECT_ROOT
    / "analysis"
    / "revision_graph_3d"
    / "scripts"
    / "run_mouse_e115_supervised_trait_mapping.sbatch"
)
PREPARATION_SCRIPT = (
    PROJECT_ROOT
    / "analysis"
    / "revision_graph_3d"
    / "scripts"
    / "prepare_mouse_e115_supervised_trait_inputs.py"
)
REPORT_SBATCH = (
    PROJECT_ROOT
    / "analysis"
    / "revision_graph_3d"
    / "scripts"
    / "run_mouse_e115_supervised_trait_reports.sbatch"
)


def test_trait_mapping_uses_one_quick_mode_pipeline():
    script = SBATCH.read_text()

    assert script.count('"${GSMAP}" quick-mode') == 1
    assert "--start-step latent2gene" in script
    assert "--stop-step cauchy" in script
    assert '"${GSMAP}" latent-to-gene' not in script
    assert '"${GSMAP}" spatial-ldsc' not in script
    assert '"${GSMAP}" cauchy-combination' not in script


def test_trait_mapping_scripts_do_not_add_future_annotations_compatibility():
    assert "from __future__ import annotations" not in PREPARATION_SCRIPT.read_text()


def test_report_job_runs_both_arms_as_report_only_quick_mode():
    script = REPORT_SBATCH.read_text()

    assert script.count('"${GSMAP}" quick-mode') == 1
    assert "--start-step report" in script
    assert "--stop-step report" in script
    assert "run_report 2d" in script
    assert "run_report 3d" in script
    assert "upstream_succeeded" in script
    assert "--no-gpu" in script

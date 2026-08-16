import json
import logging
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsMap.logging_config import (
    _reset_logging_for_tests,
    logged_stage,
    rebuild_log_index,
    stage_logging,
)


@pytest.fixture(autouse=True)
def reset_gsmap_logging():
    _reset_logging_for_tests()
    yield
    _reset_logging_for_tests()


def _run_directories(project_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in (project_dir / "logs").iterdir()
        if path.name != "latest"
        and path.is_dir()
        and (path / "run_manifest.json").is_file()
    )


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_manifest.json").read_text())


def test_console_levels_and_plain_debug_file(tmp_path, capsys):
    logger = logging.getLogger("gsMap.tests.logging")
    external_logger = logging.getLogger("test_external_dependency")

    with stage_logging(tmp_path, "latent_to_gene") as session:
        run_dir = session.run_dir
        logger.debug("debug-only-message")
        logger.info("console-info-message")
        logger.warning("console-warning-message")
        external_logger.warning("external-warning-message")
        warnings.warn("captured-python-warning", UserWarning, stacklevel=1)

    captured = capsys.readouterr()
    assert "console-info-message" in captured.out
    assert "console-warning-message" not in captured.out
    assert "console-warning-message" in captured.err
    assert "debug-only-message" not in captured.out + captured.err

    log_text = (run_dir / "02_latent_to_gene.log").read_text()
    assert "debug-only-message" in log_text
    assert "console-info-message" in log_text
    assert "console-warning-message" in log_text
    assert "external-warning-message" in log_text
    assert "captured-python-warning" in log_text
    assert "Stage resource summary" in log_text
    assert "Stage latent_to_gene completed in" in log_text
    assert "Stage latent_to_gene started | run_id=" in log_text
    for line in log_text.splitlines():
        prefix = line.split(" | ", maxsplit=1)[0]
        assert " run=" not in prefix
        assert " stage=" not in prefix
        assert " trait=" not in prefix
    assert "\x1b[" not in log_text

    manifest = _manifest(run_dir)
    assert manifest["status"] == "complete"
    assert manifest["stages"][0]["status"] == "complete"
    assert manifest["stages"][0]["log_file"] == str(run_dir / "02_latent_to_gene.log")
    assert "__latent_to_gene__" in run_dir.name
    latest_log = tmp_path / "logs" / "latest" / "02_latent_to_gene.log"
    assert latest_log.resolve() == run_dir / "02_latent_to_gene.log"
    assert latest_log.with_suffix(".manifest.json").resolve() == run_dir / "run_manifest.json"


def test_nested_stages_are_split_and_same_stage_is_idempotent(tmp_path):
    logger = logging.getLogger("gsMap.tests.logging")

    with stage_logging(tmp_path, "quick_mode") as session:
        run_dir = session.run_dir
        with stage_logging(tmp_path, "find_latent"):
            with stage_logging(tmp_path, "find_latent"):
                logger.info("nested-stage-message")

    quick_log = (run_dir / "quick_mode.log").read_text()
    stage_log = (run_dir / "01_find_latent.log").read_text()
    assert quick_log.count("nested-stage-message") == 1
    assert stage_log.count("nested-stage-message") == 1

    manifest = _manifest(run_dir)
    assert [(item["stage"], item["status"]) for item in manifest["stages"]] == [
        ("quick_mode", "complete"),
        ("find_latent", "complete"),
    ]


def test_trait_failure_has_traceback_and_marks_partial_run(tmp_path, capsys):
    logger = logging.getLogger("gsMap.tests.logging")

    def fail_trait():
        with stage_logging(tmp_path, "spatial_ldsc", trait="SCZ/test"):
            logger.info("trait-started")
            raise RuntimeError("synthetic trait failure")

    with stage_logging(tmp_path, "spatial_ldsc") as session:
        run_dir = session.run_dir
        with pytest.raises(RuntimeError, match="synthetic trait failure"):
            fail_trait()
        logger.info("continued-after-trait")

    trait_log = run_dir / "03_spatial_ldsc" / "SCZ_test.log"
    trait_text = trait_log.read_text()
    assert "synthetic trait failure" in trait_text
    assert "Traceback (most recent call last)" in trait_text
    assert "Traceback (most recent call last)" not in capsys.readouterr().err

    manifest = _manifest(run_dir)
    assert manifest["status"] == "partial_failure"
    statuses = {(item["stage"], item["trait"]): item["status"] for item in manifest["stages"]}
    assert statuses[("spatial_ldsc", None)] == "complete"
    assert statuses[("spatial_ldsc", "SCZ_test")] == "failed"


def test_trait_completion_is_unambiguous_in_parent_and_trait_logs(tmp_path):
    logger = logging.getLogger("gsMap.tests.logging")

    with stage_logging(tmp_path, "spatial_ldsc") as session:
        run_dir = session.run_dir
        with stage_logging(tmp_path, "spatial_ldsc", trait="SCZ/test"):
            logger.info("trait-finished")

    parent_text = (run_dir / "03_spatial_ldsc.log").read_text()
    trait_text = (run_dir / "03_spatial_ldsc" / "SCZ_test.log").read_text()
    completion = "Stage spatial_ldsc trait=SCZ_test completed in"
    assert completion in parent_text
    assert completion in trait_text
    assert "Stage spatial_ldsc trait=SCZ_test started | run_id=" in trait_text
    assert "Stage resource summary | stage=spatial_ldsc trait=SCZ_test" in parent_text
    assert "Stage spatial_ldsc completed in" in parent_text
    latest_trait = tmp_path / "logs" / "latest" / "03_spatial_ldsc" / "SCZ_test.log"
    assert latest_trait.resolve() == run_dir / "03_spatial_ldsc" / "SCZ_test.log"


def test_decorator_records_redacted_config_and_creates_unique_runs(tmp_path):
    @logged_stage("report")
    def sample_stage(config):
        logging.getLogger("gsMap.tests.logging").info("decorated-call")
        return "done"

    config = SimpleNamespace(
        project_dir=tmp_path,
        to_dict_with_paths_as_strings=lambda: {
            "project_dir": str(tmp_path),
            "api_token": "do-not-write-this",
        },
    )

    assert sample_stage(config) == "done"
    assert sample_stage(config) == "done"

    run_dirs = _run_directories(tmp_path)
    assert len(run_dirs) == 2
    assert run_dirs[0].name != run_dirs[1].name
    for run_dir in run_dirs:
        manifest = _manifest(run_dir)
        assert manifest["status"] == "complete"
        assert manifest["resolved_config"]["api_token"] == "<redacted>"
        assert manifest["stage_configs"]["report"]["api_token"] == "<redacted>"
        assert "do-not-write-this" not in (run_dir / "05_report.log").read_text()


def test_latest_pointer_tracks_last_run(tmp_path):
    with stage_logging(tmp_path, "format_sumstats"):
        pass
    first_run = _run_directories(tmp_path)[0]

    with stage_logging(tmp_path, "format_sumstats") as session:
        second_run = session.run_dir

    latest = json.loads((tmp_path / "logs" / "latest.json").read_text())
    assert first_run != second_run
    assert latest["run_dir"] == str(second_run)
    assert latest["status"] == "complete"
    assert (tmp_path / "logs" / "latest" / "run_manifest.json").resolve() == (
        second_run / "run_manifest.json"
    )


def test_log_index_can_be_rebuilt_from_immutable_manifests(tmp_path):
    with stage_logging(tmp_path, "spatial_ldsc", trait="SCZ") as session:
        run_dir = session.run_dir

    latest_log = tmp_path / "logs" / "latest" / "03_spatial_ldsc" / "SCZ.log"
    latest_manifest = latest_log.with_suffix(".manifest.json")
    latest_log.unlink()
    latest_manifest.unlink()

    rebuilt = rebuild_log_index(tmp_path)

    assert rebuilt["spatial_ldsc/SCZ"] == str(latest_log)
    assert latest_log.resolve() == run_dir / "03_spatial_ldsc" / "SCZ.log"
    assert latest_manifest.resolve() == run_dir / "run_manifest.json"


def test_run_directory_name_infers_single_trait(tmp_path):
    config = SimpleNamespace(project_dir=tmp_path, trait_name="SCZ/test")

    with stage_logging(config, "spatial_ldsc") as session:
        run_dir = session.run_dir

    assert "__spatial_ldsc__SCZ_test__" in run_dir.name


def test_manifest_redacts_sensitive_command_line_values(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gsmap",
            "quick-mode",
            "--api-token",
            "plain-secret",
            "--password=inline-secret",
            "--spatial-key",
            "spatial",
        ],
    )

    with stage_logging(tmp_path, "quick_mode") as session:
        run_dir = session.run_dir

    manifest = _manifest(run_dir)
    assert manifest["argv"] == [
        "gsmap",
        "quick-mode",
        "--api-token",
        "<redacted>",
        "--password=<redacted>",
        "--spatial-key",
        "spatial",
    ]
    assert "plain-secret" not in json.dumps(manifest)
    assert "inline-secret" not in json.dumps(manifest)

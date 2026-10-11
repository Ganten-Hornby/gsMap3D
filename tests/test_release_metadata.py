"""Release guards must reject mismatched tags before publishing anything."""

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_metadata", ROOT / ".github/scripts/release_metadata.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("version", "prerelease"),
    [
        ("0.1.4a1", "true"),
        ("0.1.4a2", "true"),
        ("0.1.4b1", "true"),
        ("0.1.4rc1", "true"),
        ("0.1.4.dev1", "true"),
        ("0.1.4", "false"),
        ("0.1.4.post1", "false"),
    ],
)
def test_release_classification(version, prerelease):
    assert MODULE.release_metadata(f"v{version}", version) == {
        "tag": f"v{version}",
        "version": version,
        "prerelease": prerelease,
    }


@pytest.mark.parametrize(
    "tag", ["0.1.4a1", "v0.1.3", "v0.1.4alpha1", "v0.1.4a1\n", "v0.1.4a1+local"]
)
def test_invalid_release_tags(tag):
    with pytest.raises(ValueError):
        MODULE.release_metadata(tag, "0.1.4a1")


def test_workflow_guards_and_tag_checkout():
    jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    assert jobs["build-docker"]["if"] == "needs.build-python.outputs.prerelease == 'false'"
    assert jobs["publish-pypi"]["needs"] == "build-python"
    assert "needs.publish-pypi.result == 'success'" in jobs["create-release"]["if"]
    assert "needs.build-docker.result == 'skipped'" in jobs["create-release"]["if"]
    for job in ("build-python", "build-docker", "create-release"):
        checkout = next(
            step
            for step in jobs[job]["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        )
        assert checkout["with"]["ref"].startswith("refs/tags/")
    release = next(
        step
        for step in jobs["create-release"]["steps"]
        if step.get("name") == "Create GitHub Release"
    )
    assert "--prerelease --latest=false" in release["run"]

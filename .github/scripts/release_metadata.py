"""Validate release tags against package metadata before publication."""

import os
import sys
import tomllib
from pathlib import Path

from packaging.version import Version


def release_metadata(tag, project_version):
    if not tag.startswith("v") or any(char.isspace() for char in tag):
        raise ValueError("Release tag must be v followed by a canonical PEP 440 version")
    version = Version(tag[1:])
    if str(version) != tag[1:] or version.local is not None:
        raise ValueError("Release tag must use a canonical public version")
    if tag[1:] != project_version:
        raise ValueError(f"Tag {tag} does not match package version {project_version}")
    return {
        "tag": tag,
        "version": str(version),
        "prerelease": str(version.is_prerelease).lower(),
    }


def main():
    with Path("pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    metadata = release_metadata(sys.argv[1], project["version"])
    lines = "".join(f"{key}={value}\n" for key, value in metadata.items())
    print(lines, end="")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            stream.write(lines)


if __name__ == "__main__":
    main()

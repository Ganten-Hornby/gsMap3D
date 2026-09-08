"""Public test-data downloads are verified and never extract unsafe archives."""

import hashlib
import io
import json
import tarfile

import pytest

from tests.tools.fetch_resources import digest, fetch


def make_catalog(tmp_path, members=None):
    content = b"example input\n"
    if members is None:
        members = {
            "toy/input.txt": content,
            "toy/SHA256SUMS": f"{hashlib.sha256(content).hexdigest()}  input.txt\n".encode(),
        }
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, value in members.items():
            info = tarfile.TarInfo(name)
            if isinstance(value, tarfile.TarInfo):
                output.addfile(value)
            else:
                info.size = len(value)
                output.addfile(info, io.BytesIO(value))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "resources": [
                    {
                        "id": "toy",
                        "revision": "r1",
                        "path": archive.name,
                        "sha256": digest(archive),
                    }
                ]
            }
        )
    )
    return catalog


def test_existing_extraction_is_reverified(tmp_path):
    catalog = make_catalog(tmp_path)
    destination = tmp_path / "extracted"
    fetch(str(catalog), destination, ["toy"])
    fetch(str(catalog), destination, ["toy"])
    (destination / "toy/input.txt").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum failed"):
        fetch(str(catalog), destination, ["toy"])


@pytest.mark.parametrize("name", ["../escape", "/absolute", "toy/../../escape", "other/file"])
def test_unsafe_member_is_rejected(tmp_path, name):
    catalog = make_catalog(tmp_path, {name: b"bad"})
    with pytest.raises(ValueError, match="Unsafe archive member"):
        fetch(str(catalog), tmp_path / "extracted", ["toy"])
    assert not (tmp_path / "extracted/toy").exists()


def test_symlink_archive_is_rejected(tmp_path):
    link = tarfile.TarInfo("toy/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    catalog = make_catalog(tmp_path, {"toy/link": link})
    with pytest.raises(ValueError, match="Unsafe archive member"):
        fetch(str(catalog), tmp_path / "extracted", ["toy"])


def test_failed_extraction_never_publishes_partial_directory(tmp_path):
    catalog = make_catalog(
        tmp_path,
        {
            "toy/input.txt": b"wrong",
            "toy/SHA256SUMS": f"{hashlib.sha256(b'correct').hexdigest()}  input.txt\n".encode(),
        },
    )
    destination = tmp_path / "extracted"
    for _ in range(2):
        with pytest.raises(ValueError, match="checksum failed"):
            fetch(str(catalog), destination, ["toy"])
        assert not (destination / "toy").exists()
        assert not list(destination.glob(".toy-*"))


def test_archive_checksum_is_checked_before_extraction(tmp_path):
    catalog = make_catalog(tmp_path)
    (tmp_path / "bundle.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Archive checksum failed"):
        fetch(str(catalog), tmp_path / "extracted", ["toy"])
    assert not (tmp_path / "extracted/toy").exists()


def test_cached_extraction_rejects_symlink(tmp_path):
    catalog = make_catalog(tmp_path)
    destination = tmp_path / "extracted"
    fetch(str(catalog), destination, ["toy"])
    (destination / "toy/link").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        fetch(str(catalog), destination, ["toy"])


def test_external_only_resource_has_actionable_error(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"resources": [{"id": "external"}]}))
    with pytest.raises(ValueError, match="external-only"):
        fetch(str(catalog), tmp_path / "output", ["external"])

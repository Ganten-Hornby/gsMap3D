"""Run the actual wget/SHA/tar tutorial block against a loopback release mirror."""

import argparse
import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tests.helpers.audit_commands import blocks
from tests.helpers.validate_examples import execute
from tests.tools.fetch_resources import digest as sha256


def download_blocks(path):
    """Use the exact displayed shell, not a second download implementation."""
    return [code for _, lang, code in blocks(path) if lang == "bash" and "wget -O" in code]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    base_url = json.loads((repo / "docs/resource_catalog.json").read_text())["base_url"]
    source = repo / "docs/source"
    shared = download_blocks(source / "downloads.md")
    visium = download_blocks(source / "2d_tutorial/10x.md")
    gastrulation = download_blocks(source / "3d_tutorial/human_gastrulation.md")
    e16 = download_blocks(source / "2d_tutorial/mouse_embryo.md")
    e11 = download_blocks(source / "3d_tutorial/mouse_e11.md")
    if (len(shared), len(visium), len(gastrulation), len(e16), len(e11)) != (3, 1, 1, 1, 1):
        raise ValueError("Expected three shared downloads and one download per dataset")
    requested = []

    class RecordingHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requested.append(self.path)
            super().do_GET()

    args.output.mkdir(parents=True, exist_ok=True)
    handler = partial(RecordingHandler, directory=str(args.published_data.resolve()))
    status = 0
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            mirror = f"http://127.0.0.1:{server.server_port}"
            env = dict(os.environ)
            recipes = [
                (
                    "shared-reference-download",
                    shared[0],
                    "/reference/ld_weights/1000g-hc-grch38-v50-eur/",
                ),
                ("shared-homologs-download", shared[1], "/reference/homologs/homologs/"),
                ("shared-gwas-download", shared[2], "/gwas/example-traits/"),
                ("visium-download-local-mirror", visium[0], "/datasets/st/2d/mouse-visium/"),
                (
                    "gastrulation-download-local-mirror",
                    gastrulation[0],
                    "/datasets/st/3d/human-gastrulation/",
                ),
                ("e16-download-local-mirror", e16[0], "/datasets/st/2d/mouse-e16-mosta/"),
                (
                    "e11-download-local-mirror",
                    e11[0],
                    "/datasets/st/3d/mouse-e11/",
                ),
            ]
            for name, code, prefix in recipes:
                before = len(requested)
                status = execute(
                    ["bash", "-euc", code.replace(base_url, mirror)],
                    name,
                    args.output.resolve(),
                    env,
                    cwd=args.output.resolve(),
                )
                if status:
                    break
                if prefix and (
                    len(requested) - before != 1
                    or not all(path.startswith(prefix) for path in requested[before:])
                ):
                    raise AssertionError(f"{name} did not fetch exactly its own dataset archive")
            if not status:
                # Test the checksum instructions that now live on Downloads.
                checks = [
                    code
                    for _, lang, code in blocks(source / "downloads.md")
                    if lang == "bash" and "sha256sum" in code and "wget -O" not in code
                ]
                if len(checks) != 2:
                    raise ValueError("Expected the archive and extracted-file checksum examples")
                for index, code in enumerate(checks):
                    status = execute(
                        ["bash", "-euc", code],
                        f"documented-checksum-{index}",
                        args.output.resolve(),
                        env,
                        cwd=args.output.resolve(),
                    )
                    if status:
                        break
            if not status:
                catalog = json.loads((args.published_data / "catalog.json").read_text())
                datasets = (
                    "mouse-visium",
                    "human-gastrulation",
                    "mouse-e16-mosta",
                    "mouse-e11",
                )
                archived = {
                    "1000g-hc-grch38-v50-eur": "resources",
                    "homologs": "resources",
                    "example-traits": "example_data",
                    **dict.fromkeys(datasets, "example_data"),
                }
                for identifier, parent in archived.items():
                    expected = next(e for e in catalog["resources"] if e["id"] == identifier)
                    archive = args.output / parent / f"{identifier}.tar.gz"
                    if sha256(archive) != expected["sha256"]:
                        raise ValueError(
                            f"Downloaded archive does not match catalog: {identifier}"
                        )
                # Shared installation remains intact after moving between tutorials.
                root = args.output.resolve()
                status = execute(
                    [
                        "bash",
                        "-euc",
                        "(cd resources/1000g-hc-grch38-v50-eur && sha256sum -c SHA256SUMS)\n"
                        "(cd resources/homologs && sha256sum -c SHA256SUMS)\n"
                        "(cd example_data/example-traits && sha256sum -c SHA256SUMS)\n"
                        + "\n".join(
                            f"(cd example_data/{identifier} && sha256sum -c SHA256SUMS)"
                            for identifier in datasets
                        ),
                    ],
                    "shared-resource-reuse-check",
                    root,
                    env,
                    cwd=root,
                )
            (args.output / "download_requests.json").write_text(
                json.dumps(dict(requests=requested, exit_code=status), indent=2) + "\n"
            )
        finally:
            server.shutdown()
            worker.join()
    raise SystemExit(status)

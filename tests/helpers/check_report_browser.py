"""Exercise report-view and render a report in Chromium; retain errors and screenshots."""

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=f"{repo / 'src'}:{repo}")
    command = [
        str(args.python),
        "-m",
        "gsMap.cli",
        "report-view",
        str(args.report.resolve()),
        "--port",
        str(port),
        "--no-browser",
    ]
    errors, failed_responses, failed_requests = [], [], []
    data = {}
    failure = None
    with (
        (args.output / "server.out").open("w") as stdout,
        (args.output / "server.err").open("w") as stderr,
    ):
        server = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        try:
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError("report-view terminated unexpectedly")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.25)
            else:
                raise TimeoutError("report-view did not start")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(
                    viewport={"width": 1500, "height": 1100}, device_scale_factor=1
                )
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on(
                    "requestfailed",
                    lambda request: failed_requests.append(
                        dict(url=request.url, failure=request.failure)
                    ),
                )
                page.on(
                    "response",
                    lambda response: (
                        failed_responses.append(dict(url=response.url, status=response.status))
                        if response.status >= 400 and not response.url.endswith("favicon.ico")
                        else None
                    ),
                )
                page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle", timeout=60000)
                page.wait_for_function(
                    "window.Alpine && Alpine.$data(document.body).dataLoaded", timeout=60000
                )
                page.wait_for_function(
                    "document.querySelector('#spatial2dPlotTrait canvas, #spatial3dPlotTrait canvas')",
                    timeout=60000,
                )
                page.screenshot(path=str(args.output / "overview.png"), full_page=False)
                page.locator("#sec-gsmap").scroll_into_view_if_needed()
                page.wait_for_function(
                    "document.querySelector('#spatial2dPlotTrait canvas')", timeout=60000
                )
                page.screenshot(path=str(args.output / "spatial2d.png"))
                page.locator("#trait-selector").fill("I")
                page.screenshot(path=str(args.output / "trait-search.png"))
                data = page.evaluate(
                    "({title: document.title, traits: Alpine.$data(document.body).traits, sample: Alpine.$data(document.body).selectedSample, error: Alpine.$data(document.body).spatial2dError})"
                )
                browser.close()
            record = dict(
                command=command,
                errors=errors,
                failed_responses=failed_responses,
                failed_requests=failed_requests,
                state=data,
            )
            if errors or failed_responses or failed_requests or data.get("error"):
                raise RuntimeError(f"Browser validation failed: {record}")
            print(json.dumps(record), flush=True)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            (args.output / "browser.json").write_text(
                json.dumps(
                    dict(
                        command=command,
                        errors=errors,
                        failed_responses=failed_responses,
                        failed_requests=failed_requests,
                        state=data,
                        failure=failure,
                        passed=failure is None,
                    ),
                    indent=2,
                )
                + "\n"
            )
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()

"""Run documentation data-inspection snippets against completed tutorial outputs."""

import argparse
import os
import re
import sys
from pathlib import Path

from tests.helpers.audit_commands import blocks
from tests.helpers.validate_examples import execute

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--example-data", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        PYTHONPATH=f"{repo / 'src'}:{repo}",
        CUDA_VISIBLE_DEVICES="",
        JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
    )
    failures = []
    for filename in ("key_concepts.md", "output_files.md"):
        for line, language, code in blocks(repo / "docs/source" / filename):
            if language != "python":
                continue
            roots = {
                "resources": args.resources.resolve(),
                "example_data": (args.example_data or args.resources).resolve(),
            }
            code = re.sub(
                r"\b(resources|example_data)/",
                lambda match, roots=roots: f"{roots[match[1]]}/",
                code,
            )
            code = code.replace('"results/', f'"{args.results.resolve()}/').replace(
                'Path("results")', f'Path("{args.results.resolve()}")'
            )
            result = execute(
                [sys.executable, "-c", code],
                f"{Path(filename).stem}-{line}",
                args.output.resolve(),
                env,
            )
            if result:
                failures.append(f"{filename}:{line}")
    if failures:
        raise SystemExit(f"Inspection examples failed: {failures}")

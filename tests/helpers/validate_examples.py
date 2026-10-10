"""Execute tutorial commands with persistent logs and an honest validation ledger."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from tests.helpers.audit_commands import blocks

TUTORIALS = {
    "mouse-e16-mosta": "2d_tutorial/mouse_embryo.md",
    "visium-coronal": "2d_tutorial/10x.md",
    "visium-sagittal": "2d_tutorial/10x.md",
    "visium-kidney": "2d_tutorial/10x.md",
    "human-gastrulation-cs8": "3d_tutorial/human_gastrulation.md",
    "e11-ordinal": "3d_tutorial/mouse_e11.md",
}


def tutorial_code(repo, example, interface):
    """Extract the displayed recipe, never maintain a second analysis recipe."""
    path = repo / "docs/source" / TUTORIALS[example]
    language = "bash" if interface == "cli" else "python"
    matches = []
    for _, lang, code in blocks(path):
        if lang != language:
            continue
        if interface == "cli":
            selected = re.search(
                r"--project-name\s+[\"']?" + re.escape(example) + r"(?:[\"']|\s|$)", code
            )
        else:
            selected = re.search(r"project_name\s*=\s*[\"']" + re.escape(example) + r"[\"']", code)
        if selected:
            matches.append(code)
    if len(matches) != 1:
        raise ValueError(f"Expected one {interface} recipe for {example}, found {len(matches)}")
    setups = []
    for names, code in re.findall(
        r"<!-- gsmap-setup: ([^>]+) -->\s*`{3}python\n(.*?)\n`{3}", path.read_text(), re.S
    ):
        if example in names.split():
            setups.append(code)
    return setups, matches[0]


def validation_environment(repo, environment, cpu=False):
    """Respect explicit thread budgets, capped by the Slurm task allocation.

    Never infer this budget from the visible node CPU count. Eight is only the
    fallback for an invocation without an explicit thread setting or allocation.
    """
    env = dict(environment)
    allocation = env.get("SLURM_CPUS_PER_TASK")
    budget = int(allocation) if allocation else 8
    if budget < 1:
        raise ValueError("SLURM_CPUS_PER_TASK must be positive")
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        default = 1 if variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS") else budget
        value = int(env.get(variable, default))
        if value < 1:
            raise ValueError(f"{variable} must be positive")
        env[variable] = str(min(value, budget) if allocation else value)
    env["PYTHONPATH"] = f"{repo / 'src'}{os.pathsep}{repo}"
    env["PYTHONUNBUFFERED"] = "1"
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def execute(command, name, output, env, dry_run=False, cwd=None):
    output.mkdir(parents=True, exist_ok=True)
    stdout, stderr = output / f"{name}.out", output / f"{name}.err"
    if stdout.exists() or stderr.exists():
        raise FileExistsError(f"Use a new validation directory; {name} already has logs")
    started = time.perf_counter()
    with stdout.open("w") as out, stderr.open("w") as err:
        result = subprocess.run(command, stdout=out, stderr=err, env=env, cwd=cwd)
    record = dict(
        name=name,
        command=command,
        exit_code=result.returncode,
        elapsed_seconds=time.perf_counter() - started,
        completed_utc=datetime.now(UTC).isoformat(),
        stdout=str(stdout),
        stderr=str(stderr),
        dry_run=dry_run,
        cwd=str(cwd) if cwd is not None else str(Path.cwd()),
    )
    with (output / "commands.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(
        f"{name}: exit={result.returncode}, {record['elapsed_seconds']:.1f}s; {stdout}", flush=True
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument(
        "--example-data",
        type=Path,
        help="Example ST/GWAS root; defaults to --resources for existing flat test installations",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Explicit existing workdir for testing resume; logs remain separate",
    )
    parser.add_argument("--examples", nargs="+", default=["visium-coronal"])
    parser.add_argument("--interface", choices=["cli", "python"], default="cli")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print extracted commands only; no execution or runtime validation",
    )
    parser.add_argument(
        "--e11-sections",
        type=Path,
        help="Prepared E11 dataset folder, including sample_paths.txt and gwas.yaml",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    env = validation_environment(repo, os.environ, args.cpu)
    failures = []
    for example in args.examples:
        setups, code = tutorial_code(repo, example, args.interface)
        results = (args.results_dir or args.output / "results").resolve()
        if example.startswith("e11-") and (results / example).exists() and not args.dry_run:
            raise FileExistsError(
                "Full E11 validation requires a fresh project; choose a new output directory"
            )

        def localize(text, results=results):
            # Paths/backend are the only runtime substitutions to displayed code.
            roots = {
                "resources": args.resources.resolve(),
                "example_data": (args.example_data or args.resources).resolve(),
                "inputs": args.output.resolve() / "inputs",
            }
            roots["example_data/mouse-e11"] = (
                args.e11_sections.resolve()
                if args.e11_sections
                else roots["example_data"] / "mouse-e11"
            )
            text = re.sub(
                r"\b(example_data/mouse-e11|resources|example_data|inputs)/",
                lambda match: f"{roots[match[1]]}/",
                text,
            )
            text = text.replace('"results"', f'"{results}"')
            text = text.replace("--workdir results", f'--workdir "{results}"')
            text = text.replace("mkdir -p results", f'mkdir -p "{results}"')
            return text

        code = localize(code)
        if args.cpu:
            if args.interface == "cli":
                code = code.replace("gsmap quick-mode", "gsmap quick-mode --no-gpu")
            else:
                code = code.replace("QuickModeConfig(\n", "QuickModeConfig(\n    use_gpu=False,\n")
        setup_codes = [localize(setup) for setup in setups]
        if args.dry_run:
            command = [
                sys.executable,
                "-c",
                f"print({json.dumps(dict(example=example, setup=setup_codes, code=code, dry_run=True))!r})",
            ]
        else:
            for index, setup in enumerate(setup_codes):
                if execute(
                    [sys.executable, "-c", setup],
                    f"{example}-setup-{index}",
                    args.output.resolve(),
                    env,
                ):
                    raise RuntimeError(f"Tutorial preparation failed for {example}")
            command = (
                ["bash", "-euc", code] if args.interface == "cli" else [sys.executable, "-c", code]
            )
        if execute(command, example, args.output.resolve(), env, dry_run=args.dry_run):
            failures.append(example)
    if failures:
        raise SystemExit(f"Failed examples: {failures}")


if __name__ == "__main__":
    main()

"""Plot matched one/four-trait CPU/GPU sweeps; never infer single-trait rates.

Import completed validation summaries with --benchmark-root, or redraw from
the public CSV alone. A partial/failed sweep cannot replace published data.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from docs.tools.plot_scalability import ASSETS, read_csv, save_image

GPU_NAMES = {"A100": "A100", "V100": "V100", "RTX2080Ti": "RTX 2080 Ti"}
CELLS = 1000037


def measurements(root):
    """Extract only fully validated points with matching workload and topology."""
    result = []
    identities = []
    for platform, pattern in (
        ("CPU", "cpu/runs_million/*/summary.json"),
        ("GPU", "gpu/scaling_results/*/summary.json"),
    ):
        for path in sorted(root.glob(pattern)):
            d = json.loads(path.read_text())
            assert d["complete"], f"Sweep not complete: {path}"
            assert d["all_passed"], f"Failed numerical checks: {path}"
            args = d if platform == "CPU" else d["arguments"]
            traits = args["traits"]
            if isinstance(traits, str):
                traits = list(map(int, traits.split(",")))
            assert traits in ([64], [64, 92, 2, 59])
            cells = args["cells"]
            assert cells == CELLS, f"Expected the matched million-cell workload: {path}"
            if platform == "CPU":
                match = d["matched_gpu"]
                assert match["same_cells_as_gpu"]
                assert match["cells"] == cells
                assert match["row_offset"] == 0
                assert match["source_commit"] == d["source_commit"]
                identity = (d["source_commit"], d["manifest_sha256"], match["kernel_hashes"])
            else:
                assert args["row_offset"] == 0
                identity = (d["implementation_commit"], d["prepared_sha256"], d["kernel_hashes"])
            identities.append(identity)
            model = "CPU" if platform == "CPU" else GPU_NAMES[d["label"].split("_t")[0]]
            for row in d["runs"] if platform == "CPU" else d["cases"]:
                if platform == "CPU":
                    assert row["batch"] == 400
                    assert row["cells"] == cells
                    assert row["traits"] == len(traits)
                    assert row["cores"] == row["numba_threads"] == len(row["affinity"])
                    assert row["all_passed"]
                    checks, count, batch = row["checks"], row["cores"], row["batch"]
                else:
                    assert row["method"] == "moments"
                    assert row["axis"] == "cells"
                    assert row["passed"]
                    assert args["batch"] == 100
                    checks, count, batch = row["audits"], row["gpus"], args["batch"]
                assert len(checks) == len(traits)
                assert all(
                    c["passed"] and np.isfinite(c["max_abs_logp"]) and 0 <= c["max_abs_logp"] < 0.1
                    for c in checks
                )
                delta = max(c["max_abs_logp"] for c in checks)
                assert delta < 0.1
                seconds = row["seconds"]
                rate = row["cell_trait_s"]
                assert np.isfinite(seconds)
                assert seconds > 0
                assert np.isclose(rate, cells * len(traits) / seconds)
                result.append(
                    dict(
                        hardware=model,
                        workers=count,
                        traits=len(traits),
                        batch=batch,
                        cells=cells,
                        repeat=row["repeat"],
                        seconds=seconds,
                        cell_trait_s=rate,
                        cell_s=cells / seconds,
                        max_abs_logp=delta,
                        source_run=path.parent.name,
                    )
                )
    validate_coverage(result)
    assert all(i == identities[0] for i in identities), "Mismatched source or input provenance"
    return result


def validate_coverage(rows):
    actual = [(r["hardware"], int(r["workers"]), int(r["traits"]), int(r["repeat"])) for r in rows]
    expected = {
        (model, n, t, r)
        for model in ("CPU", "A100", "V100", "RTX 2080 Ti")
        for n in ([1, 2, 4, 8, 16, 32] if model == "CPU" else [1, 2, 4])
        for t in (1, 4)
        for r in range(3 if model == "CPU" else 2)
    }
    assert len(actual) == len(set(actual)), "Duplicate measurements"
    assert set(actual) == expected, f"Missing cases: {sorted(expected - set(actual))}"


def plot(rows, platform, assets):
    cpu = platform == "CPU"
    counts = [1, 2, 4, 8, 16, 32] if cpu else [1, 2, 4]
    models = ["CPU"] if cpu else ["RTX 2080 Ti", "V100", "A100"]
    colors = {"CPU": "#007f86", "RTX 2080 Ti": "#cc7135", "V100": "#6962b4", "A100": "#007f86"}
    fig, axis = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.79, bottom=0.15)
    fig.suptitle(
        f"{platform} spatial LDSC throughput",
        x=0.12,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    subtitle = (
        "1,000,037 cells · default batch 400 · mean of 3 runs"
        if cpu
        else "1,000,037 cells · default batch 100 / GPU · mean of 2 runs"
    )
    fig.text(0.12, 0.885, subtitle, fontsize=10, color="#475569")
    for model in models:
        for traits in (1, 4):
            selected = [r for r in rows if r["hardware"] == model and int(r["traits"]) == traits]
            samples = [
                [float(r["cell_trait_s"]) for r in selected if int(r["workers"]) == n]
                for n in counts
            ]
            mean = np.array([np.mean(s) for s in samples])
            label = (f"{model} · " if not cpu else "") + (
                "1 trait (IQ)" if traits == 1 else "4 traits"
            )
            style = dict(
                color=colors[model],
                linestyle="--" if traits == 1 else "-",
                marker="s" if traits == 1 else "o",
                linewidth=2,
                label=label,
            )
            axis.plot(counts, mean, **style)
            axis.annotate(
                "≈" + format(mean[-1] / 1000, ".1f" if mean[-1] < 10000 else ".0f") + "k",
                xy=(counts[-1], mean[-1]),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=colors[model],
            )
    if cpu:
        axis.set_xscale("log", base=2)
    axis.set_xticks(counts)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlabel("Physical CPU cores" if cpu else "GPUs on one node")
    axis.set_ylabel("Cell-traits / second")
    axis.margins(x=0.12, y=0.1)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    for ext in ("png", "svg"):
        save_image(fig, assets / f"{platform.lower()}_trait_scaling.{ext}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--benchmark-root", type=Path)
    args = parser.parse_args()
    path = args.assets / "trait_scaling_measurements.csv"
    if args.benchmark_root:
        rows = measurements(args.benchmark_root)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        rows = read_csv(path)
        validate_coverage(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    for platform in ("CPU", "GPU"):
        plot(rows, platform, args.assets)


if __name__ == "__main__":
    main()

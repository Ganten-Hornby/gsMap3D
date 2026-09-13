"""Render public LDSC curves from measurements, without running a benchmark.

By default, reads the checked-in CSVs. Optional GPU summary inputs refresh the
public measurement CSV from completed production-validation sweeps; internal
paths, device identifiers and GWAS filenames are not copied into public data.
"""

import argparse
import csv
import json
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter

ASSETS = Path(__file__).resolve().parents[1] / "source/_static/benchmarks"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_image(fig, path):
    """Keep generated SVGs deterministic and free of trailing whitespace."""
    if path.suffix == ".svg":
        stream = StringIO()
        with plt.rc_context({"svg.hashsalt": "gsmap-scalability"}):
            fig.savefig(stream, format="svg", facecolor="white", metadata={"Date": None})
        path.write_text("\n".join(line.rstrip() for line in stream.getvalue().splitlines()) + "\n")
    else:
        fig.savefig(path, dpi=180, facecolor="white")


def import_gpu_summaries(paths, output):
    rows = []
    for label, path in zip(("A100", "RTX 2080 Ti"), paths, strict=True):
        data = json.loads(path.read_text())
        assert data["complete"], f"Incomplete sweep: {path}"
        args = data["arguments"]
        assert args["cells"] == 1000037
        assert args["batch"] == 100
        assert len(args["traits"].split(",")) == 4
        cases = [c for c in data["cases"] if c["method"] == "moments"]
        assert len(cases) == 6
        assert {(c["gpus"], c["repeat"]) for c in cases} == {
            (g, r) for g in (1, 2, 4) for r in (0, 1)
        }
        for case in cases:
            assert case["passed"]
            assert all(a["passed"] for a in case["audits"])
            assert case["axis"] == "cells"
            rate = float(case["cell_trait_s"])
            assert np.isfinite(rate)
            assert rate > 0
            assert np.isclose(rate, 4 * args["cells"] / case["seconds"])
            rows.append(
                dict(
                    gpu=label,
                    gpus=case["gpus"],
                    repeat=case["repeat"],
                    cells=args["cells"],
                    traits=4,
                    batch=args["batch"],
                    seconds=case["seconds"],
                    cell_trait_s=rate,
                    max_abs_logp=max(a["max_abs_logp"] for a in case["audits"]),
                )
            )
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figure(title, subtitle, counts, xlabel):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.suptitle(title, x=0.07, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.07, 0.885, subtitle, fontsize=10, color="#475569")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.77, bottom=0.18, wspace=0.52)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(counts)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Spatial LDSC throughput", loc="left", fontsize=11)
    axes[0].set_ylabel("Cell-traits / second")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    secondary = axes[0].secondary_yaxis("right", functions=(lambda x: x / 4, lambda x: x * 4))
    secondary.set_ylabel("Cells / second · all 4 traits")
    secondary.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    axes[1].set_title("Strong scaling", loc="left", fontsize=11)
    axes[1].set_ylabel("Speedup over one core / GPU")
    axes[1].plot(counts, counts, "--", color="#94a3b8", label="Ideal scaling", linewidth=1.4)
    return fig, axes


def series(axes, counts, mean, low, high, label, color, linewidth=2):
    axes[0].errorbar(
        counts,
        mean,
        yerr=[mean - low, high - mean],
        fmt="o-",
        capsize=3,
        color=color,
        label=label,
        linewidth=linewidth,
    )
    axes[1].plot(counts, mean / mean[0], "o-", label=label, color=color, linewidth=linewidth)


def save(fig, path, footer):
    for axis in fig.axes:
        axis.set_ylim(bottom=0)
    fig.text(0.075, 0.04, footer, fontsize=9, color="#475569")
    for ext in ("png", "svg"):
        save_image(fig, path.with_suffix(f".{ext}"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--a100-summary", type=Path)
    parser.add_argument("--rtx-summary", type=Path)
    args = parser.parse_args()
    if bool(args.a100_summary) != bool(args.rtx_summary):
        parser.error("Provide both GPU summaries, or neither to use existing CSVs")
    if args.a100_summary:
        import_gpu_summaries(
            (args.a100_summary, args.rtx_summary), args.assets / "gpu_scaling_measurements.csv"
        )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    cpu = read_csv(args.assets / "cpu_batch_scaling.csv")
    counts = [1, 2, 4, 8, 16, 32]
    fig, axes = figure(
        "CPU · multi-core throughput and scalability",
        "30,037 real HESTA CS20 cells × 4 traits · AMD EPYC · node-local SSD · 3 repetitions",
        counts,
        "Physical CPU cores / Numba threads",
    )
    for batch, color in ((100, "#8795a6"), (400, "#007f86"), (1600, "#cc7135")):
        rows = sorted((r for r in cpu if int(r["batch"]) == batch), key=lambda r: int(r["cores"]))
        assert [int(r["cores"]) for r in rows] == counts
        mean, low, high = (
            np.array([float(r[key]) for r in rows])
            for key in ("mean_ct_s", "min_ct_s", "max_ct_s")
        )
        series(
            axes,
            counts,
            mean,
            low,
            high,
            f"Batch {batch}" + (" · default" if batch == 400 else ""),
            color,
            linewidth=2.8 if batch == 400 else 1.6,
        )
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    save(
        fig,
        args.assets / "cpu_batch_scaling",
        "Mean throughput; bars show observed min–max. Warm pipeline includes backend setup; excludes JIT and initial staging.",
    )

    gpu = read_csv(args.assets / "gpu_scaling_measurements.csv")
    counts = [1, 2, 4]
    fig, axes = figure(
        "GPU · million-cell throughput and scalability",
        "1,000,037 real HESTA CS20 cells × 4 traits · batch 100 / GPU · node-local SSD · 2 repetitions",
        counts,
        "GPUs on one node",
    )
    for label, color in (("A100", "#007f86"), ("RTX 2080 Ti", "#cc7135")):
        samples = [
            [float(r["cell_trait_s"]) for r in gpu if r["gpu"] == label and int(r["gpus"]) == n]
            for n in counts
        ]
        assert all(len(s) == 2 for s in samples)
        mean, low, high = (np.array([fn(s) for s in samples]) for fn in (np.mean, np.min, np.max))
        series(axes, counts, mean, low, high, label, color)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    save(
        fig,
        args.assets / "gpu_scaling",
        "Mean throughput; bars show observed min–max. Warm pipeline includes output/merge; excludes setup, JIT and staging.",
    )


if __name__ == "__main__":
    main()

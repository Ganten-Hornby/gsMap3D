#!/usr/bin/env python3
"""Compare supervised and no-annotation Mouse E11.5 2D/3D embeddings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from compare_mouse_e115_heart_embeddings import (
    RASTER_DPI,
    REPRESENTATIONS,
    approximate_neighbors,
    calculate_shared_umaps,
    category_counts,
    label_purity,
    linear_cka,
    mean_cosine_similarity_on_edges,
    physical_evaluation_edges,
    plot_umap_annotation,
    procrustes_fit,
    read_embedding,
    reorder_to_reference,
    same_label_chance,
    section_neighbor_fractions,
    shared_neighbor_fraction,
    validate_match,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONDITIONS = (
    "2D supervised",
    "3D supervised",
    "2D no annotation",
    "3D no annotation",
)
PAIRS = (
    ("supervised: 2D vs 3D", "2D supervised", "3D supervised"),
    ("no annotation: 2D vs 3D", "2D no annotation", "3D no annotation"),
    ("2D: supervised vs no annotation", "2D supervised", "2D no annotation"),
    ("3D: supervised vs no annotation", "3D supervised", "3D no annotation"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised-2d", type=Path, required=True)
    parser.add_argument("--supervised-3d", type=Path, required=True)
    parser.add_argument("--unannotated-2d", type=Path, required=True)
    parser.add_argument("--unannotated-3d", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--spatial-neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    )
    return parser.parse_args()


def add_metric(
    rows: list[dict[str, Any]],
    representation: str,
    metric: str,
    method: str,
    value: float,
    n: int,
) -> None:
    rows.append(
        {
            "representation": representation,
            "metric": metric,
            "method": method,
            "value": float(value),
            "n_observations": int(n),
        }
    )


def metric_value(
    metrics: pd.DataFrame, representation: str, metric: str, method: str
) -> float:
    values = metrics.loc[
        (metrics["representation"] == representation)
        & (metrics["metric"] == metric)
        & (metrics["method"] == method),
        "value",
    ]
    if len(values) != 1:
        raise ValueError(f"Expected one value for {(representation, metric, method)}")
    return float(values.iloc[0])


def calculate_ablation_metrics(
    datasets: dict[str, Any],
    n_neighbors: int,
    spatial_neighbors: int,
    seed: int,
    threads: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = datasets["3D supervised"]
    within_source, within_target, cross_source, cross_target, physical_audit = (
        physical_evaluation_edges(
            reference.coordinates,
            reference.sections,
            spatial_neighbors,
            threads,
        )
    )
    rows: list[dict[str, Any]] = []
    neighbors: dict[tuple[str, str], np.ndarray] = {}
    n = len(reference.ids)

    for rep_index, representation in enumerate(REPRESENTATIONS):
        for condition_index, condition in enumerate(CONDITIONS):
            embedding = datasets[condition].embeddings[representation]
            latent_neighbors = approximate_neighbors(
                embedding,
                n_neighbors,
                seed + 10 * rep_index + condition_index,
                threads,
            )
            neighbors[condition, representation] = latent_neighbors
            different, adjacent = section_neighbor_fractions(
                latent_neighbors,
                reference.sections,
                reference.coordinates,
            )
            state_metrics = {
                f"Heart-region kNN purity@{n_neighbors}": label_purity(
                    latent_neighbors, reference.heart_regions
                ),
                f"ctype_user kNN purity@{n_neighbors}": label_purity(
                    latent_neighbors, reference.cell_types
                ),
                f"Different-section embedding kNN fraction@{n_neighbors}": different,
                f"Adjacent-section share among embedding kNN@{n_neighbors}": adjacent,
                "Mean cosine on within-section physical edges": mean_cosine_similarity_on_edges(
                    embedding, within_source, within_target
                ),
                "Mean cosine on cross-section radius edges": mean_cosine_similarity_on_edges(
                    embedding, cross_source, cross_target
                ),
            }
            for metric, value in state_metrics.items():
                add_metric(rows, representation, metric, condition, value, n)

        for pair_name, left_name, right_name in PAIRS:
            left = datasets[left_name].embeddings[representation]
            right = datasets[right_name].embeddings[representation]
            *_, procrustes_similarity = procrustes_fit(left, right)
            pair_metrics = {
                "Linear CKA": linear_cka(left, right),
                "Procrustes similarity": procrustes_similarity,
                f"kNN overlap@{n_neighbors}": shared_neighbor_fraction(
                    neighbors[left_name, representation],
                    neighbors[right_name, representation],
                ),
            }
            for metric, value in pair_metrics.items():
                add_metric(rows, representation, metric, pair_name, value, n)

    return pd.DataFrame(rows), {"physical_evaluation_graph": physical_audit}


def plot_metrics(metrics: pd.DataFrame, output_dir: Path, k: int) -> None:
    condition_colors = {
        "2D supervised": "#9ECAE1",
        "3D supervised": "#2171B5",
        "2D no annotation": "#FDAE6B",
        "3D no annotation": "#E6550D",
    }
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    state_panels = (
        (f"Heart-region kNN purity@{k}", "Heart-region purity", (0, 1)),
        (
            f"Adjacent-section share among embedding kNN@{k}",
            "Adjacent-section share",
            (0, 0.35),
        ),
        (
            "Mean cosine on cross-section radius edges",
            "Cross-section physical-edge cosine",
            (0, 1),
        ),
    )
    agreement_metrics = ("Linear CKA", "Procrustes similarity", f"kNN overlap@{k}")

    for row, representation in enumerate(REPRESENTATIONS):
        for col, (metric, title, ylim) in enumerate(state_panels):
            ax = axes[row, col]
            values = [metric_value(metrics, representation, metric, c) for c in CONDITIONS]
            ax.bar(
                np.arange(len(CONDITIONS)),
                values,
                color=[condition_colors[c] for c in CONDITIONS],
            )
            ax.set_xticks(
                np.arange(len(CONDITIONS)),
                ["2D\nsupervised", "3D\nsupervised", "2D\nno annotation", "3D\nno annotation"],
            )
            ax.set_ylim(*ylim)
            ax.set_title(f"{representation}: {title}")

        ax = axes[row, 3]
        x = np.arange(len(agreement_metrics))
        for offset, method, color in (
            (-0.18, "supervised: 2D vs 3D", "#2171B5"),
            (0.18, "no annotation: 2D vs 3D", "#E6550D"),
        ):
            ax.bar(
                x + offset,
                [metric_value(metrics, representation, metric, method) for metric in agreement_metrics],
                0.36,
                color=color,
                label=method,
            )
        ax.set_xticks(x, ["CKA", "Procrustes", f"kNN overlap@{k}"])
        ax.set_ylim(0, 1)
        ax.set_title(f"{representation}: 2D–3D agreement")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Mouse E11.5 heart: annotation-supervision ablation", fontsize=15)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"annotation_ablation_metrics.{suffix}", dpi=RASTER_DPI)
    plt.close(fig)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    physical_audit: dict[str, Any],
    k: int,
    heart_baseline: float,
    ctype_baseline: float,
) -> None:
    lines = [
        "# Mouse E11.5 annotation-supervision embedding ablation",
        "",
        "All four arms contain the same 98,966 observations. The supervised arms use "
        "`heart_regions` with two-stage training; the no-annotation arms use only the "
        "single-stage reconstruction objective. Graph, model, seed, feature, and observation "
        "settings are otherwise matched within the available run metadata and logs.",
        "",
        "The no-annotation `heart_regions` metrics are independent post-training label-structure "
        "summaries. The supervised values measure fit to a training target and are not independent "
        "validation.",
        "",
        "## State metrics",
        "",
    ]
    state_metrics = (
        f"Heart-region kNN purity@{k}",
        f"ctype_user kNN purity@{k}",
        f"Adjacent-section share among embedding kNN@{k}",
        "Mean cosine on cross-section radius edges",
    )
    for representation in REPRESENTATIONS:
        lines.extend(
            [
                f"### `{representation}`",
                "",
                "| Metric | 2D supervised | 3D supervised | 2D no annotation | 3D no annotation |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in state_metrics:
            values = [metric_value(metrics, representation, metric, c) for c in CONDITIONS]
            lines.append(
                f"| {metric} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Pairwise representation agreement",
            "",
            "| Representation | Comparison | Linear CKA | Procrustes | "
            f"kNN overlap@{k} |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for representation in REPRESENTATIONS:
        for pair_name, _, _ in PAIRS:
            values = [
                metric_value(metrics, representation, metric, pair_name)
                for metric in ("Linear CKA", "Procrustes similarity", f"kNN overlap@{k}")
            ]
            lines.append(
                f"| `{representation}` | {pair_name} | "
                + " | ".join(f"{value:.4f}" for value in values)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Evaluation notes",
            "",
            f"- Label-frequency baselines: heart_regions={heart_baseline:.4f}; "
            f"ctype_user={ctype_baseline:.4f}.",
            f"- Physical evaluation graph: {physical_audit['n_within_section_directed_edges']:,} "
            f"within-section and {physical_audit['n_cross_section_directed_edges']:,} "
            "cross-section directed edges; no fallback edges were forced.",
            f"- Zero qualifying cross-section-neighbor fraction: "
            f"{physical_audit['zero_cross_section_neighbor_fraction']:.2%}.",
            "- All metrics and figures use all 98,966 observations; no downsampling was used.",
            "- UMAP coordinates were computed jointly only for the two no-annotation arms; "
            "the existing supervised comparison remains in its prior report.",
        ]
    )
    (output_dir / "annotation_ablation_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "2D supervised": args.supervised_2d,
        "3D supervised": args.supervised_3d,
        "2D no annotation": args.unannotated_2d,
        "3D no annotation": args.unannotated_3d,
    }
    reference = read_embedding(args.supervised_3d)
    datasets = {"3D supervised": reference}
    match_audit: dict[str, Any] = {}
    for condition, path in input_paths.items():
        if condition == "3D supervised":
            data = reference
        else:
            data = reorder_to_reference(read_embedding(path), reference)
        datasets[condition] = data
        match_audit[condition] = validate_match(data, reference)

    metrics, metric_audit = calculate_ablation_metrics(
        datasets,
        args.neighbors,
        args.spatial_neighbors,
        args.seed,
        args.threads,
    )
    metrics.to_csv(args.output_dir / "annotation_ablation_metrics.tsv", sep="\t", index=False)
    plot_metrics(metrics, args.output_dir, args.neighbors)

    layouts, umap_audit = calculate_shared_umaps(
        datasets["2D no annotation"],
        datasets["3D no annotation"],
        args.seed,
        args.threads,
    )
    np.savez_compressed(
        args.output_dir / "no_annotation_shared_umap_coordinates.npz",
        **{
            f"{representation}_{method}": coordinates
            for representation, pair in layouts.items()
            for method, coordinates in zip(("2d", "3d"), pair, strict=True)
        },
    )
    plot_umap_annotation(
        layouts,
        reference.heart_regions,
        "heart_regions",
        "no_annotation_heart_regions_umap",
        "Mouse E11.5 no-annotation latent spaces (all observations)\n"
        "heart_regions used only for post-training evaluation",
        args.output_dir,
    )
    plot_umap_annotation(
        layouts,
        reference.cell_types,
        "ctype_user",
        "no_annotation_ctype_user_umap",
        "Mouse E11.5 no-annotation latent spaces (all observations)\n"
        "ctype_user used only for post-training evaluation",
        args.output_dir,
    )

    heart_baseline = same_label_chance(reference.heart_regions)
    ctype_baseline = same_label_chance(reference.cell_types)
    summary = {
        "inputs": {name: str(path.resolve()) for name, path in input_paths.items()},
        "parameters": {
            "neighbors": args.neighbors,
            "spatial_neighbors": args.spatial_neighbors,
            "seed": args.seed,
            "threads": args.threads,
            "raster_dpi": RASTER_DPI,
        },
        "sampling_audit": {
            "n_observations": len(reference.ids),
            "metrics_subsampling": False,
            "visualization_subsampling": False,
        },
        "annotation_audit": {
            "heart_regions_counts": category_counts(reference.heart_regions),
            "heart_regions_same_label_chance": heart_baseline,
            "ctype_user_counts": category_counts(reference.cell_types),
            "ctype_user_same_label_chance": ctype_baseline,
            "heart_regions_independent_for_no_annotation_arms": True,
            "heart_regions_independent_for_supervised_arms": False,
        },
        "match_audit": match_audit,
        "metric_audit": metric_audit,
        "umap_audit": umap_audit,
        "no_annotation_job_note": (
            "Jobs 11907274 and 11907275 completed training, inference, and H5AD output, then "
            "failed only while serializing optional annotation metadata."
        ),
    }
    (args.output_dir / "annotation_ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_report(
        args.output_dir,
        metrics,
        metric_audit["physical_evaluation_graph"],
        args.neighbors,
        heart_baseline,
        ctype_baseline,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

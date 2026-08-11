#!/usr/bin/env python3
"""Compare matched Mouse E11.5 heart sectionwise-2D and section-aware-3D embeddings."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from pynndescent import NNDescent
from umap import UMAP

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPRESENTATIONS = ("emb_cell", "emb_niche")
METHOD_2D = "sectionwise 2D"
METHOD_3D = "section-aware 3D"
COMPARISON = "2D vs 3D"
RASTER_DPI = 320


@dataclass(frozen=True)
class EmbeddingData:
    ids: np.ndarray
    sections: np.ndarray
    heart_regions: np.ndarray
    cell_types: np.ndarray
    coordinates: np.ndarray
    embeddings: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-2d", type=Path, required=True)
    parser.add_argument("--embedding-3d", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--graph-edges-2d", type=int, required=True)
    parser.add_argument("--graph-edges-3d", type=int, required=True)
    return parser.parse_args()


def _decode(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray(
            [value.decode() if isinstance(value, bytes) else str(value) for value in values],
            dtype=object,
        )
    return values.astype(str)


def read_obs_column(handle: h5py.File, key: str) -> np.ndarray:
    node = handle["obs"][key]
    if isinstance(node, h5py.Group):
        categories = _decode(node["categories"][:])
        codes = np.asarray(node["codes"][:], dtype=np.int64)
        if np.any(codes < 0):
            raise ValueError(f"obs[{key!r}] contains missing values")
        return categories[codes]
    return _decode(node[:])


def read_embedding(path: Path) -> EmbeddingData:
    with h5py.File(path, "r") as handle:
        ids = _decode(handle["obs"]["_index"][:])
        return EmbeddingData(
            ids=ids,
            sections=read_obs_column(handle, "slices"),
            heart_regions=read_obs_column(handle, "heart_regions"),
            cell_types=read_obs_column(handle, "ctype_user"),
            coordinates=np.asarray(
                handle["obsm"]["aligned_spatial_3D_mm"][:], dtype=np.float32
            ),
            embeddings={
                key: np.asarray(handle["obsm"][key][:], dtype=np.float32)
                for key in REPRESENTATIONS
            },
        )


def reorder_to_reference(data: EmbeddingData, reference: EmbeddingData) -> EmbeddingData:
    if len(np.unique(data.ids)) != len(data.ids) or len(np.unique(reference.ids)) != len(
        reference.ids
    ):
        raise ValueError("Observation IDs must be unique")
    indexer = pd.Index(data.ids).get_indexer(reference.ids)
    if np.any(indexer < 0) or len(data.ids) != len(reference.ids):
        raise ValueError("2D and 3D outputs do not contain the same observation IDs")
    return EmbeddingData(
        ids=data.ids[indexer],
        sections=data.sections[indexer],
        heart_regions=data.heart_regions[indexer],
        cell_types=data.cell_types[indexer],
        coordinates=data.coordinates[indexer],
        embeddings={key: value[indexer] for key, value in data.embeddings.items()},
    )


def validate_match(two_d: EmbeddingData, three_d: EmbeddingData) -> dict[str, Any]:
    fields = {
        "ids": np.array_equal(two_d.ids, three_d.ids),
        "sections": np.array_equal(two_d.sections, three_d.sections),
        "heart_regions": np.array_equal(two_d.heart_regions, three_d.heart_regions),
        "ctype_user": np.array_equal(two_d.cell_types, three_d.cell_types),
        "coordinates": bool(np.allclose(two_d.coordinates, three_d.coordinates, atol=1e-8)),
    }
    if not all(fields.values()):
        raise ValueError(f"Matched-input validation failed: {fields}")
    return {"n_matched_spots": int(len(two_d.ids)), **fields}


def category_counts(labels: np.ndarray) -> dict[str, int]:
    categories, counts = np.unique(labels, return_counts=True)
    return {
        str(category): int(count)
        for category, count in zip(categories, counts, strict=True)
    }


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = np.asarray(x, dtype=np.float64) - np.mean(x, axis=0, keepdims=True)
    y_centered = np.asarray(y, dtype=np.float64) - np.mean(y, axis=0, keepdims=True)
    cross = x_centered.T @ y_centered
    denominator = np.linalg.norm(x_centered.T @ x_centered, ord="fro") * np.linalg.norm(
        y_centered.T @ y_centered, ord="fro"
    )
    return float(np.linalg.norm(cross, ord="fro") ** 2 / denominator)


def procrustes_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float]:
    x_center = np.asarray(x, dtype=np.float64).mean(axis=0)
    y_center = np.asarray(y, dtype=np.float64).mean(axis=0)
    x_centered = np.asarray(x, dtype=np.float64) - x_center
    y_centered = np.asarray(y, dtype=np.float64) - y_center
    x_scale = float(np.linalg.norm(x_centered, ord="fro"))
    y_scale = float(np.linalg.norm(y_centered, ord="fro"))
    x_normalized = x_centered / x_scale
    y_normalized = y_centered / y_scale
    u, singular_values, vt = np.linalg.svd(
        x_normalized.T @ y_normalized, full_matrices=False
    )
    rotation = u @ vt
    return x_center, y_center, x_scale, y_scale, rotation, float(singular_values.sum())


def approximate_neighbors(
    embedding: np.ndarray, n_neighbors: int, seed: int, threads: int
) -> np.ndarray:
    index = NNDescent(
        embedding,
        n_neighbors=n_neighbors + 1,
        metric="cosine",
        random_state=seed,
        n_jobs=threads,
        low_memory=True,
    )
    raw_neighbors = np.asarray(index.neighbor_graph[0], dtype=np.int64)
    result = np.empty((len(embedding), n_neighbors), dtype=np.int64)
    for row, candidates in enumerate(raw_neighbors):
        external = candidates[candidates != row]
        if len(external) < n_neighbors:
            raise RuntimeError(f"Neighbor search returned {len(external)} external neighbors")
        result[row] = external[:n_neighbors]
    return result


def shared_neighbor_fraction(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("Neighbor matrices must have identical shapes")
    intersections = np.fromiter(
        (
            len(np.intersect1d(left, right, assume_unique=False))
            for left, right in zip(first, second, strict=True)
        ),
        dtype=np.float64,
        count=len(first),
    )
    return float(np.mean(intersections / first.shape[1]))


def label_purity(neighbors: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(labels[neighbors] == labels[:, None]))


def same_label_chance(labels: np.ndarray) -> float:
    """Return the same-label probability under independent frequency-matched draws."""

    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    return float(np.sum(proportions**2))


def section_neighbor_fractions(
    neighbors: np.ndarray, sections: np.ndarray, coordinates: np.ndarray
) -> tuple[float, float]:
    unique_sections = np.unique(sections)
    ordered = sorted(
        unique_sections,
        key=lambda section: float(np.median(coordinates[sections == section, 2])),
    )
    rank_by_section = {section: rank for rank, section in enumerate(ordered)}
    ranks = np.asarray([rank_by_section[section] for section in sections])
    neighbor_sections = sections[neighbors]
    different = float(np.mean(neighbor_sections != sections[:, None]))
    adjacent = float(np.mean(np.abs(ranks[neighbors] - ranks[:, None]) == 1))
    return different, adjacent


def add_metric(
    rows: list[dict[str, Any]], representation: str, metric: str, method: str, value: float, n: int
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


def metric_value(metrics: pd.DataFrame, representation: str, metric: str, method: str) -> float:
    values = metrics.loc[
        (metrics["representation"] == representation)
        & (metrics["metric"] == metric)
        & (metrics["method"] == method),
        "value",
    ]
    if len(values) != 1:
        raise ValueError(f"Expected one value for {(representation, metric, method)}")
    return float(values.iloc[0])


def calculate_metrics(
    two_d: EmbeddingData,
    three_d: EmbeddingData,
    n_neighbors: int,
    seed: int,
    threads: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    n = len(two_d.ids)
    for offset, representation in enumerate(REPRESENTATIONS):
        old = two_d.embeddings[representation]
        new = three_d.embeddings[representation]
        old_neighbors = approximate_neighbors(old, n_neighbors, seed + offset, threads)
        new_neighbors = approximate_neighbors(new, n_neighbors, seed + offset, threads)
        *_, procrustes_similarity = procrustes_fit(old, new)
        overlap = shared_neighbor_fraction(old_neighbors, new_neighbors)
        audit[representation] = {
            "knn_overlap": overlap,
            "n_neighbors": n_neighbors,
            "distance": "cosine",
            "neighbor_backend": "pynndescent.NNDescent",
        }
        add_metric(rows, representation, "Linear CKA", COMPARISON, linear_cka(old, new), n)
        add_metric(
            rows,
            representation,
            "Procrustes similarity",
            COMPARISON,
            procrustes_similarity,
            n,
        )
        add_metric(rows, representation, f"kNN overlap@{n_neighbors}", COMPARISON, overlap, n)
        for method, neighbors in ((METHOD_2D, old_neighbors), (METHOD_3D, new_neighbors)):
            add_metric(
                rows,
                representation,
                f"Training heart-region kNN purity@{n_neighbors}",
                method,
                label_purity(neighbors, three_d.heart_regions),
                n,
            )
            add_metric(
                rows,
                representation,
                f"ctype_user kNN purity@{n_neighbors}",
                method,
                label_purity(neighbors, three_d.cell_types),
                n,
            )
            different, adjacent = section_neighbor_fractions(
                neighbors, three_d.sections, three_d.coordinates
            )
            add_metric(
                rows,
                representation,
                f"Different-section latent kNN fraction@{n_neighbors}",
                method,
                different,
                n,
            )
            add_metric(
                rows,
                representation,
                f"Adjacent-section latent kNN fraction@{n_neighbors}",
                method,
                adjacent,
                n,
            )
    return pd.DataFrame(rows), audit


def plot_metric_summary(
    metrics: pd.DataFrame,
    output_dir: Path,
    k: int,
    heart_region_baseline: float,
    cell_type_baseline: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"emb_cell": "#3182BD", "emb_niche": "#E6550D"}

    agreement = ["Linear CKA", "Procrustes similarity", f"kNN overlap@{k}"]
    x = np.arange(len(agreement))
    for offset, representation in ((-0.18, "emb_cell"), (0.18, "emb_niche")):
        axes[0, 0].bar(
            x + offset,
            [metric_value(metrics, representation, metric, COMPARISON) for metric in agreement],
            0.36,
            color=colors[representation],
            label=representation,
        )
    axes[0, 0].set_xticks(x, ["Linear CKA", "Procrustes", f"kNN overlap@{k}"])
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("A  Agreement between 2D and 3D")
    axes[0, 0].legend(frameon=False)

    panels = [
        (
            axes[0, 1],
            f"Training heart-region kNN purity@{k}",
            "B  Training-label purity",
            heart_region_baseline,
        ),
        (
            axes[1, 0],
            f"ctype_user kNN purity@{k}",
            "C  Cell-type-label purity",
            cell_type_baseline,
        ),
        (
            axes[1, 1],
            f"Different-section latent kNN fraction@{k}",
            "D  Cross-section latent neighbors",
            None,
        ),
    ]
    rep_x = np.arange(len(REPRESENTATIONS))
    for ax, metric, title, baseline in panels:
        ax.bar(
            rep_x - 0.18,
            [metric_value(metrics, rep, metric, METHOD_2D) for rep in REPRESENTATIONS],
            0.36,
            color="#9ECAE1",
            label=METHOD_2D,
        )
        ax.bar(
            rep_x + 0.18,
            [metric_value(metrics, rep, metric, METHOD_3D) for rep in REPRESENTATIONS],
            0.36,
            color="#F16913",
            label=METHOD_3D,
        )
        ax.set_xticks(rep_x, REPRESENTATIONS)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        if baseline is not None:
            ax.axhline(
                baseline,
                color="#555555",
                linestyle="--",
                linewidth=1,
                label="label-frequency baseline",
            )
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Mouse E11.5 heart: matched 2D and 3D embedding comparison", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"embedding_comparison_metrics.{suffix}", dpi=RASTER_DPI)
    plt.close(fig)


def categorical_colors(labels: np.ndarray) -> dict[str, str]:
    categories = sorted(np.unique(labels))
    palette = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, len(categories)))
    return {category: to_hex(color) for category, color in zip(categories, palette, strict=True)}


def plot_scanpy_panel(
    ax: plt.Axes,
    coordinates: np.ndarray,
    labels: np.ndarray,
    annotation_key: str,
    colors: dict[str, str],
    point_size: float,
    title: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    categories = list(colors)
    obs = pd.DataFrame(
        {annotation_key: pd.Categorical(labels, categories=categories)},
        index=pd.Index(np.arange(len(labels)).astype(str), name="spot"),
    )
    panel = ad.AnnData(obs=obs)
    panel.obsm["X_umap"] = np.asarray(coordinates, dtype=np.float32)
    sc.pl.umap(
        panel,
        color=annotation_key,
        palette=[colors[category] for category in categories],
        size=point_size,
        alpha=0.7,
        frameon=False,
        legend_loc=None,
        title=title,
        ax=ax,
        show=False,
    )
    for collection in ax.collections:
        collection.set_rasterized(True)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Shared UMAP 1")
    ax.set_ylabel("Shared UMAP 2")


def calculate_shared_umaps(
    two_d: EmbeddingData,
    three_d: EmbeddingData,
    seed: int,
    threads: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    layouts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    audit: dict[str, Any] = {
        "n_spots": int(len(two_d.ids)),
        "subsampling": False,
        "coordinate_backend": "umap.UMAP",
        "plot_backend": "scanpy.pl.umap",
        "scanpy_version": version("scanpy"),
        "parameters": {
            "n_neighbors": 30,
            "min_dist": 0.25,
            "metric": "cosine",
            "random_state": seed,
            "low_memory": True,
            "n_jobs_requested": threads,
            "n_jobs_effective": 1,
        },
    }
    for representation in REPRESENTATIONS:
        old = two_d.embeddings[representation]
        new = three_d.embeddings[representation]
        x_center, y_center, x_scale, y_scale, rotation, similarity = procrustes_fit(old, new)
        old_aligned = (((old - x_center) / x_scale) @ rotation).astype(np.float32)
        new_aligned = ((new - y_center) / y_scale).astype(np.float32)
        joint = np.concatenate((old_aligned, new_aligned), axis=0)
        layout = UMAP(
            n_neighbors=30,
            min_dist=0.25,
            n_components=2,
            metric="cosine",
            random_state=seed,
            n_jobs=threads,
            low_memory=True,
        ).fit_transform(joint)
        layouts[representation] = tuple(np.split(layout.astype(np.float32), 2))
        audit[representation] = {"procrustes_similarity": similarity}
    return layouts, audit


def plot_umap_annotation(
    layouts: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    annotation_key: str,
    output_stem: str,
    title: str,
    output_dir: Path,
) -> None:
    colors = categorical_colors(labels)
    point_size = min(1.7, max(0.05, 120_000 / len(labels)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for row, representation in enumerate(REPRESENTATIONS):
        old_layout, new_layout = layouts[representation]
        joint = np.concatenate((old_layout, new_layout))
        limits = (
            (float(joint[:, 0].min()), float(joint[:, 0].max())),
            (float(joint[:, 1].min()), float(joint[:, 1].max())),
        )
        for col, (method, coordinates) in enumerate(
            ((METHOD_2D, old_layout), (METHOD_3D, new_layout))
        ):
            plot_scanpy_panel(
                axes[row, col],
                coordinates,
                labels,
                annotation_key,
                colors,
                point_size,
                f"{representation}: {method}",
                limits,
            )
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=3, color=colors[label], label=label)
        for label in colors
    ]
    fig.legend(
        handles,
        list(colors),
        loc="outside right center",
        frameon=False,
        fontsize=6 if len(colors) > 15 else 8,
        markerscale=4,
        ncols=2 if len(colors) > 20 else 1,
    )
    fig.suptitle(title, fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{output_stem}.{suffix}", dpi=RASTER_DPI, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    match_audit: dict[str, Any],
    graph_edges_2d: int,
    graph_edges_3d: int,
    k: int,
    heart_region_baseline: float,
    cell_type_baseline: float,
) -> None:
    lines = [
        "# Mouse E11.5 heart: matched sectionwise-2D versus section-aware-3D embeddings",
        "",
        "## Scope",
        "",
        f"Both arms contain the same **{match_audit['n_matched_spots']:,} observations** and were trained with identical data, `heart_regions` annotation, two-stage objective, model settings, and random seed. The sectionwise-2D graph contains "
        f"{graph_edges_2d:,} directed edges; the section-aware-3D graph contains {graph_edges_3d:,}.",
        "",
        "All metrics and UMAP panels use every matched observation. No visualization or metric is downsampled.",
        "",
        "`heart_regions` was explicitly used as the training annotation. Consequently, heart-region purity and UMAP separation quantify training-label fit and are not independent validation.",
        "",
        "## Metrics",
        "",
        f"| Representation | Linear CKA | Procrustes similarity | kNN overlap@{k} | 2D training-region purity | 3D training-region purity | 2D ctype purity | 3D ctype purity | 2D different-section kNN | 3D different-section kNN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for representation in REPRESENTATIONS:
        lines.append(
            f"| `{representation}` | "
            f"{metric_value(metrics, representation, 'Linear CKA', COMPARISON):.4f} | "
            f"{metric_value(metrics, representation, 'Procrustes similarity', COMPARISON):.4f} | "
            f"{metric_value(metrics, representation, f'kNN overlap@{k}', COMPARISON):.4f} | "
            f"{metric_value(metrics, representation, f'Training heart-region kNN purity@{k}', METHOD_2D):.4f} | "
            f"{metric_value(metrics, representation, f'Training heart-region kNN purity@{k}', METHOD_3D):.4f} | "
            f"{metric_value(metrics, representation, f'ctype_user kNN purity@{k}', METHOD_2D):.4f} | "
            f"{metric_value(metrics, representation, f'ctype_user kNN purity@{k}', METHOD_3D):.4f} | "
            f"{metric_value(metrics, representation, f'Different-section latent kNN fraction@{k}', METHOD_2D):.4f} | "
            f"{metric_value(metrics, representation, f'Different-section latent kNN fraction@{k}', METHOD_3D):.4f} |"
        )
    lines.extend(
        [
            "",
            f"`kNN overlap@{k}` is the mean shared-neighbor fraction between the two fixed-size neighbor sets. It is symmetric and does not define either arm as a positive/reference class.",
            "",
            f"The independent frequency-matched same-label baseline is {heart_region_baseline:.4f} for `heart_regions` and {cell_type_baseline:.4f} for `ctype_user`. The latter purity is dominated by the `Cardiac muscle lineages` majority class (94.79% of observations), so values near 0.90 provide little evidence of cell-type structure beyond label frequency.",
            "",
            "## Outputs",
            "",
            "- `embedding_comparison_metrics.png` / `.pdf`: metric summary.",
            "- `embedding_comparison_heart_regions_umap.png` / `.pdf`: full-data shared UMAP colored by the training annotation.",
            "- `embedding_comparison_ctype_user_umap.png` / `.pdf`: full-data shared UMAP colored by cell-type annotation.",
            "- `embedding_comparison_shared_umap_coordinates.npz`: all shared UMAP coordinates.",
            "- `embedding_comparison_metrics.tsv`: tidy metric values.",
            "- `embedding_comparison_summary.json`: input matching, parameters, and no-subsampling audit.",
            "",
        ]
    )
    (output_dir / "embedding_comparison_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    two_d = read_embedding(args.embedding_2d)
    three_d = read_embedding(args.embedding_3d)
    two_d = reorder_to_reference(two_d, three_d)
    match_audit = validate_match(two_d, three_d)

    metrics, metric_audit = calculate_metrics(
        two_d, three_d, args.neighbors, args.seed, args.threads
    )
    heart_region_baseline = same_label_chance(three_d.heart_regions)
    cell_type_baseline = same_label_chance(three_d.cell_types)
    metrics.to_csv(args.output_dir / "embedding_comparison_metrics.tsv", sep="\t", index=False)
    plot_metric_summary(
        metrics,
        args.output_dir,
        args.neighbors,
        heart_region_baseline,
        cell_type_baseline,
    )

    layouts, umap_audit = calculate_shared_umaps(two_d, three_d, args.seed, args.threads)
    np.savez_compressed(
        args.output_dir / "embedding_comparison_shared_umap_coordinates.npz",
        **{
            f"{representation}_{method}": coordinates
            for representation, pair in layouts.items()
            for method, coordinates in zip(("2d", "3d"), pair, strict=True)
        },
    )
    plot_umap_annotation(
        layouts,
        three_d.heart_regions,
        "heart_regions",
        "embedding_comparison_heart_regions_umap",
        "Mouse E11.5 heart latent spaces (all observations)\nColors are heart_regions used in two-stage training",
        args.output_dir,
    )
    plot_umap_annotation(
        layouts,
        three_d.cell_types,
        "ctype_user",
        "embedding_comparison_ctype_user_umap",
        "Mouse E11.5 heart latent spaces (all observations)\nColors are ctype_user",
        args.output_dir,
    )

    summary = {
        "inputs": {
            "embedding_2d": str(args.embedding_2d),
            "embedding_3d": str(args.embedding_3d),
        },
        "parameters": {
            "neighbors": args.neighbors,
            "seed": args.seed,
            "threads": args.threads,
            "raster_dpi": RASTER_DPI,
            "graph_edges_2d": args.graph_edges_2d,
            "graph_edges_3d": args.graph_edges_3d,
        },
        "match_audit": match_audit,
        "sampling_audit": {
            "n_observations": int(len(three_d.ids)),
            "metrics_subsampling": False,
            "visualization_subsampling": False,
        },
        "annotation_audit": {
            "training_annotation": "heart_regions",
            "training_annotation_counts": category_counts(three_d.heart_regions),
            "training_annotation_same_label_chance": heart_region_baseline,
            "cell_type_annotation": "ctype_user",
            "cell_type_annotation_counts": category_counts(three_d.cell_types),
            "cell_type_annotation_same_label_chance": cell_type_baseline,
            "training_label_evaluation_is_independent": False,
        },
        "metric_audit": metric_audit,
        "umap_audit": umap_audit,
    }
    (args.output_dir / "embedding_comparison_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    write_report(
        args.output_dir,
        metrics,
        match_audit,
        args.graph_edges_2d,
        args.graph_edges_3d,
        args.neighbors,
        heart_region_baseline,
        cell_type_baseline,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

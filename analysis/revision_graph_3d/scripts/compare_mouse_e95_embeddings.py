#!/usr/bin/env python3
"""Compare matched Mouse E9.5 section-wise 2D and section-aware 3D embeddings.

The input H5AD files are read with h5py so expression matrices are never loaded.
All comparisons use the same spots, matched by their globally unique observation IDs.
"""

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
import yaml
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scanpy as sc  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from pynndescent import NNDescent  # noqa: E402
from umap import UMAP  # noqa: E402


METHOD_2D = "Section-wise 2D"
METHOD_3D = "Section-aware 3D"
COMPARISON = "2D vs 3D"
REPRESENTATIONS = ("emb_cell", "emb_niche")
RASTER_DPI = 320


@dataclass(frozen=True)
class EmbeddingData:
    ids: np.ndarray
    sections: np.ndarray
    labels: np.ndarray
    coordinates: np.ndarray
    embeddings: dict[str, np.ndarray]


@dataclass(frozen=True)
class LatentMetricNames:
    k: int
    recall: str
    jaccard: str
    label_purity: str
    different_section: str
    adjacent_section: str

    @classmethod
    def for_k(cls, k: int) -> LatentMetricNames:
        return cls(
            k=k,
            recall=f"kNN recall@{k}",
            jaccard=f"kNN Jaccard@{k}",
            label_purity=f"Label kNN purity@{k}",
            different_section=f"Different-section latent kNN fraction@{k}",
            adjacent_section=f"Adjacent-section latent kNN fraction@{k}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--section-aware-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=30_000)
    parser.add_argument("--umap-sample-size", type=int, default=25_000)
    parser.add_argument("--latent-neighbors", type=int, default=15)
    parser.add_argument("--spatial-neighbors", type=int, default=10)
    parser.add_argument("--cross-section-cap", type=int, default=10)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--seed", type=int, default=2024)
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
        result = np.full(codes.shape, "NA", dtype=object)
        valid = codes >= 0
        result[valid] = categories[codes[valid]]
        return result
    return _decode(node[:])


def read_single_h5ad(path: Path, section_key: str) -> EmbeddingData:
    with h5py.File(path, "r") as handle:
        ids = _decode(handle["obs"]["_index"][:])
        sections = read_obs_column(handle, section_key)
        labels = read_obs_column(handle, "transf_anno_E95")
        coordinates = np.asarray(handle["obsm"]["3d_align_spatial"][:], dtype=np.float32)
        embeddings = {
            key: np.asarray(handle["obsm"][key][:], dtype=np.float32)
            for key in REPRESENTATIONS
        }
    return EmbeddingData(ids, sections, labels, coordinates, embeddings)


def read_sectionwise_baseline(directory: Path) -> EmbeddingData:
    paths = sorted(directory.glob("Mouse_E9.5_embryo_slice_*_add_latent.h5ad"))
    if len(paths) != 70:
        raise ValueError(f"Expected 70 baseline section files, found {len(paths)} in {directory}")
    pieces = [read_single_h5ad(path, "slice_id") for path in paths]
    return EmbeddingData(
        ids=np.concatenate([piece.ids for piece in pieces]),
        sections=np.concatenate([piece.sections for piece in pieces]),
        labels=np.concatenate([piece.labels for piece in pieces]),
        coordinates=np.concatenate([piece.coordinates for piece in pieces]),
        embeddings={
            key: np.concatenate([piece.embeddings[key] for piece in pieces])
            for key in REPRESENTATIONS
        },
    )


def reorder_baseline_to_section_aware(
    baseline: EmbeddingData, section_aware: EmbeddingData
) -> tuple[EmbeddingData, dict[str, Any]]:
    baseline_index = pd.Index(baseline.ids)
    if not baseline_index.is_unique or not pd.Index(section_aware.ids).is_unique:
        raise ValueError("Observation IDs must be unique in both runs")
    order = baseline_index.get_indexer(section_aware.ids)
    if np.any(order < 0):
        raise ValueError(f"Missing {int(np.sum(order < 0))} section-aware spots from baseline")
    reordered = EmbeddingData(
        ids=baseline.ids[order],
        sections=baseline.sections[order],
        labels=baseline.labels[order],
        coordinates=baseline.coordinates[order],
        embeddings={key: baseline.embeddings[key][order] for key in REPRESENTATIONS},
    )
    baseline_sections = np.asarray(reordered.sections, dtype=float)
    section_aware_sections = np.asarray(section_aware.sections, dtype=float)
    coordinate_delta = np.max(np.abs(reordered.coordinates - section_aware.coordinates), axis=1)
    audit = {
        "n_baseline_spots": int(len(baseline.ids)),
        "n_section_aware_spots": int(len(section_aware.ids)),
        "n_matched_spots": int(len(order)),
        "spot_match_fraction": float(len(order) / len(section_aware.ids)),
        "section_match_fraction": float(np.mean(baseline_sections == section_aware_sections)),
        "label_match_fraction": float(np.mean(reordered.labels == section_aware.labels)),
        "coordinate_match_fraction_at_1e-5": float(np.mean(coordinate_delta <= 1e-5)),
        "max_coordinate_absolute_difference": float(np.max(coordinate_delta)),
    }
    if audit["section_match_fraction"] != 1.0 or audit["coordinate_match_fraction_at_1e-5"] != 1.0:
        raise ValueError(f"Matched spot metadata disagree: {audit}")
    return reordered, audit


def stratified_sample(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    if size >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    target = max(1, size // len(groups))
    chosen: list[np.ndarray] = []
    remaining: list[np.ndarray] = []
    for indices in groups.values():
        take = min(target, len(indices))
        selected = rng.choice(indices, size=take, replace=False)
        chosen.append(selected)
        if take < len(indices):
            remaining.append(np.setdiff1d(indices, selected, assume_unique=False))
    result = np.concatenate(chosen)
    missing = size - len(result)
    if missing > 0:
        pool = np.concatenate(remaining)
        result = np.concatenate([result, rng.choice(pool, size=missing, replace=False)])
    rng.shuffle(result)
    return result.astype(np.int64, copy=False)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")
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
    u, singular_values, vt = np.linalg.svd(x_normalized.T @ y_normalized, full_matrices=False)
    rotation = u @ vt
    return x_center, y_center, x_scale, y_scale, rotation, float(singular_values.sum())


def aligned_embeddings(
    x: np.ndarray,
    y: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_center, y_center, x_scale, y_scale, rotation, similarity = procrustes_fit(
        x[fit_indices], y[fit_indices]
    )
    x_aligned = ((x - x_center) / x_scale) @ rotation
    y_aligned = (y - y_center) / y_scale
    return x_aligned.astype(np.float32), y_aligned.astype(np.float32), similarity


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
        candidates = candidates[candidates != row]
        if len(candidates) < n_neighbors:
            raise RuntimeError(f"Neighbor search returned only {len(candidates)} external neighbors")
        result[row] = candidates[:n_neighbors]
    return result


def neighbor_metrics(
    old_neighbors: np.ndarray,
    new_neighbors: np.ndarray,
    labels: np.ndarray,
    sections: np.ndarray,
) -> dict[str, float]:
    intersections = np.fromiter(
        (
            len(np.intersect1d(old_row, new_row, assume_unique=False))
            for old_row, new_row in zip(old_neighbors, new_neighbors, strict=True)
        ),
        dtype=np.float64,
        count=len(old_neighbors),
    )
    k = old_neighbors.shape[1]
    section_values = np.asarray(sections, dtype=float)
    ordered_sections = np.sort(np.unique(section_values))
    section_rank = {value: rank for rank, value in enumerate(ordered_sections)}
    ranks = np.asarray([section_rank[value] for value in section_values], dtype=np.int64)

    def summarize(neighbors: np.ndarray) -> tuple[float, float, float]:
        label_purity = np.mean(labels[neighbors] == labels[:, None])
        different_section = np.mean(section_values[neighbors] != section_values[:, None])
        adjacent_section = np.mean(np.abs(ranks[neighbors] - ranks[:, None]) == 1)
        return float(label_purity), float(different_section), float(adjacent_section)

    old_purity, old_cross, old_adjacent = summarize(old_neighbors)
    new_purity, new_cross, new_adjacent = summarize(new_neighbors)
    return {
        "knn_recall": float(np.mean(intersections / k)),
        "knn_jaccard": float(np.mean(intersections / (2 * k - intersections))),
        "old_label_purity": old_purity,
        "new_label_purity": new_purity,
        "old_different_section_fraction": old_cross,
        "new_different_section_fraction": new_cross,
        "old_adjacent_section_fraction": old_adjacent,
        "new_adjacent_section_fraction": new_adjacent,
    }


def cosine_similarity_on_edges(embedding: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = embedding[source]
    y = embedding[target]
    numerator = np.einsum("ij,ij->i", x, y)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def physical_evaluation_edges(
    coordinates: np.ndarray,
    sections: np.ndarray,
    focal_indices: np.ndarray,
    within_k: int,
    cross_cap: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    section_values = np.asarray(sections, dtype=float)
    unique_sections = np.sort(np.unique(section_values))
    positions_by_section = {
        section: np.flatnonzero(section_values == section) for section in unique_sections
    }
    trees = {
        section: cKDTree(coordinates[positions, :3])
        for section, positions in positions_by_section.items()
    }
    rank = {section: i for i, section in enumerate(unique_sections)}
    within_sources: list[np.ndarray] = []
    within_targets: list[np.ndarray] = []
    cross_sources: list[np.ndarray] = []
    cross_targets: list[np.ndarray] = []
    focal_has_cross = np.zeros(len(focal_indices), dtype=bool)
    radii = np.empty(len(focal_indices), dtype=np.float64)

    for section in unique_sections:
        focal_mask = section_values[focal_indices] == section
        focal_offsets = np.flatnonzero(focal_mask)
        if not len(focal_offsets):
            continue
        focal = focal_indices[focal_offsets]
        section_positions = positions_by_section[section]
        k_query = min(within_k + 1, len(section_positions))
        distances, local_neighbors = trees[section].query(coordinates[focal, :3], k=k_query)
        distances = np.atleast_2d(distances)
        local_neighbors = np.atleast_2d(local_neighbors)
        if distances.shape[0] != len(focal):
            distances = distances.T
            local_neighbors = local_neighbors.T
        external_local = local_neighbors[:, 1:k_query]
        external_distances = distances[:, 1:k_query]
        if external_local.shape[1] < within_k:
            raise ValueError(f"Section {section:g} has fewer than {within_k + 1} spots")
        within_sources.append(np.repeat(focal, within_k))
        within_targets.append(section_positions[external_local[:, :within_k]].reshape(-1))
        section_radii = external_distances[:, within_k - 1]
        radii[focal_offsets] = section_radii

        adjacent_ranks = [rank[section] - 1, rank[section] + 1]
        for adjacent_rank in adjacent_ranks:
            if not 0 <= adjacent_rank < len(unique_sections):
                continue
            adjacent = unique_sections[adjacent_rank]
            adjacent_positions = positions_by_section[adjacent]
            candidates = trees[adjacent].query_ball_point(
                coordinates[focal, :3], r=section_radii, workers=threads
            )
            for local_focal, candidate_local in enumerate(candidates):
                if not candidate_local:
                    continue
                candidate_local_array = np.asarray(candidate_local, dtype=np.int64)
                if len(candidate_local_array) > cross_cap:
                    delta = (
                        coordinates[adjacent_positions[candidate_local_array], :3]
                        - coordinates[focal[local_focal], :3]
                    )
                    nearest = np.argpartition(
                        np.einsum("ij,ij->i", delta, delta), cross_cap - 1
                    )[:cross_cap]
                    candidate_local_array = candidate_local_array[nearest]
                targets = adjacent_positions[candidate_local_array]
                cross_sources.append(np.full(len(targets), focal[local_focal], dtype=np.int64))
                cross_targets.append(targets)
                focal_has_cross[focal_offsets[local_focal]] = True

    within_source = np.concatenate(within_sources)
    within_target = np.concatenate(within_targets)
    cross_source = np.concatenate(cross_sources) if cross_sources else np.empty(0, dtype=np.int64)
    cross_target = np.concatenate(cross_targets) if cross_targets else np.empty(0, dtype=np.int64)
    audit = {
        "within_section_neighbors": int(within_k),
        "cross_section_candidate_cap_per_adjacent_section": int(cross_cap),
        "n_focal_spots": int(len(focal_indices)),
        "n_within_section_directed_edges": int(len(within_source)),
        "n_cross_section_directed_edges": int(len(cross_source)),
        "zero_cross_section_neighbor_fraction": float(np.mean(~focal_has_cross)),
        "within_section_radius_median": float(np.median(radii)),
        "within_section_radius_q25": float(np.quantile(radii, 0.25)),
        "within_section_radius_q75": float(np.quantile(radii, 0.75)),
    }
    return within_source, within_target, cross_source, cross_target, audit


def add_metric(
    rows: list[dict[str, Any]], representation: str, metric: str, method: str, value: float, n: int
) -> None:
    rows.append(
        {
            "representation": representation,
            "metric": metric,
            "method": method,
            "value": float(value),
            "n": int(n),
        }
    )


def metric_value(
    metrics: pd.DataFrame, representation: str, metric: str, method: str
) -> float:
    matches = metrics.loc[
        (metrics.representation == representation)
        & (metrics.metric == metric)
        & (metrics.method == method),
        "value",
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one metric row for {representation=}, {metric=}, {method=}; "
            f"found {len(matches)}"
        )
    return float(matches.iloc[0])


def plot_metric_summary(
    metrics: pd.DataFrame, output_dir: Path, latent_metric_names: LatentMetricNames
) -> None:
    representation_colors = {"emb_cell": "#3B6FB6", "emb_niche": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)

    agreement_names = ["Linear CKA", "Procrustes similarity", latent_metric_names.recall]
    ax = axes[0, 0]
    x = np.arange(len(agreement_names))
    width = 0.34
    for offset, representation in zip((-width / 2, width / 2), REPRESENTATIONS, strict=True):
        values = [
            metric_value(metrics, representation, name, COMPARISON)
            for name in agreement_names
        ]
        ax.bar(x + offset, values, width, color=representation_colors[representation], label=representation)
    ax.set_title("A  Global and local agreement")
    ax.set_xticks(
        x,
        ["Linear\nCKA", "Procrustes\nsimilarity", latent_metric_names.recall.replace(" ", "\n")],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend(frameon=False)

    def paired_panel(ax: plt.Axes, metric: str, title: str, ylabel: str) -> None:
        x = np.arange(len(REPRESENTATIONS))
        width = 0.34
        for offset, method, color in [(-width / 2, METHOD_2D, "#8C8C8C"), (width / 2, METHOD_3D, "#6A3D9A")]:
            values = [
                metric_value(metrics, representation, metric, method)
                for representation in REPRESENTATIONS
            ]
            ax.bar(x + offset, values, width, color=color, label=method)
        ax.set_title(title)
        ax.set_xticks(x, ["Cell", "Niche"])
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

    paired_panel(
        axes[0, 1],
        latent_metric_names.label_purity,
        "B  Annotation structure",
        "Same-label neighbor fraction",
    )
    axes[0, 1].set_ylim(0, 1)
    paired_panel(
        axes[1, 0],
        latent_metric_names.different_section,
        "C  Section mixing in latent space",
        "Different-section neighbor fraction",
    )
    axes[1, 0].set_ylim(0, 1)

    ax = axes[1, 1]
    groups = [("Within-section physical edges", "Within"), ("Cross-section radius edges", "Cross")]
    x = np.arange(len(groups))
    width = 0.19
    placements = [
        (-1.5 * width, METHOD_2D, "emb_cell", "#A6BDD8"),
        (-0.5 * width, METHOD_3D, "emb_cell", "#2B8CBE"),
        (0.5 * width, METHOD_2D, "emb_niche", "#FDBB84"),
        (1.5 * width, METHOD_3D, "emb_niche", "#E34A33"),
    ]
    for offset, method, representation, color in placements:
        values = [
            metric_value(metrics, representation, metric, method)
            for metric, _ in groups
        ]
        ax.bar(x + offset, values, width, color=color, label=f"{representation}: {method}")
    ax.set_title("D  Spatial-edge embedding continuity")
    ax.set_xticks(x, [short for _, short in groups])
    ax.set_ylabel("Mean cosine similarity")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Mouse E9.5 embedding comparison (matched spots)", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"embedding_comparison_metrics.{suffix}",
            dpi=RASTER_DPI,
            bbox_inches="tight",
        )
    plt.close(fig)


def categorical_colors(labels: np.ndarray) -> dict[str, str]:
    categories = sorted(np.unique(labels))
    palette = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, len(categories)))
    return {category: to_hex(color) for category, color in zip(categories, palette, strict=True)}


def plot_scanpy_umap_panel(
    ax: plt.Axes,
    coordinates: np.ndarray,
    labels: np.ndarray,
    colors: dict[str, str],
    title: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """Plot precomputed shared-UMAP coordinates through Scanpy's UMAP API."""
    categories = list(colors)
    obs = pd.DataFrame(
        {"transf_anno_E95": pd.Categorical(labels, categories=categories)},
        index=pd.Index(np.arange(len(labels)).astype(str), name="spot"),
    )
    panel = ad.AnnData(obs=obs)
    panel.obsm["X_umap"] = np.asarray(coordinates, dtype=np.float32)
    sc.pl.umap(
        panel,
        color="transf_anno_E95",
        palette=[colors[category] for category in categories],
        size=1.7,
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


def plot_shared_umap(
    baseline: EmbeddingData,
    section_aware: EmbeddingData,
    sample_indices: np.ndarray,
    output_dir: Path,
    seed: int,
    threads: int,
) -> dict[str, Any]:
    labels = section_aware.labels[sample_indices]
    colors = categorical_colors(labels)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    umap_audit: dict[str, Any] = {
        "n_spots": int(len(sample_indices)),
        "shared_axes_after_procrustes": True,
        "coordinate_backend": "umap.UMAP",
        "plot_backend": "scanpy.pl.umap",
        "scanpy_version": version("scanpy"),
        "parameters": {
            "n_neighbors": 30,
            "min_dist": 0.25,
            "n_components": 2,
            "metric": "cosine",
            "random_state": seed,
            "n_jobs_requested": threads,
            "n_jobs_effective": 1,
            "n_jobs_note": "umap-learn forces n_jobs=1 when random_state is set",
            "low_memory": True,
        },
    }
    for row, representation in enumerate(REPRESENTATIONS):
        old_aligned, new_aligned, similarity = aligned_embeddings(
            baseline.embeddings[representation], section_aware.embeddings[representation], sample_indices
        )
        joint = np.concatenate([old_aligned[sample_indices], new_aligned[sample_indices]])
        layout = UMAP(
            n_neighbors=30,
            min_dist=0.25,
            n_components=2,
            metric="cosine",
            random_state=seed,
            n_jobs=threads,
            low_memory=True,
        ).fit_transform(joint)
        old_layout, new_layout = np.split(layout, 2)
        umap_audit[representation] = {"procrustes_similarity": similarity}
        limits = (
            (float(layout[:, 0].min()), float(layout[:, 0].max())),
            (float(layout[:, 1].min()), float(layout[:, 1].max())),
        )
        for col, (method, coordinates) in enumerate(((METHOD_2D, old_layout), (METHOD_3D, new_layout))):
            ax = axes[row, col]
            plot_scanpy_umap_panel(
                ax,
                coordinates,
                labels,
                colors,
                f"{representation}: {method}",
                limits,
            )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=3,
            color=colors[label],
            label=label,
        )
        for label in colors
    ]
    fig.legend(
        handles,
        list(colors),
        loc="outside right center",
        frameon=False,
        fontsize=7,
        markerscale=4,
    )
    fig.suptitle(
        "Matched Mouse E9.5 latent spaces\n"
        "Shared UMAP per representation after rotation/scale alignment; colors are transf_anno_E95",
        fontsize=14,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"embedding_comparison_shared_umap.{suffix}",
            dpi=RASTER_DPI,
            bbox_inches="tight",
        )
    plt.close(fig)
    return umap_audit


def read_metadata(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return yaml.safe_load(stream)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    audit: dict[str, Any],
    spatial_audit: dict[str, Any],
    baseline_metadata: dict[str, Any],
    section_aware_metadata: dict[str, Any],
    latent_metric_names: LatentMetricNames,
) -> None:
    config_2d = baseline_metadata["config"]
    config_3d = section_aware_metadata["config"]
    model_2d = baseline_metadata["model_info"]
    model_3d = section_aware_metadata["model_info"]
    lines = [
        "# Mouse E9.5 section-wise 2D versus section-aware 3D embeddings",
        "",
        "## Scope and comparability",
        "",
        f"All **{audit['n_matched_spots']:,} spots** were matched by globally unique observation ID. "
        f"Section labels, `transf_anno_E95`, and 3D coordinates agree for "
        f"{audit['section_match_fraction']:.1%}, {audit['label_match_fraction']:.1%}, and "
        f"{audit['coordinate_match_fraction_at_1e-5']:.1%} of matched spots, respectively.",
        "",
        "This is a descriptive comparison, **not an isolated graph-only ablation**. The uploaded baseline and "
        "section-aware run differ in additional settings:",
        "",
        "| Setting | Uploaded section-wise 2D | Section-aware 3D |",
        "|---|---:|---:|",
        f"| Graph mode | per-section 2D kNN | {config_3d.get('graph_mode')} |",
        f"| Training annotation | {config_2d.get('annotation')} | {config_3d.get('annotation')} |",
        f"| Model class size | {model_2d.get('class_size')} | {model_3d.get('class_size')} |",
        f"| Genes used/common | {baseline_metadata['training_info'].get('n_common_genes')} | {section_aware_metadata['training_info'].get('n_common_genes')} |",
        f"| Two-stage training | {config_2d.get('two_stage')} | {config_3d.get('two_stage')} |",
        "",
        "## Metrics",
        "",
        f"| Representation | Linear CKA | Procrustes similarity | {latent_metric_names.recall} | 2D label purity | 3D label purity | 2D different-section kNN | 3D different-section kNN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for representation in REPRESENTATIONS:
        lines.append(
            f"| `{representation}` | "
            f"{metric_value(metrics, representation, 'Linear CKA', COMPARISON):.4f} | "
            f"{metric_value(metrics, representation, 'Procrustes similarity', COMPARISON):.4f} | "
            f"{metric_value(metrics, representation, latent_metric_names.recall, COMPARISON):.4f} | "
            f"{metric_value(metrics, representation, latent_metric_names.label_purity, METHOD_2D):.4f} | "
            f"{metric_value(metrics, representation, latent_metric_names.label_purity, METHOD_3D):.4f} | "
            f"{metric_value(metrics, representation, latent_metric_names.different_section, METHOD_2D):.4f} | "
            f"{metric_value(metrics, representation, latent_metric_names.different_section, METHOD_3D):.4f} |"
        )
    niche_purity_2d = metric_value(
        metrics, "emb_niche", latent_metric_names.label_purity, METHOD_2D
    )
    niche_purity_3d = metric_value(
        metrics, "emb_niche", latent_metric_names.label_purity, METHOD_3D
    )
    niche_purity_absolute_pp = 100 * (niche_purity_3d - niche_purity_2d)
    niche_purity_relative_percent = 100 * (niche_purity_3d / niche_purity_2d - 1)
    lines.extend(
        [
            "",
            f"`{latent_metric_names.label_purity}` is the mean, across focal spots, of the "
            f"fraction of the {latent_metric_names.k} nearest latent neighbors carrying the same "
            "`transf_anno_E95` label. It is a neighborhood-structure statistic, not prediction accuracy.",
            "",
            f"For `emb_niche`, label purity changed from {niche_purity_2d:.2%} to "
            f"{niche_purity_3d:.2%}: an absolute increase of {niche_purity_absolute_pp:.2f} "
            f"percentage points and a relative increase of {niche_purity_relative_percent:.1f}%. "
            f"This does not mean that purity is {niche_purity_relative_percent:.1f}%, and the descriptive comparison does not "
            "isolate a causal effect of the 3D graph because other training settings also differ.",
            "",
            f"`{latent_metric_names.recall}` is the mean fraction of latent neighbors shared by the two runs on the same stratified sample. "
            "Linear CKA and Procrustes similarity are invariant to arbitrary orthogonal rotations of the latent axes.",
            "",
            "## Physical-neighborhood continuity",
            "",
            f"The evaluation uses exactly {spatial_audit['n_within_section_directed_edges']:,} directed within-section edges "
            f"and {spatial_audit['n_cross_section_directed_edges']:,} directed cross-section edges. Cross-section candidates "
            f"are accepted only within the radius inferred from each focal spot's "
            f"{spatial_audit['within_section_neighbors']}th external within-section neighbor, then capped at the "
            f"{spatial_audit['cross_section_candidate_cap_per_adjacent_section']} nearest qualifying spots per adjacent section. "
            "The radius matches the cross-section sampling scale to the within-section neighborhood; the cap only removes "
            "excess qualifying spots and never creates an edge. No edge is forced across a gap. "
            f"The zero-cross-section-neighbor rate is {spatial_audit['zero_cross_section_neighbor_fraction']:.2%}.",
            "",
            "| Representation | Edge type | 2D mean cosine | 3D mean cosine |",
            "|---|---|---:|---:|",
        ]
    )
    for representation in REPRESENTATIONS:
        for metric, label in [
            (
                "Within-section physical edges",
                f"Within-section {spatial_audit['within_section_neighbors']}-NN",
            ),
            (
                "Cross-section radius edges",
                "Adjacent-section, within radius and per-section cap",
            ),
        ]:
            lines.append(
                f"| `{representation}` | {label} | "
                f"{metric_value(metrics, representation, metric, METHOD_2D):.4f} | "
                f"{metric_value(metrics, representation, metric, METHOD_3D):.4f} |"
            )
    lines.extend(
        [
            "",
            "Higher spatial-edge cosine means a smoother latent representation along those physical edges. It is not, by "
            "itself, proof of better biological accuracy because excessive smoothing can also raise this score.",
            "",
            "## Outputs",
            "",
            "- `embedding_comparison_shared_umap.png` / `.pdf`: matched spots in a shared UMAP for each representation after orthogonal Procrustes alignment, plotted with `scanpy.pl.umap`.",
            "- `embedding_comparison_metrics.png` / `.pdf`: compact metric summary.",
            "- `embedding_comparison_metrics.tsv`: tidy metric values.",
            "- `embedding_comparison_summary.json`: matching, sampling, and spatial-evaluation audit data.",
            "",
        ]
    )
    (output_dir / "embedding_comparison_report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = read_sectionwise_baseline(args.baseline_dir)
    section_aware = read_single_h5ad(args.section_aware_h5ad, "section_id")
    baseline, match_audit = reorder_baseline_to_section_aware(baseline, section_aware)
    sample_indices = stratified_sample(section_aware.labels, args.sample_size, args.seed)
    umap_indices = stratified_sample(section_aware.labels, args.umap_sample_size, args.seed + 1)
    latent_metric_names = LatentMetricNames.for_k(args.latent_neighbors)

    within_source, within_target, cross_source, cross_target, spatial_audit = physical_evaluation_edges(
        section_aware.coordinates,
        section_aware.sections,
        sample_indices,
        args.spatial_neighbors,
        args.cross_section_cap,
        args.threads,
    )

    metric_rows: list[dict[str, Any]] = []
    representation_audit: dict[str, Any] = {}
    for representation in REPRESENTATIONS:
        old = baseline.embeddings[representation]
        new = section_aware.embeddings[representation]
        old_sample = old[sample_indices]
        new_sample = new[sample_indices]
        old_neighbors = approximate_neighbors(
            old_sample, args.latent_neighbors, args.seed, args.threads
        )
        new_neighbors = approximate_neighbors(
            new_sample, args.latent_neighbors, args.seed, args.threads
        )
        local = neighbor_metrics(
            old_neighbors,
            new_neighbors,
            section_aware.labels[sample_indices],
            section_aware.sections[sample_indices],
        )
        *_, procrustes_similarity = procrustes_fit(old_sample, new_sample)
        cka = linear_cka(old_sample, new_sample)
        representation_audit[representation] = {"linear_cka": cka, "procrustes_similarity": procrustes_similarity, **local}

        add_metric(metric_rows, representation, "Linear CKA", COMPARISON, cka, len(sample_indices))
        add_metric(
            metric_rows,
            representation,
            "Procrustes similarity",
            COMPARISON,
            procrustes_similarity,
            len(sample_indices),
        )
        add_metric(
            metric_rows,
            representation,
            latent_metric_names.recall,
            COMPARISON,
            local["knn_recall"],
            len(sample_indices),
        )
        add_metric(
            metric_rows,
            representation,
            latent_metric_names.jaccard,
            COMPARISON,
            local["knn_jaccard"],
            len(sample_indices),
        )
        for method, prefix in ((METHOD_2D, "old"), (METHOD_3D, "new")):
            add_metric(
                metric_rows,
                representation,
                latent_metric_names.label_purity,
                method,
                local[f"{prefix}_label_purity"],
                len(sample_indices),
            )
            add_metric(
                metric_rows,
                representation,
                latent_metric_names.different_section,
                method,
                local[f"{prefix}_different_section_fraction"],
                len(sample_indices),
            )
            add_metric(
                metric_rows,
                representation,
                latent_metric_names.adjacent_section,
                method,
                local[f"{prefix}_adjacent_section_fraction"],
                len(sample_indices),
            )
            embedding = old if method == METHOD_2D else new
            within_cosine = cosine_similarity_on_edges(embedding, within_source, within_target)
            cross_cosine = cosine_similarity_on_edges(embedding, cross_source, cross_target)
            add_metric(
                metric_rows,
                representation,
                "Within-section physical edges",
                method,
                float(np.mean(within_cosine)),
                len(within_cosine),
            )
            add_metric(
                metric_rows,
                representation,
                "Cross-section radius edges",
                method,
                float(np.mean(cross_cosine)),
                len(cross_cosine),
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "embedding_comparison_metrics.tsv", sep="\t", index=False)
    plot_metric_summary(metrics, args.output_dir, latent_metric_names)
    umap_audit = plot_shared_umap(
        baseline,
        section_aware,
        umap_indices,
        args.output_dir,
        args.seed,
        args.threads,
    )

    baseline_metadata = read_metadata(args.baseline_dir / "find_latent_metadata.yaml")
    section_aware_metadata = read_metadata(args.section_aware_h5ad.parent / "find_latent_metadata.yaml")
    summary = {
        "inputs": {
            "baseline_dir": str(args.baseline_dir),
            "section_aware_h5ad": str(args.section_aware_h5ad),
        },
        "parameters": vars(args)
        | {
            "baseline_dir": str(args.baseline_dir),
            "section_aware_h5ad": str(args.section_aware_h5ad),
            "output_dir": str(args.output_dir),
            "raster_dpi": RASTER_DPI,
        },
        "match_audit": match_audit,
        "spatial_evaluation_audit": spatial_audit,
        "representation_metrics": representation_audit,
        "umap_audit": umap_audit,
        "comparability_warning": (
            "Descriptive comparison only: graph construction, annotation used for training, gene space, "
            "and two-stage training differ between runs."
        ),
    }
    (args.output_dir / "embedding_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        args.output_dir,
        metrics,
        match_audit,
        spatial_audit,
        baseline_metadata,
        section_aware_metadata,
        latent_metric_names,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

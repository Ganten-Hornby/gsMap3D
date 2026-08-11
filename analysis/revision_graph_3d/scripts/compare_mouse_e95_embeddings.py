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
DOMAIN_KEY = "heart_region"
DOMAIN_MISSING_LABEL = "NA"


@dataclass(frozen=True)
class EmbeddingData:
    ids: np.ndarray
    sections: np.ndarray
    labels: np.ndarray
    domains: np.ndarray
    cell_types: np.ndarray
    coordinates: np.ndarray
    embeddings: dict[str, np.ndarray]


@dataclass(frozen=True)
class LatentMetricNames:
    k: int
    overlap: str
    jaccard: str
    label_purity: str
    different_section: str
    adjacent_section: str

    @classmethod
    def for_k(cls, k: int) -> LatentMetricNames:
        return cls(
            k=k,
            overlap=f"kNN overlap@{k}",
            jaccard=f"kNN Jaccard@{k}",
            label_purity=f"Heart-region kNN purity@{k}",
            different_section=f"Different-section latent kNN fraction@{k}",
            adjacent_section=f"Adjacent-section latent kNN fraction@{k}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--section-aware-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=30_000)
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
        domains = read_obs_column(handle, DOMAIN_KEY)
        cell_types = read_obs_column(handle, "mapped_celltype")
        coordinates = np.asarray(handle["obsm"]["3d_align_spatial"][:], dtype=np.float32)
        embeddings = {
            key: np.asarray(handle["obsm"][key][:], dtype=np.float32)
            for key in REPRESENTATIONS
        }
    return EmbeddingData(ids, sections, labels, domains, cell_types, coordinates, embeddings)


def read_sectionwise_baseline(directory: Path) -> EmbeddingData:
    paths = sorted(directory.glob("Mouse_E9.5_embryo_slice_*_add_latent.h5ad"))
    if len(paths) != 70:
        raise ValueError(f"Expected 70 baseline section files, found {len(paths)} in {directory}")
    pieces = [read_single_h5ad(path, "slice_id") for path in paths]
    return EmbeddingData(
        ids=np.concatenate([piece.ids for piece in pieces]),
        sections=np.concatenate([piece.sections for piece in pieces]),
        labels=np.concatenate([piece.labels for piece in pieces]),
        domains=np.concatenate([piece.domains for piece in pieces]),
        cell_types=np.concatenate([piece.cell_types for piece in pieces]),
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
        domains=baseline.domains[order],
        cell_types=baseline.cell_types[order],
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
        "heart_region_match_fraction": float(
            np.mean(reordered.domains == section_aware.domains)
        ),
        "coordinate_match_fraction_at_1e-5": float(np.mean(coordinate_delta <= 1e-5)),
        "max_coordinate_absolute_difference": float(np.max(coordinate_delta)),
    }
    if (
        audit["section_match_fraction"] != 1.0
        or audit["heart_region_match_fraction"] != 1.0
        or audit["coordinate_match_fraction_at_1e-5"] != 1.0
    ):
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


def domain_cell_type_composition(data: EmbeddingData) -> pd.DataFrame:
    """Count mapped cell types in each defined heart-region annotation."""
    defined = data.domains != DOMAIN_MISSING_LABEL
    table = (
        pd.DataFrame(
            {
                DOMAIN_KEY: data.domains[defined],
                "mapped_celltype": data.cell_types[defined],
            }
        )
        .value_counts(sort=False)
        .rename("n")
        .reset_index()
    )
    table["fraction_within_region"] = table["n"] / table.groupby(DOMAIN_KEY)["n"].transform(
        "sum"
    )
    return table.sort_values([DOMAIN_KEY, "n"], ascending=[True, False]).reset_index(drop=True)


def domain_annotation_audit(data: EmbeddingData) -> dict[str, Any]:
    defined = data.domains != DOMAIN_MISSING_LABEL
    cardiomyocyte = np.fromiter(
        ("cardiomyocytes" in cell_type.lower() for cell_type in data.cell_types),
        dtype=bool,
        count=len(data.cell_types),
    )
    categories, counts = np.unique(data.domains[defined], return_counts=True)
    n_defined = int(np.sum(defined))
    n_cardiomyocyte = int(np.sum(defined & cardiomyocyte))
    return {
        "annotation_key": DOMAIN_KEY,
        "excluded_value": DOMAIN_MISSING_LABEL,
        "n_total": int(len(data.ids)),
        "n_defined": n_defined,
        "defined_fraction": float(n_defined / len(data.ids)),
        "category_counts": {
            str(category): int(count) for category, count in zip(categories, counts, strict=True)
        },
        "n_mapped_cardiomyocytes_among_defined": n_cardiomyocyte,
        "mapped_cardiomyocyte_fraction_among_defined": float(n_cardiomyocyte / n_defined),
    }


def category_counts(labels: np.ndarray) -> dict[str, int]:
    categories, counts = np.unique(labels, return_counts=True)
    return {
        str(category): int(count) for category, count in zip(categories, counts, strict=True)
    }


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
    full_index = len(fit_indices) == len(x) and np.array_equal(
        fit_indices, np.arange(len(x), dtype=fit_indices.dtype)
    )
    fit_x = x if full_index else x[fit_indices]
    fit_y = y if full_index else y[fit_indices]
    x_center, y_center, x_scale, y_scale, rotation, similarity = procrustes_fit(fit_x, fit_y)
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
        "knn_overlap": float(np.mean(intersections / k)),
        "knn_jaccard": float(np.mean(intersections / (2 * k - intersections))),
        "old_label_purity": old_purity,
        "new_label_purity": new_purity,
        "old_different_section_fraction": old_cross,
        "new_different_section_fraction": new_cross,
        "old_adjacent_section_fraction": old_adjacent,
        "new_adjacent_section_fraction": new_adjacent,
    }


def label_purity(neighbors: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean fraction of external neighbors with the focal label."""
    return float(np.mean(labels[neighbors] == labels[:, None]))


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

    agreement_names = ["Linear CKA", "Procrustes similarity", latent_metric_names.overlap]
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
        ["Linear\nCKA", "Procrustes\nsimilarity", latent_metric_names.overlap.replace(" ", "\n")],
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
        "B  Heart-region structure (defined regions only)",
        "Same-region neighbor fraction",
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
    annotation_key: str,
    point_size: float,
    title: str,
    limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """Plot precomputed shared-UMAP coordinates through Scanpy's UMAP API."""
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


def plot_annotation_umap(
    baseline: EmbeddingData,
    section_aware: EmbeddingData,
    observation_indices: np.ndarray,
    labels: np.ndarray,
    annotation_key: str,
    output_stem: str,
    title: str,
    output_dir: Path,
    seed: int,
    threads: int,
) -> dict[str, Any]:
    colors = categorical_colors(labels)
    point_size = min(1.7, max(0.05, 120_000 / len(labels)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    umap_audit: dict[str, Any] = {
        "n_spots": int(len(observation_indices)),
        "annotation_key": annotation_key,
        "subsampling": False,
        "category_counts": category_counts(labels),
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
            "point_size": point_size,
        },
    }
    full_index = len(observation_indices) == len(section_aware.ids) and np.array_equal(
        observation_indices, np.arange(len(section_aware.ids), dtype=observation_indices.dtype)
    )
    for row, representation in enumerate(REPRESENTATIONS):
        old_aligned, new_aligned, similarity = aligned_embeddings(
            baseline.embeddings[representation],
            section_aware.embeddings[representation],
            observation_indices,
        )
        old_selected = old_aligned if full_index else old_aligned[observation_indices]
        new_selected = new_aligned if full_index else new_aligned[observation_indices]
        joint = np.concatenate([old_selected, new_selected])
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
                annotation_key,
                point_size,
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
        title,
        fontsize=14,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"{output_stem}.{suffix}",
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
    domain_audit: dict[str, Any],
    domain_composition: pd.DataFrame,
    sampling_audit: dict[str, Any],
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
        f"Section labels, `heart_region`, `transf_anno_E95`, and 3D coordinates agree for "
        f"{audit['section_match_fraction']:.1%}, {audit['heart_region_match_fraction']:.1%}, "
        f"{audit['label_match_fraction']:.1%}, and "
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
        "## Heart-region annotation scope",
        "",
        f"`heart_region` is defined for **{domain_audit['n_defined']:,} / "
        f"{domain_audit['n_total']:,} spots ({domain_audit['defined_fraction']:.2%})**. "
        f"The two defined regions contain {domain_audit['category_counts']['LV/atrial region']:,} "
        "LV/atrial observations and "
        f"{domain_audit['category_counts']['Right/outflow tract region']:,} right/outflow-tract "
        "observations. All heart-region purity and UMAP results below exclude `NA`.",
        "",
        f"All {sampling_audit['heart_region_observation_count']:,} observations with a defined "
        "`heart_region` are used for both the purity calculation and UMAP; there is no subsampling. "
        f"This preserves the observed region counts of "
        f"{sampling_audit['heart_region_category_counts']['LV/atrial region']:,} LV/atrial and "
        f"{sampling_audit['heart_region_category_counts']['Right/outflow tract region']:,} "
        "right/outflow-tract observations.",
        "",
        f"The cell-type UMAP likewise uses all {sampling_audit['celltype_umap_observation_count']:,} "
        "matched observations colored by `transf_anno_E95`, with no subsampling.",
        "",
        f"This annotation is not restricted to cardiomyocytes: only "
        f"{domain_audit['n_mapped_cardiomyocytes_among_defined']:,} / "
        f"{domain_audit['n_defined']:,} defined observations "
        f"({domain_audit['mapped_cardiomyocyte_fraction_among_defined']:.1%}) have a "
        "`mapped_celltype` containing `cardiomyocytes`. The remainder includes "
        "cardiopharyngeal mesoderm, primitive erythroid cells, endocardial cells, "
        "proepicardium, and arterial endothelial cells.",
        "",
        "| heart_region | mapped_celltype | n | Fraction within region |",
        "|---|---|---:|---:|",
        *[
            f"| {row.heart_region} | {row.mapped_celltype} | {row.n:,} | "
            f"{row.fraction_within_region:.1%} |"
            for row in domain_composition.itertuples(index=False)
        ],
        "",
        "## Metrics",
        "",
        f"| Representation | Linear CKA | Procrustes similarity | {latent_metric_names.overlap} | 2D heart-region purity | 3D heart-region purity | 2D different-section kNN | 3D different-section kNN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for representation in REPRESENTATIONS:
        lines.append(
            f"| `{representation}` | "
            f"{metric_value(metrics, representation, 'Linear CKA', COMPARISON):.4f} | "
            f"{metric_value(metrics, representation, 'Procrustes similarity', COMPARISON):.4f} | "
            f"{metric_value(metrics, representation, latent_metric_names.overlap, COMPARISON):.4f} | "
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
            "non-`NA` `heart_region` label. Both focal spots and the latent-neighbor candidate pool "
            "are restricted to observations with a defined heart region. It is a neighborhood-structure "
            "statistic, not prediction accuracy.",
            "",
            f"For `emb_niche`, heart-region purity changed from {niche_purity_2d:.2%} to "
            f"{niche_purity_3d:.2%}: an absolute increase of {niche_purity_absolute_pp:.2f} "
            f"percentage points and a relative increase of {niche_purity_relative_percent:.1f}%. "
            f"This does not mean that purity is {niche_purity_relative_percent:.1f}%, and the descriptive comparison does not "
            "isolate a causal effect of the 3D graph because other training settings also differ.",
            "",
            f"`{latent_metric_names.overlap}` is the mean fraction of latent neighbors shared by the two runs on the same stratified sample. "
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
            "- `embedding_comparison_shared_umap.png` / `.pdf`: all matched observations in a shared UMAP colored by `transf_anno_E95`, with no subsampling.",
            "- `embedding_comparison_heart_region_umap.png` / `.pdf`: all non-`NA` heart-region spots in a shared UMAP for each representation after orthogonal Procrustes alignment, colored by `heart_region` and plotted with `scanpy.pl.umap`.",
            "- `embedding_comparison_metrics.png` / `.pdf`: compact metric summary.",
            "- `embedding_comparison_metrics.tsv`: tidy metric values.",
            "- `heart_region_celltype_composition.tsv`: mapped cell-type composition within each defined heart region.",
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
    all_indices = np.arange(len(section_aware.ids), dtype=np.int64)
    domain_indices = np.flatnonzero(section_aware.domains != DOMAIN_MISSING_LABEL)
    if not len(domain_indices):
        raise ValueError("No observations have a defined heart_region")
    latent_metric_names = LatentMetricNames.for_k(args.latent_neighbors)
    domain_audit = domain_annotation_audit(section_aware)
    sampling_audit = {
        "global_metric_sample_size": int(len(sample_indices)),
        "celltype_umap_observation_count": int(len(all_indices)),
        "celltype_umap_subsampling": False,
        "heart_region_observation_count": int(len(domain_indices)),
        "heart_region_category_counts": category_counts(
            section_aware.domains[domain_indices]
        ),
        "heart_region_subsampling": False,
    }
    domain_composition = domain_cell_type_composition(section_aware)
    domain_composition.to_csv(
        args.output_dir / "heart_region_celltype_composition.tsv", sep="\t", index=False
    )

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
        old_domain_neighbors = approximate_neighbors(
            old[domain_indices], args.latent_neighbors, args.seed + 1, args.threads
        )
        new_domain_neighbors = approximate_neighbors(
            new[domain_indices], args.latent_neighbors, args.seed + 1, args.threads
        )
        domain_labels = section_aware.domains[domain_indices]
        old_domain_purity = label_purity(old_domain_neighbors, domain_labels)
        new_domain_purity = label_purity(new_domain_neighbors, domain_labels)
        *_, procrustes_similarity = procrustes_fit(old_sample, new_sample)
        cka = linear_cka(old_sample, new_sample)
        representation_audit[representation] = {
            "linear_cka": cka,
            "procrustes_similarity": procrustes_similarity,
            **{
                key: value
                for key, value in local.items()
                if key not in {"old_label_purity", "new_label_purity"}
            },
            "old_heart_region_purity": old_domain_purity,
            "new_heart_region_purity": new_domain_purity,
        }

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
            latent_metric_names.overlap,
            COMPARISON,
            local["knn_overlap"],
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
                old_domain_purity if prefix == "old" else new_domain_purity,
                len(domain_indices),
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
    celltype_umap_audit = plot_annotation_umap(
        baseline,
        section_aware,
        all_indices,
        section_aware.labels,
        "transf_anno_E95",
        "embedding_comparison_shared_umap",
        "Mouse E9.5 latent spaces (all matched observations)\n"
        "Shared UMAP per representation after rotation/scale alignment; colors are transf_anno_E95",
        args.output_dir,
        args.seed,
        args.threads,
    )
    heart_region_umap_audit = plot_annotation_umap(
        baseline,
        section_aware,
        domain_indices,
        section_aware.domains[domain_indices],
        DOMAIN_KEY,
        "embedding_comparison_heart_region_umap",
        "Mouse E9.5 heart-region latent spaces (all defined observations)\n"
        "Shared UMAP per representation after rotation/scale alignment; colors are heart_region",
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
        "heart_region_annotation_audit": domain_audit,
        "sampling_audit": sampling_audit,
        "spatial_evaluation_audit": spatial_audit,
        "representation_metrics": representation_audit,
        "celltype_umap_audit": celltype_umap_audit,
        "heart_region_umap_audit": heart_region_umap_audit,
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
        domain_audit,
        domain_composition,
        sampling_audit,
        spatial_audit,
        baseline_metadata,
        section_aware_metadata,
        latent_metric_names,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

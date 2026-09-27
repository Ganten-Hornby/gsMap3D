from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops, degree


@dataclass(frozen=True)
class SpatialGraph:
    """Directed edge representation of a spatial graph.

    Symmetric graphs contain both directed representations of every external
    relation. Self-loops are deliberately absent and are added exactly once by
    :class:`GCN` during normalization.
    """

    edges: np.ndarray
    weights: np.ndarray | None = None


def _validate_coordinates(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError(f"coords must be a 2D array, got shape {coords.shape}")
    if coords.shape[1] == 0:
        raise ValueError("coords must contain at least one spatial dimension")
    if not np.isfinite(coords).all():
        raise ValueError("coords must contain only finite values")
    return np.ascontiguousarray(coords)


def _canonicalize_nodes(
    coords: np.ndarray, stable_ids: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return coordinates in a row-order-independent canonical order.

    Stable observation IDs are the preferred tie-breaker. If IDs are omitted,
    unique coordinates are ordered lexicographically; exact duplicate
    coordinates then retain their input order and cannot be guaranteed to be
    permutation invariant.
    """

    n_nodes = coords.shape[0]
    if stable_ids is not None:
        stable_ids = np.asarray(stable_ids).astype(str)
        if stable_ids.shape != (n_nodes,):
            raise ValueError(f"stable_ids must have shape ({n_nodes},), got {stable_ids.shape}")
        if np.unique(stable_ids).size != n_nodes:
            raise ValueError("stable_ids must be unique")
        canonical_order = np.argsort(stable_ids, kind="stable")
    else:
        # np.lexsort uses the last key as primary. Reverse the coordinate
        # columns so x, then y, then z, ... define the canonical order.
        canonical_order = np.lexsort(
            tuple(coords[:, dim] for dim in range(coords.shape[1] - 1, -1, -1))
        )

    return np.ascontiguousarray(coords[canonical_order]), canonical_order


def _external_knn_indices(
    canonical_coords: np.ndarray,
    n_external_neighbors: int,
    workers: int,
) -> np.ndarray:
    """Select exact external KNNs using distance then canonical ID for ties."""

    n_nodes = canonical_coords.shape[0]
    if n_external_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    if n_nodes <= 1:
        return np.empty((n_nodes, 0), dtype=np.int64)

    effective_k = min(n_external_neighbors, n_nodes - 1)

    # A larger candidate pool permits exact stable-ID ordering of the usual
    # grid-distance ties without a query-ball call per node. If the K boundary
    # is still at the edge of this pool (e.g. many duplicate coordinates), the
    # affected rows fall back to an exhaustive radius query.
    candidate_k = min(n_nodes, max(effective_k + 2, 4 * effective_k + 1))
    tree = cKDTree(canonical_coords, balanced_tree=True, compact_nodes=True)
    distances, indices = tree.query(canonical_coords, k=candidate_k, workers=workers)
    if candidate_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    focal = np.arange(n_nodes, dtype=np.int64)[:, None]
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    external_distances = distances.copy()
    external_distances[indices == focal] = np.inf

    # canonical node index is equivalent to stable-ID rank and is therefore
    # the secondary key after distance.
    candidate_order = np.lexsort((indices, external_distances), axis=1)
    sorted_indices = np.take_along_axis(indices, candidate_order, axis=1)
    sorted_distances = np.take_along_axis(external_distances, candidate_order, axis=1)
    selected = sorted_indices[:, :effective_k].copy()

    if candidate_k < n_nodes:
        kth_distance = sorted_distances[:, effective_k - 1]
        max_returned_external = np.max(np.where(indices == focal, -np.inf, distances), axis=1)
        unresolved = np.isclose(
            kth_distance,
            max_returned_external,
            rtol=1e-12,
            atol=1e-12,
        )

        for row in np.flatnonzero(unresolved):
            radius = np.nextafter(kth_distance[row], np.inf)
            candidates = np.asarray(
                tree.query_ball_point(canonical_coords[row], radius), dtype=np.int64
            )
            candidates = candidates[candidates != row]
            candidate_distances = np.linalg.norm(
                canonical_coords[candidates] - canonical_coords[row], axis=1
            )
            order = np.lexsort((candidates, candidate_distances))
            selected[row] = candidates[order[:effective_k]]

    return selected


def build_spatial_graph(
    coords: np.ndarray,
    n_neighbors: int,
    undirected: bool = True,
    stable_ids: np.ndarray | None = None,
    workers: int = -1,
) -> np.ndarray:
    """Build a deterministic KNN graph with K external neighbors.

    Parameters
    ----------
    coords
        Spatial coordinates with shape ``(n_nodes, n_dimensions)``.
    n_neighbors
        Number of *external* neighbors selected by every eligible focal node
        before symmetrization. The focal node is removed explicitly.
    undirected
        If true, materialize both ``i -> j`` and ``j -> i`` for every selected
        relation and coalesce duplicate directed edges.
    stable_ids
        Unique observation IDs used to make tie handling invariant to AnnData
        row order. Supplying these IDs is strongly recommended.
    workers
        Number of SciPy KD-tree query workers.

    Returns
    -------
    np.ndarray
        Directed edge pairs with shape ``(n_edges, 2)``. No self-loops are
        returned. Final symmetric degree may exceed K because the union of KNN
        selections is used.
    """

    coords = _validate_coordinates(coords)
    n_nodes = coords.shape[0]
    if n_nodes <= 1:
        return np.empty((0, 2), dtype=np.int64)

    canonical_coords, canonical_order = _canonicalize_nodes(coords, stable_ids)
    selected = _external_knn_indices(canonical_coords, n_neighbors, workers)
    effective_k = selected.shape[1]

    canonical_sources = np.repeat(np.arange(n_nodes, dtype=np.int64), effective_k)
    canonical_targets = selected.reshape(-1)
    sources = canonical_order[canonical_sources]
    targets = canonical_order[canonical_targets]
    edges = np.column_stack((sources, targets))

    if undirected:
        edges = np.concatenate((edges, edges[:, ::-1]), axis=0)

    return np.unique(edges, axis=0).astype(np.int64, copy=False)


def build_legacy_spatial_graph(
    coords: np.ndarray, query_neighbors: int = 10, undirected: bool = True
) -> np.ndarray:
    """Reproduce the historical graph implementation for impact analysis.

    ``query_neighbors`` includes the focal node and, when ``undirected=True``,
    unordered pairs are collapsed to only the lower-index-to-higher-index edge.
    This function intentionally retains the defect and must not be used as a
    production default.
    """

    coords = np.ascontiguousarray(coords, dtype=np.float32)
    n_nodes = coords.shape[0]
    if n_nodes == 0:
        return np.empty((0, 2), dtype=np.int64)
    query_neighbors = min(query_neighbors, n_nodes)
    tree = cKDTree(coords, balanced_tree=True, compact_nodes=True)
    _, indices = tree.query(coords, k=query_neighbors, workers=-1)
    if query_neighbors == 1:
        indices = indices[:, None]

    source = np.repeat(np.arange(n_nodes), query_neighbors)
    target = indices.reshape(-1)
    if undirected:
        all_edges = np.column_stack(
            [np.concatenate([source, target]), np.concatenate([target, source])]
        )
        edge_set = {tuple(sorted([i, j])) for i, j in all_edges}
        return np.asarray(list(edge_set), dtype=np.int64).reshape(-1, 2)
    return np.column_stack([source, target]).astype(np.int64)


def build_legacy_query_symmetric_graph(
    coords: np.ndarray, query_neighbors: int = 10
) -> np.ndarray:
    """Symmetrize the exact historical query result while excluding self.

    With the historical default ``query_neighbors=10``, this normally selects
    nine external neighbors. It is the G1 arm that isolates the direction bug
    from the subsequent change to ten explicit external neighbors.
    """

    directed = build_legacy_spatial_graph(
        coords, query_neighbors=query_neighbors, undirected=False
    )
    directed = directed[directed[:, 0] != directed[:, 1]]
    bidirectional = np.concatenate((directed, directed[:, ::-1]), axis=0)
    return np.unique(bidirectional, axis=0).astype(np.int64, copy=False)


def build_radius_spatial_graph(
    coords: np.ndarray,
    radius: float,
    stable_ids: np.ndarray | None = None,
    sigma: float | None = None,
) -> SpatialGraph:
    """Build an explicitly bidirectional fixed-radius graph.

    If ``sigma`` is supplied, edge weights are Gaussian physical-distance
    weights. A caller can use ``radius=3*sigma`` for the prespecified cutoff.
    """

    coords = _validate_coordinates(coords)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive")
    if sigma is not None and (not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("sigma must be positive")
    if coords.shape[0] <= 1:
        return SpatialGraph(np.empty((0, 2), dtype=np.int64), None)

    canonical_coords, canonical_order = _canonicalize_nodes(coords, stable_ids)
    tree = cKDTree(canonical_coords, balanced_tree=True, compact_nodes=True)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if pairs.size == 0:
        return SpatialGraph(np.empty((0, 2), dtype=np.int64), None)

    pair_distances = np.linalg.norm(
        canonical_coords[pairs[:, 0]] - canonical_coords[pairs[:, 1]], axis=1
    )
    original_pairs = canonical_order[pairs]
    edges = np.concatenate((original_pairs, original_pairs[:, ::-1]), axis=0)
    weights = None
    if sigma is not None:
        pair_weights = np.exp(-(pair_distances**2) / (2 * sigma**2))
        weights = np.concatenate((pair_weights, pair_weights)).astype(np.float32)
    return SpatialGraph(edges.astype(np.int64, copy=False), weights)


def _nearest_target_indices(
    query_coords: np.ndarray,
    target_coords: np.ndarray,
    n_neighbors: int,
    workers: int,
) -> np.ndarray:
    """Return cross-set KNNs with deterministic distance/ID tie-breaking."""

    effective_k = min(n_neighbors, target_coords.shape[0])
    if query_coords.shape[0] == 0 or effective_k == 0:
        return np.empty((query_coords.shape[0], 0), dtype=np.int64)

    candidate_k = min(target_coords.shape[0], max(effective_k + 2, 4 * effective_k))
    tree = cKDTree(target_coords, balanced_tree=True, compact_nodes=True)
    distances, indices = tree.query(query_coords, k=candidate_k, workers=workers)
    if candidate_k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    candidate_order = np.lexsort((indices, distances), axis=1)
    sorted_indices = np.take_along_axis(indices, candidate_order, axis=1)
    sorted_distances = np.take_along_axis(distances, candidate_order, axis=1)
    selected = sorted_indices[:, :effective_k].copy()

    if candidate_k < target_coords.shape[0]:
        kth_distance = sorted_distances[:, effective_k - 1]
        max_returned_distance = np.max(distances, axis=1)
        unresolved = np.isclose(
            kth_distance,
            max_returned_distance,
            rtol=1e-12,
            atol=1e-12,
        )
        for row in np.flatnonzero(unresolved):
            radius = np.nextafter(kth_distance[row], np.inf)
            candidates = np.asarray(tree.query_ball_point(query_coords[row], radius))
            candidate_distances = np.linalg.norm(
                target_coords[candidates] - query_coords[row], axis=1
            )
            order = np.lexsort((candidates, candidate_distances))
            selected[row] = candidates[order[:effective_k]]

    return selected


def order_sections_by_median_coordinate(
    coords: np.ndarray, section_labels: np.ndarray
) -> np.ndarray:
    """Return section labels ordered by median Z (the third coordinate)."""

    coords = _validate_coordinates(coords)
    if coords.shape[1] != 3:
        raise ValueError("Physical section ordering requires XYZ coordinates with an explicit z")
    section_labels = np.asarray(section_labels).astype(str)
    if section_labels.shape != (coords.shape[0],):
        raise ValueError(
            f"section_labels must have shape ({coords.shape[0]},), got {section_labels.shape}"
        )
    unique_sections = np.unique(section_labels)
    section_z = np.asarray(
        [np.median(coords[section_labels == section, 2]) for section in unique_sections]
    )
    return unique_sections[np.argsort(section_z, kind="stable")]


def sectionwise_2d_slice_labels(coords: np.ndarray) -> np.ndarray:
    """Resolve slice identity for a sectionwise 2D graph.

    Every input H5AD represents one slice, independently of whether its stored
    coordinates are XY or XYZ, so all rows receive the same local identity.
    """

    coords = _validate_coordinates(coords)
    if coords.shape[1] not in {2, 3}:
        raise ValueError("sectionwise 2D graphs require XY or XYZ coordinates")
    return np.zeros(coords.shape[0], dtype=np.int8)


def build_section_aware_3d_graph(
    coords: np.ndarray,
    section_labels: np.ndarray,
    n_neighbors: int,
    undirected: bool = True,
    stable_ids: np.ndarray | None = None,
    workers: int = -1,
    section_order: np.ndarray | None = None,
) -> SpatialGraph:
    """Build a physical 3D graph using a within-section KNN scale.

    Each spot first selects K external neighbors from its own section. The
    distance to its farthest selected within-section neighbor is its local
    physical radius. Cross-section edges connect the spot to every cell in the
    immediately preceding or following physical section within that radius.
    If neither neighboring section contains a cell within the radius, the spot
    falls back to at most ``max(1, K // 2)`` nearest cells from each available
    neighboring section, but only when they are within twice the focal spot's
    within-section radius. This explicit ceiling prevents forced edges across
    large physical section gaps.

    Coordinates must already use a common physical scale, including a
    calibrated z axis. ``section_order`` supplies the physical input-file
    order when slice identity comes from an ordered H5AD list. Without it,
    sections retain the legacy median-z ordering. Self-loops are absent; the
    GCN normalization adds them separately.
    """

    coords = _validate_coordinates(coords)
    if coords.shape[1] != 3:
        raise ValueError("section-aware 3D graphs require exactly three XYZ columns")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")

    section_labels = np.asarray(section_labels)
    if section_labels.shape != (coords.shape[0],):
        raise ValueError(
            f"section_labels must have shape ({coords.shape[0]},), got {section_labels.shape}"
        )
    section_labels = section_labels.astype(str)
    if coords.shape[0] <= 1:
        return SpatialGraph(np.empty((0, 2), dtype=np.int64), None)

    canonical_coords, canonical_order = _canonicalize_nodes(coords, stable_ids)
    canonical_sections = section_labels[canonical_order]
    if section_order is None:
        ordered_sections = order_sections_by_median_coordinate(
            canonical_coords, canonical_sections
        )
    else:
        ordered_sections = np.asarray(section_order).astype(str)
        observed_sections = np.unique(canonical_sections)
        if ordered_sections.size != observed_sections.size or set(ordered_sections) != set(
            observed_sections
        ):
            raise ValueError("section_order must contain every section label exactly once")

    within_edges: list[np.ndarray] = []
    within_radii = np.zeros(coords.shape[0], dtype=np.float64)
    section_indices: dict[str, np.ndarray] = {}
    for section in ordered_sections:
        global_indices = np.flatnonzero(canonical_sections == section)
        section_indices[section] = global_indices
        local_coords = canonical_coords[global_indices]
        selected = _external_knn_indices(local_coords, n_neighbors, workers)
        effective_k = selected.shape[1]
        if effective_k == 0:
            continue

        sources = np.repeat(global_indices, effective_k)
        targets = global_indices[selected.reshape(-1)]
        within_edges.append(np.column_stack((sources, targets)))
        selected_distances = np.linalg.norm(
            local_coords[:, None, :] - local_coords[selected], axis=2
        )
        within_radii[global_indices] = np.max(selected_distances, axis=1)

    cross_edges: list[np.ndarray] = []
    fallback_k = max(1, n_neighbors // 2)
    for position, section in enumerate(ordered_sections):
        focal_indices = section_indices[section]
        focal_coords = canonical_coords[focal_indices]
        neighboring_sections = []
        if position > 0:
            neighboring_sections.append(ordered_sections[position - 1])
        if position + 1 < ordered_sections.size:
            neighboring_sections.append(ordered_sections[position + 1])
        if not neighboring_sections:
            continue

        has_radius_neighbor = np.zeros(focal_indices.size, dtype=bool)
        for neighbor_section in neighboring_sections:
            target_indices = section_indices[neighbor_section]
            target_coords = canonical_coords[target_indices]
            tree = cKDTree(target_coords, balanced_tree=True, compact_nodes=True)
            matches = tree.query_ball_point(
                focal_coords,
                within_radii[focal_indices],
                workers=workers,
                return_sorted=True,
            )
            counts = np.fromiter((len(match) for match in matches), dtype=np.int64)
            has_radius_neighbor |= counts > 0
            if counts.sum() == 0:
                continue
            matched_targets = np.concatenate(
                [np.asarray(match, dtype=np.int64) for match in matches if len(match)]
            )
            cross_edges.append(
                np.column_stack(
                    (
                        np.repeat(focal_indices, counts),
                        target_indices[matched_targets],
                    )
                )
            )

        fallback_indices = focal_indices[~has_radius_neighbor]
        fallback_coords = canonical_coords[fallback_indices]
        for neighbor_section in neighboring_sections:
            target_indices = section_indices[neighbor_section]
            selected = _nearest_target_indices(
                fallback_coords,
                canonical_coords[target_indices],
                fallback_k,
                workers,
            )
            if selected.shape[1] == 0:
                continue
            selected_targets = target_indices[selected]
            selected_distances = np.linalg.norm(
                fallback_coords[:, None, :] - canonical_coords[selected_targets],
                axis=2,
            )
            within_ceiling = selected_distances <= (2.0 * within_radii[fallback_indices, None])
            if not np.any(within_ceiling):
                continue
            cross_edges.append(
                np.column_stack(
                    (
                        np.broadcast_to(fallback_indices[:, None], selected.shape)[within_ceiling],
                        selected_targets[within_ceiling],
                    )
                )
            )

    edge_blocks = within_edges + cross_edges
    if not edge_blocks:
        return SpatialGraph(np.empty((0, 2), dtype=np.int64), None)

    canonical_edges = np.concatenate(edge_blocks)
    edges = canonical_order[canonical_edges]
    if undirected:
        edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    edges = np.unique(edges, axis=0).astype(np.int64, copy=False)
    return SpatialGraph(edges, None)


def build_sectionwise_2d_graph(
    coords: np.ndarray,
    section_labels: np.ndarray,
    n_neighbors: int,
    undirected: bool = True,
    stable_ids: np.ndarray | None = None,
    workers: int = -1,
) -> SpatialGraph:
    """Build independent within-slice 2D KNN graphs.

    Each spot selects up to ``n_neighbors`` external neighbors from its own
    physical slice. Slice identity is resolved before this function from one
    H5AD file per slice. No cross-slice edge is
    created, and only XY participates in neighbor search. Self-loops remain a
    separate GCN operation.
    """

    coords = _validate_coordinates(coords)
    if coords.shape[1] not in {2, 3}:
        raise ValueError("sectionwise 2D graphs require XY or XYZ coordinates")
    coords = coords[:, :2]
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")

    section_labels = np.asarray(section_labels)
    if section_labels.shape != (coords.shape[0],):
        raise ValueError(
            f"section_labels must have shape ({coords.shape[0]},), got {section_labels.shape}"
        )
    if coords.shape[0] <= 1:
        return SpatialGraph(np.empty((0, 2), dtype=np.int64), None)

    canonical_coords, canonical_order = _canonicalize_nodes(coords, stable_ids)
    canonical_sections = section_labels[canonical_order].astype(str)
    edge_blocks: list[np.ndarray] = []
    for section in np.unique(canonical_sections):
        global_indices = np.flatnonzero(canonical_sections == section)
        selected = _external_knn_indices(canonical_coords[global_indices], n_neighbors, workers)
        effective_k = selected.shape[1]
        if effective_k == 0:
            continue
        edge_blocks.append(
            np.column_stack(
                (
                    np.repeat(global_indices, effective_k),
                    global_indices[selected.reshape(-1)],
                )
            )
        )

    if not edge_blocks:
        return SpatialGraph(np.empty((0, 2), dtype=np.int64), None)
    edges = canonical_order[np.concatenate(edge_blocks)]
    if undirected:
        edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    edges = np.unique(edges, axis=0).astype(np.int64, copy=False)
    return SpatialGraph(edges, None)


def build_graph_for_mode(
    coords: np.ndarray,
    stable_ids: np.ndarray,
    mode: str,
    n_neighbors: int,
    radius: float | None = None,
    sigma: float | None = None,
    section_labels: np.ndarray | None = None,
    section_order: np.ndarray | None = None,
) -> SpatialGraph:
    """Dispatch graph construction for production and revision experiment arms."""

    mode = mode.lower()
    if mode == "symmetric_knn":
        return SpatialGraph(build_spatial_graph(coords, n_neighbors, stable_ids=stable_ids), None)
    if mode == "legacy_bug":
        return SpatialGraph(build_legacy_spatial_graph(coords, n_neighbors), None)
    if mode == "legacy_symmetric":
        return SpatialGraph(build_legacy_query_symmetric_graph(coords, n_neighbors), None)
    if mode in {"radius", "weighted_radius"}:
        if radius is None:
            raise ValueError(f"graph_radius is required for graph_mode={mode!r}")
        if mode == "weighted_radius" and sigma is None:
            raise ValueError("graph_sigma is required for weighted_radius mode")
        return build_radius_spatial_graph(
            coords,
            radius=radius,
            stable_ids=stable_ids,
            sigma=sigma if mode == "weighted_radius" else None,
        )
    if mode == "section_aware_3d":
        if section_labels is None:
            raise ValueError("section_labels are required for section_aware_3d mode")
        return build_section_aware_3d_graph(
            coords,
            section_labels,
            n_neighbors,
            stable_ids=stable_ids,
            section_order=section_order,
        )
    if mode == "sectionwise_2d":
        if section_labels is None:
            raise ValueError("section_labels are required for sectionwise_2d mode")
        return build_sectionwise_2d_graph(
            coords,
            section_labels,
            n_neighbors,
            stable_ids=stable_ids,
        )
    if mode == "no_graph":
        return SpatialGraph(np.empty((0, 2), dtype=np.int64), None)
    raise ValueError(
        "graph_mode must be one of symmetric_knn, legacy_bug, "
        "legacy_symmetric, radius, weighted_radius, sectionwise_2d, "
        "section_aware_3d, or no_graph"
    )


class GCN(torch.nn.Module):
    """Parameter-free K-hop GCN feature propagation."""

    def __init__(self, K: int = 1):
        super().__init__()
        self.K = K

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        legacy_normalization: bool = False,
    ) -> torch.Tensor:
        if legacy_normalization:
            if edge_weight is not None:
                raise ValueError("Legacy normalization supports only unweighted edges")
            # Exact historical normalization, retained only for G0 impact
            # reproduction. It uses source-row degree on the defective directed
            # graph rather than PyG's target-degree convention.
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            normalized_weight = (deg[row] * deg[col]).pow(-0.5)
            normalized_weight[normalized_weight == float("inf")] = 0
        else:
            edge_index, normalized_weight = gcn_norm(
                edge_index,
                edge_weight,
                num_nodes=x.size(0),
                add_self_loops=True,
                flow="source_to_target",
                dtype=x.dtype,
            )

        adjacency = torch.sparse_coo_tensor(
            edge_index.flip(0),
            normalized_weight,
            size=(x.size(0), x.size(0)),
            device=x.device,
            dtype=x.dtype,
            check_invariants=True,
        ).coalesce()

        xs = [x]
        for _ in range(self.K):
            xs.append(torch.sparse.mm(adjacency, xs[-1]))

        return torch.cat(xs[1:], dim=1)

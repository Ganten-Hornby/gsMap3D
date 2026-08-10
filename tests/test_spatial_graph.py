from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from gsMap.find_latent.gnn.gcn import (
    GCN,
    build_legacy_spatial_graph,
    build_radius_spatial_graph,
    build_section_aware_3d_graph,
    build_spatial_graph,
)
from gsMap.find_latent.st_process import TrainingData


def _edge_id_set(edges, stable_ids):
    stable_ids = np.asarray(stable_ids)
    return {(stable_ids[source], stable_ids[target]) for source, target in edges}


def test_corrected_knn_selects_exact_external_k_before_symmetrization():
    rng = np.random.default_rng(42)
    coords = rng.normal(size=(30, 2))
    stable_ids = np.asarray([f"spot-{i:02d}" for i in range(coords.shape[0])])

    edges = build_spatial_graph(
        coords,
        n_neighbors=5,
        undirected=False,
        stable_ids=stable_ids,
        workers=1,
    )

    assert not np.any(edges[:, 0] == edges[:, 1])
    np.testing.assert_array_equal(np.bincount(edges[:, 0], minlength=30), 5)


def test_corrected_graph_is_explicitly_bidirectional():
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(40, 3))
    stable_ids = np.asarray([f"id-{i}" for i in range(40)])
    edges = build_spatial_graph(coords, 10, stable_ids=stable_ids, workers=1)
    edge_set = {tuple(edge) for edge in edges}

    assert not np.any(edges[:, 0] == edges[:, 1])
    assert all((target, source) in edge_set for source, target in edges)


def test_regular_lattice_is_invariant_to_row_permutation():
    coords = np.asarray([(x, y) for x in range(5) for y in range(5)], dtype=float)
    stable_ids = np.asarray([f"grid-{i:02d}" for i in range(coords.shape[0])])
    reference = build_spatial_graph(coords, 6, stable_ids=stable_ids, workers=1)

    rng = np.random.default_rng(2026)
    for _ in range(20):
        permutation = rng.permutation(coords.shape[0])
        permuted = build_spatial_graph(
            coords[permutation],
            6,
            stable_ids=stable_ids[permutation],
            workers=1,
        )
        assert _edge_id_set(reference, stable_ids) == _edge_id_set(
            permuted, stable_ids[permutation]
        )


def test_duplicate_coordinates_use_stable_ids_for_permutation_invariance():
    coords = np.asarray([[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    stable_ids = np.asarray(["d", "b", "a", "g", "c", "f", "e"])
    reference = build_spatial_graph(coords, 2, stable_ids=stable_ids, workers=1)

    permutation = np.asarray([6, 3, 1, 5, 0, 4, 2])
    permuted = build_spatial_graph(
        coords[permutation],
        2,
        stable_ids=stable_ids[permutation],
        workers=1,
    )
    assert _edge_id_set(reference, stable_ids) == _edge_id_set(permuted, stable_ids[permutation])


def test_knn_handles_one_node_and_k_larger_than_section():
    one = build_spatial_graph(np.asarray([[0.0, 0.0]]), 10, stable_ids=np.asarray(["a"]))
    assert one.shape == (0, 2)

    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    edges = build_spatial_graph(
        coords, 10, undirected=False, stable_ids=np.asarray(["a", "b", "c"]), workers=1
    )
    np.testing.assert_array_equal(np.bincount(edges[:, 0], minlength=3), 2)


def test_legacy_graph_retains_documented_direction_defect():
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    edges = build_legacy_spatial_graph(coords, query_neighbors=3)
    edge_set = {tuple(edge) for edge in edges if edge[0] != edge[1]}

    assert edge_set
    assert all(source < target for source, target in edge_set)
    assert any((target, source) not in edge_set for source, target in edge_set)


def test_weighted_radius_graph_has_symmetric_edges_and_weights():
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    stable_ids = np.asarray(["a", "b", "c"])
    graph = build_radius_spatial_graph(coords, radius=1.5, stable_ids=stable_ids, sigma=1.0)
    edge_weight = {
        tuple(edge): weight for edge, weight in zip(graph.edges, graph.weights, strict=True)
    }

    assert set(edge_weight) == {(0, 1), (1, 0)}
    assert np.isclose(edge_weight[(0, 1)], np.exp(-0.5))
    assert edge_weight[(0, 1)] == edge_weight[(1, 0)]


def test_section_aware_3d_graph_uses_each_spots_within_section_knn_radius():
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [8.0, 0.0, 0.5],
        ]
    )
    sections = np.asarray(["a", "a", "b", "b", "b"])
    stable_ids = np.asarray(["a0", "a1", "b0", "b1", "b8"])

    graph = build_section_aware_3d_graph(
        coords,
        sections,
        n_neighbors=1,
        stable_ids=stable_ids,
        undirected=False,
        workers=1,
    )
    edges = _edge_id_set(graph.edges, stable_ids)

    # a0's within-section radius is 2. Both nearby spots in section b are
    # inside that physical radius, whereas b8 is outside it.
    assert ("a0", "b0") in edges
    assert ("a0", "b1") in edges
    assert ("a0", "b8") not in edges


def test_section_aware_3d_graph_falls_back_to_half_k_per_neighboring_section():
    coords = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 10.0],
            [2.0, 0.0, 10.0],
            [-1.0, 0.0, 20.0],
            [1.0, 0.0, 20.0],
        ]
    )
    sections = np.asarray(["lower", "lower", "middle", "middle", "upper", "upper"])
    stable_ids = np.asarray(["l-1", "l1", "m0", "m2", "u-1", "u1"])

    graph = build_section_aware_3d_graph(
        coords,
        sections,
        n_neighbors=2,
        stable_ids=stable_ids,
        undirected=False,
        workers=1,
    )
    edges = _edge_id_set(graph.edges, stable_ids)

    # No cross-section spot is inside m0's within-section radius (2), so the
    # fallback contributes K/2=1 nearest spot from both adjacent sections.
    assert ("m0", "l-1") in edges
    assert ("m0", "u-1") in edges
    assert ("m0", "l1") not in edges
    assert ("m0", "u1") not in edges


def test_section_aware_3d_graph_is_bidirectional_and_permutation_invariant():
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [1.0, 0.0, 4.0],
            [2.0, 0.0, 4.0],
        ]
    )
    sections = np.asarray([0, 0, 0, 1, 1, 1])
    stable_ids = np.asarray(["a", "b", "c", "d", "e", "f"])
    reference = build_section_aware_3d_graph(coords, sections, 2, stable_ids=stable_ids, workers=1)
    reference_edges = _edge_id_set(reference.edges, stable_ids)

    assert not np.any(reference.edges[:, 0] == reference.edges[:, 1])
    assert all((target, source) in reference_edges for source, target in reference_edges)

    permutation = np.asarray([4, 0, 5, 2, 1, 3])
    permuted = build_section_aware_3d_graph(
        coords[permutation],
        sections[permutation],
        2,
        stable_ids=stable_ids[permutation],
        workers=1,
    )
    assert reference_edges == _edge_id_set(permuted.edges, stable_ids[permutation])


def test_gcn_propagation_is_row_permutation_equivariant():
    coords = np.asarray([(x, y) for x in range(4) for y in range(4)], dtype=float)
    stable_ids = np.asarray([f"s-{i:02d}" for i in range(coords.shape[0])])
    features = torch.arange(coords.shape[0] * 3, dtype=torch.float32).reshape(-1, 3)
    gcn = GCN(K=3)

    edges = build_spatial_graph(coords, 4, stable_ids=stable_ids, workers=1)
    reference = gcn(features, torch.from_numpy(edges.T).long()).detach().numpy()

    permutation = np.random.default_rng(1).permutation(coords.shape[0])
    permuted_edges = build_spatial_graph(
        coords[permutation],
        4,
        stable_ids=stable_ids[permutation],
        workers=1,
    )
    permuted_result = (
        gcn(features[permutation], torch.from_numpy(permuted_edges.T).long()).detach().numpy()
    )
    mapped_back = np.empty_like(permuted_result)
    mapped_back[permutation] = permuted_result
    np.testing.assert_allclose(reference, mapped_back, rtol=1e-6, atol=1e-6)


def test_gcn_normalization_on_two_node_graph():
    features = torch.asarray([[1.0], [3.0]])
    edges = torch.asarray([[0, 1], [1, 0]])
    result = GCN(K=1)(features, edges)
    torch.testing.assert_close(result, torch.asarray([[2.0], [2.0]]))


def test_training_data_scatter_is_invariant_to_interleaved_batches():
    expression = np.asarray([[1, 0], [4, 0], [2, 1], [5, 1], [3, 2], [6, 2]], dtype=np.float32)
    obs_names = np.asarray(["a0", "b0", "a1", "b1", "a2", "b2"])
    obs = pd.DataFrame({"batch_id": ["a", "b", "a", "b", "a", "b"]}, index=obs_names)
    adata = anndata.AnnData(
        X=sp.csr_matrix(expression),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    adata.obsm["spatial"] = np.asarray(
        [[0, 0], [0, 0], [1, 0], [1, 0], [2, 0], [2, 0]], dtype=float
    )
    params = SimpleNamespace(
        K=2,
        annotation=None,
        spatial_key="spatial",
        graph_mode="symmetric_knn",
        n_neighbors=1,
        graph_radius=None,
        graph_sigma=None,
    )

    original = TrainingData(params)
    original.prepare(adata, ["g0", "g1"])

    permutation = np.asarray([5, 2, 0, 3, 1, 4])
    shuffled = TrainingData(params)
    shuffled.prepare(adata[permutation].copy(), ["g0", "g1"])
    mapped_back = np.empty_like(shuffled.expression_gcn_merge.numpy())
    mapped_back[permutation] = shuffled.expression_gcn_merge.numpy()

    np.testing.assert_allclose(original.expression_gcn_merge.numpy(), mapped_back)

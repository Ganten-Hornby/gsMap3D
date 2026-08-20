from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from gsMap.find_latent.gnn.gcn import (
    GCN,
    build_legacy_spatial_graph,
    build_radius_spatial_graph,
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


@pytest.mark.parametrize("radius", [0.0, -1.0, np.nan, np.inf])
def test_radius_graph_rejects_nonpositive_or_nonfinite_radius(radius):
    with pytest.raises(ValueError, match="radius must be positive"):
        build_radius_spatial_graph(np.asarray([[0.0, 0.0], [1.0, 0.0]]), radius=radius)


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_weighted_radius_graph_rejects_nonpositive_or_nonfinite_sigma(sigma):
    with pytest.raises(ValueError, match="sigma must be positive"):
        build_radius_spatial_graph(
            np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            radius=2.0,
            sigma=sigma,
        )


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


def test_training_batch_codes_follow_input_order_after_s10():
    batch_order = [
        "S10",
        "S2",
        "S0",
        "S11",
        "S1",
        "S9",
        "S3",
        "S8",
        "S4",
        "S7",
        "S5",
        "S6",
    ]
    n_batches = len(batch_order)
    adata = anndata.AnnData(
        X=sp.csr_matrix(np.ones((n_batches, 1), dtype=np.float32)),
        obs=pd.DataFrame(
            {"batch_id": batch_order},
            index=[f"spot-{batch_id}" for batch_id in batch_order],
        ),
        var=pd.DataFrame(index=["g0"]),
    )
    adata.obsm["spatial"] = np.arange(n_batches * 2, dtype=float).reshape(n_batches, 2)
    params = SimpleNamespace(
        K=1,
        annotation=None,
        spatial_key="spatial",
        graph_mode="no_graph",
        n_neighbors=1,
        graph_radius=None,
        graph_sigma=None,
        sample_h5ad_dict={f"sample-{i}": None for i in range(n_batches)},
    )

    training_data = TrainingData(params)
    training_data.prepare(adata, ["g0"])

    expected_codes = np.asarray(
        [int(batch_id[1:]) for batch_id in batch_order], dtype=np.float32
    )
    np.testing.assert_array_equal(training_data.batch_merge.numpy(), expected_codes)
    assert training_data.batch_size == n_batches

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_comparison_module():
    script = (
        Path(__file__).parents[1]
        / "analysis"
        / "revision_graph_3d"
        / "scripts"
        / "compare_mouse_e115_heart_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location("mouse_e115_embedding_comparison", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shared_neighbor_fraction_is_symmetric_overlap():
    comparison = load_comparison_module()
    first = np.asarray([[1, 2], [0, 2], [0, 1]])
    second = np.asarray([[1, 2], [0, 3], [1, 3]])

    forward = comparison.shared_neighbor_fraction(first, second)
    reverse = comparison.shared_neighbor_fraction(second, first)

    assert forward == reverse == 2 / 3


def test_label_purity_uses_supplied_labels():
    comparison = load_comparison_module()
    neighbors = np.asarray([[1], [0], [1]])
    labels = np.asarray(["A", "A", "B"], dtype=object)

    assert comparison.label_purity(neighbors, labels) == 2 / 3


def test_same_label_chance_uses_label_frequencies():
    comparison = load_comparison_module()
    labels = np.asarray(["A", "A", "A", "B"], dtype=object)

    assert comparison.same_label_chance(labels) == 10 / 16


def test_physical_evaluation_graph_uses_radius_without_forced_fallback():
    comparison = load_comparison_module()
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 10.0],
        ]
    )
    sections = np.asarray(["a", "a", "b", "b", "c", "c"])

    within_source, _, cross_source, cross_target, audit = (
        comparison.physical_evaluation_edges(
            coordinates, sections, within_k=1, threads=1
        )
    )

    assert len(within_source) == 6
    assert set(zip(cross_source, cross_target, strict=True)) == {
        (0, 2),
        (1, 3),
        (2, 0),
        (3, 1),
    }
    assert audit["zero_cross_section_neighbor_fraction"] == 2 / 6
    assert audit["cross_section_fallback"] is False


def test_edge_cosine_mean_is_chunked_and_signed():
    comparison = load_comparison_module()
    embedding = np.asarray([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    source = np.asarray([0, 0])
    target = np.asarray([1, 2])

    assert comparison.mean_cosine_similarity_on_edges(
        embedding, source, target, chunk_size=1
    ) == 0.0

"""Diagnostic rendering remains safe after threaded numerical runtimes start."""

import os
from concurrent.futures import ThreadPoolExecutor

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from gsMap.report.report_data import (
    _calculate_gene_annotation_stats,
    _diagnostic_plot_pool,
    _render_single_sample_gene_plot_task,
)


def test_plot_worker_starts_from_thread_after_jax_without_fork(tmp_path, monkeypatch):
    jnp.arange(4).sum().block_until_ready()

    def unsafe_fork():
        pytest.fail("Diagnostic rendering must not fork the initialized JAX parent")

    monkeypatch.setattr(os, "fork", unsafe_fork, raising=False)
    output = tmp_path / "diagnostic.png"
    task = {
        "gene": "Gene",
        "sample_name": "Sample",
        "plot_type": "gss",
        "coords": np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float),
        "values": np.asarray([0, 1, 2, 3], dtype=float),
        "output_path": output,
        "fig_width": 2,
        "dpi": 40,
    }
    with _diagnostic_plot_pool(1) as pool:
        with ThreadPoolExecutor(max_workers=1) as submitters:
            future = submitters.submit(pool.submit, _render_single_sample_gene_plot_task, task)
            assert future.result(timeout=30).result(timeout=60)
        assert pool.submit(os.getpid).result(timeout=10) != os.getpid()
        assert pool.submit(os.getenv, "OMP_NUM_THREADS").result(timeout=10) == "1"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(tmp_path.glob("*.tmp.png"))


def test_gene_annotation_stats_ignore_unobserved_categories():
    spots = np.asarray(["a", "b", "c"])
    annotations = pd.Series(
        pd.Categorical(["B", "A", "B"], categories=["unused", "A", "B"]), index=spots
    )
    scores = np.asarray([[4, 1], [0, 7], [6, 3]], dtype=float)
    result = _calculate_gene_annotation_stats(scores, spots, ["gene1", "gene2"], annotations)
    assert result["Annotation"].tolist() == ["B", "A"]
    np.testing.assert_array_equal(result["Median_GSS"], [5, 7])

"""Four-column fast path: exact trait masks/weights and bounded CPU/GPU paths."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.special import log_ndtr

from gsMap.spatial_ldsc.moment_cpu import CPUMomentRegression
from gsMap.spatial_ldsc.moment_design import gene_hsq, prepare_moment_design, supports_moments
from gsMap.spatial_ldsc.multitrait import _select_moments


def fixture_data(blocks=7, cells=17, traits=3):
    rng = np.random.default_rng(987)
    rows, genes = blocks * 140, 23
    a = sparse.random(rows, genes, density=0.2, random_state=rng, dtype=np.float32).tocsr()
    marker = rng.uniform(0.1, 4, (genes, cells)).astype(np.float32)
    prepared = []
    for t in range(traits):
        size = rows - t * blocks * 2
        idx = rng.permutation(rows)[:size].astype(np.int32)
        n = rng.uniform(10000, 50000, size).astype(np.float32)
        nbar = np.float32(n.mean() + 123)  # pre-truncation mean, not recomputed
        bs = rng.uniform(0.1, 4, (size, 2)).astype(np.float32)
        y = rng.chisquare(2, size).astype(np.float32)
        if t == 1:
            y *= 0.1  # initializer clips at zero
        if t == 2:
            y *= 100  # initializer clips at one
        base = np.column_stack((bs * n[:, None] / nbar, np.ones(size, np.float32)))
        prepared.append(
            dict(
                indices=idx,
                N=n,
                Nbar=nbar,
                baseline=base,
                baseline_sum=bs.sum(1),
                chisq=y,
                w_ld=rng.uniform(0.2, 8, size).astype(np.float32),
            )
        )
    return a, marker, prepared


def reference(a, marker, prepared, blocks):
    import jax
    import jax.numpy as jnp

    from gsMap.spatial_ldsc.multitrait import PARAMETER_KEYS
    from gsMap.spatial_ldsc.regression_kernel import process_spatial_ldsc_batch

    ld = np.asarray(a @ marker, dtype=np.float32)
    return np.asarray(
        [
            jax.device_get(
                process_spatial_ldsc_batch(
                    blocks,
                    jnp.asarray(ld[d["indices"]]),
                    *(jnp.asarray(d[k]) for k in PARAMETER_KEYS),
                )
            )
            for d in prepared
        ]
    )


def compare(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert (actual[:, 1] > 0).all()
    logp = -log_ndtr(-actual[:, 0] / actual[:, 1]) / np.log(10)
    old = -log_ndtr(-expected[:, 0] / expected[:, 1]) / np.log(10)
    assert np.max(np.abs(logp - old)) < 0.1
    np.testing.assert_allclose(actual, expected, rtol=0.008, atol=2e-7)


@pytest.mark.parametrize(
    ("blocks", "cells", "traits"), [(3, 1, 1), (7, 17, 3), (11, 37, 5), (200, 7, 2)]
)
def test_cpu_moments(blocks, cells, traits):
    from numba import get_num_threads

    a, marker, prepared = fixture_data(blocks, cells, traits)
    old_threads = get_num_threads()
    backend = CPUMomentRegression(a, prepared, blocks, threads=2, trait_batch=2)
    expected = reference(a, marker, prepared, blocks)
    result = backend.compute(marker)
    compare(result, expected)
    assert get_num_threads() == old_threads
    # Tail changes and buffer reuse cannot mutate a completed result.
    saved = result.copy()
    compare(backend.compute(marker[:, :1]), reference(a, marker[:, :1], prepared, blocks))
    np.testing.assert_array_equal(result, saved)
    backend.close()


def test_gene_projection_and_validation():
    a, marker, prepared = fixture_data()
    design = prepare_moment_design(a, prepared, 7)
    expected = []
    for d in prepared:
        ld = (a @ marker)[d["indices"]]
        expected.append(
            np.clip(
                10000
                * (d["chisq"].mean() - 1)
                / np.mean(d["N"][:, None] * (ld + d["baseline_sum"][:, None]), axis=0),
                0,
                1,
            )
        )
    np.testing.assert_allclose(gene_hsq(design, marker), expected, rtol=2e-5, atol=1e-7)
    np.testing.assert_array_equal(design["nbar"], [d["Nbar"] for d in prepared])
    for invalid in (-1, a.shape[0]):
        bad = [{**prepared[0], "indices": prepared[0]["indices"].copy()}]
        bad[0]["indices"][0] = invalid
        with pytest.raises(ValueError, match="mapping"):
            prepare_moment_design(a, bad, 7)
    bad = [{**prepared[0], "indices": np.zeros_like(prepared[0]["indices"])}]
    with pytest.raises(ValueError, match="mapping"):
        prepare_moment_design(a, bad, 7)
    with pytest.raises(ValueError, match="blocks"):
        prepare_moment_design(a, prepared, 1)


def test_general_fallback():
    a, marker, prepared = fixture_data()
    config = SimpleNamespace(ldsc_regression_backend="auto", n_blocks=7, ldsc_trait_batch_size=4)
    extra = [
        {**d, "baseline": np.column_stack((d["baseline"], np.ones(len(d["N"]))))} for d in prepared
    ]
    assert not supports_moments(a, extra)
    assert _select_moments(config, a, extra, None, "cpu") is None
    assert _select_moments(config, None, prepared, None, "gpu") is None
    config.ldsc_regression_backend = "moments"
    with pytest.raises(ValueError, match="Moments require"):
        _select_moments(config, a, extra, None, "cpu")
    backend = CPUMomentRegression(a, prepared, 7)
    with pytest.raises(ValueError):
        backend.compute(marker[:-1])
    with pytest.raises(ValueError, match="Non-finite"):
        backend.compute(marker * np.nan)


def test_moment_cli_controls():
    import click
    from typer.main import get_command

    from gsMap.cli import app
    from gsMap.config.spatial_ldsc_config import SpatialLDSCComputeConfig

    assert SpatialLDSCComputeConfig().ldsc_regression_backend == "auto"
    for command_name in ("spatial-ldsc", "quick-mode"):
        command = get_command(app).commands[command_name]
        options = {p.name: p for p in command.params}
        with click.Context(command) as ctx:
            option = options["ldsc_regression_backend"]
            assert option.get_default(ctx) == "auto"
            for value in ("jax", "moments"):
                assert option.process_value(ctx, value) == value
            with pytest.raises(click.BadParameter):
                option.process_value(ctx, "typo")


def test_prefetch_is_bounded_ordered_and_joins_on_close():
    import threading

    from gsMap.spatial_ldsc.multitrait import _prefetched_chunks

    calls = []
    lock = threading.Lock()

    def fetch(index, deferred):
        assert deferred
        with lock:
            calls.append(index)
        return index

    source = SimpleNamespace(
        marker_score_adata=SimpleNamespace(X=np.ones((20, 2))),
        chunk_starts=list(range(20)),
        fetch_chunk=fetch,
    )
    stream = _prefetched_chunks(source, SimpleNamespace(ldsc_read_workers=4))
    assert next(stream) == 0
    stream.close()
    assert len(calls) <= 9  # depth8 plus the next slot, never all20
    assert len(calls) == len(set(calls))
    assert set(calls).issubset(range(9))
    assert list(_prefetched_chunks(source, SimpleNamespace(ldsc_read_workers=1))) == list(
        range(20)
    )


def test_failure_discards_partial_results(tmp_path, monkeypatch):
    from gsMap.spatial_ldsc import multitrait
    from gsMap.spatial_ldsc.chunk_source import SpatialLDChunkSource

    a, marker, prepared = fixture_data(blocks=3, cells=7, traits=1)
    prepared[0]["name"] = "trait"
    config = SimpleNamespace(
        project_name="bad",
        sample_filter=None,
        cell_indices_range=None,
        n_blocks=3,
        ldsc_trait_batch_size=4,
        ldsc_regression_backend="auto",
    )
    source = SpatialLDChunkSource(
        "weight_matrix",
        [0, 3, 6],
        3,
        7,
        np.arange(7),
        np.array([f"s{i}" for i in range(7)]),
        ldscore_snp_positions=np.arange(a.shape[0]),
        marker_score_adata=SimpleNamespace(X=marker.T),
        marker_score_gene_indices=np.arange(marker.shape[0]),
    )
    calls = []

    class InvalidMoments:
        def compute(self, g):
            return np.full((1, 2, g.shape[1]), np.nan)

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(multitrait, "_select_moments", lambda *args: InvalidMoments())
    with pytest.raises(FloatingPointError, match="invalid"):
        multitrait.run_shared_chunks(source, config, prepared, a, tmp_path, 7)
    assert calls == ["closed"]
    assert not list(tmp_path.glob("*.csv.gz"))
    assert not list(tmp_path.glob("*.partial*"))


@pytest.mark.skipif(os.environ.get("JAX_PLATFORMS") != "cuda", reason="CUDA opt-in required")
@pytest.mark.parametrize(
    ("blocks", "cells", "traits"), [(3, 1, 1), (7, 17, 3), (11, 37, 5), (200, 100, 4), (7, 7, 19)]
)
def test_gpu_moments(blocks, cells, traits):
    import jax
    import torch

    from gsMap.spatial_ldsc.ldscore_backend import TorchLDScoreBackend
    from gsMap.spatial_ldsc.moment_gpu import GPUMomentRegression

    assert jax.default_backend() == "gpu"
    a, marker, prepared = fixture_data(blocks, cells, traits)
    ld = TorchLDScoreBackend(a)
    backend = GPUMomentRegression(a, prepared, blocks, ld, trait_batch=4)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        results, owners = backend.compute(marker)
        assert torch.backends.cuda.matmul.allow_tf32
        compare(np.concatenate(jax.device_get(results)), reference(a, marker, prepared, blocks))
        torch.cuda.empty_cache()
        tail, tail_owners = backend.compute(marker[:, :1])
        compare(
            np.concatenate(jax.device_get(tail)), reference(a, marker[:, :1], prepared, blocks)
        )
    finally:
        jax.effects_barrier()
        backend.close()
        ld.close()

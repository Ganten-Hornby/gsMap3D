"""Tests for spatial-LDSC regression-kernel selection and equivalence."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from gsMap.spatial_ldsc import regression_kernel as kernel_module


class ShapeOnly:
    """Array stub sufficient for wrapper selection tests."""

    def __init__(self, shape):
        self.shape = shape


def wrapper_args(n_snps=1_000, n_spots=50, n_baseline=3):
    return (
        200,
        ShapeOnly((n_snps, n_spots)),
        None,
        None,
        None,
        ShapeOnly((n_snps, n_baseline)),
        None,
        None,
    )


def install_kernel_stubs(monkeypatch):
    monkeypatch.setattr(
        kernel_module,
        "process_chunk_batched_lowmem_jit",
        lambda *args, **kwargs: "lowmem",
    )

    def per_spot(*args, **kwargs):
        assert args[1] == 1
        return "per_spot"

    monkeypatch.setattr(kernel_module, "process_chunk_jit", per_spot)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_wrapper_prefers_lowmem_for_normal_chunks(monkeypatch, backend):
    install_kernel_stubs(monkeypatch)
    monkeypatch.setattr(kernel_module.jax, "default_backend", lambda: backend)
    monkeypatch.setattr(kernel_module, "_estimate_available_device_gb", lambda: 40.0)
    result = kernel_module.wrapper_of_process_chunk_jit(
        *wrapper_args(n_snps=1_059_600, n_spots=50)
    )
    assert result == "lowmem"


def test_wrapper_keeps_large_chunk_batched_on_large_gpu(monkeypatch):
    install_kernel_stubs(monkeypatch)
    monkeypatch.setattr(kernel_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(kernel_module, "_estimate_available_device_gb", lambda: 40.0)
    result = kernel_module.wrapper_of_process_chunk_jit(
        *wrapper_args(n_snps=1_000_000, n_spots=1_000)
    )
    assert result == "lowmem"


@pytest.mark.parametrize(
    ("backend", "available_device_gb"),
    [("cpu", None), ("gpu", 2.0)],
)
def test_wrapper_uses_per_spot_when_lowmem_estimate_exceeds_budget(
    monkeypatch, backend, available_device_gb
):
    install_kernel_stubs(monkeypatch)
    monkeypatch.setattr(kernel_module.jax, "default_backend", lambda: backend)
    monkeypatch.setattr(
        kernel_module,
        "_estimate_available_device_gb",
        lambda: available_device_gb,
    )
    result = kernel_module.wrapper_of_process_chunk_jit(
        *wrapper_args(n_snps=1_000_000, n_spots=1_000)
    )
    assert result == "per_spot"


def test_lowmem_matches_fullsolve_on_small_problem():
    rng = np.random.default_rng(20260815)
    n_blocks, n_snps, n_spots = 4, 80, 4
    spatial_ld = jnp.asarray(rng.uniform(0.2, 2.0, size=(n_snps, n_spots)), dtype=jnp.float32)
    baseline_ld_sum = jnp.asarray(rng.uniform(0.2, 1.2, size=n_snps), dtype=jnp.float32)
    chisq = jnp.asarray(rng.uniform(1.0, 2.5, size=n_snps), dtype=jnp.float32)
    sample_size = jnp.asarray(rng.uniform(80_000, 120_000, size=n_snps), dtype=jnp.float32)
    baseline_ann = jnp.asarray(
        np.column_stack(
            [
                rng.uniform(0.1, 1.0, size=n_snps),
                rng.uniform(0.1, 1.0, size=n_snps),
                np.ones(n_snps),
            ]
        ),
        dtype=jnp.float32,
    )
    w_ld = jnp.asarray(rng.uniform(1.0, 2.0, size=n_snps), dtype=jnp.float32)
    nbar = 100_000.0
    args = (
        n_blocks,
        spatial_ld,
        baseline_ld_sum,
        chisq,
        sample_size,
        baseline_ann,
        w_ld,
        nbar,
    )

    full_beta, full_se = kernel_module.process_chunk_batched_fullsolve_jit(*args)
    low_beta, low_se = kernel_module.process_chunk_batched_lowmem_jit(*args)
    np.testing.assert_allclose(np.asarray(low_beta), np.asarray(full_beta), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(low_se), np.asarray(full_se), rtol=1e-4, atol=1e-6)


def test_all_kernels_scale_focal_ld_by_per_snp_sample_size():
    rng = np.random.default_rng(20260816)
    n_blocks, n_snps = 4, 80
    sample_size = np.linspace(50_000.0, 500_000.0, n_snps, dtype=np.float32)
    nbar = float(sample_size.mean())

    raw_spatial = rng.uniform(0.2, 2.0, n_snps)
    raw_spatial *= np.power(sample_size / nbar, 1.5)
    expected_beta = np.asarray([1.5e-6, 2.0e-6, 3.0e-6], dtype=np.float32)
    common_signal = expected_beta[1] * raw_spatial
    spatial_ld = (common_signal[:, None] / expected_beta[None, :]).astype(np.float32)

    # Each spot obeys chisq = intercept + N * beta_spot * spatial_ld_spot.
    chisq = (1.25 + sample_size * common_signal).astype(np.float32)
    baseline_ld_sum = rng.uniform(0.2, 1.2, n_snps).astype(np.float32)
    baseline_ann = np.ones((n_snps, 1), dtype=np.float32)
    w_ld = rng.uniform(1.0, 2.0, n_snps).astype(np.float32)
    arrays = tuple(
        jnp.asarray(value)
        for value in (
            spatial_ld,
            baseline_ld_sum,
            chisq,
            sample_size,
            baseline_ann,
            w_ld,
        )
    )
    kernels = (
        lambda: kernel_module.process_chunk_jit(n_blocks, 0, *arrays, nbar),
        lambda: kernel_module.process_chunk_batched_lowmem_jit(n_blocks, *arrays, nbar),
        lambda: kernel_module.process_chunk_batched_fullsolve_jit(n_blocks, *arrays, nbar),
        lambda: kernel_module.process_chunk_batched_materialized_jit(n_blocks, *arrays, nbar),
    )

    for run_kernel in kernels:
        beta, se = run_kernel()
        np.testing.assert_allclose(np.asarray(beta), expected_beta, rtol=2e-3, atol=1e-9)
        assert np.max(np.asarray(se)) < 1e-8

"""Numerical tests for the single production spatial-LDSC kernel."""

import jax.numpy as jnp
import numpy as np
import pytest

from gsMap.spatial_ldsc.kernels.jax_regression import process_spatial_ldsc_batch


def _direct_per_spot_reference(
    n_blocks,
    spatial_ld,
    baseline_ld_sum,
    chisq,
    sample_size,
    baseline_ann,
    w_ld,
    nbar,
):
    estimates = []
    errors = []
    block_size = spatial_ld.shape[0] // n_blocks
    n_snps_used = block_size * n_blocks

    for spot_ld in spatial_ld.T:
        x_tot = spot_ld + baseline_ld_sum
        hsq = 10000.0 * (np.mean(chisq) - 1.0) / np.mean(x_tot * sample_size)
        hsq = np.clip(hsq, 0.0, 1.0)
        weights = np.sqrt(
            1.0
            / (
                2
                * np.square(1.0 + hsq * sample_size / 10000.0 * np.maximum(x_tot, 1.0))
                * np.maximum(w_ld, 1.0)
            )
        )
        weights /= weights.sum()
        design = np.column_stack([spot_ld * sample_size / nbar, baseline_ann])
        weighted_design = design * weights[:, None]
        weighted_response = chisq * weights
        x_blocks = weighted_design[:n_snps_used].reshape(n_blocks, block_size, -1)
        y_blocks = weighted_response[:n_snps_used].reshape(n_blocks, block_size)
        xty_blocks = np.einsum("nbp,nb->np", x_blocks, y_blocks)
        xtx_blocks = np.einsum("nbp,nbq->npq", x_blocks, x_blocks)
        xty_total = xty_blocks.sum(axis=0)
        xtx_total = xtx_blocks.sum(axis=0)
        estimate = np.linalg.solve(xtx_total, xty_total)
        delete_estimates = np.linalg.solve(
            xtx_total - xtx_blocks,
            (xty_total - xty_blocks)[..., None],
        ).squeeze(-1)
        pseudovalues = n_blocks * estimate - (n_blocks - 1) * delete_estimates
        jackknife_estimate = pseudovalues.mean(axis=0)
        centered = pseudovalues[:, 0] - jackknife_estimate[0]
        jackknife_se = np.sqrt(np.sum(np.square(centered)) / (n_blocks * (n_blocks - 1)))
        estimates.append(jackknife_estimate[0] / nbar)
        errors.append(jackknife_se / nbar)
    return np.asarray(estimates), np.asarray(errors)


def test_production_kernel_matches_direct_per_spot_reference():
    rng = np.random.default_rng(20260815)
    n_blocks, n_snps, n_spots = 4, 80, 4
    spatial_ld = rng.uniform(0.2, 2.0, size=(n_snps, n_spots)).astype(np.float32)
    baseline_ld_sum = rng.uniform(0.2, 1.2, size=n_snps).astype(np.float32)
    chisq = rng.uniform(1.0, 2.5, size=n_snps).astype(np.float32)
    sample_size = rng.uniform(80_000, 120_000, size=n_snps).astype(np.float32)
    baseline_ann = np.column_stack(
        [
            rng.uniform(0.1, 1.0, size=n_snps),
            rng.uniform(0.1, 1.0, size=n_snps),
            np.ones(n_snps),
        ]
    ).astype(np.float32)
    w_ld = rng.uniform(1.0, 2.0, size=n_snps).astype(np.float32)
    nbar = float(sample_size.mean())

    expected_beta, expected_se = _direct_per_spot_reference(
        n_blocks,
        spatial_ld,
        baseline_ld_sum,
        chisq,
        sample_size,
        baseline_ann,
        w_ld,
        nbar,
    )
    beta, se = process_spatial_ldsc_batch(
        n_blocks,
        *map(
            jnp.asarray,
            (spatial_ld, baseline_ld_sum, chisq, sample_size, baseline_ann, w_ld),
        ),
        nbar,
    )

    np.testing.assert_allclose(np.asarray(beta), expected_beta, rtol=2e-4, atol=1e-7)
    np.testing.assert_allclose(np.asarray(se), expected_se, rtol=2e-4, atol=1e-7)


def test_production_kernel_scales_focal_ld_by_per_snp_sample_size():
    rng = np.random.default_rng(20260816)
    n_blocks, n_snps = 4, 80
    sample_size = np.linspace(50_000.0, 500_000.0, n_snps, dtype=np.float32)
    nbar = float(sample_size.mean())
    raw_spatial = rng.uniform(0.2, 2.0, n_snps)
    raw_spatial *= np.power(sample_size / nbar, 1.5)
    expected_beta = np.asarray([1.5e-6, 2.0e-6, 3.0e-6], dtype=np.float32)
    common_signal = expected_beta[1] * raw_spatial
    spatial_ld = (common_signal[:, None] / expected_beta[None, :]).astype(np.float32)
    chisq = (1.25 + sample_size * common_signal).astype(np.float32)
    baseline_ld_sum = rng.uniform(0.2, 1.2, n_snps).astype(np.float32)
    baseline_ann = np.ones((n_snps, 1), dtype=np.float32)
    w_ld = rng.uniform(1.0, 2.0, n_snps).astype(np.float32)

    beta, se = process_spatial_ldsc_batch(
        n_blocks,
        *map(
            jnp.asarray,
            (spatial_ld, baseline_ld_sum, chisq, sample_size, baseline_ann, w_ld),
        ),
        nbar,
    )

    np.testing.assert_allclose(np.asarray(beta), expected_beta, rtol=2e-3, atol=1e-9)
    assert np.max(np.asarray(se)) < 1e-8


@pytest.mark.parametrize(
    ("n_spots", "n_baseline", "response_scale"),
    [(1, 1, 2), (7, 3, 2), (100, 3, 2), (7, 4, 2), (7, 5, 2), (7, 8, 2), (3, 3, 0.5), (3, 3, 50)],
)
def test_block_equations_match_float64_with_partial_blocks(n_spots, n_baseline, response_scale):
    """Cover both contraction paths, small batches, h2 clipping and tail SNPs."""
    rng = np.random.default_rng(82)
    n_snps, n_blocks = 403, 4
    spatial = rng.uniform(0.2, 2, (n_snps, n_spots)).astype(np.float32)
    baseline_sum = rng.uniform(0.2, 1.2, n_snps).astype(np.float32)
    chisq = rng.uniform(0.8, 1.2, n_snps).astype(np.float32) * response_scale
    sample_size = rng.uniform(50_000, 150_000, n_snps).astype(np.float32)
    baseline = np.column_stack(
        [rng.uniform(0.1, 1, (n_snps, n_baseline - 1)), np.ones(n_snps)]
    ).astype(np.float32)
    weights = rng.uniform(0.5, 2, n_snps).astype(np.float32)
    arrays = (spatial, baseline_sum, chisq, sample_size, baseline, weights)
    nbar = float(sample_size.mean())
    expected = _direct_per_spot_reference(n_blocks, *[x.astype(np.float64) for x in arrays], nbar)
    observed = process_spatial_ldsc_batch(n_blocks, *map(jnp.asarray, arrays), nbar)
    for result, reference in zip(observed, expected, strict=True):
        np.testing.assert_allclose(np.asarray(result), reference, rtol=5e-4, atol=1e-9)

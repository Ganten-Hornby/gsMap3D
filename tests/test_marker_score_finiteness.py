"""Regression tests for finite marker-score generation."""

import jax.numpy as jnp
import numpy as np

from gsMap.latent2gene.marker_scores import (
    compute_marker_scores_3d_hierarchical_pool_jax,
    compute_marker_scores_jax,
)

FLOAT16_MAX = np.finfo(np.float16).max


def _standard_marker_scores(enrichment: float) -> np.ndarray:
    return np.asarray(
        compute_marker_scores_jax(
            jnp.array([[np.log(enrichment)]], dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            1,
            1,
            jnp.zeros(1, dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
            True,
        )
    )


def _hierarchical_marker_scores(enrichments: list[float]) -> np.ndarray:
    n_slices = len(enrichments)
    return np.asarray(
        compute_marker_scores_3d_hierarchical_pool_jax(
            jnp.array([[np.log(value)] for value in enrichments], dtype=jnp.float32),
            jnp.ones((1, n_slices), dtype=jnp.float32),
            1,
            n_slices,
            1,
            jnp.zeros(1, dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
            True,
        )
    )


def test_marker_score_transform_preserves_representable_values():
    result = _standard_marker_scores(2.0)
    expected = np.float16(np.expm1(2.0**1.5))

    assert result.dtype == np.float16
    assert result[0, 0] == expected


def test_marker_score_transform_saturates_instead_of_overflowing():
    result = _standard_marker_scores(100.0)

    assert result.dtype == np.float16
    assert np.isfinite(result).all()
    assert result[0, 0] == FLOAT16_MAX


def test_hierarchical_marker_score_transform_is_finite():
    result = _hierarchical_marker_scores([100.0, 100.0])

    assert result.dtype == np.float16
    assert np.isfinite(result).all()
    assert result[0, 0] == FLOAT16_MAX


def test_hierarchical_transform_preserves_representable_pooled_value():
    transformed_score = 100_000.0
    enrichment = np.log1p(transformed_score) ** (2.0 / 3.0)

    result = _hierarchical_marker_scores([enrichment, 0.5])

    # The first slice exceeds float16, but its mean with the zero-scoring second
    # slice is representable. It must therefore be pooled before final saturation.
    expected = np.float16(transformed_score / 2.0)
    assert abs(float(result[0, 0]) - transformed_score / 2.0) <= float(np.spacing(expected))

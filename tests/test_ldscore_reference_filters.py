from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from gsMap.config.ldscore_config import LDScoreConfig
from gsMap.ldscore.batch_construction import construct_batches, resolve_ld_window
from gsMap.ldscore.compute import (
    compute_batch_weights_and_ld_scores,
    compute_batch_weights_sparse,
    compute_unbiased_l2_batch,
)
from gsMap.ldscore.io import biallelic_snp_mask, calculate_variant_statistics
from gsMap.ldscore.pipeline import exclude_mhc_regression_snps


def test_ldscore_defaults_use_strict_mac_above_five() -> None:
    fields = LDScoreConfig.__dataclass_fields__
    assert fields["snps_only"].default is True
    assert fields["mac_min"].default == 5
    assert fields["maf_min"].default is None
    assert fields["exclude_mhc_from_regression"].default is True


def test_biallelic_snp_mask_excludes_indels_and_noncanonical_alleles() -> None:
    observed = biallelic_snp_mask(
        np.array(["A", "c", "AT", "A", "0", "G"]),
        np.array(["G", "T", "A", "A", "C", "<DEL>"]),
    )
    np.testing.assert_array_equal(observed, [True, True, False, False, False, False])


def test_variant_statistics_use_called_alleles_for_mac() -> None:
    genotypes = np.array(
        [
            [0, 0, 2, np.nan],
            [0, 1, 2, np.nan],
            [1, 1, 1, np.nan],
            [1, 1, 1, np.nan],
            [2, 1, 0, np.nan],
            [2, 2, 0, np.nan],
            [2, 2, np.nan, np.nan],
            [2, 2, np.nan, np.nan],
        ],
        dtype=np.float32,
    )

    stats = calculate_variant_statistics(genotypes)

    np.testing.assert_array_equal(stats.called_samples, [8, 8, 6, 0])
    np.testing.assert_array_equal(stats.mac, [6, 6, 6, 0])
    np.testing.assert_allclose(stats.maf, [6 / 16, 6 / 16, 6 / 12, 0])
    assert stats.mac[2] == 6  # Uses 12 called alleles, not 16 total alleles.


def test_mhc_is_removed_from_targets_but_retained_in_reference() -> None:
    bim = pd.DataFrame(
        {
            "SNP": ["outside_left", "mhc_start", "mhc_middle", "mhc_end", "outside_right"],
            "BP": [28_510_119, 28_510_120, 30_000_000, 33_480_577, 33_480_578],
        }
    )
    original = bim.copy(deep=True)

    retained, removed, reference_mhc = exclude_mhc_regression_snps(
        hm3_snps=bim["SNP"].tolist(),
        bim_df=bim,
        chromosome=6,
        mhc_start_bp=28_510_120,
        mhc_end_bp=33_480_577,
    )

    assert retained == ["outside_left", "outside_right"]
    assert removed == 3
    assert reference_mhc == 3
    pd.testing.assert_frame_equal(bim, original)


def test_mhc_filter_is_not_applied_to_other_chromosomes() -> None:
    bim = pd.DataFrame({"SNP": ["rs1"], "BP": [30_000_000]})
    retained, removed, reference_mhc = exclude_mhc_regression_snps(
        hm3_snps=["rs1"],
        bim_df=bim,
        chromosome=5,
        mhc_start_bp=28_510_120,
        mhc_end_bp=33_480_577,
    )
    assert retained == ["rs1"]
    assert removed == 0
    assert reference_mhc == 0


def test_in_place_unbiased_l2_matches_reference_formula() -> None:
    rng = np.random.default_rng(20260901)
    x_hm3 = rng.normal(size=(503, 23)).astype(np.float32)
    x_ref = rng.normal(size=(503, 101)).astype(np.float32)
    x_hm3 = ((x_hm3 - x_hm3.mean(axis=0)) / x_hm3.std(axis=0)).astype(np.float32)
    x_ref = ((x_ref - x_ref.mean(axis=0)) / x_ref.std(axis=0)).astype(np.float32)

    r = np.dot(x_hm3.T, x_ref) / x_hm3.shape[0]
    expected = r**2 - (1 - r**2) / (x_hm3.shape[0] - 2)
    observed = compute_unbiased_l2_batch(x_hm3, x_ref)

    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-7)


def test_exact_ld_windows_make_results_batch_size_invariant() -> None:
    rng = np.random.default_rng(42)
    genotypes = rng.normal(size=(101, 20)).astype(np.float32)
    genotypes = ((genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)).astype(np.float32)
    bim = pd.DataFrame(
        {
            "CHR": 1,
            "SNP": [f"rs{i}" for i in range(20)],
            "CM": 0.0,
            "BP": np.arange(20) * 100,
        }
    )
    hm3 = ["rs3", "rs7", "rs12", "rs17"]
    mapping = scipy.sparse.eye(20, format="csr", dtype=np.float32)
    coordinates, max_distance = resolve_ld_window(bim, ld_wind=0.25, ld_unit="KB")

    def calculate(batch_size: int) -> np.ndarray:
        results = []
        for batch in construct_batches(bim, hm3, batch_size, 0.25, "KB"):
            ref_slice = slice(batch.ref_start_idx, batch.ref_end_idx)
            mask = (
                np.abs(coordinates[batch.hm3_indices, None] - coordinates[None, ref_slice])
                <= max_distance
            )
            results.append(
                compute_batch_weights_sparse(
                    genotypes[:, batch.hm3_indices],
                    genotypes[:, ref_slice],
                    mapping[ref_slice],
                    ld_window_mask=mask,
                )
            )
        return np.vstack(results)

    one_at_a_time = calculate(1)
    all_at_once = calculate(4)
    np.testing.assert_allclose(one_at_a_time, all_at_once, rtol=2e-6, atol=2e-7)

    # rs3 at 300 bp has a 250-bp window, so only rs1-rs5 can contribute.
    assert np.count_nonzero(one_at_a_time[0, :1]) == 0
    assert np.count_nonzero(one_at_a_time[0, 1:6]) == 5
    assert np.count_nonzero(one_at_a_time[0, 6:]) == 0


@pytest.mark.parametrize(
    ("coordinates", "match"),
    [
        ([0.0, 0.0, 0.0], "do not define a genetic map"),
        ([0.0, np.nan, 1.0], "not finite"),
        ([0.0, 2.0, 1.0], "not sorted"),
    ],
)
def test_cm_windows_reject_invalid_maps(coordinates: list[float], match: str) -> None:
    bim = pd.DataFrame(
        {
            "CHR": 1,
            "SNP": [f"rs{i}" for i in range(len(coordinates))],
            "CM": coordinates,
            "BP": np.arange(len(coordinates)) * 100,
        }
    )
    with pytest.raises(ValueError, match=match):
        resolve_ld_window(bim, ld_wind=1, ld_unit="CM")


def test_cm_window_is_invariant_to_map_origin() -> None:
    bim = pd.DataFrame(
        {
            "CHR": 22,
            "SNP": [f"rs{i}" for i in range(5)],
            "CM": [-0.1, 0.0, 0.5, 1.0, 2.0],
            "BP": np.arange(5) * 100,
        }
    )
    before = construct_batches(bim, bim.SNP.tolist(), 2, ld_wind=1, ld_unit="CM")
    shifted = bim.assign(CM=bim.CM + 100.0)
    after = construct_batches(shifted, shifted.SNP.tolist(), 2, ld_wind=1, ld_unit="CM")
    assert [(b.ref_start_idx, b.ref_end_idx) for b in before] == [
        (b.ref_start_idx, b.ref_end_idx) for b in after
    ]
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left.hm3_indices, right.hm3_indices)


def test_fused_annotation_and_weight_ld_share_exact_l2_block() -> None:
    rng = np.random.default_rng(20260902)
    genotypes = rng.normal(size=(101, 12)).astype(np.float32)
    genotypes = ((genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)).astype(np.float32)
    target_indices = np.array([2, 6, 9])
    mapping = scipy.sparse.csr_matrix(
        rng.integers(0, 2, size=(12, 5), dtype=np.int8).astype(np.float32)
    )
    window_mask = np.abs(target_indices[:, None] - np.arange(12)[None, :]) <= 3
    regression_mask = np.zeros(12, dtype=bool)
    regression_mask[[1, 2, 6, 9, 10]] = True

    observed_weights, observed_w_ld = compute_batch_weights_and_ld_scores(
        genotypes[:, target_indices],
        genotypes,
        mapping,
        ld_window_mask=window_mask,
        ld_score_reference_mask=regression_mask,
    )

    expected_l2 = compute_unbiased_l2_batch(genotypes[:, target_indices], genotypes)
    expected_l2 *= window_mask
    expected_weights = expected_l2 @ mapping
    expected_w_ld = np.sum(expected_l2[:, regression_mask], axis=1, dtype=np.float64)

    np.testing.assert_allclose(observed_weights, expected_weights, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(observed_w_ld, expected_w_ld, rtol=2e-6, atol=2e-7)

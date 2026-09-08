from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from scipy import stats

if TYPE_CHECKING:
    from pathlib import Path

from gsMap.cauchy_combination_test import (
    _acat_test,
    _build_group_index,
    _load_trait_matrix,
    _matrix_positions,
    _read_metadata,
    _result_frame,
    _summarize_trait_groups,
    _update_hash_from_arrow_strings,
    _update_hash_from_python_strings,
    _write_combined_parquet,
    run_Cauchy_combination,
    run_cauchy_on_dataframe,
)


def test_identifier_hash_preserves_boundaries_and_chunk_invariance():
    values = np.array(["ab", "c", "δ"], dtype=object)
    python_hash = hashlib.blake2b()
    _update_hash_from_python_strings(python_hash, values)
    arrow_hash = hashlib.blake2b()
    for chunk in [pa.array(values[:1]), pa.array(values[1:])]:
        _update_hash_from_arrow_strings(arrow_hash, chunk)
    assert arrow_hash.digest() == python_hash.digest()
    different = hashlib.blake2b()
    _update_hash_from_python_strings(different, np.array(["a", "bc", "δ"], dtype=object))
    assert different.digest() != python_hash.digest()


def test_identifier_hash_handles_empty_arrow_chunk_and_rejects_nul():
    empty = hashlib.blake2b()
    _update_hash_from_arrow_strings(empty, pa.array([], type=pa.string()))
    assert empty.digest() == hashlib.blake2b().digest()
    for values in (np.array(["a\0b"]), np.array([b"a\0b"])):
        with pytest.raises(ValueError, match="NUL"):
            _update_hash_from_python_strings(hashlib.blake2b(), values)


def _legacy_equal_weight_acat(p_values: np.ndarray) -> float:
    weights = np.full(len(p_values), 1 / len(p_values))
    is_small = p_values < 1e-16
    statistic = 0.0
    if np.any(is_small):
        statistic += np.sum((weights[is_small] / p_values[is_small]) / np.pi)
    if np.any(~is_small):
        statistic += np.sum(weights[~is_small] * np.tan((0.5 - p_values[~is_small]) * np.pi))
    if statistic > 1e15:
        return (1 / statistic) / np.pi
    return 1 - stats.cauchy.cdf(statistic)


def _write_ldsc(path: Path, spots: list[str], p_values: np.ndarray) -> None:
    frame = pd.DataFrame(
        {
            "spot": spots,
            "beta": np.linspace(0.1, 0.2, len(spots)),
            "se": np.full(len(spots), 0.1),
            "z": np.linspace(1.0, 2.0, len(spots)),
            "p": p_values,
            "neg_log10_p": -np.log10(p_values),
        }
    )
    frame.to_csv(path, index=False, compression="gzip")


@pytest.fixture
def compact_cauchy_inputs(tmp_path):
    spots = [f"spot_{index:02d}" for index in range(18)]
    obs = pd.DataFrame(
        {
            "annotation": pd.Categorical(
                ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
                categories=["A", "B", "C"],
            ),
            "second_annotation": pd.Categorical(
                ["X", "Y", "Z"] * 6,
                categories=["X", "Y", "Z"],
            ),
            "sample_name": pd.Categorical(
                ["sample_1"] * 9 + ["sample_2"] * 9,
                categories=["sample_1", "sample_2"],
            ),
        },
        index=spots,
    )
    h5ad_path = tmp_path / "latent.h5ad"
    ad.AnnData(X=np.zeros((len(spots), 1), dtype=np.float32), obs=obs).write_h5ad(h5ad_path)

    p_trait_1 = np.geomspace(0.9, 1e-8, len(spots))
    p_trait_2 = np.array(
        [0.8, 0.7, 0.5, 0.2, 0.1, 1e-20, 0.9, 0.4, 0.3] * 2,
        dtype=np.float64,
    )
    paths = {
        "trait_1": tmp_path / "trait_1.csv.gz",
        "trait_2": tmp_path / "trait_2.csv.gz",
    }
    _write_ldsc(paths["trait_1"], spots, p_trait_1)
    _write_ldsc(paths["trait_2"], spots, p_trait_2)
    return h5ad_path, spots, paths, (p_trait_1, p_trait_2)


def test_compact_loader_and_bounded_parquet_preserve_values(compact_cauchy_inputs, tmp_path):
    h5ad_path, spots, paths, p_values = compact_cauchy_inputs
    metadata = _read_metadata(h5ad_path, ["annotation", "second_annotation"])
    matrix = _load_trait_matrix(paths, metadata, max_workers=2)

    assert matrix.values.shape == (2, len(spots))
    assert matrix.valid_spots.all()
    np.testing.assert_allclose(matrix.values[0], -np.log10(p_values[0]))
    np.testing.assert_allclose(matrix.values[1], -np.log10(p_values[1]))

    parquet_path = tmp_path / "combined.parquet"
    _write_combined_parquet(parquet_path, matrix, metadata)
    combined = pd.read_parquet(parquet_path)

    assert combined.columns.tolist() == [
        "spot",
        "trait_1",
        "trait_2",
        "annotation",
        "second_annotation",
        "sample_name",
    ]
    assert combined["spot"].tolist() == spots
    np.testing.assert_allclose(combined["trait_1"], matrix.values[0])
    np.testing.assert_allclose(combined["trait_2"], matrix.values[1])
    assert (
        combined["annotation"].astype(str).tolist()
        == metadata.annotations["annotation"].astype(str).tolist()
    )


@pytest.mark.parametrize("invalid", [-0.1, 1.01, np.inf, -np.inf])
def test_compact_loader_rejects_invalid_probabilities(compact_cauchy_inputs, invalid):
    h5ad_path, spots, paths, p_values = compact_cauchy_inputs
    values = p_values[0].copy()
    values[0] = invalid
    with np.errstate(invalid="ignore"):
        _write_ldsc(paths["trait_1"], spots, values)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        _load_trait_matrix(paths, _read_metadata(h5ad_path, ["annotation"]), max_workers=1)


def test_equal_weight_acat_avoids_weight_array_without_changing_result():
    p_values = np.concatenate(
        (
            np.geomspace(0.99, 1e-14, 10_000),
            np.array([1e-18, 1e-30, 1e-100]),
        )
    )
    assert _acat_test(p_values) == pytest.approx(
        _legacy_equal_weight_acat(p_values),
        rel=1e-14,
        abs=1e-300,
    )


@pytest.mark.parametrize("sample_level", [False, True])
def test_compact_cauchy_matches_dataframe_reference(compact_cauchy_inputs, sample_level):
    h5ad_path, _, paths, _ = compact_cauchy_inputs
    metadata = _read_metadata(h5ad_path, ["annotation"])
    matrix = _load_trait_matrix(paths, metadata, max_workers=2)
    positions = _matrix_positions(matrix.valid_spots)

    groups = _build_group_index(
        metadata.annotations["annotation"],
        positions,
        sample_name=metadata.sample_name if sample_level else None,
    )
    compact_rows = []
    for trait_index, trait in enumerate(matrix.traits):
        compact_rows.extend(
            _summarize_trait_groups(
                trait,
                matrix.values[trait_index],
                "annotation",
                groups,
            )
        )
    compact = _result_frame(compact_rows, sample_level=sample_level).reset_index(drop=True)

    reference_input = pd.DataFrame(
        matrix.values.T,
        columns=matrix.traits,
    )
    reference_input["annotation"] = metadata.annotations["annotation"]
    reference_input["sample_name"] = metadata.sample_name
    reference = run_cauchy_on_dataframe(
        reference_input,
        annotation_col="annotation",
        trait_cols=list(matrix.traits),
        extra_group_col="sample_name" if sample_level else None,
    ).reset_index(drop=True)

    key_columns = ["trait", "annotation"]
    if sample_level:
        key_columns.append("sample_name")
    compact.sort_values(key_columns, inplace=True, ignore_index=True)
    reference.sort_values(key_columns, inplace=True, ignore_index=True)
    pd.testing.assert_frame_equal(
        compact,
        reference,
        check_dtype=False,
        rtol=1e-13,
        atol=1e-300,
    )


def test_compact_loader_aligns_reordered_spots(compact_cauchy_inputs):
    h5ad_path, spots, paths, p_values = compact_cauchy_inputs
    _write_ldsc(paths["trait_2"], list(reversed(spots)), p_values[1][::-1])
    metadata = _read_metadata(h5ad_path, ["annotation"])

    matrix = _load_trait_matrix(paths, metadata, max_workers=2)
    np.testing.assert_allclose(matrix.values[1], -np.log10(p_values[1]), equal_nan=True)


def test_compact_loader_rejects_changed_spot_universe(compact_cauchy_inputs):
    h5ad_path, spots, paths, p_values = compact_cauchy_inputs
    changed = list(spots)
    changed[0] = "unknown-spot"
    _write_ldsc(paths["trait_2"], changed, p_values[1])
    with pytest.raises(ValueError, match="does not match the canonical"):
        _load_trait_matrix(paths, _read_metadata(h5ad_path, ["annotation"]), max_workers=2)


def test_compact_loader_rejects_duplicate_result_spots(compact_cauchy_inputs):
    h5ad_path, spots, paths, p_values = compact_cauchy_inputs
    changed = list(spots)
    changed[0] = changed[1]
    _write_ldsc(paths["trait_2"], changed, p_values[1])
    with pytest.raises(ValueError, match="Duplicate spot"):
        _load_trait_matrix(paths, _read_metadata(h5ad_path, ["annotation"]), max_workers=2)


def test_compact_cauchy_end_to_end_preserves_output_contract(
    compact_cauchy_inputs, tmp_path, monkeypatch
):
    h5ad_path, _, paths, _ = compact_cauchy_inputs
    monkeypatch.chdir(tmp_path)

    class Config:
        ldsc_traits_result_path_dict = paths
        concatenated_latent_adata_path = h5ad_path
        annotation_list = ["annotation", "second_annotation"]
        ldsc_combined_parquet_path = tmp_path / "cauchy" / "combined.parquet"

        @staticmethod
        def get_cauchy_result_file(trait, annotation, all_samples):
            suffix = "cauchy.csv" if all_samples else "sample_cauchy.csv"
            return tmp_path / "cauchy" / f"{trait}.{annotation}.{suffix}"

    run_Cauchy_combination(Config())

    assert Config.ldsc_combined_parquet_path.is_file()
    combined = pd.read_parquet(Config.ldsc_combined_parquet_path)
    assert len(combined) == 18
    for trait in paths:
        for annotation in Config.annotation_list:
            aggregated = Config.get_cauchy_result_file(trait, annotation, True)
            sample_level = Config.get_cauchy_result_file(trait, annotation, False)
            assert aggregated.is_file()
            assert sample_level.is_file()
            assert set(pd.read_csv(aggregated)["trait"]) == {trait}
            assert set(pd.read_csv(sample_level)["trait"]) == {trait}

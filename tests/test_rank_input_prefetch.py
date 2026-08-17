"""Tests for selective rank input loading and bounded prefetching."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

from gsMap.latent2gene import rank_calculator
from gsMap.latent2gene.rank_calculator import (
    RankCalculator,
    _prefetched_rank_inputs,
    _RankInputRequest,
    _read_rank_input,
)

pytestmark = pytest.mark.filterwarnings("ignore:Observation names are not unique.*")


def _write_rank_fixture(path, *, include_counts: bool = True) -> None:
    obs = pd.DataFrame(
        {
            "celltype": pd.Categorical(["a", "drop", "b", "b"]),
            "quality": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        },
        index=pd.Index(["duplicate", "drop", "duplicate", "keep"], name="cell"),
    )
    var = pd.DataFrame(index=pd.Index(["g1", "g2", "g3"], name="gene"))
    X = csr_matrix(
        np.asarray(
            [[100, 0, 0], [0, 200, 0], [0, 0, 300], [400, 0, 500]],
            dtype=np.float32,
        )
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if include_counts:
        adata.layers["counts"] = csr_matrix(
            np.asarray(
                [[1, 0, 2], [0, 3, 0], [4, 5, 0], [0, 6, 7]],
                dtype=np.float32,
            )
        )
    adata.obsm["array_embedding"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    adata.obsm["frame_embedding"] = pd.DataFrame(
        np.arange(12, dtype=np.float32).reshape(4, 3),
        index=obs.index.copy(),
        columns=["e1", "e2", "e3"],
    )
    adata.raw = ad.AnnData(
        X=csr_matrix(np.full((4, 3), 9, dtype=np.float32)),
        obs=obs.copy(),
        var=var.copy(),
    )
    adata.write_h5ad(path)


def _expected_full_load(path, keep: np.ndarray, sample_name: str, slice_id: int):
    expected = ad.read_h5ad(path)
    expected.obs["slice_id"] = slice_id
    expected.obs["slice_name"] = sample_name
    expected.obs["sample_name"] = sample_name
    expected.obs_names_make_unique()
    expected.obs_names = (
        expected.obs_names.astype(str) + "|" + expected.obs["slice_name"].astype(str)
    )
    return expected[keep].copy()


def test_selective_reader_matches_full_anndata_semantics(tmp_path, monkeypatch):
    path = tmp_path / "rank_input.h5ad"
    _write_rank_fixture(path)
    keep = np.asarray([True, False, True, True])
    expected = _expected_full_load(path, keep, "sample-a", 7)

    read_paths = []
    original_read_elem = rank_calculator.read_elem

    def recording_read_elem(element):
        read_paths.append(element.name)
        return original_read_elem(element)

    monkeypatch.setattr(rank_calculator, "read_elem", recording_read_elem)
    result = _read_rank_input(
        _RankInputRequest(
            sample_name="sample-a",
            h5ad_path=path,
            slice_id=7,
            keep_mask=pd.Series(keep),
            data_layer="counts",
        )
    )

    assert result.matrix_source == "layers/counts"
    assert "/layers/counts" in read_paths
    assert "/X" not in read_paths
    assert "/raw" not in read_paths
    assert result.var_names == expected.var_names.tolist()
    pd.testing.assert_frame_equal(result.obs, expected.obs)
    np.testing.assert_array_equal(result.X.toarray(), expected.layers["counts"].toarray())
    assert set(result.obsm) == set(expected.obsm)
    np.testing.assert_array_equal(result.obsm["array_embedding"], expected.obsm["array_embedding"])
    pd.testing.assert_frame_equal(result.obsm["frame_embedding"], expected.obsm["frame_embedding"])


def test_selective_reader_falls_back_to_x(tmp_path):
    path = tmp_path / "rank_input_without_layer.h5ad"
    _write_rank_fixture(path, include_counts=False)
    keep = np.ones(4, dtype=bool)
    result = _read_rank_input(
        _RankInputRequest(
            sample_name="sample-b",
            h5ad_path=path,
            slice_id=0,
            keep_mask=pd.Series(keep),
            data_layer="counts",
        )
    )

    assert result.matrix_source == "X"
    np.testing.assert_array_equal(result.X.toarray(), ad.read_h5ad(path).X.toarray())


def test_selective_reader_rejects_wrong_mask_length(tmp_path):
    path = tmp_path / "rank_input.h5ad"
    _write_rank_fixture(path)

    with pytest.raises(ValueError, match=r"Filter mask.*expected \(4,\)"):
        _read_rank_input(
            _RankInputRequest(
                sample_name="sample-c",
                h5ad_path=path,
                slice_id=0,
                keep_mask=pd.Series([True, False]),
                data_layer="counts",
            )
        )


def test_prefetcher_loads_exactly_one_future_input():
    started = [threading.Event() for _ in range(3)]
    release = [threading.Event() for _ in range(3)]
    active = 0
    active_high_water = 0
    lock = threading.Lock()

    def loader(value):
        nonlocal active, active_high_water
        with lock:
            active += 1
            active_high_water = max(active_high_water, active)
        started[value].set()
        if value > 0:
            assert release[value].wait(2.0)
        with lock:
            active -= 1
        return value

    with _prefetched_rank_inputs([0, 1, 2], loader=loader) as prefetched:
        value, wait_seconds = next(prefetched)
        assert value == 0
        assert wait_seconds >= 0

        # Input 1 is loading while the caller processes input 0. Input 2 has
        # not been submitted, proving the future buffer is bounded at one.
        assert started[1].wait(2.0)
        assert not started[2].is_set()

        release[1].set()
        value, wait_seconds = next(prefetched)
        assert value == 1
        assert wait_seconds >= 0
        assert started[2].wait(2.0)

        release[2].set()
        value, wait_seconds = next(prefetched)
        assert value == 2
        assert wait_seconds >= 0
        with pytest.raises(StopIteration):
            next(prefetched)

    assert active_high_water == 1


def test_prefetcher_propagates_loader_failure():
    def loader(value):
        if value == 1:
            raise RuntimeError("selective read failed")
        return value

    with _prefetched_rank_inputs([0, 1], loader=loader) as prefetched:
        assert next(prefetched)[0] == 0
        with pytest.raises(RuntimeError, match="selective read failed"):
            next(prefetched)


def test_rank_calculator_runs_end_to_end_with_prefetched_inputs(tmp_path):
    genes = [f"g{index}" for index in range(25)]
    sample_paths = {}
    expected_counts = []
    for sample_index, sample_name in enumerate(("section-a", "section-b")):
        counts = np.zeros((3, len(genes)), dtype=np.float32)
        counts[:, :20] = (np.arange(60, dtype=np.float32).reshape(3, 20) + sample_index) % 7 + 1
        expected_counts.append(counts)
        adata = ad.AnnData(
            X=csr_matrix(counts),
            obs=pd.DataFrame(
                {"celltype": pd.Categorical(["type", "type", "type"])},
                index=[f"cell-{row}" for row in range(3)],
            ),
            var=pd.DataFrame(index=genes),
        )
        adata.layers["counts"] = csr_matrix(counts)
        adata.obsm["emb_cell"] = np.full((3, 2), sample_index + 1, dtype=np.float32)
        sample_path = tmp_path / f"{sample_name}.h5ad"
        adata.write_h5ad(sample_path)
        sample_paths[sample_name] = sample_path

    output_dir = tmp_path / "latent2gene"
    config = SimpleNamespace(
        latent_dir=tmp_path,
        latent2gene_dir=output_dir,
        concatenated_latent_adata_path=output_dir / "concatenated.h5ad",
        rank_memmap_path=output_dir / "ranks",
        mean_frac_path=output_dir / "mean_frac.parquet",
        sample_h5ad_dict=sample_paths,
        annotation="celltype",
        min_cells_per_type=1,
        rank_batch_size=2,
        rank_write_interval=1,
        mkscore_write_workers=1,
        memmap_tmp_dir=None,
    )

    outputs = RankCalculator(config).calculate_ranks_and_concatenate()
    rank_handle = outputs["rank_memmap_handle"]
    try:
        actual_ranks = np.asarray(rank_handle.memmap).copy()
    finally:
        rank_handle.close()

    all_counts = np.vstack(expected_counts)
    expected_ranks = np.log(rankdata(all_counts, method="average", axis=1)).astype(np.float16)
    np.testing.assert_array_equal(actual_ranks, expected_ranks)

    mean_frac = pd.read_parquet(config.mean_frac_path)
    np.testing.assert_allclose(
        mean_frac.loc[genes, "frac"].to_numpy(),
        np.count_nonzero(all_counts, axis=0) / all_counts.shape[0],
    )

    concatenated = ad.read_h5ad(config.concatenated_latent_adata_path)
    assert concatenated.shape == (6, 25)
    assert concatenated.obs["sample_name"].tolist() == [
        "section-a",
        "section-a",
        "section-a",
        "section-b",
        "section-b",
        "section-b",
    ]
    np.testing.assert_array_equal(
        concatenated.obsm["emb_cell"],
        np.vstack(
            [
                np.full((3, 2), 1, dtype=np.float32),
                np.full((3, 2), 2, dtype=np.float32),
            ]
        ),
    )

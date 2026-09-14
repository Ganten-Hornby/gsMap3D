"""GWAS readers bound memory/concurrency and preserve serial preparation semantics."""

import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gsMap.spatial_ldsc.data.loading import _read_sumstats
from gsMap.spatial_ldsc.execution.trait_loading import bounded_trait_map, resolve_gwas_workers


@pytest.mark.parametrize("value", [0, 17, True, 1.5, "2"])
def test_gwas_worker_limit_rejects_invalid_requests(value):
    with pytest.raises(ValueError, match="ldsc_gwas_workers"):
        resolve_gwas_workers(value, 127)


def test_worker_count_respects_affinity(monkeypatch):
    from gsMap.spatial_ldsc.execution import trait_loading

    monkeypatch.setattr(trait_loading, "available_cpu_count", lambda: 1)
    assert resolve_gwas_workers(4, 127) == 1


@pytest.mark.parametrize("workers", [1, 2, 4, 8, 16])
def test_only_bounded_work_is_submitted_and_order_is_preserved(workers):
    started, lock = [], threading.Lock()

    def prepare(index):
        with lock:
            started.append(index)
        return index

    results = bounded_trait_map(prepare, range(127), workers)
    assert next(results) == 0
    with lock:
        assert len(started) <= workers  # even when a completed future could prefetch more
    assert [0, *results] == list(range(127))
    assert not any(t.name.startswith("ldsc-gwas-reader") for t in threading.enumerate())


def test_reader_failure_and_early_close_join_workers():
    def fail(index):
        if index == 1:
            raise ValueError("bad GWAS")
        return index

    with pytest.raises(ValueError, match="bad GWAS"):
        list(bounded_trait_map(fail, range(127), 2))
    result = bounded_trait_map(lambda i: i, range(127), 2)
    next(result)
    result.close()
    assert not any(t.name.startswith("ldsc-gwas-reader") for t in threading.enumerate())


def test_unordered_readers_refill_without_waiting_for_slow_first_file():
    release = threading.Event()

    def prepare(index):
        if index == 0:
            assert release.wait(timeout=5)
        return index

    result = bounded_trait_map(prepare, range(4), 2, ordered=False)
    try:
        assert next(result) == 1
        assert next(result) == 2  # starts even though file zero remains blocked
        release.set()
        assert sorted(result) == [0, 3]
    finally:
        release.set()
        result.close()
    assert not any(t.name.startswith("ldsc-gwas-reader") for t in threading.enumerate())


@pytest.mark.parametrize(("n_traits", "n_ref"), [(1, 10000), (4, 127), (127, 10000)])
def test_union_indexing_preserves_sorted_union_and_each_traits_order(n_traits, n_ref):
    from gsMap.spatial_ldsc.execution.shared import _index_trait_union

    rng = np.random.default_rng(91)
    rows = [rng.permutation(n_ref)[: n_ref // 2] for _ in range(n_traits)]
    expected = np.unique(np.concatenate(rows))
    prepared = [{"positions": positions.copy()} for positions in rows]
    union = _index_trait_union(prepared, n_ref)
    np.testing.assert_array_equal(union, expected)
    for data, positions in zip(prepared, rows, strict=True):
        assert "positions" not in data
        assert data["indices"].dtype == np.int32
        np.testing.assert_array_equal(data["indices"], np.searchsorted(expected, positions))


def test_fp32_parse_preserves_conversion_duplicates_and_missing_values(tmp_path):
    path = tmp_path / "gwas.txt.gz"
    rng = np.random.default_rng(913)
    size = 2000
    data = pd.DataFrame(
        {
            "SNP": [f"rs{i % 1000}" for i in range(size)],
            "N": rng.uniform(100, 1000000, size),
            "Z": rng.normal(size=size),
            "A1": "A",
            "A2": "G",
            "unused": rng.random(size),
        }
    )
    data.loc[2, "Z"] = np.nan
    data.to_csv(path, sep="\t", index=False, na_rep=".")
    old = _read_sumstats(path).astype({"N": np.float32, "Z": np.float32})
    new = _read_sumstats(path, float_dtype=np.float32)
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    assert len(new) == 1000
    assert set(new.columns) == {"SNP", "N", "Z"}


def test_serial_and_parallel_full_trait_preparation_match(tmp_path, monkeypatch):
    import anndata as ad
    from scipy import sparse

    from gsMap.spatial_ldsc.execution import shared

    config = SimpleNamespace(
        project_name="p",
        sample_filter=None,
        cell_indices_range=None,
        n_blocks=3,
        chisq_max=None,
        ldsc_gwas_workers=1,
    )
    rng = np.random.default_rng(6)
    names = [f"rs{i}" for i in range(103)]
    ref = ad.AnnData(sparse.eye(103, format="csr"))
    ref.obs_names = names
    engine = SimpleNamespace(
        mode="weight_matrix",
        snp_gene_weight_adata=ref,
        config=config,
        spot_indices=np.arange(7),
        chunk_starts=[0],
        _cell_end=7,
        n_spots=7,
        output_dir=tmp_path,
        weight_gene_indices=np.arange(103),
        baseline_ld=pd.DataFrame(rng.uniform(1, 5, (103, 2)), index=names),
        w_ld=pd.DataFrame({"LD_weights": rng.uniform(1, 5, 103)}, index=names),
    )
    traits = []
    for i in range(4):
        path = tmp_path / f"trait{i}.sumstats.gz"
        selected = rng.permutation(103)[: 95 + i]
        pd.DataFrame(
            {
                "SNP": np.array(names)[selected],
                "N": rng.uniform(10000, 20000, len(selected)),
                "Z": rng.normal(size=len(selected)),
            }
        ).to_csv(path, sep="\t", index=False)
        traits.append((f"trait{i}", path))
    traits.insert(1, ("broken", tmp_path / "absent.gz"))
    reference = shared.prepare_shared_traits(engine, traits)
    config.ldsc_gwas_workers = 2
    candidate = shared.prepare_shared_traits(engine, traits)
    np.testing.assert_array_equal(candidate[0], reference[0])
    np.testing.assert_array_equal(candidate[1].toarray(), reference[1].toarray())
    assert len(candidate[2]) == 4
    assert candidate[3][0][0] == "broken"
    assert candidate[3][0][1].__traceback__ is None
    for actual, expected in zip(candidate[2], reference[2], strict=True):
        for key in expected:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]

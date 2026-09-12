"""Shared-LD kernels and streaming keep each trait's statistical design."""

from types import SimpleNamespace

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gsMap.spatial_ldsc.data.chunks import SpatialLDChunkSource
from gsMap.spatial_ldsc.execution import shared as multitrait
from gsMap.spatial_ldsc.kernels.jax_multitrait import (
    mean_regression,
    prepare_mean_design,
    shared_hsq,
)
from gsMap.spatial_ldsc.kernels.jax_regression import process_spatial_ldsc_batch


@pytest.mark.parametrize("cells", [3, 7])
@pytest.mark.parametrize("n_baseline", [2, 6])
def test_heterogeneous_traits_and_custom_baselines(cells, n_baseline):
    rng = np.random.default_rng(523)
    ld = rng.gamma(2, 3, size=(217, cells)).astype("float32")
    raw_base = rng.uniform(0.3, 5, size=(217, n_baseline)).astype("float32")
    wld = rng.uniform(1, 20, size=217).astype("float32")
    indices, sizes, sums, responses, traits, expected, expected_h = [], [], [], [], [], [], []
    for i, count in enumerate([200, 160, 130]):
        idx = rng.choice(217, count, replace=False).astype("int32")
        n = rng.uniform(50000, 200000, size=count).astype("float32")
        nbar = np.float32(n.mean() * 1.003)
        y = (
            rng.uniform(0.1, 0.7, size=count).astype("float32")
            if i == 0
            else rng.uniform(0.5, 3, size=count).astype("float32")
        )
        bs = raw_base[idx].sum(axis=1)
        base = np.concatenate(
            [raw_base[idx] * n[:, None] / nbar, np.ones((count, 1), dtype="float32")], axis=1
        )
        params = tuple(map(jnp.asarray, (bs, y, n, base, wld[idx], nbar)))
        traits.append((jnp.asarray(idx), params))
        expected.append(process_spatial_ldsc_batch(10, jnp.asarray(ld[idx]), *params))
        expected_h.append(
            np.clip(
                10000 * (y.mean() - 1) / np.mean((ld[idx] + bs[:, None]) * n[:, None], axis=0),
                0,
                1,
            )
        )
        indices.append(idx)
        sizes.append(n)
        sums.append(bs)
        responses.append(y)
    h = shared_hsq(jnp.asarray(ld), *prepare_mean_design(len(ld), indices, sizes, sums, responses))
    np.testing.assert_allclose(h, expected_h, rtol=3e-6, atol=2e-8)
    result = mean_regression(10, jnp.asarray(ld), tuple(traits), h)
    reference = np.asarray(jax.device_get(expected))
    np.testing.assert_allclose(result, reference, rtol=3e-3, atol=1e-8)
    assert np.isfinite(result).all()
    assert (np.asarray(result[:, 1]) > 0).all()


@pytest.mark.parametrize("idx", [[1, 1, 3], [3, 1, 3], [-1, 0, 1], [1, 2, 10], []])
def test_invalid_trait_mapping(idx):
    idx = np.asarray(idx, dtype=np.int32)
    vals = np.ones(len(idx), dtype=np.float32)
    with pytest.raises(ValueError, match="SNP mapping"):
        prepare_mean_design(10, [idx], [vals], [vals], [vals])


@pytest.mark.parametrize("order", ["sorted", "reverse", "shuffled"])
def test_design_preserves_mapping_order(order):
    idx = np.array([0, 3, 5, 8])
    if order == "reverse":
        idx = idx[::-1]
    elif order == "shuffled":
        idx = idx[[1, 3, 0, 2]]
    n = np.arange(1, 5, dtype=np.float32)
    design, counts, _, _ = prepare_mean_design(10, [idx], [n], [n], [n])
    np.testing.assert_array_equal(np.asarray(design)[0, idx], n)
    assert counts[0] == 4


def test_prepare_traits_preserves_order_nbar_and_skips_existing(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        project_name="p", sample_filter=None, cell_indices_range=(0, 3), n_blocks=2
    )
    ref = ad.AnnData(sp.eye(10, format="csr"))
    engine = SimpleNamespace(
        mode="weight_matrix",
        snp_gene_weight_adata=ref,
        config=cfg,
        spot_indices=np.arange(3),
        chunk_starts=[0],
        _cell_end=3,
        n_spots=3,
        output_dir=tmp_path,
        baseline_ld=None,
        w_ld=None,
        weight_gene_indices=np.arange(10),
    )
    (tmp_path / "p_done_cells_0_3.csv.gz").touch()
    positions = np.array([7, 1, 4, 3, 2])
    data = dict(
        snp_positions=positions,
        N=np.array([1, 2, 3, 4, 100], dtype=np.float32),
        Nbar=np.float32(22),
        chisq=np.ones(5),
        baseline_ld=pd.DataFrame(np.ones((5, 2))),
        baseline_ld_sum=np.full(5, 2),
        w_ld=np.ones(5),
    )

    def prepare(*args):
        if args[1] == "bad":
            raise ValueError("bad trait")
        if args[1] == "done":
            pytest.fail("An existing trait was re-read")
        return data, None

    monkeypatch.setattr(multitrait, "prepare_trait_data", prepare)
    union, sparse, traits, failures = multitrait.prepare_shared_traits(
        engine, [("done", ""), ("good", ""), ("bad", "")]
    )
    np.testing.assert_array_equal(union[traits[0]["indices"]], positions[:4])
    np.testing.assert_array_equal(sparse.toarray(), ref.X.toarray()[union])
    assert traits[0]["Nbar"] == 22
    assert len(traits[0]["N"]) == 4
    assert failures[0][0] == "bad"


@pytest.mark.parametrize("mode", ["weight_matrix", "direct_ldscore"])
def test_shared_stream_tail_order_and_reference(tmp_path, mode):
    rng = np.random.default_rng(56)
    marker = rng.uniform(0, 3, (7, 6)).astype(np.float32)
    sparse = sp.csr_matrix(rng.uniform(0, 2, (120, 6)).astype(np.float32))
    ld = sparse @ marker.T
    names = np.array([f"cell{i}" for i in range(7)])
    selected = np.array([0, 2, 3, 5, 6]) if mode == "weight_matrix" else np.arange(2, 7)
    source = SpatialLDChunkSource(
        mode,
        [0, 3],
        3,
        5,
        selected,
        names[selected],
        ldscore_adata=ad.AnnData(ld),
        ldscore_snp_positions=np.arange(120),
        marker_score_adata=SimpleNamespace(X=marker),
        marker_score_gene_indices=np.arange(6),
    )
    cfg = SimpleNamespace(
        project_name="p",
        sample_filter=None,
        cell_indices_range=None,
        n_blocks=4,
        ldsc_trait_batch_size=2,
    )
    traits = []
    for name, count in [("first", 120), ("second", 80), ("third", 60)]:
        idx = rng.choice(120, count, replace=False).astype(np.int32)
        n = rng.uniform(10000, 30000, count).astype(np.float32)
        raw = rng.uniform(0.5, 3, (count, 2)).astype(np.float32)
        traits.append(
            dict(
                name=name,
                indices=idx,
                N=n,
                Nbar=np.float32(n.mean()),
                chisq=rng.uniform(0.5, 3, count).astype(np.float32),
                baseline_sum=raw.sum(axis=1),
                baseline=np.c_[raw * n[:, None] / n.mean(), np.ones(count)].astype(np.float32),
                w_ld=np.ones(count, dtype=np.float32),
            )
        )
    paths = multitrait.run_shared_chunks(
        source, cfg, traits, sparse if mode == "weight_matrix" else None, tmp_path, 7
    )
    for data, path in zip(traits, paths, strict=True):
        frame = pd.read_csv(path)
        np.testing.assert_array_equal(frame.spot, names[selected])
        baseline, se = process_spatial_ldsc_batch(
            4,
            jnp.asarray(ld[data["indices"]][:, selected]),
            *[jnp.asarray(data[key]) for key in multitrait.PARAMETER_KEYS],
        )
        np.testing.assert_allclose(frame.beta, baseline, rtol=0.005, atol=1e-8)
        np.testing.assert_allclose(frame.se, se, rtol=0.005, atol=1e-8)
    assert not list(tmp_path.glob("*.partial*"))

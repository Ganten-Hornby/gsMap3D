"""Early SNP pruning must not change full-file QC or reference alignment."""

import bz2
import gzip
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gsMap.spatial_ldsc.data import loading
from gsMap.spatial_ldsc.data.sumstats import read_reference_sumstats


@pytest.mark.parametrize("suffix", [".txt", ".gz", ".bz2", ".concat.gz"])
def test_fast_reader_preserves_first_duplicates_global_qc_and_fp32(tmp_path, suffix):
    text = (
        b"SNP\tA1\tN\tZ\n"
        b"outside\tA\t1000000\t3\n"
        b"rs1\tC\t1000\t10\n"
        b"rs2\tT\t2000\t0.123456789\n"
        b"rs1\tG\t90000000\t100\n"
        b"rs3\tA\t.\t.\n"
        b"outside\tC\t999999999\t999\n"
        b"rs4\tA\t45678.123456789\t-1.23456789\n"
        b"rs5\tC\t12345\t2.3\n"
    )
    path = tmp_path / ("trait" + suffix)
    if suffix == ".concat.gz":
        split = text.index(b"rs1")
        payload = gzip.compress(text[:split]) + gzip.compress(text[split:])
    elif suffix == ".gz":
        payload = gzip.compress(text)
    elif suffix == ".bz2":
        payload = bz2.compress(text)
    else:
        payload = text
    path.write_bytes(payload)
    reference = pd.Index(["rs5", "rs3", "rs2", "absent", "rs1", "rs4", "rs2"])
    actual, nmax, gc = read_reference_sumstats(path, reference)
    old = loading._read_sumstats(path, float_dtype=np.float32)
    expected = old.set_index("SNP").loc[["rs5", "rs3", "rs2", "rs1", "rs4"]]
    pd.testing.assert_frame_equal(actual.set_index("SNP")[expected.columns], expected)
    assert nmax == old.N.max() == 1000000
    assert np.isnan(gc)
    assert np.isnan(np.median(old.Z**2))


def test_pruning_does_not_lower_automatic_chisq_threshold(tmp_path, monkeypatch):
    path = tmp_path / "trait.gz"
    pd.DataFrame(
        {
            "SNP": ["outside", "r1", "r2", "r3", "r4"],
            "N": [1000000, 1000, 2000, 3000, 4000],
            "Z": [3, 10, 2, 1, 2],
        }
    ).to_csv(path, sep="\t", index=False)
    reference = pd.Index(["r4", "r1", "r2", "r3"])
    baseline = pd.DataFrame({"base": [1.0, 2.0, 3.0, 4.0]}, index=reference)
    weights = pd.DataFrame({"LD_weights": [1.0, 2.0, 3.0, 4.0]}, index=reference)
    config = SimpleNamespace(chisq_max=None)
    ref = SimpleNamespace(obs_names=reference)
    actual, common = loading.prepare_trait_data(config, "trait", path, baseline, weights, ref)
    monkeypatch.setattr(loading, "read_reference_sumstats", lambda *a: None)
    expected, old_common = loading.prepare_trait_data(
        config, "trait", path, baseline, weights, ref
    )
    pd.testing.assert_index_equal(common, old_common)
    assert "r1" in common  # chi2=100 passes full-file cutoff=1000, not subset cutoff=80
    for key, value in expected.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(actual[key], value[actual[key].columns])
        else:
            np.testing.assert_equal(actual[key], value)


@pytest.mark.parametrize("separator", [" ", "  ", " \t "])
def test_nonstandard_whitespace_falls_back(tmp_path, separator):
    path = tmp_path / "trait.txt"
    path.write_text(separator.join(["SNP", "N", "Z"]) + "\nr1 100 1\n")
    assert read_reference_sumstats(path, pd.Index(["r1"])) is None
    assert len(loading._read_sumstats(path)) == 1


def test_header_only_and_no_reference_matches(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("SNP\tN\tZ\n")
    frame, nmax, gc = read_reference_sumstats(path, pd.Index(["r1"]))
    assert frame.empty
    assert np.isnan(nmax)
    assert np.isnan(gc)
    path.write_text("SNP\tN\tZ\nr1\t1000\t2\n")
    frame, nmax, gc = read_reference_sumstats(path, pd.Index(["r2"]))
    assert frame.empty
    assert nmax == 1000
    assert gc == np.float32(4) / 0.4559364


def test_malformed_tab_data_returns_to_existing_parser(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("SNP\tN\tZ\nr1\tnot_a_number\t2\n")
    assert read_reference_sumstats(path, pd.Index(["r1"])) is None
    with pytest.raises(ValueError, match="Improperly formatted"):
        loading._read_sumstats(path)

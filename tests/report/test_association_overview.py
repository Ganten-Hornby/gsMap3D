import json

import numpy as np
import pandas as pd
import pytest

from gsMap.report.association_overview import N_BINS, export_association_overview


def payload(js, entry):
    text = (js.parent / entry["file"]).read_text()
    return json.loads(text.split(f"[{json.dumps(entry['key'])}]=", 1)[1].rstrip(";\n"))


def test_overview_uses_all_rows_exact_counts_and_scope_specific_cauchy(tmp_path):
    data, js = tmp_path / "report_data", tmp_path / "web/js_data"
    data.mkdir()
    js.mkdir(parents=True)
    metadata = pd.DataFrame(
        {
            "spot": list("abcdef"),
            "sample_name": ["s1"] * 3 + ["s2"] * 3,
            "celltype": ["01", "01", "NA", "01", "NA", "2"],
            "IQ": [0.0, 2.0, np.nan, 10.0, 4.0, 10.0],
            "flat": [0.0] * 6,
        }
    )
    metadata.to_csv(data / "spot_metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "trait": "IQ",
                "annotation_name": "celltype",
                "annotation": "01",
                "type": "aggregated",
                "sample_name": "All Samples",
                "mlog10_p_cauchy": 5.0,
                "mlog10_p_median": 1.234,
                "top_95_quantile": 9.25,
                "total_spots": 3,
            },
            {
                "trait": "IQ",
                "annotation_name": "celltype",
                "annotation": "01",
                "type": "sample",
                "sample_name": "s1",
                "mlog10_p_cauchy": 3.0,
                "mlog10_p_median": 0.296709,
                "top_95_quantile": 1.9,
                "total_spots": 2,
            },
        ]
    ).to_csv(data / "cauchy_results.csv", index=False)
    index = export_association_overview(metadata, data, js, ["IQ", "flat"], ["celltype"])
    entries = index["traits"]["IQ"]["celltype"]
    all_data = payload(js, entries["all"])
    rows = {row["category"]: row for row in all_data["rows"]}
    assert all_data["n_spots"] == 6
    assert all_data["sampling"] == "all"
    assert rows["01"]["n_spots"] == rows["01"]["n_scored"] == 3
    assert rows["01"]["cauchy_logp"] == 5
    # Join Enrichment values exactly, not histogram estimates or median(logP).
    assert rows["01"]["median_logp"] == 1.234
    assert rows["01"]["q95_logp"] == 9.25
    assert rows["01"]["cauchy_n_spots"] == 3
    assert rows["NA"]["n_spots"] == 2
    assert rows["NA"]["n_scored"] == 1
    assert rows["NA"]["cauchy_logp"] is None
    assert rows["NA"]["median_logp"] is None
    assert rows["NA"]["q95_logp"] is None
    assert rows["2"]["histogram"][-1] == 1  # exact maximum is in the last bin
    s1, s2 = [payload(js, entries["samples"][sample]) for sample in ["s1", "s2"]]
    assert s1["x_max"] == s2["x_max"] == all_data["x_max"] == 10
    assert s1["rows"][0]["cauchy_logp"] == 3
    assert s1["rows"][0]["median_logp"] == 0.296709
    assert s1["rows"][0]["q95_logp"] == 1.9
    assert s1["rows"][0]["cauchy_n_spots"] == 2
    assert s2["rows"][0]["cauchy_logp"] is None  # never use pooled Cauchy for a section
    assert s2["rows"][0]["median_logp"] is None
    assert s2["rows"][0]["q95_logp"] is None
    for category, row in rows.items():
        combined = np.zeros(N_BINS, dtype=int)
        for section in [s1, s2]:
            for local in section["rows"]:
                if local["category"] == category:
                    combined += local["histogram"]
        np.testing.assert_array_equal(combined, row["histogram"])
        assert sum(row["histogram"]) == row["n_scored"]
    constant = payload(js, index["traits"]["flat"]["celltype"]["all"])
    assert constant["x_max"] == 1
    assert sum(row["histogram"][0] for row in constant["rows"]) == 6
    index_file = js / "association_overview_index.js"
    before = index_file.stat().st_mtime_ns
    assert export_association_overview(metadata, data, js, ["IQ", "flat"], ["celltype"]) == index
    assert index_file.stat().st_mtime_ns == before
    shard = js.parent / entries["all"]["file"]
    shard.write_text("broken")
    export_association_overview(metadata, data, js, ["IQ", "flat"], ["celltype"])
    assert payload(js, entries["all"]) == all_data


@pytest.mark.parametrize("values", [[-1.0], [np.inf]])
def test_overview_rejects_invalid_scores(tmp_path, values):
    frame = pd.DataFrame({"sample_name": ["s"], "celltype": [0], "trait": values})
    frame.to_csv(tmp_path / "spot_metadata.csv", index=False)
    with pytest.raises(ValueError, match="nonnegative"):
        export_association_overview(frame, tmp_path, tmp_path / "js", ["trait"], ["celltype"])


def test_overview_missing_annotation_and_scores_remain_visible(tmp_path):
    frame = pd.DataFrame(
        {"sample_name": ["s", "s"], "celltype": [None, 0], "trait": [np.nan, np.nan]}
    )
    frame.to_csv(tmp_path / "spot_metadata.csv", index=False)
    js = tmp_path / "web/js_data"
    index = export_association_overview(frame, tmp_path, js, ["trait"], ["celltype"])
    data = payload(js, index["traits"]["trait"]["celltype"]["all"])
    assert data["n_spots"] == 2
    assert "NaN" in {row["category"] for row in data["rows"]}
    assert all(row["n_scored"] == 0 for row in data["rows"])
    assert all(
        row[key] is None
        for row in data["rows"]
        for key in ["cauchy_logp", "median_logp", "q95_logp"]
    )

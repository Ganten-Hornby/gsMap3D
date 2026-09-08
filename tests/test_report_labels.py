import json
from types import SimpleNamespace

import pandas as pd
import pytest

from gsMap.report.report import _render_report_html
from gsMap.report.report_data import (
    ReportDataManager,
    _collect_cauchy_results,
    _export_cauchy_js,
    _export_umap_js,
    _read_cauchy_report_result,
    _read_report_csv,
)


def test_csv_preserves_numeric_and_na_like_labels(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"annotation": ["01", "1", "NA", "null"], "value": [1, 2, None, 4]}).to_csv(
        path, index=False
    )
    result = _read_report_csv(path, labels=["annotation"])
    assert result.annotation.tolist() == ["01", "1", "NA", "null"]
    assert pd.isna(result.value.iloc[2])


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf")])
def test_invalid_cauchy_probability_cannot_be_shown_as_significant(tmp_path, value):
    path = tmp_path / "invalid.csv"
    pd.DataFrame(
        {
            "annotation": ["01"],
            "p_cauchy": [value],
            "p_median": [0.1],
            "top_95_quantile": [1],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        _read_cauchy_report_result(path)


def test_sample_alias_is_normalized_before_combining_results(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "annotation": ["01"],
            "p_cauchy": [0],
            "p_median": [0.1],
            "top_95_quantile": [2],
            "sample": ["001"],
        }
    ).to_csv(path, index=False)
    result = _read_cauchy_report_result(path)
    assert result.sample_name.tolist() == ["001"]
    assert result.mlog10_p_cauchy.tolist() == [300]


def test_cauchy_export_preserves_labels_and_sample_identifiers(tmp_path):
    data = tmp_path / "data"
    js = tmp_path / "js_data"
    data.mkdir()
    labels = ["01", "1", "NA", "null"]
    pd.DataFrame(
        {
            "trait": ["01"] * 4,
            "annotation_name": ["domain"] * 4,
            "annotation": labels,
            "type": ["sample"] * 4,
            "sample_name": ["001"] * 4,
            "mlog10_p_cauchy": [1, 2, 3, 4],
            "mlog10_p_median": [1, 2, 3, 4],
            "top_95_quantile": [2, 3, 4, 5],
        }
    ).to_csv(data / "cauchy_results.csv", index=False)
    _export_cauchy_js(data, js)
    index = json.loads((js / "cauchy_index.js").read_text().split("=", 1)[1].rstrip(";"))
    entry = index["table"]["sample"]["01"]["domain"]
    payload = json.loads((tmp_path / entry["file"]).read_text().rsplit("=", 1)[1].rstrip(";"))
    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    assert frame.annotation.tolist() == labels
    assert frame.sample_name.tolist() == ["001"] * 4


def test_umap_numeric_annotations_remain_categorical(tmp_path):
    js = tmp_path / "js"
    pd.DataFrame(
        {
            "spot": ["01", "02"],
            "sample_name": ["001", "001"],
            "domain": ["01", "02"],
            "umap_cell_x": [0, 1],
            "umap_cell_y": [1, 0],
            "IQ": [3.0, 4.0],
        }
    ).to_csv(tmp_path / "umap_data.csv", index=False)
    _export_umap_js(tmp_path, js, {"annotations": ["domain"]})
    payload = json.loads((js / "umap_data.js").read_text().split("=", 1)[1].rstrip(";"))
    assert payload["domain"] == ["01", "02"]
    assert payload["spot"] == ["01", "02"]
    assert set(payload["color_maps"]["domain"]) == {"01", "02"}
    assert isinstance(payload["color_maps"]["IQ"], list)


def test_empty_cauchy_collection_exports_valid_empty_index(tmp_path):
    config = SimpleNamespace(
        annotation_list=["domain"],
        get_cauchy_result_file=lambda *args, **kwargs: tmp_path / "absent.csv",
    )
    _collect_cauchy_results(config, ["IQ"], tmp_path)
    _export_cauchy_js(tmp_path, tmp_path / "js")
    index = json.loads((tmp_path / "js/cauchy_index.js").read_text().split("=", 1)[1].rstrip(";"))
    assert index["table"] == index["heatmap"] == {"all": {}, "sample": {}}


def test_cauchy_cache_recollects_changed_source_and_reuses_unchanged_csv(tmp_path):
    source = tmp_path / "source.csv"

    def write(value):
        pd.DataFrame(
            {
                "annotation": ["01"],
                "p_cauchy": [value],
                "p_median": [value],
                "top_95_quantile": [3],
                "sample_name": ["001"],
            }
        ).to_csv(source, index=False)

    write(0.01)
    manager = ReportDataManager.__new__(ReportDataManager)
    manager.report_config = SimpleNamespace(
        annotation_list=["domain"], get_cauchy_result_file=lambda *args, **kwargs: source
    )
    manager.report_data_dir = tmp_path
    manager.js_data_dir = tmp_path / "js"
    manager.traits = ["IQ"]
    manager.force_re_run = False
    manager.collect_cauchy_results()
    cached = tmp_path / "cauchy_results.csv"
    modified = cached.stat().st_mtime_ns
    manager.collect_cauchy_results()
    assert cached.stat().st_mtime_ns == modified
    write(0.00001)
    manager.collect_cauchy_results()
    result = _read_report_csv(cached, labels=["annotation"])
    assert result.annotation.tolist() == ["01", "01"]
    assert result.mlog10_p_cauchy.tolist() == [5, 5]


def test_report_title_is_html_escaped():
    html = _render_report_html('test <img src=x onerror="alert(1)"> & demo')
    assert '<img src=x onerror="alert(1)">' not in html
    assert "&lt;img" in html
    assert "&amp; demo" in html


def test_missing_template_raises_instead_of_silent_success(monkeypatch):
    from pathlib import Path

    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.name == "template.html":
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(FileNotFoundError):
        _render_report_html("test")

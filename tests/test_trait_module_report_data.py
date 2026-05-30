import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gsMap.report.report_data import (
    _export_per_sample_spatial_js,
    _merge_trait_module_metadata,
)


def _report_config(tmp_path):
    project_name = "module_report"
    return SimpleNamespace(
        project_name=project_name,
        project_dir=tmp_path / project_name,
        trait_name_list=["IQ"],
    )


def test_merge_trait_module_metadata_adds_trait_specific_columns(tmp_path):
    config = _report_config(tmp_path)
    module_dir = config.project_dir / "trait_modules"
    module_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "spot": ["cell_1", "cell_3"],
            "trait_module": [2, 1],
            "module_specificity": [0.9234, 0.5123],
            "is_module_cell": [True, True],
        }
    ).to_csv(module_dir / "module_report_IQ.cell_modules.csv", index=False)

    metadata = pd.DataFrame(
        {
            "spot": ["cell_1", "cell_2", "cell_3"],
            "sample_name": ["sampleA", "sampleA", "sampleB"],
            "IQ": [8.1, 0.2, 6.4],
        }
    )

    merged = _merge_trait_module_metadata(metadata, config)

    assert merged["IQ_trait_module"].tolist() == [2, -1, 1]
    assert merged["IQ_module_specificity"].iloc[[0, 2]].tolist() == [0.923, 0.512]
    assert np.isnan(merged["IQ_module_specificity"].iloc[1])
    assert merged["IQ_is_module_cell"].tolist() == [True, False, True]


def test_export_per_sample_spatial_js_includes_trait_module_columns(tmp_path):
    data_dir = tmp_path / "report_data"
    js_dir = tmp_path / "js_data"
    data_dir.mkdir()
    js_dir.mkdir()

    pd.DataFrame(
        {
            "spot": ["cell_1", "cell_2"],
            "sample_name": ["sampleA", "sampleA"],
            "sx": [1.0, 2.0],
            "sy": [3.0, 4.0],
            "annotation": ["A", "B"],
            "IQ": [8.1, 0.2],
            "IQ_trait_module": [2, -1],
            "IQ_module_specificity": [0.9234, None],
            "IQ_is_module_cell": ["True", "False"],
        }
    ).to_csv(data_dir / "spot_metadata.csv", index=False)

    _export_per_sample_spatial_js(
        data_dir,
        js_dir,
        {
            "samples": ["sampleA"],
            "traits": ["IQ"],
            "annotations": ["annotation"],
            "downsampling_n_spots_2d": 10,
        },
    )

    js_text = (js_dir / "sample_sampleA_spatial.js").read_text()
    payload = js_text.split("=", 1)[1].rstrip(";")
    data = json.loads(payload)

    assert data["IQ_trait_module"] == [2, -1]
    assert data["IQ_module_specificity"] == [0.923, None]
    assert data["IQ_is_module_cell"] == [True, False]

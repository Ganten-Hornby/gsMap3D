from types import SimpleNamespace

import pandas as pd
import pytest

from gsMap.report import report_data


def test_prepare_3d_visualization_propagates_export_failure(monkeypatch, tmp_path):
    metadata = pd.DataFrame(
        {
            "spot": ["spot-1", "spot-2", "spot-3"],
            "sample_name": ["section-1", "section-1", "section-2"],
            "3d_x": [0.0, 1.0, 0.0],
            "3d_y": [0.0, 0.0, 1.0],
            "3d_z": [0.0, 0.0, 1.0],
            "ManualAnnotation": ["a", "b", "a"],
            "Height": [1.0, 2.0, 3.0],
        }
    )
    config = SimpleNamespace(
        annotation_list=["ManualAnnotation"],
        downsampling_n_spots_3d=3,
    )
    report_data_dir = tmp_path / "report_data"
    web_report_dir = tmp_path / "web_report"
    report_data_dir.mkdir()
    web_report_dir.mkdir()

    monkeypatch.setattr(report_data, "three_d_plot", lambda **kwargs: object())

    def fail_export(*args, **kwargs):
        raise ImportError("No module named 'nest_asyncio2'")

    monkeypatch.setattr(report_data, "three_d_plot_save", fail_export)

    with pytest.raises(ImportError, match="nest_asyncio2"):
        report_data._prepare_3d_visualization(
            config,
            metadata,
            ["Height"],
            report_data_dir,
            web_report_dir,
        )

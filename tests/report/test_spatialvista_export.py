import json
from types import SimpleNamespace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from gsMap.cli import app
from gsMap.export_spatialvista import export_spatialvista


@pytest.fixture
def project(tmp_path):
    project = tmp_path / "embryo"
    (project / "report_data").mkdir(parents=True)
    (project / "gsmap_web_report").mkdir()
    metadata = pd.DataFrame(
        {
            "spot": ["d|S2", "a|S1", "c|S2", "b|S1"],
            "sample_name": ["S2", "S1", "S2", "S1"],
            "annotation": ["01", "NA", "Heart", ""],
            "3d_x": [40.0, 10.0, 30.0, 20.0],
            "3d_y": [1.0, 2.0, 3.0, 4.0],
            "3d_z": [20.0, 10.0, 20.0, 10.0],
            "IQ": [4.0, 1.0, np.nan, 2.0],
            "AF": [1.0, 2.0, 3.0, 4.0],
            "a/b": [1.0, 2.0, 3.0, 4.0],
            "a_b": [4.0, 3.0, 2.0, 1.0],
        }
    )
    metadata.to_csv(project / "report_data/spot_metadata.csv", index=False)
    (project / "gsmap_web_report/report_meta.json").write_text(
        json.dumps(
            {
                "traits": ["IQ", "AF", "a/b", "a_b"],
                "annotations": ["annotation"],
                "samples": ["S1", "S2"],
                "is_3d": True,
            }
        )
    )
    (project / "report_data/spatial_3d_coordinate_info.json").write_text(
        json.dumps({"z_source": "ordinal_section_order", "z_is_physical": False})
    )
    return project


def test_export_preserves_rows_labels_values_and_geometry(project):
    output = export_spatialvista(project)
    result = ad.read_h5ad(output)
    assert result.X is None and result.shape == (4, 0)
    assert result.obs_names.tolist() == ["d|S2", "a|S1", "c|S2", "b|S1"]
    assert result.obs.annotation.tolist() == ["01", "NA", "Heart", "Unknown"]
    assert result.obs.sample_name.cat.categories.tolist() == ["S1", "S2"]
    np.testing.assert_equal(result.obsm["spatial"][:, 0], [40, 10, 30, 20])
    schema = result.uns["gsmap"]
    assert schema["schema_version"] == 1 and schema["value_scale"] == "neg_log10_p"
    assert schema["sampling"] == "all" and len(schema["traits"]) == 4
    key = next(key for key, label in schema["traits"].items() if label == "IQ")
    np.testing.assert_allclose(result.obs[key], [4, 1, np.nan, 2], equal_nan=True)
    assert all("/" not in key for key in schema["traits"])
    assert not json.loads(schema["coordinate_provenance_json"])["z_is_physical"]


def test_export_overwrite_guard_and_cleanup(project):
    output = export_spatialvista(project)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_spatialvista(project)
    assert output.read_bytes() == before
    export_spatialvista(project, overwrite=True)
    assert not list(project.glob(".embryo.*.h5ad"))


def test_trait_selection_and_deterministic_sampling(project, tmp_path):
    a = ad.read_h5ad(export_spatialvista(project, tmp_path / "a.h5ad", traits=["AF"], max_spots=2))
    b = ad.read_h5ad(export_spatialvista(project, tmp_path / "b.h5ad", traits=["AF"], max_spots=2))
    assert a.obs_names.tolist() == b.obs_names.tolist()
    assert list(a.uns["gsmap"]["traits"].values()) == ["AF"]
    assert a.uns["gsmap"]["source_n_spots"] == 4 and a.n_obs == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"traits": []},
        {"traits": ["missing"]},
        {"traits": ["IQ", "IQ"]},
        {"annotations": []},
        {"annotations": ["missing"]},
        {"max_spots": 0},
        {"max_spots": True},
        {"output": "bad.txt"},
    ],
)
def test_bad_options(project, kwargs):
    with pytest.raises(ValueError):
        export_spatialvista(project, **kwargs)


@pytest.mark.parametrize(
    "column,value", [("3d_x", np.inf), ("IQ", -1), ("IQ", np.inf), ("sample_name", "unknown")]
)
def test_bad_metadata(project, column, value):
    path = project / "report_data/spot_metadata.csv"
    data = pd.read_csv(path)
    data.loc[0, column] = value
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        export_spatialvista(project)
    assert not list(project.glob("*.h5ad"))


def test_duplicate_rows_rejected(project):
    path = project / "report_data/spot_metadata.csv"
    data = pd.read_csv(path)
    data.loc[0, "spot"] = data.loc[1, "spot"]
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        export_spatialvista(project)


def test_cli_export(project):
    result = CliRunner().invoke(app, ["export-spatialvista", str(project), "--trait", "IQ"])
    assert result.exit_code == 0, result.output
    assert "spatialvista --input" in result.output
    exported = ad.read_h5ad(project / "embryo.spatialvista.h5ad")
    assert list(exported.uns["gsmap"]["traits"].values()) == ["IQ"]


@pytest.mark.parametrize("values", [[True] * 4, [1 + 2j] * 4, ["1"] * 4])
def test_export_rejects_non_real_scores(project, values):
    metadata = pd.read_csv(project / "report_data/spot_metadata.csv")
    metadata["IQ"] = values
    with pytest.raises(ValueError, match="real numeric"):
        export_spatialvista(project, metadata=metadata)


def test_report_hook_exports_in_memory_metadata(project, monkeypatch):
    from gsMap.report import report

    config = SimpleNamespace(
        export_spatialvista=True,
        dataset_type="spatial3D",
        project_dir=project,
        project_name=project.name,
        report_data_dir=project / "report_data",
    )
    manager = Mock()
    manager.metadata = pd.read_csv(project / "report_data/spot_metadata.csv")
    # The manager starts with is_3d=False, before run() loads coordinates.
    manager.is_3d = False
    manager.run.return_value = project / "gsmap_web_report"
    monkeypatch.setattr(report, "ReportDataManager", Mock(return_value=manager))
    report.run_report.__wrapped__(config)
    manager.run.assert_called_once()
    exported = ad.read_h5ad(project / f"{project.name}.spatialvista.h5ad")
    assert exported.n_obs == 4


def test_non_3d_report_fails_before_preparation(monkeypatch):
    from gsMap.report import report

    manager = Mock()
    monkeypatch.setattr(report, "ReportDataManager", manager)
    with pytest.raises(ValueError, match="spatial3D"):
        report.run_report.__wrapped__(
            SimpleNamespace(export_spatialvista=True, dataset_type="spatial2D")
        )
    manager.assert_not_called()


def test_export_flag_requires_export_for_report_completion(project):
    from gsMap.config.quick_mode_config import get_report_missing_files

    config = SimpleNamespace(
        dataset_type="scRNA",
        project_dir=project,
        report_data_dir=project / "report_data",
        web_report_dir=project / "gsmap_web_report",
        trait_name_list=[],
        annotation_list=[],
        generate_multi_sample_plots=False,
        concatenated_latent_adata_path=None,
        export_spatialvista=True,
    )
    output = project / f"{project.name}.spatialvista.h5ad"
    assert output in get_report_missing_files(config)[0]
    export_spatialvista(project)
    assert output not in get_report_missing_files(config)[0]


def test_export_cli_help_and_quick_mode_flag():
    runner = CliRunner()
    result = runner.invoke(app, ["export-spatialvista", "--help"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["quick-mode", "--help"], env={"COLUMNS": "240"})
    assert result.exit_code == 0
    assert "--export-spatialvista" in result.output

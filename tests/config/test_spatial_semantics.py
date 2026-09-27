"""Spatial mode describes geometry; cross-section matching is optional."""

from collections import OrderedDict
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pytest
import torch

from gsMap.config import FindLatentRepresentationsConfig, LatentToGeneConfig, QuickModeConfig
from gsMap.config.spatial import (
    PROBE_ROWS,
    guard_spatial_contract,
    is_thick_tissue,
    probe_coordinates,
    spatial_coordinates,
    validate_spatial_config,
)
from gsMap.find_latent.gnn.gcn import GCN
from gsMap.find_latent.st_process import propagate_spatial_features
from gsMap.latent2gene.connectivity import find_spatial_neighbors_with_slices


def write_input(path, coords):
    data = ad.AnnData(np.ones((len(coords), 4), dtype=np.float32))
    data.obs["celltype"] = "typeA"
    data.obsm["spatial"] = np.asarray(coords)
    data.obsm["emb_cell"] = np.ones((len(coords), 3), dtype=np.float32)
    data.write_h5ad(path)
    return path


@pytest.fixture
def inputs(tmp_path):
    (tmp_path / "IQ.sumstats.gz").touch()
    rng = np.random.default_rng(14)
    return [
        write_input(
            tmp_path / f"s{i}.h5ad", np.column_stack((rng.random((12, 2)), np.full(12, i)))
        )
        for i in range(3)
    ]


@pytest.mark.parametrize(
    "cls", [FindLatentRepresentationsConfig, LatentToGeneConfig, QuickModeConfig]
)
@pytest.mark.parametrize(
    ("mode", "n_files", "adjacent", "volume"),
    [
        ("spatial2D", 1, 0, False),
        ("spatial2D", 3, 0, False),
        ("spatial3D", 1, 0, True),
        ("spatial3D", 3, 2, False),
    ],
)
def test_config_mode_matrix(tmp_path, inputs, monkeypatch, cls, mode, n_files, adjacent, volume):
    monkeypatch.setattr("gsMap.config.latent2gene_config.configure_jax_platform", lambda _: None)
    options = dict(
        workdir=tmp_path, project_name="test", h5ad_path=inputs[:n_files], dataset_type=mode
    )
    if cls is not FindLatentRepresentationsConfig:
        options["use_gpu"] = False
    if cls is QuickModeConfig:
        options["sumstats_config_dict"] = {"IQ": tmp_path / "IQ.sumstats.gz"}
        options["annotation"] = "celltype"
    config = cls(**options)
    assert is_thick_tissue(config) is volume
    if hasattr(config, "n_adjacent_slices"):
        assert config.n_adjacent_slices == adjacent
    if mode == "spatial2D":
        assert config._spatial_layout == "sections"


def test_inactive_cross_settings(tmp_path, inputs, monkeypatch, caplog):
    monkeypatch.setattr("gsMap.config.latent2gene_config.configure_jax_platform", lambda _: None)
    config = LatentToGeneConfig(
        workdir=tmp_path,
        project_name="test",
        h5ad_path=inputs,
        dataset_type="spatial3D",
        n_adjacent_slices=0,
        cross_slice_z_scale=-1,
        cross_slice_selection_mode="physical_window",
        cross_slice_max_distance=None,
        adjacent_slice_spatial_neighbors=1,
    )
    assert config.n_adjacent_slices == 0
    assert config.adjacent_slice_spatial_neighbors == 0
    assert not config.fix_cross_slice_homogenous_neighbors
    assert "full XYZ distances remain" in caplog.text


def test_thick_tissue_ignores_nondefault_cross_matching(tmp_path, inputs, caplog):
    config = QuickModeConfig(
        workdir=tmp_path,
        project_name="volume",
        h5ad_path=inputs[:1],
        annotation="celltype",
        sumstats_config_dict={"IQ": tmp_path / "IQ.sumstats.gz"},
        dataset_type="spatial3D",
        n_adjacent_slices=8,
        cross_slice_selection_mode="physical_window",
    )
    assert config.n_adjacent_slices == 0
    assert is_thick_tissue(config)
    assert "Thick tissue ignores" in caplog.text
    assert "cross_slice_selection_mode" in caplog.text


def test_early_3d_validation_precedes_jax(tmp_path, monkeypatch):
    path = write_input(tmp_path / "xy.h5ad", np.zeros((5, 2)))

    def forbidden(*_):
        raise AssertionError("JAX initialized before coordinate validation")

    monkeypatch.setattr("gsMap.config.latent2gene_config.configure_jax_platform", forbidden)
    with pytest.raises(ValueError, match="XYZ"):
        LatentToGeneConfig(
            workdir=tmp_path, project_name="invalid", h5ad_path=[path], dataset_type="spatial3D"
        )


@pytest.mark.parametrize(
    "cls", [FindLatentRepresentationsConfig, LatentToGeneConfig, QuickModeConfig]
)
def test_removed_axis_parameter_is_not_accepted(cls):
    assert "cross_slice_z_coordinate" not in cls.__dataclass_fields__
    with pytest.raises(TypeError, match="cross_slice_z_coordinate"):
        cls(workdir="unused", project_name="unused", cross_slice_z_coordinate=2)


@pytest.mark.parametrize("command", ["quick-mode", "find-latent", "latent-to-gene"])
def test_removed_axis_flag_is_not_advertised_or_accepted(command):
    from click import Context
    from typer.main import get_command
    from typer.testing import CliRunner

    from gsMap.cli import app

    command_object = get_command(app).commands[command]
    flags = [
        flag
        for option in command_object.get_params(Context(command_object))
        for flag in option.opts
    ]
    assert "--cross-slice-z-coordinate" not in flags
    result = CliRunner().invoke(app, [command, "--cross-slice-z-coordinate", "2"])
    assert result.exit_code != 0
    assert "No such option" in result.output


def section_coords(spans, z_offset=0.0):
    xyz = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    xyz *= spans
    xyz[:, 2] += z_offset
    return xyz


@pytest.mark.parametrize(
    "cls", [FindLatentRepresentationsConfig, LatentToGeneConfig, QuickModeConfig]
)
@pytest.mark.parametrize("adjacent", [0, 2])
@pytest.mark.parametrize(
    "spans", [(10, 20, 10), (20, 10, 10), (10, 20, 30), (0, 20, 0), (0, 0, 0)]
)
def test_bad_section_geometry_fails_early(tmp_path, monkeypatch, cls, adjacent, spans):
    def forbidden(*_):
        raise AssertionError("JAX initialized before coordinate validation")

    monkeypatch.setattr("gsMap.config.latent2gene_config.configure_jax_platform", forbidden)
    paths = [
        write_input(tmp_path / f"s{i}.h5ad", section_coords((10, 20, 0.2), i)) for i in range(2)
    ]
    bad = write_input(tmp_path / "bad_section.h5ad", section_coords(spans, 2))
    options = dict(
        workdir=tmp_path, project_name="test", h5ad_path=[*paths, bad], dataset_type="spatial3D"
    )
    if cls is not FindLatentRepresentationsConfig:
        options["n_adjacent_slices"] = adjacent
    if cls is QuickModeConfig:
        gwas = tmp_path / "IQ.sumstats.gz"
        gwas.touch()
        options.update(annotation="celltype", sumstats_config_dict={"IQ": gwas})
    with pytest.raises(ValueError, match="Z span must be strictly smaller") as error:
        cls(**options)
    assert str(bad) in str(error.value)
    assert "X=" in str(error.value)
    assert "Y=" in str(error.value)
    assert "Z=" in str(error.value)


@pytest.mark.parametrize("spans", [(10, 20, 0), (10, 20, 9.99), (20, 10, 9.99)])
def test_valid_section_spans_need_no_arbitrary_ratio_cutoff(tmp_path, spans):
    paths = [write_input(tmp_path / f"s{i}.h5ad", section_coords(spans, 50 * i)) for i in range(3)]
    config = FindLatentRepresentationsConfig(
        workdir=tmp_path, project_name="test", h5ad_path=paths, dataset_type="spatial3D"
    )
    assert not is_thick_tissue(config)


@pytest.mark.parametrize(("mode", "n_files"), [("spatial3D", 1), ("spatial2D", 3)])
def test_section_thickness_check_excludes_volume_and_2d(tmp_path, mode, n_files):
    paths = [
        write_input(tmp_path / f"s{i}.h5ad", section_coords((10, 20, 100), i))
        for i in range(n_files)
    ]
    config = FindLatentRepresentationsConfig(
        workdir=tmp_path, project_name="test", h5ad_path=paths, dataset_type=mode
    )
    assert is_thick_tissue(config) == (mode == "spatial3D")


def test_section_span_check_is_robust_to_an_isolated_outlier(tmp_path):
    coords = np.tile(section_coords((10, 20, 0.1)), (100, 1))
    coords[0, 2] = 10000
    paths = [write_input(tmp_path / f"s{i}.h5ad", coords + (0, 0, i)) for i in range(3)]
    FindLatentRepresentationsConfig(
        workdir=tmp_path, project_name="test", h5ad_path=paths, dataset_type="spatial3D"
    )


def test_probe_is_bounded_and_runtime_catches_unsampled_nan(tmp_path, monkeypatch):
    n = PROBE_ROWS * 3
    values = np.random.default_rng(4).random((n, 3))
    sampled = np.linspace(0, n - 1, PROBE_ROWS, dtype=np.int64)
    missing = np.setdiff1d(np.arange(n), sampled)[0]
    values[missing, 2] = np.nan
    path = write_input(tmp_path / "large.h5ad", values)
    getitem = h5py.Dataset.__getitem__
    reads = []

    def observed(dataset, selection):
        assert dataset.name != "/X", "Coordinate preflight must not read expression"
        result = getitem(dataset, selection)
        if dataset.name == "/obsm/spatial":
            reads.append(len(result))
            assert len(result) <= PROBE_ROWS
        return result

    monkeypatch.setattr(h5py.Dataset, "__getitem__", observed)
    assert probe_coordinates(path, "spatial", "spatial3D")["shape"] == (n, 3)
    assert reads == [PROBE_ROWS]
    with pytest.raises(ValueError, match="NaN"):
        spatial_coordinates(values, "spatial3D")
    # An unused Z is not a 2D validity condition.
    np.testing.assert_array_equal(spatial_coordinates(values, "spatial2D"), values[:, :2])


def test_no_adjacent_matching_skips_physical_requirements():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 5.0], [0.0, 0.0, 6.0], [1.0, 0.0, 6.0]])
    ids = np.array([0, 0, 1, 1])
    actual, mapping = find_spatial_neighbors_with_slices(
        coords,
        ids,
        k_central=2,
        k_adjacent=0,
        n_adjacent_slices=0,
        cross_slice_selection_mode="physical_window",
        cross_slice_max_distance=None,
        cross_slice_z_scale=-1,
    )
    assert actual.shape == (4, 2)
    assert np.all(ids[actual] == ids[:, None])
    assert set(mapping) == {0, 1}


def test_full_xyz_neighbors_differ_from_xy():
    xyz = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 10.0], [1.0, 0.0, 0.0]])

    def neighbors(mode):
        return find_spatial_neighbors_with_slices(
            spatial_coordinates(xyz, mode),
            np.zeros(3),
            k_central=2,
            k_adjacent=0,
            n_adjacent_slices=0,
        )[0]

    assert neighbors("spatial2D")[0].tolist() == [0, 1]
    assert neighbors("spatial3D")[0].tolist() == [0, 2]


def test_2d_xyz_embedding_graph_equals_xy():
    xyz = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 10.0], [1.0, 0.0, 0.0]])
    params = SimpleNamespace(
        dataset_type="spatial2D",
        graph_mode="symmetric_knn",
        graph_z_scale=1.0,
        graph_radius=None,
        graph_sigma=None,
        n_neighbors=1,
        K=1,
    )

    def propagate(coords):
        return propagate_spatial_features(torch.eye(3), coords, np.arange(3), params, GCN(K=1))

    first, first_graph = propagate(xyz)
    second, second_graph = propagate(xyz[:, :2])
    torch.testing.assert_close(first, second)
    np.testing.assert_equal(first_graph.edges, second_graph.edges)


def test_quick_reuses_probes_and_warns_once(tmp_path, inputs, monkeypatch, caplog):
    import gsMap.config.spatial as spatial

    original = spatial.probe_coordinates
    calls = []

    def counted(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(spatial, "probe_coordinates", counted)
    config = QuickModeConfig(
        workdir=tmp_path,
        project_name="quick",
        sumstats_config_dict={"IQ": tmp_path / "IQ.sumstats.gz"},
        h5ad_path=inputs,
        annotation="celltype",
    )
    child = config.find_latent_config
    assert len(calls) == 3
    assert child.dataset_type == "spatial2D"
    assert caplog.text.count("spatial2D received XYZ") == 1


def test_contract_refuses_changed_semantics_but_not_display_limits(tmp_path, inputs):
    config = QuickModeConfig(
        workdir=tmp_path,
        project_name="contract",
        sumstats_config_dict={"IQ": tmp_path / "IQ.sumstats.gz"},
        h5ad_path=inputs,
        annotation="celltype",
        dataset_type="spatial3D",
    )
    guard_spatial_contract(config, write=True)
    config.downsampling_n_spots_3d = 999
    guard_spatial_contract(config)
    config.n_adjacent_slices = 0
    with pytest.raises(ValueError, match="new project"):
        guard_spatial_contract(config)


def test_indistinguishable_and_nonmonotonic_order_only_warn(tmp_path, inputs, caplog):
    config = SimpleNamespace(
        dataset_type="spatial3D",
        spatial_key="spatial",
        sample_h5ad_dict=OrderedDict((str(i), inputs[i]) for i in [0, 2, 1]),
    )
    validate_spatial_config(config)
    assert "not monotonic" in caplog.text
    assert list(config.sample_h5ad_dict) == ["0", "2", "1"]


def test_explicit_section_graph_is_independent_and_scales_third_column():
    coords = np.random.default_rng(28).random((12, 3))
    labels = np.repeat(np.arange(3), 4)
    coords[:, 2] = labels * 0.2
    params = SimpleNamespace(
        dataset_type="spatial3D",
        graph_mode="section_aware_3d",
        graph_z_scale=2.0,
        n_adjacent_slices=0,
        n_neighbors=2,
        graph_radius=None,
        graph_sigma=None,
        K=1,
    )
    expression = torch.eye(12)

    def graph(values):
        return propagate_spatial_features(
            expression,
            values,
            np.arange(12),
            params,
            GCN(K=1),
            section_labels=labels,
        )

    expected, before = graph(coords)
    assert np.any(labels[before.edges[:, 0]] != labels[before.edges[:, 1]])
    scaled = coords.copy()
    scaled[:, 2] *= params.graph_z_scale
    params.graph_z_scale = 1.0
    observed, after = graph(scaled)
    np.testing.assert_array_equal(before.edges, after.edges)
    torch.testing.assert_close(expected, observed)


def test_find_latent_cli_supports_volume_mode(tmp_path, inputs, monkeypatch):
    from typer.testing import CliRunner

    from gsMap.cli import app

    seen = []
    monkeypatch.setattr(
        "gsMap.find_latent.run_find_latent_representation", lambda config: seen.append(config)
    )
    result = CliRunner().invoke(
        app,
        [
            "find-latent",
            "--workdir",
            str(tmp_path),
            "--project-name",
            "cli",
            "--h5ad-path",
            str(inputs[0]),
            "--dataset-type",
            "spatial3D",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(seen) == 1
    assert is_thick_tissue(seen[0])


def test_contract_rejects_unverified_artifacts_without_overwriting(tmp_path, inputs):
    config = FindLatentRepresentationsConfig(
        workdir=tmp_path, project_name="unknown", h5ad_path=inputs
    )
    config.model_path.write_bytes(b"unverified-checkpoint")
    with pytest.raises(ValueError, match="no spatial computation contract"):
        guard_spatial_contract(config, write=True)
    assert config.model_path.read_bytes() == b"unverified-checkpoint"
    assert not (config.project_dir / "spatial_contract.json").exists()


@pytest.mark.parametrize("corruption", ["rows", "dtype", "empty", "missing"])
def test_invalid_h5ad_geometry_metadata(tmp_path, corruption):
    path = write_input(tmp_path / "invalid.h5ad", np.ones((5, 3)))
    with h5py.File(path, "r+") as handle:
        del handle["obsm/spatial"]
        if corruption == "rows":
            handle["obsm/spatial"] = np.ones((4, 3))
        elif corruption == "dtype":
            handle["obsm/spatial"] = np.full((5, 3), b"a")
        elif corruption == "empty":
            handle["obsm/spatial"] = np.ones((0, 3))
    with pytest.raises(ValueError):
        probe_coordinates(path, "spatial", "spatial3D")

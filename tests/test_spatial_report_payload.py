import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from gsMap.config.latent2gene_config import DatasetType
from gsMap.report.report_data import (
    _apply_section_plane_coordinates,
    _derive_section_plane_coordinates,
    _load_coordinates,
    _spatial_coordinate_info_is_current,
)
from gsMap.report.spatial_payload import (
    SPATIAL_PAYLOAD_VERSION,
    _point_size_2d,
    _stratified_positions,
    _top_trait_positions,
    decode_typed_array,
    encode_nonnegative_quantized_array,
    encode_typed_array,
    export_sample_spatial_payloads,
    export_spatial_3d_payloads,
)


def _assignment_payload(path):
    content = path.read_text(encoding="utf-8")
    assignment = content.split("]=", 1)[1] if "]=" in content else content.split("=", 1)[1]
    return json.loads(assignment.rstrip(";"))


def _metadata(n_rows=20):
    return pd.DataFrame(
        {
            "spot": [f"spot_{index}" for index in range(n_rows)],
            "sample_name": [
                "sample_a" if index < n_rows // 2 else "sample_b" for index in range(n_rows)
            ],
            "sx": np.arange(n_rows, dtype=np.float32),
            "sy": np.arange(n_rows, dtype=np.float32) * 2,
            "3d_x": np.arange(n_rows, dtype=np.float32),
            "3d_y": np.arange(n_rows, dtype=np.float32) * 2,
            "3d_z": np.repeat([0.0, 10.0], [n_rows // 2, n_rows - n_rows // 2]),
            "region": [
                "rare" if index in (0, n_rows - 1) else "common" for index in range(n_rows)
            ],
            "trait": np.arange(n_rows, dtype=np.float32),
        }
    )


def test_typed_array_payload_round_trip():
    values = np.array([1.25, np.nan, -3.5], dtype=np.float64)
    payload = encode_typed_array(values)

    result = decode_typed_array(payload)

    assert payload["dtype"] == "float32"
    assert result.dtype == np.dtype("<f4")
    np.testing.assert_allclose(result[[0, 2]], [1.25, -3.5])
    assert np.isnan(result[1])


def test_quantized_trait_payload_round_trip():
    values = np.array([0.04, 1.26, np.nan, 7000.0, np.inf, -np.inf], dtype=np.float32)

    payload = encode_nonnegative_quantized_array(values)
    result = decode_typed_array(payload)

    assert payload["dtype"] == "uint16"
    assert payload["scale"] == 0.1
    np.testing.assert_allclose(result[:2], [0.0, 1.3], atol=1e-6)
    assert np.isnan(result[2])
    assert result[3] == pytest.approx(6553.4, abs=1e-3)
    assert result[4] == pytest.approx(6553.4, abs=1e-3)
    assert result[5] == 0


def test_stratified_sampling_is_exact_and_reproducible():
    metadata = _metadata(20)

    first = _stratified_positions(metadata, 8, ["region"])
    second = _stratified_positions(metadata, 8, ["region"])

    assert len(first) == 8
    np.testing.assert_array_equal(first, second)
    assert set(metadata.iloc[first]["sample_name"]) == {"sample_a", "sample_b"}
    assert "rare" in set(metadata.iloc[first]["region"])


def test_large_stratified_sampling_uses_bounded_reproducible_candidates():
    n_rows = 300_000
    metadata = pd.DataFrame(
        {
            "sample_name": pd.Categorical(
                np.asarray([f"sample_{index % 30}" for index in range(n_rows)])
            ),
            "region": pd.Categorical(
                np.asarray([f"region_{index % 12}" for index in range(n_rows)])
            ),
        }
    )

    first = _stratified_positions(metadata, 10_000, ["region"])
    second = _stratified_positions(metadata, 10_000, ["region"])

    assert len(first) == 10_000
    np.testing.assert_array_equal(first, second)
    assert metadata.iloc[first]["sample_name"].nunique() == 30
    assert metadata.iloc[first]["region"].nunique() == 12


def test_large_stratified_sampling_preserves_rare_sample_and_annotation():
    n_rows = 300_000
    metadata = pd.DataFrame(
        {
            "sample_name": pd.Categorical(["common"] * (n_rows - 1) + ["singleton"]),
            "region": pd.Categorical(["common"] * (n_rows - 2) + ["rare", "common"]),
        }
    )

    positions = _stratified_positions(metadata, 10_000, ["region"])

    assert set(metadata.iloc[positions]["sample_name"]) == {"common", "singleton"}
    assert set(metadata.iloc[positions]["region"]) == {"common", "rare"}


def test_top_trait_positions_include_display_sample():
    values = np.arange(10, dtype=np.float32)

    result = _top_trait_positions(values, 3)

    assert result.tolist() == [9, 8, 7]


def test_2d_point_size_has_visible_density_aware_bounds():
    dense_x, dense_y = np.meshgrid(
        np.linspace(0, 100, 400, dtype=np.float32),
        np.linspace(0, 100, 400, dtype=np.float32),
    )
    sparse_x, sparse_y = np.meshgrid(
        np.linspace(0, 100, 200, dtype=np.float32),
        np.linspace(0, 100, 200, dtype=np.float32),
    )
    dense_grid = np.column_stack((dense_x.ravel(), dense_y.ravel()))
    sparse_grid = np.column_stack((sparse_x.ravel(), sparse_y.ravel()))

    dense_size = _point_size_2d(dense_grid)
    sparse_size = _point_size_2d(sparse_grid)

    assert 1.25 <= dense_size <= 6.0
    assert dense_size < sparse_size <= 6.0


def test_spatial_3d_export_uses_shared_geometry_and_lazy_features(tmp_path):
    metadata = _metadata(20)
    report_data = tmp_path / "report_data"
    web_report = tmp_path / "web"
    report_data.mkdir()
    metadata.to_csv(report_data / "spot_metadata.csv", index=False)

    viewer = export_spatial_3d_payloads(
        metadata,
        traits=["trait"],
        annotations=["region"],
        report_data_dir=report_data,
        web_report_dir=web_report,
        max_points=8,
        extreme_points_per_trait=3,
    )

    assert viewer == {
        "schema_version": SPATIAL_PAYLOAD_VERSION,
        "index_file": "js_data/spatial_3d/spatial_3d_index.js",
        "n_points": 8,
        "n_points_original": 20,
        "downsampled": True,
        "point_size": 4.0,
        "coordinate_info": None,
    }
    base = _assignment_payload(web_report / "js_data/spatial_3d/spatial_3d_base.js")
    trait = _assignment_payload(web_report / "js_data/spatial_3d/trait_trait.js")
    index = _assignment_payload(web_report / "js_data/spatial_3d/spatial_3d_index.js")
    region = _assignment_payload(web_report / "js_data/spatial_3d/annotation_region.js")
    assert len(decode_typed_array(base["x"])) == 8
    assert base["coordinate_ranges"] == [19.0, 38.0, 10.0]
    assert len(decode_typed_array(trait["values"])) == 8
    assert trait["n_top"] == 3
    assert trait["q95"] == pytest.approx(18.05)
    assert trait["cmax"] == pytest.approx(18.981)
    assert set(trait["top"]["annotation_codes"]) == {"region"}
    top_region_codes = decode_typed_array(trait["top"]["annotation_codes"]["region"])
    assert len(top_region_codes) == trait["n_top"]
    top_positions = decode_typed_array(trait["top"]["x"]).astype(int)
    assert top_positions.tolist() == [19, 18, 17]
    assert [region["categories"][code] for code in top_region_codes] == [
        metadata.loc[position, "region"] for position in top_positions
    ]
    assert index["trait_top_annotation_codes"] == ["region"]
    provenance = pd.read_parquet(report_data / "spatial_3d/spatial_3d_display_spots.parquet")
    assert len(provenance) == 8


def test_sample_export_splits_base_and_traits(tmp_path):
    metadata = _metadata(20)
    coordinate_info = {
        "section_planes": {
            "sample_a": {
                "axis_titles": ["Dorsal-ventral", "Medial-lateral"],
                "cut_axis_title": "Anterior-posterior",
            }
        }
    }

    index, outputs = export_sample_spatial_payloads(
        metadata,
        samples=["sample_a", "sample_b"],
        traits=["trait"],
        annotations=["region"],
        js_data_dir=tmp_path,
        max_points=6,
        extreme_points_per_trait=2,
        coordinate_info=coordinate_info,
    )

    assert index["sample_a"]["n_spots"] == 6
    assert index["sample_a"]["n_spots_original"] == 10
    assert index["sample_a"]["base_file"] == "spatial_2d/sample_sample_a_base.js"
    assert index["sample_a"]["trait_files"]["trait"]["file"].endswith(
        "sample_sample_a_trait_trait.js"
    )
    assert len(outputs) == 4
    base = _assignment_payload(tmp_path / index["sample_a"]["base_file"])
    assert len(decode_typed_array(base["sx"])) == 6
    assert base["coordinate_ranges"] == [9.0, 18.0]
    assert base["axis_titles"] == ["Dorsal-ventral", "Medial-lateral"]
    assert base["cut_axis_title"] == "Anterior-posterior"
    assert len(base["spot"]) == 6
    sample_a_trait = _assignment_payload(
        tmp_path / index["sample_a"]["trait_files"]["trait"]["file"]
    )
    sample_b_trait = _assignment_payload(
        tmp_path / index["sample_b"]["trait_files"]["trait"]["file"]
    )
    assert sample_a_trait["cmax"] == sample_b_trait["cmax"] == pytest.approx(18.981)
    assert sample_a_trait["q95"] == sample_b_trait["q95"] == pytest.approx(18.05)
    assert sample_a_trait["scale_scope"] == "all_samples"
    assert sample_a_trait["values"]["dtype"] == "uint16"
    assert sample_a_trait["values"]["scale"] == 0.1
    assert sample_a_trait["n_top"] == 2
    assert set(sample_a_trait["top"]["annotation_codes"]) == {"region"}
    assert decode_typed_array(sample_a_trait["top"]["x"]).tolist() == [9.0, 8.0]
    assert decode_typed_array(sample_a_trait["top"]["values"]).tolist() == pytest.approx(
        [9.0, 8.0]
    )
    region_codes = decode_typed_array(sample_a_trait["top"]["annotation_codes"]["region"])
    region_categories = base["annotations"]["region"]["categories"]
    assert [region_categories[code] for code in region_codes] == [
        metadata.loc[position, "region"] for position in [9, 8]
    ]


def test_2d_sections_use_explicitly_ordinal_nonphysical_z(tmp_path):
    path = tmp_path / "sections.h5ad"
    adata = ad.AnnData(np.ones((4, 1), dtype=np.float32))
    adata.obs["sample_name"] = ["slice_b", "slice_b", "slice_a", "slice_a"]
    adata.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 2.0], [10.0, 10.0], [11.0, 12.0]])
    adata.write_h5ad(path)
    config = SimpleNamespace(
        concatenated_latent_adata_path=path,
        spatial_key="spatial",
        dataset_type=DatasetType.SPATIAL_3D,
        sample_h5ad_dict={"slice_b": path, "slice_a": path},
        cross_slice_z_scale=9.0,
    )

    coordinates, is_3d, z_axis = _load_coordinates(config)

    assert is_3d
    assert z_axis == 2
    assert coordinates.groupby("sample_name", observed=True)["3d_z"].first().to_dict() == {
        "slice_a": 1.0,
        "slice_b": 0.0,
    }
    info = coordinates.attrs["spatial_3d_coordinate_info"]
    assert info["z_source"] == "ordinal_section_order"
    assert info["z_is_physical"] is False
    assert info["default_aspect_mode"] == "expanded"
    assert "not represented" in info["note"]


def test_registered_z_is_detected_reordered_and_scaled(tmp_path):
    path = tmp_path / "volume.h5ad"
    adata = ad.AnnData(np.ones((4, 1), dtype=np.float32))
    adata.obs["sample_name"] = ["slice_a", "slice_a", "slice_b", "slice_b"]
    adata.obsm["spatial"] = np.array(
        [
            [0.0, 0.0, 99.0, 0.0],
            [1.0, 2.0, 99.0, 0.0],
            [10.0, 10.0, 99.0, 5.0],
            [11.0, 12.0, 99.0, 5.0],
        ]
    )
    adata.write_h5ad(path)
    config = SimpleNamespace(
        concatenated_latent_adata_path=path,
        spatial_key="spatial",
        dataset_type=DatasetType.SPATIAL_3D,
        sample_h5ad_dict={"slice_a": path, "slice_b": path},
        cross_slice_z_coordinate=3,
        cross_slice_z_scale=4.0,
    )

    coordinates, is_3d, z_axis = _load_coordinates(config)

    assert is_3d
    assert z_axis == 3
    assert coordinates["3d_z"].tolist() == [0.0, 0.0, 20.0, 20.0]
    info = coordinates.attrs["spatial_3d_coordinate_info"]
    assert info["z_source"] == "registered_coordinate"
    assert info["z_scale"] == 4.0
    assert info["z_is_physical"] is True
    assert info["default_aspect_mode"] == "physical"


def test_physical_2d_sections_drop_their_sample_specific_cut_normal_axis(tmp_path):
    path = tmp_path / "mixed_section_orientations.h5ad"
    adata = ad.AnnData(np.ones((8, 1), dtype=np.float32))
    adata.obs["sample_name"] = ["coronal"] * 4 + ["sagittal"] * 4
    adata.obsm["spatial"] = np.array(
        [
            [5.00, 0.0, 0.0],
            [5.10, 0.0, 20.0],
            [5.20, 10.0, 0.0],
            [5.30, 10.0, 20.0],
            [0.0, 0.0, 7.00],
            [20.0, 0.0, 7.10],
            [0.0, 10.0, 7.20],
            [20.0, 10.0, 7.30],
        ]
    )
    adata.write_h5ad(path)
    config = SimpleNamespace(
        concatenated_latent_adata_path=path,
        spatial_key="spatial",
        dataset_type=DatasetType.SPATIAL_3D,
        sample_h5ad_dict={"coronal": path, "sagittal": path},
        cross_slice_z_coordinate=2,
        cross_slice_z_scale=1.0,
    )

    coordinates, is_3d, z_axis = _load_coordinates(config)

    assert is_3d
    assert z_axis == 2
    coronal = coordinates[coordinates["sample_name"] == "coronal"]
    sagittal = coordinates[coordinates["sample_name"] == "sagittal"]
    assert coronal["sx"].tolist() == [0.0, 0.0, 10.0, 10.0]
    assert coronal["sy"].tolist() == [0.0, 20.0, 0.0, 20.0]
    assert sagittal["sx"].tolist() == [0.0, 20.0, 0.0, 20.0]
    assert sagittal["sy"].tolist() == [0.0, 0.0, 10.0, 10.0]
    planes = coordinates.attrs["spatial_3d_coordinate_info"]["section_planes"]
    assert planes["coronal"]["cut_axis_title"] == "X"
    assert planes["coronal"]["axis_titles"] == ["Y", "Z"]
    assert planes["sagittal"]["cut_axis_title"] == "Z"
    assert planes["sagittal"]["axis_titles"] == ["X", "Y"]


def test_section_plane_schema_change_does_not_invalidate_coordinate_storage(tmp_path):
    path = tmp_path / "spatial_3d_coordinate_info.json"
    stored = {
        "schema_version": 2,
        "z_source": "registered_coordinate",
        "z_axis_input": 2,
        "z_scale": 1.0,
        "z_is_physical": True,
    }
    path.write_text(json.dumps(stored))
    current = {
        **stored,
        "schema_version": 3,
        "section_plane_method": "per_sample_1st_to_99th_percentile_span",
        "section_planes": {"section": {"x_axis": 1, "y_axis": 2, "cut_axis": 0}},
    }

    assert _spatial_coordinate_info_is_current(path, current)
    assert not _spatial_coordinate_info_is_current(path, {**current, "z_scale": 2.0})


def test_cached_metadata_section_plane_is_corrected_in_memory():
    metadata = pd.DataFrame(
        {
            "sample_name": ["coronal", "coronal", "sagittal", "sagittal"],
            "3d_x": [5.0, 5.1, 0.0, 20.0],
            "3d_y": [0.0, 10.0, 0.0, 10.0],
            "3d_z": [0.0, 20.0, 7.0, 7.1],
            "sx": [5.0, 5.1, 0.0, 20.0],
            "sy": [0.0, 10.0, 0.0, 10.0],
        }
    )
    coordinate_info = {
        "section_planes": {
            "coronal": {"x_axis": 1, "y_axis": 2, "cut_axis": 0},
            "sagittal": {"x_axis": 0, "y_axis": 1, "cut_axis": 2},
        }
    }

    _apply_section_plane_coordinates(metadata, coordinate_info)

    assert metadata["sx"].tolist() == [0.0, 10.0, 0.0, 20.0]
    assert metadata["sy"].tolist() == [0.0, 20.0, 0.0, 10.0]


def test_volumetric_sample_without_thin_axis_retains_xy_projection():
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [10.0, 9.0, 0.0],
            [0.0, 0.0, 8.0],
            [10.0, 0.0, 8.0],
            [0.0, 9.0, 8.0],
            [10.0, 9.0, 8.0],
        ]
    )

    section_x, section_y, planes = _derive_section_plane_coordinates(
        coordinates,
        pd.Series(["volume"] * len(coordinates)),
        ["X", "Y", "Z"],
    )

    np.testing.assert_array_equal(section_x, coordinates[:, 0])
    np.testing.assert_array_equal(section_y, coordinates[:, 1])
    assert planes["volume"]["cut_axis_detected"] is False
    assert planes["volume"]["axis_titles"] == ["X", "Y"]

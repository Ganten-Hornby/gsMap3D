"""Graph controls and spatial defaults agree across CLI, configs, and search."""

from dataclasses import fields
from inspect import signature
from types import SimpleNamespace

import numpy as np
from typer.testing import CliRunner

from gsMap.cli import app
from gsMap.config import LatentToGeneConfig, QuickModeConfig
from gsMap.latent2gene.connectivity import find_spatial_neighbors_with_slices


def test_ordinal_default_is_two_in_both_configs_and_search():
    for config in (LatentToGeneConfig, QuickModeConfig):
        assert next(f for f in fields(config) if f.name == "n_adjacent_slices").default == 2
    assert (
        signature(find_spatial_neighbors_with_slices).parameters["n_adjacent_slices"].default == 2
    )


def test_default_ordinal_search_includes_four_external_sections_at_boundary():
    coords = np.column_stack((np.arange(5, dtype=float), np.zeros(5), np.arange(5)))
    neighbors, _ = find_spatial_neighbors_with_slices(
        coords, slice_ids=np.arange(5), k_central=1, k_adjacent=1
    )
    assert neighbors.shape == (5, 5)
    assert set(neighbors[0, 1:]) == {1, 2, 3, 4}


def test_spatial2d_still_disables_cross_section_matching():
    config = SimpleNamespace(
        n_adjacent_slices=2,
        adjacent_slice_spatial_neighbors=200,
        latent_representation_niche="emb_niche",
        homogeneous_neighbors=20,
        spatial_neighbors=300,
    )
    LatentToGeneConfig._configure_spatial_2d(config)
    assert config.n_adjacent_slices == 0
    assert config.adjacent_slice_spatial_neighbors == 0


def test_quick_mode_exposes_complete_radius_controls():
    result = CliRunner().invoke(app, ["quick-mode", "--help"], env={"COLUMNS": "240"})
    assert result.exit_code == 0
    for option in ("--n-neighbors", "--graph-radius", "--graph-sigma", "--n-adjacent-slices"):
        assert option in result.output

"""Stable GWAS IDs, optional display labels, and old YAML compatibility."""

from dataclasses import fields
from pathlib import Path

import pytest

from gsMap.config.gwas import load_gwas_mapping, normalize_gwas_mapping
from gsMap.config.spatial_ldsc_config import GWASSumstatsConfig


def test_mixed_yaml_preserves_ids_and_resolves_paths(tmp_path):
    stats = tmp_path / "GWAS with spaces.sumstats.gz"
    stats.touch()
    config_file = tmp_path / "gwas.yaml"
    config_file.write_text(
        'IQ:\n  path: GWAS with spaces.sumstats.gz\n  label: "Intelligence — cognition"\n'
        '"Type 2 diabetes": GWAS with spaces.sumstats.gz\n'
    )
    config = GWASSumstatsConfig(sumstats_config_file=config_file)
    assert config.trait_name_list == ["IQ", "Type 2 diabetes"]
    assert config.sumstats_config_dict == dict.fromkeys(config.trait_name_list, stats)
    assert config.trait_label("IQ") == "Intelligence — cognition"
    assert config.trait_label("Type 2 diabetes") == "Type 2 diabetes"
    # QuickModeConfig passes init fields into child configurations this way.
    child = GWASSumstatsConfig(
        **{f.name: getattr(config, f.name) for f in fields(config) if f.init}
    )
    assert child.trait_labels == config.trait_labels
    assert all(isinstance(path, Path) for path in child.sumstats_config_dict.values())


def test_python_api_and_single_trait_labels(tmp_path):
    stats = tmp_path / "IQ.sumstats.gz"
    stats.touch()
    config = GWASSumstatsConfig(sumstats_config_dict={"IQ": {"path": stats, "label": "Cognition"}})
    assert config.trait_labels == {"IQ": "Cognition"}
    single = GWASSumstatsConfig(
        trait_name="IQ", sumstats_file=stats, trait_labels={"IQ": "Cognition"}
    )
    assert single.trait_labels == config.trait_labels


@pytest.mark.parametrize(
    "contents",
    [
        "IQ: a\nIQ: b\n",
        "IQ:\n  path: a\n  path: b\n",
        "123: a\n",
    ],
)
def test_duplicate_and_non_string_yaml_keys_are_rejected(tmp_path, contents):
    config = tmp_path / "gwas.yaml"
    config.write_text(contents)
    with pytest.raises(ValueError):
        load_gwas_mapping(config)


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        [],
        {"IQ": {"file": "x"}},
        {"IQ": {"path": "x", "label": None}},
        {"IQ": {"path": "x", "label": " "}},
        {"IQ": 3},
        {"a/b": "x"},
        {" IQ": "x"},
        {"__proto__": "x"},
        {False: "x"},
    ],
)
def test_invalid_entries_fail_early(mapping):
    with pytest.raises(ValueError):
        normalize_gwas_mapping(mapping)


def test_missing_files_fail_early(tmp_path):
    with pytest.raises(FileNotFoundError):
        normalize_gwas_mapping({"IQ": {"path": tmp_path / "missing", "label": "Intelligence"}})


def test_labels_can_be_identical_without_merging_traits(tmp_path):
    stats = tmp_path / "data"
    stats.touch()
    paths, labels = normalize_gwas_mapping(
        {
            key: {"path": stats, "label": "Same display name"}
            for key in ["A B", "A_B", "A-B", "智力"]
        }
    )
    assert len(paths) == len(labels) == 4

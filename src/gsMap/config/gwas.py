"""GWAS inputs keep stable result IDs separate from presentation labels."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import yaml


def trait_file_token(trait: str) -> str:
    """Preserve simple IDs; hash others to avoid lossy report filename collisions."""
    if all(c.isalnum() or c == "_" for c in trait) and not trait.startswith("__trait_"):
        return trait
    return "__trait_" + hashlib.sha256(trait.encode("utf-8")).hexdigest()


def trait_labels(config, traits):
    labels = getattr(config, "trait_labels", {})
    return {trait: labels.get(trait, trait) for trait in traits}


def trait_label(config, trait):
    return getattr(config, "trait_labels", {}).get(trait, trait)


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("GWAS configuration keys must be strings; quote numeric IDs")
            if key in result:
                raise ValueError(f"Duplicate GWAS configuration key: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def normalize_gwas_mapping(mapping, *, base_dir=None, labels=None):
    """Accept path-only and {path, label} entries without renaming trait IDs.

    YAML paths are relative to the YAML directory. Direct Python paths retain
    their usual current-working-directory interpretation. Labels are carried
    separately so child pipeline configurations do not lose them.
    """
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("Summary-statistics config must be a non-empty mapping")
    paths, resolved_labels = {}, {}
    for trait, entry in mapping.items():
        if (
            not isinstance(trait, str)
            or not trait.strip()
            or trait != trait.strip()
            or any(c in "/\\" or ord(c) < 32 for c in trait)
            or trait in {".", "..", "__proto__", "constructor", "prototype"}
        ):
            raise ValueError(
                f"Invalid GWAS trait ID: {trait!r}; use a non-empty, filename-safe ID"
            )
        label = (labels or {}).get(trait, trait)
        if isinstance(entry, Mapping):
            unknown = set(entry) - {"path", "label"}
            if unknown or "path" not in entry:
                raise ValueError(f"GWAS entry {trait!r} requires 'path' and optional 'label' only")
            label = entry.get("label", label)
            entry = entry["path"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"GWAS label for {trait!r} must be a non-empty string")
        if not isinstance(entry, str | Path) or not str(entry).strip():
            raise ValueError(f"GWAS path for {trait!r} must be a non-empty path")
        path = Path(entry)
        if base_dir is not None:
            path = (path if path.is_absolute() else Path(base_dir) / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Summary-statistics file does not exist: {path}")
        paths[trait] = path
        resolved_labels[trait] = label.strip()
    return paths, resolved_labels


def load_gwas_mapping(path, *, labels=None):
    """Load safe YAML, rejecting duplicate keys instead of dropping traits."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        mapping = yaml.load(handle, Loader=_UniqueKeyLoader)
    return normalize_gwas_mapping(mapping, base_dir=path.parent, labels=labels)

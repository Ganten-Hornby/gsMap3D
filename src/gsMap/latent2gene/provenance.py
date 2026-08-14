"""Small provenance primitives for restartable latent-to-gene artifacts."""

import hashlib
import json
import os
from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path


def canonical_sha256(payload: object) -> str:
    """Hash a JSON-serializable payload with deterministic encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def gene_names_sha256(gene_names: Iterable[object]) -> str:
    """Return an order-sensitive checksum for a gene axis."""

    return canonical_sha256([str(name) for name in gene_names])


def artifact_signature(path: str | Path) -> dict[str, int | str]:
    """Identify one local artifact without reading a potentially huge file."""

    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def rank_source_fingerprint(rank_memmap_path: str | Path) -> str:
    """Fingerprint a rank memmap and its completion metadata."""

    rank_memmap_path = Path(rank_memmap_path)
    return canonical_sha256(
        {
            "data": artifact_signature(rank_memmap_path.with_suffix(".dat")),
            "metadata": artifact_signature(rank_memmap_path.with_suffix(".meta.json")),
        }
    )


def rank_inputs_fingerprint(
    sample_h5ad_dict: Mapping[str, str | Path],
    *,
    annotation_key: str | None,
    data_layer: str,
    min_cells_per_type: int | None,
    min_genes: int,
) -> str:
    """Fingerprint ordered rank inputs and every filtering choice."""

    return canonical_sha256(
        {
            "samples": [
                {
                    "name": str(sample_name),
                    "artifact": artifact_signature(h5ad_path),
                }
                for sample_name, h5ad_path in sample_h5ad_dict.items()
            ],
            "annotation_key": annotation_key,
            "data_layer": data_layer,
            "min_cells_per_type": min_cells_per_type,
            "min_genes": min_genes,
        }
    )


def atomic_json_dump(path: str | Path, payload: dict) -> None:
    """Atomically replace one JSON sidecar."""

    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.partial")
    with open(temporary_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary_path, path)

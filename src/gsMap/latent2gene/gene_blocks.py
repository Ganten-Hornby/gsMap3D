"""Gene-block marker-score storage and annotated-LD utilities."""

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import anndata as ad
import h5py
import jax.numpy as jnp
import numpy as np
import pandas as pd
from rich.progress import track

from .marker_scores import (
    compute_marker_scores_3d_hierarchical_pool_jax,
    compute_marker_scores_jax,
)
from .memmap_io import MemMapDense
from .provenance import (
    artifact_signature,
    atomic_json_dump,
    canonical_sha256,
    gene_names_sha256,
    rank_source_fingerprint,
)

logger = logging.getLogger(__name__)

GENE_BLOCK_FORMAT_VERSION = 2
MAX_GENE_BLOCK_BYTES = 32 * 1024**3


def gene_block_input_fingerprint(
    rank_memmap_path: str | Path,
    mean_frac_path: str | Path,
    neighbor_adata_path: str | Path,
) -> str:
    """Fingerprint every persisted input that determines marker-score blocks."""

    return canonical_sha256(
        {
            "rank": rank_source_fingerprint(rank_memmap_path),
            "mean_frac": artifact_signature(mean_frac_path),
            "neighbors": artifact_signature(neighbor_adata_path),
        }
    )


def _read_marker_gene_names(marker_metadata_path: str | Path) -> list[str]:
    metadata_adata = ad.read_h5ad(marker_metadata_path, backed="r")
    try:
        return metadata_adata.var_names.astype(str).tolist()
    finally:
        metadata_adata.file.close()


def gene_block_store_is_read_compatible(
    manifest_path: str | Path,
    *,
    rank_memmap_path: str | Path,
    mean_frac_path: str | Path,
    marker_metadata_path: str | Path,
) -> bool:
    """Accept complete legacy stores when their spot and gene axes still match.

    Older gene-block manifests fingerprinted the entire metadata H5AD. Rewriting
    homogeneous-neighbor diagnostics changes that file signature even though
    spatial LDSC only needs the unchanged observation count and gene order to
    read the already-computed marker-score blocks.
    """

    try:
        metadata_adata = ad.read_h5ad(marker_metadata_path, backed="r")
        try:
            n_obs = metadata_adata.n_obs
            gene_names = metadata_adata.var_names.astype(str).tolist()
        finally:
            metadata_adata.file.close()

        store = GeneBlockMarkerStore(manifest_path)
        try:
            if store.shape[0] != n_obs:
                return False
            store.validate_gene_names(gene_names)
            current_fingerprint = gene_block_input_fingerprint(
                rank_memmap_path,
                mean_frac_path,
                marker_metadata_path,
            )
            if store.source_fingerprint != current_fingerprint:
                logger.warning(
                    "Gene-block input fingerprint changed, but the complete store has "
                    "the same spot count and gene order; reading it as a legacy artifact"
                )
            return True
        finally:
            store.close()
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False


def gene_block_store_matches_inputs(
    manifest_path: str | Path,
    *,
    rank_memmap_path: str | Path,
    mean_frac_path: str | Path,
    marker_metadata_path: str | Path,
) -> bool:
    """Return whether a complete store belongs to the current upstream artifacts."""

    try:
        source_fingerprint = gene_block_input_fingerprint(
            rank_memmap_path,
            mean_frac_path,
            marker_metadata_path,
        )
        gene_names = _read_marker_gene_names(marker_metadata_path)
        return GeneBlockMarkerStore.check_complete(
            manifest_path,
            source_fingerprint=source_fingerprint,
            gene_names=gene_names,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return False


def compute_marker_score_gene_block(
    *,
    rank_rows: np.ndarray,
    rank_inverse: np.ndarray,
    rank_row_min: np.ndarray,
    neighbor_weights: np.ndarray,
    global_log_gmean: np.ndarray,
    global_expr_frac: np.ndarray,
    n_slices: int,
    homogeneous_neighbors: int,
    hierarchical_pool: bool,
    no_expression_fraction: bool = False,
) -> np.ndarray:
    """Compute one independent gene block without changing full-gene filtering.

    ``rank_row_min`` must contain the minimum over the complete gene axis for
    every row in ``rank_rows``.  Supplying that precomputed value preserves the
    expression-fraction mask when the marker calculation is split by gene.
    """

    rank_rows = np.asarray(rank_rows)
    rank_inverse = np.asarray(rank_inverse, dtype=np.int32).reshape(-1)
    rank_row_min = np.asarray(rank_row_min)
    neighbor_weights = np.asarray(neighbor_weights)

    if rank_rows.ndim != 2:
        raise ValueError("rank_rows must be a two-dimensional row-by-gene array")
    if rank_row_min.shape != (rank_rows.shape[0],):
        raise ValueError("rank_row_min must contain one full-gene minimum per rank row")
    if rank_inverse.size != neighbor_weights.size:
        raise ValueError("rank_inverse and neighbor_weights describe different neighbor counts")
    if rank_inverse.size and (rank_inverse.min() < 0 or rank_inverse.max() >= len(rank_rows)):
        raise IndexError("rank_inverse contains an index outside rank_rows")

    batch_size = neighbor_weights.shape[0]
    expected_neighbors = n_slices * homogeneous_neighbors
    if neighbor_weights.shape[1] != expected_neighbors:
        raise ValueError("neighbor_weights width does not equal n_slices * homogeneous_neighbors")

    inverse_jax = jnp.asarray(rank_inverse)
    gathered_ranks = jnp.asarray(rank_rows)[inverse_jax]
    gathered_row_min = jnp.asarray(rank_row_min)[inverse_jax]

    if hierarchical_pool:
        result = compute_marker_scores_3d_hierarchical_pool_jax(
            gathered_ranks,
            jnp.asarray(neighbor_weights),
            batch_size,
            n_slices,
            homogeneous_neighbors,
            jnp.asarray(global_log_gmean),
            jnp.asarray(global_expr_frac),
            no_expression_fraction,
            gathered_row_min,
        )
    else:
        result = compute_marker_scores_jax(
            gathered_ranks,
            jnp.asarray(neighbor_weights),
            batch_size,
            expected_neighbors,
            jnp.asarray(global_log_gmean),
            jnp.asarray(global_expr_frac),
            no_expression_fraction,
            gathered_row_min,
        )

    marker_scores = np.array(result, dtype=np.float16, copy=True)
    marker_scores[neighbor_weights.sum(axis=1) <= 0] = 0
    return marker_scores


def accumulate_annotated_ldscore_blocks(
    snp_gene_weight,
    marker_blocks: Iterable[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Accumulate ``SNP × spot`` annotated LD scores over gene blocks.

    Each item contains column positions into ``snp_gene_weight`` and a
    ``gene × spot`` marker-score block.  The result is mathematically identical
    to multiplying the full matrices, without materializing the full marker
    matrix.
    """

    annotated_ldscore = None
    for weight_columns, marker_block in marker_blocks:
        weight_columns = np.asarray(weight_columns, dtype=np.int64)
        marker_block = np.asarray(marker_block, dtype=np.float32)
        if marker_block.ndim != 2:
            raise ValueError("marker blocks must be gene-by-spot matrices")
        if marker_block.shape[0] != len(weight_columns):
            raise ValueError("marker block rows do not match their weight columns")

        partial = snp_gene_weight[:, weight_columns] @ marker_block
        if hasattr(partial, "toarray"):
            partial = partial.toarray()
        partial = np.asarray(partial, dtype=np.float32)
        if annotated_ldscore is None:
            annotated_ldscore = np.zeros_like(partial, dtype=np.float32)
        annotated_ldscore += partial

    if annotated_ldscore is None:
        raise ValueError("at least one marker-score gene block is required")
    return annotated_ldscore


def _read_rank_metadata(rank_memmap_path: str | Path) -> tuple[tuple[int, int], np.dtype]:
    rank_memmap_path = Path(rank_memmap_path)
    with open(rank_memmap_path.with_suffix(".meta.json")) as handle:
        metadata = json.load(handle)
    return tuple(metadata["shape"]), np.dtype(metadata["dtype"])


def ensure_rank_row_min(
    rank_memmap_path: str | Path,
    output_path: str | Path,
    *,
    row_batch_size: int = 2048,
) -> Path:
    """Persist the full-gene minimum rank for every spot using sequential reads."""

    rank_memmap_path = Path(rank_memmap_path)
    output_path = Path(output_path)
    shape, dtype = _read_rank_metadata(rank_memmap_path)
    n_spots, _ = shape
    source_fingerprint = rank_source_fingerprint(rank_memmap_path)
    metadata_path = output_path.with_suffix(".meta.json")

    if output_path.exists() and metadata_path.exists():
        try:
            with open(metadata_path) as handle:
                metadata = json.load(handle)
            existing = np.load(output_path, mmap_mode="r")
            reusable = (
                existing.shape == (n_spots,)
                and existing.dtype == dtype
                and metadata.get("source_fingerprint") == source_fingerprint
                and tuple(metadata.get("shape", ())) == (n_spots,)
                and np.dtype(metadata.get("dtype")) == dtype
                and metadata.get("output_signature") == artifact_signature(output_path)
            )
            del existing
            if reusable:
                return output_path
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.partial")
    temporary_path.unlink(missing_ok=True)

    ranks = np.memmap(
        rank_memmap_path.with_suffix(".dat"),
        mode="r",
        dtype=dtype,
        shape=shape,
    )
    row_min = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=dtype,
        shape=(n_spots,),
    )
    try:
        for start in track(
            range(0, n_spots, row_batch_size),
            description="Computing full-gene row minima",
        ):
            end = min(start + row_batch_size, n_spots)
            row_min[start:end] = np.min(ranks[start:end], axis=1)
        row_min.flush()
        del row_min
        os.replace(temporary_path, output_path)
        atomic_json_dump(
            metadata_path,
            {
                "shape": [n_spots],
                "dtype": dtype.name,
                "source_fingerprint": source_fingerprint,
                "output_signature": artifact_signature(output_path),
                "complete": True,
                "completed_at": time.time(),
            },
        )
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        del ranks
    return output_path


def _preallocate_file(path: Path, byte_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w+b") as handle:
        os.posix_fallocate(handle.fileno(), 0, byte_size)


def _block_filename(start: int, end: int) -> str:
    return f"marker_scores_genes_{start:05d}_{end:05d}.dat"


class GeneBlockMarkerStore:
    """Read a complete set of dense marker-score gene blocks."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path) as handle:
            self.manifest = json.load(handle)
        if not self.manifest.get("complete", False):
            raise ValueError(f"Gene-block marker store is incomplete: {self.manifest_path}")
        if self.manifest.get("format_version") != GENE_BLOCK_FORMAT_VERSION:
            raise ValueError(f"Unsupported gene-block format: {self.manifest_path}")

        self.shape = tuple(self.manifest["shape"])
        self.dtype = np.dtype(self.manifest["dtype"])
        self.blocks = self.manifest["blocks"]
        self.source_fingerprint = self.manifest.get("source_fingerprint")
        self.gene_names_checksum = self.manifest.get("gene_names_sha256")
        self._subset_staging_lock = threading.Lock()
        self._validate_manifest_and_files()

    def _validate_manifest_and_files(self) -> None:
        if len(self.shape) != 2 or any(int(size) < 1 for size in self.shape):
            raise ValueError("Gene-block marker store has an invalid shape")
        if not isinstance(self.gene_names_checksum, str) or len(self.gene_names_checksum) != 64:
            raise ValueError("Gene-block marker store is missing its gene-order checksum")
        if not isinstance(self.source_fingerprint, str) or len(self.source_fingerprint) != 64:
            raise ValueError("Gene-block marker store is missing its input fingerprint")
        if not isinstance(self.blocks, list) or not self.blocks:
            raise ValueError("Gene-block marker store contains no blocks")

        expected_start = 0
        store_root = self.manifest_path.parent.resolve()
        for block in self.blocks:
            start = block.get("start")
            end = block.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("Gene-block boundaries must be integers")
            if start != expected_start or end <= start or end > self.shape[1]:
                raise ValueError("Gene blocks must provide ordered, non-overlapping gene coverage")
            if not block.get("complete", False):
                raise ValueError("Gene-block manifest contains an incomplete block")

            relative_path = Path(block.get("path", ""))
            if relative_path.is_absolute():
                raise ValueError("Gene-block paths must be relative to the manifest")
            block_path = (self.manifest_path.parent / relative_path).resolve()
            if not block_path.is_relative_to(store_root):
                raise ValueError("Gene-block path escapes the marker-store directory")
            expected_size = self.shape[0] * (end - start) * self.dtype.itemsize
            if block_path.stat().st_size != expected_size:
                raise ValueError(f"Gene block has the wrong size: {block_path}")
            expected_start = end

        if expected_start != self.shape[1]:
            raise ValueError("Gene blocks do not cover the complete gene axis")

    @classmethod
    def check_complete(
        cls,
        manifest_path: str | Path,
        *,
        source_fingerprint: str | None = None,
        gene_names: Iterable[object] | None = None,
    ) -> bool:
        try:
            store = cls(manifest_path)
            if (
                source_fingerprint is not None
                and store.source_fingerprint != source_fingerprint
            ):
                return False
            if gene_names is not None:
                store.validate_gene_names(gene_names)
            store.close()
            return True
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return False

    def _open_block(self, block: dict) -> np.memmap:
        block_path = self.manifest_path.parent / block["path"]
        return np.memmap(
            block_path,
            mode="r",
            dtype=self.dtype,
            shape=(self.shape[0], block["end"] - block["start"]),
        )

    def validate_gene_names(self, gene_names: Iterable[object]) -> None:
        """Fail before multiplication when the current gene axis was reordered."""

        if gene_names_sha256(gene_names) != self.gene_names_checksum:
            raise ValueError(
                "Gene-block marker-score gene order does not match the current marker metadata"
            )

    def iter_spot_marker_blocks(
        self,
        spot_start: int,
        spot_end: int,
        marker_gene_indices: np.ndarray,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield common-gene marker blocks as ``(weight columns, gene × spot)``."""

        marker_gene_indices = np.asarray(marker_gene_indices, dtype=np.int64)
        if not 0 <= spot_start <= spot_end <= self.shape[0]:
            raise IndexError("Requested spot chunk is outside the marker store")
        if marker_gene_indices.ndim != 1:
            raise ValueError("marker_gene_indices must be one-dimensional")
        if len(marker_gene_indices) and (
            marker_gene_indices[0] < 0
            or marker_gene_indices[-1] >= self.shape[1]
            or np.any(np.diff(marker_gene_indices) <= 0)
        ):
            raise ValueError("marker_gene_indices must be unique, ordered, and in range")
        for block in self.blocks:
            in_block = (marker_gene_indices >= block["start"]) & (
                marker_gene_indices < block["end"]
            )
            weight_columns = np.flatnonzero(in_block)
            if not len(weight_columns):
                continue
            local_columns = marker_gene_indices[in_block] - block["start"]
            memmap = self._open_block(block)
            marker = memmap[spot_start:spot_end, local_columns]
            marker_copy = np.array(marker.T, dtype=np.float32, copy=True)
            del marker, memmap
            yield weight_columns, marker_copy

    def read_subset(
        self,
        spot_indices: np.ndarray,
        gene_indices: np.ndarray,
        *,
        staging_dir: str | Path | None = None,
    ) -> np.ndarray:
        """Read a spot-by-gene subset, optionally staging each block on local storage."""

        spot_indices = np.asarray(spot_indices, dtype=np.int64)
        gene_indices = np.asarray(gene_indices, dtype=np.int64)
        if spot_indices.ndim != 1 or gene_indices.ndim != 1:
            raise ValueError("spot_indices and gene_indices must be one-dimensional")
        if len(spot_indices) and (
            spot_indices.min() < 0 or spot_indices.max() >= self.shape[0]
        ):
            raise IndexError("Requested spot subset is outside the marker store")
        if len(gene_indices) and (
            gene_indices.min() < 0 or gene_indices.max() >= self.shape[1]
        ):
            raise IndexError("Requested gene subset is outside the marker store")

        if staging_dir is None:
            return self._read_subset(spot_indices, gene_indices, staging_dir=None)

        # A report can load samples concurrently. Serialize staged reads so the
        # job-local filesystem never holds more than one full gene block.
        with self._subset_staging_lock:
            return self._read_subset(
                spot_indices,
                gene_indices,
                staging_dir=Path(staging_dir),
            )

    def _read_subset(
        self,
        spot_indices: np.ndarray,
        gene_indices: np.ndarray,
        *,
        staging_dir: Path | None,
    ) -> np.ndarray:
        result = np.empty((len(spot_indices), len(gene_indices)), dtype=np.float32)
        if staging_dir is not None:
            staging_dir.mkdir(parents=True, exist_ok=True)

        for block in self.blocks:
            in_block = (gene_indices >= block["start"]) & (gene_indices < block["end"])
            output_columns = np.flatnonzero(in_block)
            if not len(output_columns):
                continue
            local_columns = gene_indices[in_block] - block["start"]

            source_path = self.manifest_path.parent / block["path"]
            staged_path = None
            partial_path = None
            memmap = None
            try:
                if staging_dir is not None:
                    staging_token = f"{os.getpid()}-{id(self):x}"
                    staged_path = staging_dir / f".gsmap-{staging_token}-{source_path.name}"
                    partial_path = staging_dir / (
                        f".gsmap-{staging_token}-{source_path.name}.partial"
                    )
                    staged_path.unlink(missing_ok=True)
                    partial_path.unlink(missing_ok=True)
                    required_bytes = source_path.stat().st_size
                    reserve_bytes = 1024**3
                    free_bytes = shutil.disk_usage(staging_dir).free
                    if free_bytes < required_bytes + reserve_bytes:
                        raise OSError(
                            f"Insufficient local staging space in {staging_dir}: "
                            f"need {(required_bytes + reserve_bytes) / 1024**3:.1f} GiB "
                            f"including reserve, found {free_bytes / 1024**3:.1f} GiB"
                        )
                    logger.info(
                        "Staging marker-score block %s (%.1f GiB) in %s",
                        source_path.name,
                        required_bytes / 1024**3,
                        staging_dir,
                    )
                    shutil.copyfile(source_path, partial_path)
                    os.replace(partial_path, staged_path)

                if staged_path is None:
                    memmap = self._open_block(block)
                else:
                    memmap = np.memmap(
                        staged_path,
                        mode="r",
                        dtype=self.dtype,
                        shape=(self.shape[0], block["end"] - block["start"]),
                    )
                result[:, output_columns] = np.asarray(
                    memmap[np.ix_(spot_indices, local_columns)],
                    dtype=np.float32,
                )
            finally:
                del memmap
                if staged_path is not None:
                    staged_path.unlink(missing_ok=True)
                    if partial_path is not None:
                        partial_path.unlink(missing_ok=True)
                    logger.info("Removed staged marker-score block %s", staged_path)
        return result

    def close(self) -> None:
        """Compatibility no-op: each block mapping is released after it is copied."""


def ensure_consolidated_marker_memmap(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    spot_batch_size: int = 4096,
) -> Path:
    """Consolidate a complete gene-block store into the standard row-major memmap."""
    store = GeneBlockMarkerStore(manifest_path)
    output_path = Path(output_path)
    try:
        complete, metadata = MemMapDense.check_complete(output_path)
        if complete:
            if metadata is None or tuple(metadata["shape"]) != store.shape:
                raise ValueError(
                    f"Consolidated marker-score shape does not match {manifest_path}"
                )
            if np.dtype(metadata["dtype"]) != store.dtype:
                raise ValueError(
                    f"Consolidated marker-score dtype does not match {manifest_path}"
                )
            logger.info("Using existing consolidated marker-score memmap: %s", output_path)
            return output_path

        logger.info(
            "Consolidating %s gene-block marker-score files into %s",
            len(store.blocks),
            output_path,
        )
        output = MemMapDense(
            output_path,
            shape=store.shape,
            dtype=store.dtype,
            mode="w",
            num_write_workers=0,
            tmp_dir=None,
        )
        block_maps: list[np.memmap] = []
        try:
            block_maps = [store._open_block(block) for block in store.blocks]
            for spot_start in range(0, store.shape[0], spot_batch_size):
                spot_end = min(spot_start + spot_batch_size, store.shape[0])
                for block, block_map in zip(store.blocks, block_maps, strict=True):
                    output.memmap[
                        spot_start:spot_end,
                        block["start"] : block["end"],
                    ] = block_map[spot_start:spot_end]
            output.close(
                metadata_updates={
                    "consolidated_from_gene_blocks": artifact_signature(manifest_path),
                }
            )
        except BaseException:
            output.abort()
            raise
        finally:
            for block_map in block_maps:
                block_map._mmap.close()
        return output_path
    finally:
        store.close()


def calculate_gene_block_marker_scores(
    *,
    rank_memmap_path: str | Path,
    rank_row_min_path: str | Path,
    mean_frac_path: str | Path,
    neighbor_adata_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    gene_block_size: int,
    spot_batch_size: int,
    n_slices: int,
    homogeneous_neighbors: int,
    hierarchical_pool: bool,
    no_expression_fraction: bool,
    tmp_dir: str | Path,
) -> Path:
    """Materialize marker scores as independently allocated gene-block files."""

    rank_memmap_path = Path(rank_memmap_path)
    rank_row_min_path = Path(rank_row_min_path)
    mean_frac_path = Path(mean_frac_path)
    neighbor_adata_path = Path(neighbor_adata_path)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    tmp_dir = Path(tmp_dir)
    if gene_block_size < 1 or spot_batch_size < 1:
        raise ValueError("gene_block_size and spot_batch_size must both be positive")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    shape, rank_dtype = _read_rank_metadata(rank_memmap_path)
    n_spots, n_genes = shape
    if n_spots < 1 or n_genes < 1:
        raise ValueError("rank matrix must contain at least one spot and one gene")
    block_width = min(gene_block_size, n_genes)
    largest_block_bytes = n_spots * block_width * max(
        rank_dtype.itemsize,
        np.dtype(np.float16).itemsize,
    )
    if largest_block_bytes > MAX_GENE_BLOCK_BYTES:
        max_width = max(
            1,
            MAX_GENE_BLOCK_BYTES
            // (n_spots * max(rank_dtype.itemsize, np.dtype(np.float16).itemsize)),
        )
        raise ValueError(
            f"marker_score_gene_block_size={gene_block_size} would allocate a "
            f"{largest_block_bytes / 1024**3:.1f} GiB block; reduce it to at most "
            f"{max_width} genes so no rank or marker mmap exceeds 32 GiB"
        )
    gene_names = _read_marker_gene_names(neighbor_adata_path)
    if len(gene_names) != n_genes:
        raise ValueError("Marker metadata gene count does not match the rank matrix")

    source_fingerprint = gene_block_input_fingerprint(
        rank_memmap_path,
        mean_frac_path,
        neighbor_adata_path,
    )
    gene_names_checksum = gene_names_sha256(gene_names)
    row_min_path = ensure_rank_row_min(rank_memmap_path, rank_row_min_path)
    row_min = np.load(row_min_path, mmap_mode="r")
    global_stats = pd.read_parquet(mean_frac_path)
    global_log_gmean = global_stats["G_Mean"].to_numpy(dtype=np.float32)
    global_expr_frac = global_stats["frac"].to_numpy(dtype=np.float32)
    if len(global_log_gmean) != n_genes:
        raise ValueError("Global gene statistics do not match the rank matrix")
    global_gene_names = (
        global_stats["gene_name"].astype(str).tolist()
        if "gene_name" in global_stats.columns
        else global_stats.index.astype(str).tolist()
    )
    if global_gene_names != gene_names:
        raise ValueError("Global gene statistics do not match the marker gene order")

    blocks: list[dict] = [
        {
            "start": start,
            "end": min(start + gene_block_size, n_genes),
            "path": _block_filename(start, min(start + gene_block_size, n_genes)),
            "complete": False,
        }
        for start in range(0, n_genes, gene_block_size)
    ]
    calculation = {
        "rank_dtype": rank_dtype.str,
        "n_slices": n_slices,
        "homogeneous_neighbors": homogeneous_neighbors,
        "hierarchical_pool": hierarchical_pool,
        "no_expression_fraction": no_expression_fraction,
    }
    manifest = {
        "format_version": GENE_BLOCK_FORMAT_VERSION,
        "shape": [n_spots, n_genes],
        "dtype": "float16",
        "gene_block_size": gene_block_size,
        "complete": False,
        "created_at": time.time(),
        "source_fingerprint": source_fingerprint,
        "gene_names_sha256": gene_names_checksum,
        "calculation": calculation,
        "blocks": blocks,
    }

    if manifest_path.exists():
        try:
            with open(manifest_path) as handle:
                existing = json.load(handle)
        except (OSError, json.JSONDecodeError):
            existing = {}
        expected_layout = [
            (block["start"], block["end"], block["path"]) for block in blocks
        ]
        existing_layout = [
            (block.get("start"), block.get("end"), block.get("path"))
            for block in existing.get("blocks", [])
            if isinstance(block, dict)
        ]
        compatible = (
            existing.get("format_version") == GENE_BLOCK_FORMAT_VERSION
            and tuple(existing.get("shape", ())) == shape
            and existing.get("dtype") == "float16"
            and existing.get("gene_block_size") == gene_block_size
            and existing.get("calculation") == calculation
            and existing.get("source_fingerprint") == source_fingerprint
            and existing.get("gene_names_sha256") == gene_names_checksum
            and existing_layout == expected_layout
        )
        if compatible and GeneBlockMarkerStore.check_complete(
            manifest_path,
            source_fingerprint=source_fingerprint,
            gene_names=gene_names,
        ):
            logger.info("Gene-block marker store is already complete: %s", manifest_path)
            del row_min
            return manifest_path
        if compatible:
            manifest = existing
            blocks = manifest["blocks"]

    # A repair or replacement must be visibly incomplete before any block is
    # written.  A crash can therefore never leave a stale global complete flag.
    manifest["complete"] = False
    manifest.pop("completed_at", None)
    atomic_json_dump(manifest_path, manifest)

    rank_memmap = np.memmap(
        rank_memmap_path.with_suffix(".dat"),
        mode="r",
        dtype=rank_dtype,
        shape=shape,
    )
    with h5py.File(neighbor_adata_path, "r") as adata_file:
        neighbor_indices = adata_file["obsm"]["gsMap_homo_indices"]
        neighbor_weights = adata_file["obsm"]["gsMap_homo_cell_sims"]
        if (
            neighbor_indices.shape[0] != n_spots
            or neighbor_weights.shape != neighbor_indices.shape
        ):
            raise ValueError("Homogeneous-neighbor matrices do not match the rank matrix")

        for block in blocks:
            destination = output_dir / block["path"]
            block_width = block["end"] - block["start"]
            expected_size = n_spots * block_width * np.dtype(np.float16).itemsize
            if (
                block.get("complete")
                and destination.exists()
                and destination.stat().st_size == expected_size
            ):
                logger.info("Skipping complete marker gene block %s", destination)
                continue

            block["complete"] = False
            atomic_json_dump(manifest_path, manifest)

            with tempfile.TemporaryDirectory(
                prefix=f"gsmap_genes_{block['start']:05d}_{block['end']:05d}_",
                dir=tmp_dir,
            ) as temporary_directory:
                temporary_directory = Path(temporary_directory)
                local_rank_path = temporary_directory / "rank_block.dat"
                local_marker_path = temporary_directory / "marker_block.dat"
                rank_bytes = n_spots * block_width * rank_dtype.itemsize
                marker_bytes = n_spots * block_width * np.dtype(np.float16).itemsize
                _preallocate_file(local_rank_path, rank_bytes)
                _preallocate_file(local_marker_path, marker_bytes)
                local_rank = np.memmap(
                    local_rank_path,
                    mode="r+",
                    dtype=rank_dtype,
                    shape=(n_spots, block_width),
                )
                local_marker = np.memmap(
                    local_marker_path,
                    mode="r+",
                    dtype=np.float16,
                    shape=(n_spots, block_width),
                )

                for start in track(
                    range(0, n_spots, spot_batch_size * 8),
                    description=(
                        f"Staging rank genes {block['start']}:{block['end']} to local storage"
                    ),
                ):
                    end = min(start + spot_batch_size * 8, n_spots)
                    local_rank[start:end] = rank_memmap[
                        start:end,
                        block["start"] : block["end"],
                    ]
                local_rank.flush()

                for start in track(
                    range(0, n_spots, spot_batch_size),
                    description=f"Marker genes {block['start']}:{block['end']}",
                ):
                    end = min(start + spot_batch_size, n_spots)
                    batch_neighbors = np.asarray(neighbor_indices[start:end], dtype=np.int64)
                    batch_weights = np.asarray(neighbor_weights[start:end], dtype=np.float32)
                    safe_neighbors = np.where(batch_neighbors >= 0, batch_neighbors, 0)
                    unique_neighbors, inverse = np.unique(safe_neighbors, return_inverse=True)
                    local_marker[start:end] = compute_marker_score_gene_block(
                        rank_rows=np.asarray(local_rank[unique_neighbors]),
                        rank_inverse=inverse,
                        rank_row_min=np.asarray(row_min[unique_neighbors]),
                        neighbor_weights=batch_weights,
                        global_log_gmean=global_log_gmean[block["start"] : block["end"]],
                        global_expr_frac=global_expr_frac[block["start"] : block["end"]],
                        n_slices=n_slices,
                        homogeneous_neighbors=homogeneous_neighbors,
                        hierarchical_pool=hierarchical_pool,
                        no_expression_fraction=no_expression_fraction,
                    )
                local_marker.flush()
                del local_rank, local_marker

                destination_partial = destination.with_name(f".{destination.name}.partial")
                destination_partial.unlink(missing_ok=True)
                try:
                    shutil.copyfile(local_marker_path, destination_partial)
                    os.replace(destination_partial, destination)
                except BaseException:
                    destination_partial.unlink(missing_ok=True)
                    raise

            block["complete"] = True
            atomic_json_dump(manifest_path, manifest)

    del rank_memmap, row_min
    manifest["complete"] = all(block.get("complete", False) for block in blocks)
    if not manifest["complete"]:
        atomic_json_dump(manifest_path, manifest)
        raise RuntimeError("Gene-block marker-score calculation ended with incomplete blocks")
    manifest["completed_at"] = time.time()
    atomic_json_dump(manifest_path, manifest)
    if not GeneBlockMarkerStore.check_complete(
        manifest_path,
        source_fingerprint=source_fingerprint,
        gene_names=gene_names,
    ):
        manifest["complete"] = False
        manifest.pop("completed_at", None)
        atomic_json_dump(manifest_path, manifest)
        raise RuntimeError("Completed gene-block marker store failed validation")
    return manifest_path

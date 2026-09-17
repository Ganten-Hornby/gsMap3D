"""One expression-QC population shared by HVG, sampling, graph building and inference."""

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
from anndata.io import read_elem, sparse_dataset

from gsMap.utils.expression_qc import canonical_expression_block
from gsMap.utils.h5ad import read_obsm_rows

logger = logging.getLogger(__name__)
SOURCE_ROW = "_gsmap_source_obs_index"
SOURCE_NAME = "_gsmap_source_obs_name"
GENE_EXCLUSION = r"^(HB.-|MT-)"
QC_VERSION = 1
# Bit flags in the per-original-row audit cache (multiple reasons may apply).
ZERO_EXPRESSION, LOW_GENES, LOW_COUNTS = 1, 2, 4


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def qc_contract(path, params):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "version": QC_VERSION,
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "data_layer": params.data_layer,
        "min_genes": getattr(params, "cell_qc_min_genes", 20),
        "min_counts": getattr(params, "cell_qc_min_counts", None),
        "gene_exclusion": GENE_EXCLUSION,
    }


class SectionDataSource:
    """Lazy, bounded CSR reads; cache QC metrics, not another expression H5AD.

    The first load computes QC while collecting only accepted expression rows.
    Subsequent loads reuse the same mask and read only accepted rows. Original
    row IDs are provenance, never positions in the filtered graph.
    """

    def __init__(self, path, params):
        self.path = Path(path).resolve()
        self.contract = qc_contract(path, params)
        self.layer = params.data_layer
        self.spatial_key = getattr(params, "spatial_key", "spatial")
        self.chunk_rows = getattr(params, "cell_qc_chunk_size", 8192)
        if self.contract["min_genes"] < 0 or self.chunk_rows < 1:
            raise ValueError("cell_qc_min_genes must be nonnegative and chunk size positive")
        min_counts = self.contract["min_counts"]
        if min_counts is not None and (not np.isfinite(min_counts) or min_counts < 0):
            raise ValueError("cell_qc_min_counts must be finite and nonnegative")
        with h5py.File(self.path, "r") as handle:
            var = read_elem(handle["var"])
            obs_key = handle["obs"].attrs["_index"]
            self.n_source_rows = len(handle["obs"][obs_key])
        self.gene_columns = np.flatnonzero(~var.index.str.match(re.compile(GENE_EXCLUSION, re.I)))
        names = ad.AnnData(var=var.iloc[self.gene_columns].copy())
        names.var_names_make_unique()
        self.var = names.var
        self.keep_indices = None
        self.cache_path = None
        if getattr(params, "latent_dir", None) is not None:
            cache_dir = Path(params.latent_dir) / "cell_qc"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path = cache_dir / f"{self.path.stem}.{_digest(self.contract)[:20]}.npz"
        if self.cache_path is not None and self.cache_path.exists():
            self._load_cache()

    def _matrix(self, handle):
        if self.layer in handle.get("layers", {}):
            node = handle["layers"][self.layer]
        elif self.layer == "X":
            node = handle["X"]
        else:
            raise ValueError(f"Data layer {self.layer!r} not found in {self.path}")
        return self._matrix_node(node)

    @staticmethod
    def _matrix_node(node):
        if isinstance(node, h5py.Dataset):
            return node
        if node.attrs.get("encoding-type") == "csr_matrix":
            return sparse_dataset(node)
        # CSC row slicing repeatedly scans columns. Convert the selected sparse
        # layer once per load instead; no dense matrix or unrelated layers.
        return read_elem(node).tocsr()

    def _blocks(self, matrix, rows, columns):
        for start in range(0, len(rows), self.chunk_rows):
            indices = rows[start : start + self.chunk_rows]
            ordered, inverse = np.unique(indices, return_inverse=True)
            block = canonical_expression_block(sp.csr_matrix(matrix[ordered])[:, columns])[inverse]
            yield indices, block

    def _scan(self, matrix, *, collect):
        self.n_genes = np.zeros(self.n_source_rows, dtype=np.int32)
        self.total_counts = np.zeros(self.n_source_rows, dtype=np.float64)
        self.reasons = np.zeros(self.n_source_rows, dtype=np.uint8)
        all_integer = True
        parts = []
        for rows, block in self._blocks(matrix, np.arange(self.n_source_rows), self.gene_columns):
            all_integer &= bool(np.equal(block.data, np.floor(block.data)).all())
            self.n_genes[rows] = np.diff(block.indptr)
            self.total_counts[rows] = np.asarray(block.sum(axis=1, dtype=np.float64)).ravel()
            self.reasons[rows] |= np.where(
                self.total_counts[rows] <= 0, ZERO_EXPRESSION, 0
            ).astype(np.uint8)
            self.reasons[rows] |= np.where(
                self.n_genes[rows] < self.contract["min_genes"], LOW_GENES, 0
            ).astype(np.uint8)
            if self.contract["min_counts"] is not None:
                self.reasons[rows] |= np.where(
                    self.total_counts[rows] < self.contract["min_counts"], LOW_COUNTS, 0
                ).astype(np.uint8)
            if collect:
                parts.append(block[self.reasons[rows] == 0])
        named_counts = self.layer in {"count", "counts", "raw_counts", "impute_count"}
        if named_counts and not all_integer:
            raise ValueError(f"Cell QC {self.path.stem}: count layer must be integer-valued")
        self.is_count_data = all_integer
        if self.contract["min_counts"] is not None and not all_integer:
            raise ValueError(
                "cell_qc_min_counts requires a raw-count data_layer, not continuous expression"
            )
        self.keep_indices = np.flatnonzero(self.reasons == 0)
        self._validate_population()
        self._save_cache()
        logger.info(
            "Cell QC %s: %d/%d retained",
            self.path.stem,
            len(self.keep_indices),
            self.n_source_rows,
        )
        return sp.vstack(parts, format="csr") if collect else None

    def _validate_population(self):
        if len(self.keep_indices) < 2:
            raise ValueError(
                f"Cell QC {self.path.stem}: {len(self.keep_indices)} cells retained; "
                "need at least two. Refusing to silently remove/reindex a physical section."
            )

    def _cache_hash(self):
        digest = hashlib.sha256()
        for value in (self.n_genes, self.total_counts, self.reasons):
            digest.update(value.tobytes())
        digest.update(str(self.is_count_data).encode())
        return digest.hexdigest()

    def _load_cache(self):
        with np.load(self.cache_path, allow_pickle=False) as cache:
            if json.loads(str(cache["contract"])) != self.contract:
                raise ValueError(f"Cell QC cache contract mismatch: {self.cache_path}")
            self.n_genes = cache["n_genes"]
            self.total_counts = cache["total_counts"]
            self.reasons = cache["reasons"]
            self.is_count_data = bool(cache["is_count_data"])
            if any(
                x.shape != (self.n_source_rows,)
                for x in (self.n_genes, self.total_counts, self.reasons)
            ) or self._cache_hash() != str(cache["qc_hash"]):
                raise ValueError(f"Invalid cell QC cache: {self.cache_path}")
        self.keep_indices = np.flatnonzero(self.reasons == 0)
        self._validate_population()

    def _save_cache(self):
        if self.cache_path is None:
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_path.parent, delete=False) as handle:
                temporary = handle.name
                np.savez_compressed(
                    handle,
                    contract=json.dumps(self.contract),
                    n_genes=self.n_genes,
                    total_counts=self.total_counts,
                    reasons=self.reasons,
                    is_count_data=self.is_count_data,
                    qc_hash=self._cache_hash(),
                )
            os.replace(temporary, self.cache_path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def _assert_unchanged(self):
        if self.contract != qc_contract(self.path, self._contract_params()):
            raise ValueError(f"Source changed during cell QC run: {self.path}")

    def _ensure_population(self):
        self._assert_unchanged()
        if self.keep_indices is None:
            with h5py.File(self.path, "r") as handle:
                self._scan(self._matrix(handle), collect=False)
        # A previous failed load must not make an invalid population reusable.
        self._validate_population()

    def observations(self):
        """Metadata-only sampling pool; expression is read after focal rows are chosen."""
        self._ensure_population()
        with h5py.File(self.path, "r") as handle:
            obs = read_elem(handle["obs"]).iloc[self.keep_indices].copy()
        obs[SOURCE_ROW] = self.keep_indices
        obs[SOURCE_NAME] = obs.index.astype(str)
        return obs

    def load(self, *, genes=None, all_obsm=False, original_rows=None):
        self._assert_unchanged()
        with h5py.File(self.path, "r") as handle:
            matrix = self._matrix(handle)
            if self.keep_indices is None:
                expression = self._scan(matrix, collect=genes is None and original_rows is None)
            else:
                expression = None
            rows = (
                self.keep_indices
                if original_rows is None
                else np.asarray(original_rows, dtype=np.int64)
            )
            self.filtered_positions(rows)
            positions = (
                np.arange(len(self.var)) if genes is None else self.var.index.get_indexer(genes)
            )
            if (positions < 0).any():
                raise ValueError(f"Requested genes absent from {self.path}")
            if expression is None:
                expression = sp.vstack(
                    [
                        block
                        for _, block in self._blocks(matrix, rows, self.gene_columns[positions])
                    ],
                    format="csr",
                )
            obs = read_elem(handle["obs"]).iloc[rows].copy()
            obsm_keys = list(handle.get("obsm", {})) if all_obsm else [self.spatial_key]
            # Preserve other expression layers in training/output H5ADs. In
            # particular, normalized X must not silently discard a counts layer
            # needed by standalone latent2gene. Graph-only reads omit these.
            layers = {}
            if all_obsm:
                ordered_rows, inverse = np.unique(rows, return_inverse=True)
                for name, node in handle.get("layers", {}).items():
                    if name == self.layer:
                        continue
                    other = self._matrix_node(node)
                    blocks = [
                        sp.csr_matrix(other[ordered_rows[start : start + self.chunk_rows]])[
                            :, self.gene_columns[positions]
                        ]
                        for start in range(0, len(ordered_rows), self.chunk_rows)
                    ]
                    layers[name] = sp.vstack(blocks, format="csr")[inverse]
                uns = read_elem(handle["uns"]) if "uns" in handle else {}
            else:
                uns = {}
        obs[SOURCE_ROW] = rows
        obs[SOURCE_NAME] = obs.index.astype(str)
        obs["cell_qc_n_genes"] = self.n_genes[rows]
        obs["cell_qc_total_counts"] = self.total_counts[rows]
        result = ad.AnnData(
            X=expression,
            obs=obs,
            var=self.var.iloc[positions].copy(),
            obsm=read_obsm_rows(self.path, obsm_keys, rows),
            layers=layers,
            uns=uns,
        )
        if self.layer != "X":
            result.layers[self.layer] = result.X
        result.uns["cell_qc"] = self.summary()
        return result

    def _contract_params(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            data_layer=self.layer,
            cell_qc_min_genes=self.contract["min_genes"],
            cell_qc_min_counts=self.contract["min_counts"],
        )

    def filtered_positions(self, original_rows):
        self._ensure_population()
        original_rows = np.asarray(original_rows, dtype=np.int64)
        positions = np.searchsorted(self.keep_indices, original_rows)
        if (positions >= len(self.keep_indices)).any() or not np.array_equal(
            self.keep_indices[positions], original_rows
        ):
            raise RuntimeError("Training rows contain cells absent from the QC population")
        return positions

    def summary(self):
        self._ensure_population()
        return {
            "contract": self.contract,
            "n_input": self.n_source_rows,
            "n_retained": len(self.keep_indices),
            "n_zero_expression": int(((self.reasons & ZERO_EXPRESSION) != 0).sum()),
            "n_low_genes": int(((self.reasons & LOW_GENES) != 0).sum()),
            "n_low_counts": int(((self.reasons & LOW_COUNTS) != 0).sum()),
            "keep_sha256": hashlib.sha256(self.keep_indices.astype("<i8").tobytes()).hexdigest(),
            "cache_path": str(self.cache_path) if self.cache_path else "",
            "reason_bits": {
                "zero_expression": ZERO_EXPRESSION,
                "low_genes": LOW_GENES,
                "low_counts": LOW_COUNTS,
            },
        }


def section_source(path, params):
    key = str(Path(path).resolve())
    registry = getattr(params, "_section_sources", None)
    if registry is None:
        registry = {}
        params._section_sources = registry
    contract = qc_contract(path, params)
    if key not in registry:
        registry[key] = SectionDataSource(path, params)
    elif registry[key].contract != contract:
        raise ValueError(
            "Cell QC inputs/config changed during this run; create a fresh run/config"
        )
    return registry[key]


def cell_qc_summary(config):
    return {
        str(name): section_source(path, config).summary()
        for name, path in config.sample_h5ad_dict.items()
    }

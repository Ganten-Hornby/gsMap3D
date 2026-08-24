"""Cell chunk sources for spatial LDSC."""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gsMap.latent2gene.gene_blocks import accumulate_annotated_ldscore_blocks


@dataclass(slots=True)
class MarkerScoreWindow:
    """One contiguous marker-score window shared by several compute chunks."""

    chunk_indices: tuple[int, ...]
    source_start: int
    values: np.ndarray


class SpatialLDChunkSource:
    """Fetch LD-score or marker-score chunks for one prepared trait."""

    def __init__(
        self,
        mode: str,
        chunk_starts: list[int],
        chunk_size: int,
        cell_end: int,
        spot_indices: np.ndarray,
        spot_names_filtered: np.ndarray,
        ldscore_adata=None,
        ldscore_snp_positions=None,
        marker_score_adata=None,
        marker_score_gene_indices=None,
        snp_gene_weight_sparse=None,
        marker_score_store=None,
    ):
        self.mode = mode
        self.chunk_starts = chunk_starts
        self.chunk_size = chunk_size
        self.cell_end = cell_end
        self.spot_indices = spot_indices
        self.spot_names_filtered = spot_names_filtered
        self.ldscore_adata = ldscore_adata
        self.ldscore_snp_positions = ldscore_snp_positions
        self.marker_score_adata = marker_score_adata
        self.marker_score_gene_indices = marker_score_gene_indices
        self.snp_gene_weight_sparse = snp_gene_weight_sparse
        self.marker_score_store = marker_score_store
        self._marker_read_fd: int | None = None

    def _chunk_selection(
        self, chunk_index: int
    ) -> tuple[int, int, pd.Index, np.ndarray, int, int]:
        """Resolve logical and source rows for one compute chunk."""
        if chunk_index >= len(self.chunk_starts):
            raise ValueError(f"Invalid chunk index {chunk_index}")

        start = self.chunk_starts[chunk_index]
        end = min(start + self.chunk_size, self.cell_end)
        spot_names = pd.Index(self.spot_names_filtered[start:end])
        selected_rows = self.spot_indices[start:end]
        absolute_start = int(selected_rows[0]) if selected_rows.size else start
        absolute_end = int(selected_rows[-1]) + 1 if selected_rows.size else absolute_start
        return start, end, spot_names, selected_rows, absolute_start, absolute_end

    def fetch_chunk(self, chunk_index: int, defer_ldscore_to_compute: bool):
        """Fetch one chunk as either LD scores or marker scores."""
        _, _, spot_names, selected_rows, absolute_start, absolute_end = self._chunk_selection(
            chunk_index
        )
        if selected_rows.size and selected_rows[-1] - selected_rows[0] + 1 == selected_rows.size:
            row_selector: slice | np.ndarray = slice(
                int(selected_rows[0]),
                int(selected_rows[-1]) + 1,
            )
        else:
            row_selector = selected_rows

        if self.mode == "direct_ldscore":
            if not isinstance(row_selector, slice):
                raise RuntimeError("Direct LD-score chunks require contiguous cell columns")
            ldscore_chunk = self._fetch_direct_ldscore_chunk(
                row_selector.start,
                row_selector.stop,
            )
        elif self.mode == "gene_block":
            ldscore_chunk = self._fetch_gene_block_ldscore_chunk(row_selector)
        elif self.mode == "weight_matrix":
            marker_score_chunk = self._fetch_marker_score_chunk(row_selector)
            if defer_ldscore_to_compute:
                return (
                    None,
                    spot_names,
                    absolute_start,
                    absolute_end,
                    marker_score_chunk,
                )
            ldscore_chunk = self.compute_ldscore_from_marker_chunk(marker_score_chunk)
        else:
            raise ValueError(f"Unsupported spatial LD chunk mode: {self.mode}")

        return (
            ldscore_chunk.astype(np.float32, copy=False),
            spot_names,
            absolute_start,
            absolute_end,
        )

    def supports_sequential_marker_windows(self) -> bool:
        """Return whether marker rows can be read as large contiguous windows.

        Standard row-major memmaps and complete gene-block stores support the
        optimized path. Backed H5AD and Feather objects retain their existing
        readers, while a non-contiguous sample filter retains exact fancy-index
        semantics.
        """
        if self.spot_indices.size == 0:
            return False
        if self.mode == "gene_block":
            if self.marker_score_store is None:
                return False
        elif self.mode == "weight_matrix" and self.marker_score_adata is not None:
            marker_matrix = self.marker_score_adata.X
            if (
                not isinstance(marker_matrix, np.memmap)
                or not marker_matrix.flags.c_contiguous
            ):
                return False
        else:
            return False

        # spot_indices originates from arange/where and is therefore sorted and
        # unique. Equal span and size is sufficient to prove contiguity without
        # allocating another array for datasets containing millions of cells.
        return (
            int(self.spot_indices[-1]) - int(self.spot_indices[0]) + 1
            == self.spot_indices.size
        )

    def marker_row_bytes(self) -> int:
        """Return persisted bytes per complete marker-score row."""
        if self.mode == "gene_block":
            return (
                int(self.marker_score_store.shape[1])
                * self.marker_score_store.dtype.itemsize
            )
        marker_matrix = self.marker_score_adata.X
        return int(marker_matrix.shape[1]) * marker_matrix.dtype.itemsize

    def recommended_marker_window_chunks(self, target_bytes: int = 32 * 1024**2) -> int:
        """Choose a bounded I/O window near ``target_bytes`` automatically."""
        if not self.supports_sequential_marker_windows():
            return 1

        row_bytes = self.marker_row_bytes()
        rows_per_window = max(self.chunk_size, target_bytes // max(1, row_bytes))
        # The upper bound keeps queue bookkeeping bounded for unusually narrow
        # matrices while the byte target controls memory for normal datasets.
        return max(1, min(256, rows_per_window // self.chunk_size))

    def advise_sequential_marker_access(self) -> bool:
        """Open the descriptor used for large positional marker reads."""
        if not self.supports_sequential_marker_windows():
            return False

        if self.mode == "gene_block":
            return self.marker_score_store.open_sequential_access()

        marker_matrix = self.marker_score_adata.X
        try:
            self._marker_read_fd = os.open(os.fspath(marker_matrix.filename), os.O_RDONLY)
        except (OSError, ValueError):
            return False

        posix_fadvise = getattr(os, "posix_fadvise", None)
        sequential = getattr(os, "POSIX_FADV_SEQUENTIAL", None)
        if posix_fadvise is not None and sequential is not None:
            try:
                posix_fadvise(self._marker_read_fd, 0, 0, sequential)
            except OSError:
                pass
        return True

    def close_sequential_marker_access(self) -> None:
        """Close the positional-read descriptor, if one was opened."""
        if self._marker_read_fd is not None:
            os.close(self._marker_read_fd)
            self._marker_read_fd = None
        if self.mode == "gene_block" and self.marker_score_store is not None:
            self.marker_score_store.close_sequential_access()

    def _read_contiguous_marker_rows(self, source_start: int, source_end: int) -> np.ndarray:
        """Read rows directly into RAM, bypassing synchronous mmap page faults."""
        if self.mode == "gene_block":
            return self.marker_score_store.read_spot_window(source_start, source_end)

        marker_matrix = self.marker_score_adata.X
        values = np.empty(
            (source_end - source_start, marker_matrix.shape[1]),
            dtype=marker_matrix.dtype,
            order="C",
        )

        if self._marker_read_fd is None or not hasattr(os, "preadv"):
            values[:] = marker_matrix[source_start:source_end, :]
            return values

        byte_view = memoryview(values).cast("B")
        row_bytes = marker_matrix.shape[1] * marker_matrix.dtype.itemsize
        file_offset = int(getattr(marker_matrix, "offset", 0)) + source_start * row_bytes
        bytes_read = 0
        while bytes_read < byte_view.nbytes:
            count = os.preadv(
                self._marker_read_fd,
                [byte_view[bytes_read:]],
                file_offset + bytes_read,
            )
            if count <= 0:
                raise EOFError(
                    f"Unexpected end of marker-score memmap after {bytes_read:,}/"
                    f"{byte_view.nbytes:,} bytes"
                )
            bytes_read += count
        return values

    def load_marker_window(self, chunk_indices: list[int]) -> MarkerScoreWindow:
        """Read consecutive chunks with one large, contiguous memmap access."""
        if not chunk_indices:
            raise ValueError("Marker-score window cannot be empty")
        if not self.supports_sequential_marker_windows():
            raise RuntimeError("Sequential marker-score windows are unavailable")
        if any(
            right != left + 1
            for left, right in zip(chunk_indices, chunk_indices[1:], strict=False)
        ):
            raise ValueError("Marker-score window chunks must be consecutive")

        first_start, _, _, first_rows, _, _ = self._chunk_selection(chunk_indices[0])
        _, last_end, _, last_rows, _, _ = self._chunk_selection(chunk_indices[-1])
        if first_rows.size == 0 or last_rows.size == 0:
            raise ValueError("Marker-score window contains an empty chunk")

        selected_rows = self.spot_indices[first_start:last_end]
        source_start = int(selected_rows[0])
        source_end = int(selected_rows[-1]) + 1
        if source_end - source_start != selected_rows.size:
            raise RuntimeError("Marker-score window rows are not contiguous")

        # Read all columns once. Selecting thousands of columns directly from a
        # row-major memmap turns each tiny compute chunk into scattered page
        # faults. A large preadv fills RAM without faulting one page at a time;
        # selection from this buffer is then predictable and cheap.
        values = self._read_contiguous_marker_rows(source_start, source_end)
        return MarkerScoreWindow(tuple(chunk_indices), source_start, values)

    def fetch_chunk_from_marker_window(
        self,
        window: MarkerScoreWindow,
        chunk_index: int,
        defer_ldscore_to_compute: bool,
    ):
        """Materialize one normal pipeline item from a shared RAM window."""
        if chunk_index not in window.chunk_indices:
            raise ValueError(f"Chunk {chunk_index} is not present in its marker-score window")

        _, _, spot_names, _, absolute_start, absolute_end = self._chunk_selection(chunk_index)
        relative_start = absolute_start - window.source_start
        relative_end = absolute_end - window.source_start
        if relative_start < 0 or relative_end > window.values.shape[0]:
            raise RuntimeError("Marker-score chunk falls outside its loaded window")

        marker_score_chunk = window.values[relative_start:relative_end][
            :, self.marker_score_gene_indices
        ].T.astype(np.float32)
        if defer_ldscore_to_compute:
            return (
                None,
                spot_names,
                absolute_start,
                absolute_end,
                marker_score_chunk,
            )

        ldscore_chunk = self.compute_ldscore_from_marker_chunk(marker_score_chunk)
        return (
            ldscore_chunk.astype(np.float32, copy=False),
            spot_names,
            absolute_start,
            absolute_end,
        )

    def _fetch_direct_ldscore_chunk(self, memmap_start: int, memmap_end: int) -> np.ndarray:
        if self.ldscore_adata is None or self.ldscore_snp_positions is None:
            raise RuntimeError("Direct LD-score chunk source is not initialized")

        # Read a contiguous cell-column slice first to avoid slow HDF5 fancy row indexing.
        chunk_raw = self.ldscore_adata.X[:, memmap_start:memmap_end]
        if hasattr(chunk_raw, "toarray"):
            chunk_all_snps = chunk_raw.toarray()
        else:
            chunk_all_snps = np.asarray(chunk_raw)
        return np.take(chunk_all_snps, self.ldscore_snp_positions, axis=0)

    def _fetch_marker_score_chunk(self, row_selector: slice | np.ndarray) -> np.ndarray:
        if self.marker_score_adata is None or self.marker_score_gene_indices is None:
            raise RuntimeError("Marker-score chunk source is not initialized")

        if isinstance(row_selector, slice):
            marker_score_chunk = self.marker_score_adata.X[
                row_selector, self.marker_score_gene_indices
            ]
        else:
            # Select rows first so two advanced index arrays do not trigger
            # NumPy's pairwise-indexing semantics.
            marker_score_chunk = self.marker_score_adata.X[row_selector][
                :, self.marker_score_gene_indices
            ]
        return marker_score_chunk.T.astype(np.float32)

    def _fetch_gene_block_ldscore_chunk(
        self,
        row_selector: slice | np.ndarray,
    ) -> np.ndarray:
        """Accumulate annotated LD scores without materializing all marker genes."""
        if (
            self.marker_score_store is None
            or self.marker_score_gene_indices is None
            or self.snp_gene_weight_sparse is None
        ):
            raise RuntimeError("Gene-block chunk source is not initialized")

        if isinstance(row_selector, slice):
            marker_blocks = self.marker_score_store.iter_spot_marker_blocks(
                int(row_selector.start),
                int(row_selector.stop),
                self.marker_score_gene_indices,
            )
            return accumulate_annotated_ldscore_blocks(
                self.snp_gene_weight_sparse,
                marker_blocks,
            )

        marker_scores = self.marker_score_store.read_subset(
            row_selector,
            self.marker_score_gene_indices,
        )
        return self.compute_ldscore_from_marker_chunk(marker_scores.T)

    def compute_ldscore_from_marker_chunk(self, marker_score_chunk: np.ndarray) -> np.ndarray:
        """Compute SNP x spot LD scores from a gene x spot marker-score chunk on CPU."""
        if self.snp_gene_weight_sparse is None:
            raise RuntimeError("SNP-gene sparse weight matrix is not initialized")

        ldscore_chunk = self.snp_gene_weight_sparse @ marker_score_chunk
        if hasattr(ldscore_chunk, "toarray"):
            ldscore_chunk = ldscore_chunk.toarray()
        return ldscore_chunk.astype(np.float32, copy=False)

"""Cell chunk sources for spatial LDSC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gsMap.spatial_ldsc.backends.cpu_sparse import CPUSparseLDScore
from gsMap.utils.matrix_io import MatrixRowReader


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
    ):
        self.mode = mode
        self.chunk_starts = chunk_starts
        self.chunk_size = chunk_size
        self.cell_end = cell_end
        self.spot_indices = spot_indices
        self._contiguous_spot_rows = bool(
            spot_indices.size <= 1
            or (
                spot_indices[-1] - spot_indices[0] + 1 == spot_indices.size
                and np.all(np.diff(spot_indices) == 1)
            )
        )
        self.spot_names_filtered = spot_names_filtered
        self.ldscore_adata = ldscore_adata
        self.ldscore_snp_positions = ldscore_snp_positions
        self.marker_score_adata = marker_score_adata
        self.marker_score_gene_indices = marker_score_gene_indices
        self.snp_gene_weight_sparse = snp_gene_weight_sparse
        self.cpu_ldscore_backend = (
            CPUSparseLDScore(snp_gene_weight_sparse)
            if snp_gene_weight_sparse is not None
            else None
        )
        self._matrix_reader: MatrixRowReader | None = None

    def _chunk_selection(
        self, chunk_index: int
    ) -> tuple[int, int, pd.Index, np.ndarray, int, int]:
        """Resolve logical and source rows for one compute chunk."""
        if not 0 <= chunk_index < len(self.chunk_starts):
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
        if selected_rows.size and np.all(np.diff(selected_rows) == 1):
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
        else:
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

        return (
            ldscore_chunk.astype(np.float32, copy=False),
            spot_names,
            absolute_start,
            absolute_end,
        )

    def supports_sequential_marker_windows(self) -> bool:
        """Return whether marker rows can be read as large contiguous windows.

        The optimized path is intentionally limited to the row-major NumPy
        memmap format. Backed H5AD and Feather objects retain their existing
        readers. Non-contiguous sample filters retain the existing fancy-index
        fallback; the streaming path targets full, contiguous matrix scans.
        """
        if self.mode != "weight_matrix" or self.marker_score_adata is None:
            return False

        marker_matrix = self.marker_score_adata.X
        if (
            not isinstance(marker_matrix, np.memmap)
            or not marker_matrix.flags.c_contiguous
            or self.spot_indices.size == 0
        ):
            return False

        return self._contiguous_spot_rows

    def recommended_marker_window_chunks(self, target_bytes: int = 32 * 1024**2) -> int:
        """Choose a bounded I/O window near ``target_bytes`` automatically."""
        if not self.supports_sequential_marker_windows():
            return 1

        marker_matrix = self.marker_score_adata.X
        row_bytes = int(marker_matrix.shape[1]) * marker_matrix.dtype.itemsize
        rows_per_window = max(self.chunk_size, target_bytes // max(1, row_bytes))
        # The upper bound keeps queue bookkeeping bounded for unusually narrow
        # matrices while the byte target controls memory for normal datasets.
        return max(1, min(256, rows_per_window // self.chunk_size))

    def advise_sequential_marker_access(self) -> bool:
        """Open the descriptor used for large positional marker reads."""
        if not self.supports_sequential_marker_windows():
            return False

        marker_matrix = self.marker_score_adata.X
        if self._matrix_reader is not None:
            return self._matrix_reader.fd is not None
        try:
            self._matrix_reader = MatrixRowReader(marker_matrix, sequential=True)
        except (OSError, ValueError):
            return False

        return self._matrix_reader.fd is not None

    def close_sequential_marker_access(self) -> None:
        """Close the positional reader after its source thread stops."""
        if self._matrix_reader is not None:
            self._matrix_reader.close()
            self._matrix_reader = None

    def _read_contiguous_marker_rows(self, source_start: int, source_end: int) -> np.ndarray:
        """Read a contiguous RAM buffer without mmap faults when possible."""
        if self._matrix_reader is not None:
            return self._matrix_reader.read_contiguous(source_start, source_end)
        return np.array(self.marker_score_adata.X[source_start:source_end, :], copy=True)

    def load_marker_window(self, chunk_indices: list[int]) -> MarkerScoreWindow:
        """Read consecutive chunks with one large positional read."""
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
        if source_end - source_start != selected_rows.size or not np.all(
            np.diff(selected_rows) == 1
        ):
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

    def compute_ldscore_from_marker_chunk(self, marker_score_chunk: np.ndarray) -> np.ndarray:
        """Compute SNP x spot LD scores from a gene x spot marker-score chunk on CPU."""
        if self.cpu_ldscore_backend is None:
            raise RuntimeError("SNP-gene sparse weight matrix is not initialized")

        return self.cpu_ldscore_backend.compute(marker_score_chunk)

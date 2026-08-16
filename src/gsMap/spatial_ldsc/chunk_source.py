"""Cell chunk sources for spatial LDSC."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
        self.spot_names_filtered = spot_names_filtered
        self.ldscore_adata = ldscore_adata
        self.ldscore_snp_positions = ldscore_snp_positions
        self.marker_score_adata = marker_score_adata
        self.marker_score_gene_indices = marker_score_gene_indices
        self.snp_gene_weight_sparse = snp_gene_weight_sparse

    def fetch_chunk(self, chunk_index: int, defer_ldscore_to_compute: bool):
        """Fetch one chunk as either LD scores or marker scores."""
        if chunk_index >= len(self.chunk_starts):
            raise ValueError(f"Invalid chunk index {chunk_index}")

        start = self.chunk_starts[chunk_index]
        end = min(start + self.chunk_size, self.cell_end)

        spot_names = pd.Index(self.spot_names_filtered[start:end])
        selected_rows = self.spot_indices[start:end]
        absolute_start = selected_rows[0] if selected_rows.size else start
        absolute_end = selected_rows[-1] + 1 if selected_rows.size else absolute_start
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

    def _fetch_direct_ldscore_chunk(self, memmap_start: int, memmap_end: int) -> np.ndarray:
        if self.ldscore_adata is None or self.ldscore_snp_positions is None:
            raise RuntimeError("Direct LD-score chunk source is not initialized")

        # Read a contiguous cell-column slice first to avoid slow HDF5 fancy row indexing.
        chunk_raw = self.ldscore_adata.X[:, memmap_start:memmap_end]
        if hasattr(chunk_raw, "toarray"):
            chunk_all_snps = chunk_raw.toarray()
        else:
            chunk_all_snps = np.asarray(chunk_raw)
        return chunk_all_snps[self.ldscore_snp_positions, :]

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
        if self.snp_gene_weight_sparse is None:
            raise RuntimeError("SNP-gene sparse weight matrix is not initialized")

        ldscore_chunk = self.snp_gene_weight_sparse @ marker_score_chunk
        if hasattr(ldscore_chunk, "toarray"):
            ldscore_chunk = ldscore_chunk.toarray()
        return ldscore_chunk.astype(np.float32, copy=False)

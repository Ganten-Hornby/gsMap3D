"""Bounded, ordered result persistence for spatial LDSC."""

from __future__ import annotations

import gzip
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm

if TYPE_CHECKING:
    from typing import TextIO

    from gsMap.config import SpatialLDSCConfig

logger = logging.getLogger("gsMap.spatial_ldsc.result_writer")


@dataclass(slots=True)
class ChunkResult:
    """Regression output for one cell chunk."""

    chunk_idx: int
    betas: np.ndarray
    ses: np.ndarray
    spot_names: pd.Index
    abs_start: int
    abs_end: int


class ResultWriter:
    """Write out-of-order compute results as one ordered atomic CSV stream.

    Compute workers may finish out of order. Results are held only until the
    next consecutive chunk is available, then coalesced into a moderate output
    batch and written to a temporary gzip stream. The completed file is
    atomically published only after every expected chunk and row is validated.
    """

    def __init__(
        self,
        config: SpatialLDSCConfig,
        trait_name: str,
        output_dir: Path,
        n_spots: int,
        total_chunks: int,
        expected_cells: int,
        output_batch_rows: int = 50_000,
        max_pending_chunks: int = 64,
    ):
        self.config = config
        self.trait_name = trait_name
        self.output_dir = output_dir
        self.n_spots = n_spots
        self.total_chunks = total_chunks
        self.expected_cells = expected_cells
        self.output_batch_rows = output_batch_rows
        self.max_pending_chunks = max(1, max_pending_chunks)
        self.condition = threading.Condition()

        self.pending: dict[int, ChunkResult] = {}
        self.ready: list[ChunkResult] = []
        self.ready_rows = 0
        self.next_chunk = 0
        self.seen_chunks = np.zeros(total_chunks, dtype=np.bool_)
        self.min_spot_start = float("inf")
        self.max_spot_end = 0

        self.p_values = np.empty(expected_cells, dtype=np.float64)
        self.rows_written = 0
        self.nominal_significant = 0
        self.max_neg_log10_p = float("-inf")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".spatial_ldsc_",
            suffix=".partial.csv.gz",
            dir=self.output_dir,
        )
        os.close(descriptor)
        self.temporary_path = Path(temporary_name)
        self.handle: TextIO | None = None
        self.header_written = False
        self.final_path: Path | None = None
        self.closed = False

    def add_chunk_result(
        self,
        chunk_idx: int,
        betas: np.ndarray,
        ses: np.ndarray,
        spot_names: pd.Index,
        abs_start: int,
        abs_end: int,
    ) -> None:
        """Accept one compute result and drain newly consecutive chunks."""
        betas = np.asarray(betas).reshape(-1)
        ses = np.asarray(ses).reshape(-1)
        if not (betas.size == ses.size == len(spot_names)):
            raise ValueError(
                f"Chunk {chunk_idx} output length mismatch: beta={betas.size}, "
                f"se={ses.size}, spots={len(spot_names)}"
            )

        with self.condition:
            if self.closed:
                raise RuntimeError("Cannot add a result after the LDSC writer was closed")
            if not 0 <= chunk_idx < self.total_chunks:
                raise ValueError(
                    f"Invalid result chunk {chunk_idx}; expected 0..{self.total_chunks - 1}"
                )
            if self.seen_chunks[chunk_idx]:
                raise ValueError(f"Duplicate result chunk {chunk_idx}")

            while (
                len(self.pending) >= self.max_pending_chunks
                and chunk_idx != self.next_chunk
                and not self.closed
            ):
                self.condition.wait(timeout=0.1)
            if self.closed:
                raise RuntimeError("Cannot add a result after the LDSC writer was closed")

            self.seen_chunks[chunk_idx] = True
            self.min_spot_start = min(self.min_spot_start, abs_start)
            self.max_spot_end = max(self.max_spot_end, abs_end)
            self.pending[chunk_idx] = ChunkResult(
                chunk_idx=chunk_idx,
                betas=betas,
                ses=ses,
                spot_names=spot_names,
                abs_start=abs_start,
                abs_end=abs_end,
            )
            self._drain_consecutive()
            self.condition.notify_all()

    def _drain_consecutive(self) -> None:
        while self.next_chunk in self.pending:
            result = self.pending.pop(self.next_chunk)
            self.ready.append(result)
            self.ready_rows += len(result.spot_names)
            self.next_chunk += 1
            if self.ready_rows >= self.output_batch_rows:
                self._flush_ready()

    def _flush_ready(self) -> None:
        if not self.ready:
            return

        betas = np.concatenate([result.betas for result in self.ready]).astype(
            np.float64, copy=False
        )
        ses = np.concatenate([result.ses for result in self.ready]).astype(np.float64, copy=False)
        spot_names = np.concatenate(
            [np.asarray(result.spot_names, dtype=object) for result in self.ready]
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            z_scores = betas / ses
        p_values = norm.sf(z_scores)
        p_values = np.nan_to_num(p_values, nan=1.0, posinf=0.0, neginf=1.0)
        neg_log10_p = -np.log10(np.maximum(p_values, 1e-300))

        end_row = self.rows_written + p_values.size
        if end_row > self.expected_cells:
            raise ValueError(
                f"LDSC writer received {end_row:,} rows, exceeding expected "
                f"{self.expected_cells:,}"
            )
        self.p_values[self.rows_written : end_row] = p_values
        self.rows_written = end_row
        self.nominal_significant += int(np.count_nonzero(p_values < 0.05))
        if neg_log10_p.size:
            self.max_neg_log10_p = max(
                self.max_neg_log10_p,
                float(np.max(neg_log10_p)),
            )

        frame = pd.DataFrame(
            {
                "spot": spot_names,
                "beta": betas.astype(np.float32),
                "se": ses.astype(np.float32),
                "z": z_scores.astype(np.float32),
                "p": p_values,
                "neg_log10_p": neg_log10_p,
            }
        )
        if self.handle is None:
            self.handle = gzip.open(self.temporary_path, mode="wt", newline="")
        frame.to_csv(self.handle, index=False, header=not self.header_written)
        self.header_written = True
        self.ready.clear()
        self.ready_rows = 0

    def finalize(self) -> Path:
        """Validate completeness and atomically publish the final result."""
        with self.condition:
            if self.closed:
                if self.final_path is None:
                    raise RuntimeError("LDSC result writer was aborted")
                return self.final_path

            try:
                self._drain_consecutive()
                self._flush_ready()
                if self.next_chunk != self.total_chunks:
                    missing = np.flatnonzero(~self.seen_chunks)
                    preview = ", ".join(map(str, missing[:10]))
                    raise RuntimeError(
                        f"Missing {missing.size:,}/{self.total_chunks:,} LDSC chunks"
                        f" (first: {preview})"
                    )
                if self.rows_written != self.expected_cells:
                    raise RuntimeError(
                        f"LDSC result row mismatch: wrote {self.rows_written:,}, "
                        f"expected {self.expected_cells:,}"
                    )
                if self.handle is None:
                    raise RuntimeError("No LDSC results were written")

                self.handle.close()
                self.handle = None
                output_path = self.output_dir / self._generate_output_filename()
                os.replace(self.temporary_path, output_path)
                self.final_path = output_path
                self.closed = True
                self.condition.notify_all()
                self._log_statistics(output_path)
                return output_path
            except BaseException:
                self._abort_locked()
                raise

    def abort(self) -> None:
        """Discard an incomplete stream; no final-looking output is retained."""
        with self.condition:
            self._abort_locked()

    def _abort_locked(self) -> None:
        """Abort while holding ``condition`` and release blocked producers."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.temporary_path.unlink(missing_ok=True)
        self.closed = True
        self.condition.notify_all()

    def _generate_output_filename(self) -> str:
        base_name = f"{self.config.project_name}_{self.trait_name}"

        if self.config.cell_indices_range:
            start_cell, end_cell = self.config.cell_indices_range
            actual_start = max(int(self.min_spot_start), start_cell)
            actual_end = min(self.max_spot_end, end_cell)
            return f"{base_name}_cells_{actual_start}_{actual_end}.csv.gz"

        if self.min_spot_start == 0 and self.max_spot_end == self.n_spots:
            return f"{base_name}.csv.gz"

        if self.config.sample_filter:
            return (
                f"{base_name}_{self.config.sample_filter}_start{self.min_spot_start}"
                f"_end{self.max_spot_end}_total{self.n_spots}.csv.gz"
            )

        return (
            f"{base_name}_start{self.min_spot_start}_end{self.max_spot_end}"
            f"_total{self.n_spots}.csv.gz"
        )

    def _fdr_significant_count(self, alpha: float = 0.001) -> int:
        sorted_p = np.sort(self.p_values[: self.rows_written])
        n_values = sorted_p.size
        maximum_rank = 0
        block_size = 1_000_000
        for start in range(0, n_values, block_size):
            end = min(start + block_size, n_values)
            ranks = np.arange(start + 1, end + 1, dtype=np.float64)
            passing = np.flatnonzero(sorted_p[start:end] <= alpha * ranks / n_values)
            if passing.size:
                maximum_rank = start + int(passing[-1]) + 1
        return maximum_rank

    def _log_statistics(self, output_path: Path) -> None:
        bonferroni_threshold = 0.05 / self.rows_written
        p_values = self.p_values[: self.rows_written]
        n_bonferroni = int(np.count_nonzero(p_values < bonferroni_threshold))
        n_fdr = self._fdr_significant_count(alpha=0.001)

        logger.info("=" * 70)
        logger.info("STATISTICAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total spots: {self.rows_written:,}")
        logger.info(f"Cell range processed: [{self.min_spot_start}, {self.max_spot_end})")
        logger.info(f"Max -log10(p): {self.max_neg_log10_p:.2f}")
        logger.info("-" * 70)
        logger.info(f"Nominally significant (p < 0.05): {self.nominal_significant:,}")
        logger.info(f"Bonferroni threshold: {bonferroni_threshold:.2e}")
        logger.info(f"Bonferroni significant: {n_bonferroni:,}")
        logger.info(f"FDR significant (alpha=0.001): {n_fdr:,}")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {output_path}")

"""Result assembly and persistence for spatial LDSC."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from pathlib import Path

    from gsMap.config import SpatialLDSCConfig

logger = logging.getLogger("gsMap.spatial_ldsc.result_writer")


@dataclass
class ChunkResult:
    """Regression output for one cell chunk."""

    chunk_idx: int
    betas: np.ndarray
    ses: np.ndarray
    spot_names: pd.Index
    abs_start: int
    abs_end: int


class ResultWriter:
    """Validate, merge, save, and summarize spatial LDSC chunk results."""

    def __init__(
        self,
        config: SpatialLDSCConfig,
        trait_name: str,
        output_dir: Path,
        n_spots: int,
    ):
        self.config = config
        self.trait_name = trait_name
        self.output_dir = output_dir
        self.n_spots = n_spots
        self.results: list[ChunkResult] = []
        self.processed_chunks: set[int] = set()
        self.min_spot_start = float("inf")
        self.max_spot_end = 0

    def add_chunk_result(
        self,
        chunk_idx: int,
        betas: np.ndarray,
        ses: np.ndarray,
        spot_names: pd.Index,
        abs_start: int,
        abs_end: int,
    ):
        """Add processed chunk output to the result accumulator."""
        self.min_spot_start = min(self.min_spot_start, abs_start)
        self.max_spot_end = max(self.max_spot_end, abs_end)

        self.results.append(
            ChunkResult(
                chunk_idx=chunk_idx,
                betas=betas,
                ses=ses,
                spot_names=spot_names,
                abs_start=abs_start,
                abs_end=abs_end,
            )
        )
        self.processed_chunks.add(chunk_idx)

    def validate_merge_and_save(self, total_chunks: int) -> pd.DataFrame:
        """Validate completeness, merge chunk outputs, save, and return results."""
        if not self.results:
            raise ValueError("No results to merge")

        missing_chunks = set(range(total_chunks)) - self.processed_chunks
        if missing_chunks:
            logger.warning(f"Missing chunks: {sorted(missing_chunks)}")
            logger.warning(f"Processed {len(self.processed_chunks)}/{total_chunks} chunks")

        sorted_results = sorted(self.results, key=lambda x: x.chunk_idx)
        merged_df = pd.concat(
            [self._chunk_to_dataframe(result) for result in sorted_results],
            ignore_index=True,
        )

        output_path = self.output_dir / self._generate_output_filename()
        logger.info(f"Saving results to {output_path}")
        merged_df.to_csv(output_path, index=False, compression="gzip")

        self._log_statistics(merged_df, output_path, total_chunks)
        return merged_df

    def _chunk_to_dataframe(self, result: ChunkResult) -> pd.DataFrame:
        betas = result.betas.astype(np.float64)
        ses = result.ses.astype(np.float64)

        z_scores = betas / ses
        p_values = norm.sf(z_scores)
        log10_p = -np.log10(np.maximum(p_values, 1e-300))

        return pd.DataFrame(
            {
                "spot": result.spot_names,
                "beta": result.betas,
                "se": result.ses,
                "z": z_scores.astype(np.float32),
                "p": p_values,
                "neg_log10_p": log10_p,
            }
        )

    def _generate_output_filename(self) -> str:
        base_name = f"{self.config.project_name}_{self.trait_name}"

        if self.config.cell_indices_range:
            start_cell, end_cell = self.config.cell_indices_range
            actual_start = max(self.min_spot_start, start_cell)
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

    def _log_statistics(self, df: pd.DataFrame, output_path: Path, total_chunks: int):
        n_spots = len(df)
        bonferroni_threshold = 0.05 / n_spots
        n_bonferroni_sig = (df["p"] < bonferroni_threshold).sum()

        reject, _, _, _ = multipletests(df["p"], alpha=0.001, method="fdr_bh")
        n_fdr_sig = reject.sum()

        logger.info("=" * 70)
        logger.info("STATISTICAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total spots: {n_spots:,}")
        logger.info(f"Cell range processed: [{self.min_spot_start}, {self.max_spot_end})")
        logger.info(f"Max -log10(p): {df['neg_log10_p'].max():.2f}")
        logger.info("-" * 70)
        logger.info(f"Nominally significant (p < 0.05): {(df['p'] < 0.05).sum():,}")
        logger.info(f"Bonferroni threshold: {bonferroni_threshold:.2e}")
        logger.info(f"Bonferroni significant: {n_bonferroni_sig:,}")
        logger.info(f"FDR significant (alpha=0.001): {n_fdr_sig:,}")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {output_path}")

        if len(self.processed_chunks) < total_chunks:
            logger.warning(
                f"WARNING: Only processed {len(self.processed_chunks)}/{total_chunks} chunks"
            )

"""Durable throughput counters for completed (not merely dispatched) results."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("gsMap.spatial_ldsc.progress")


class LDSCProgress:
    """Count real cells, including partial tails, across pending traits only.

    Completion means synchronized regression output handed to the writers.
    It does not mean gzip output has been finalized. The final pipeline summary
    separately includes writer drain/publication time.
    """

    interval = 30.0

    def __init__(self, cells, traits, chunks, *, started=None, label="Shared LDSC", callback=None):
        self.total_cells, self.traits, self.total_chunks = cells, traits, chunks
        self.started = time.monotonic() if started is None else started
        self.last_time = self.started
        self.last_cells = 0
        self.label = label
        self.callback = callback

    def update(self, cells, chunks, *, dispatched=None, force=False):
        now = time.monotonic()
        if not force and now - self.last_time < self.interval:
            return
        elapsed = max(now - self.started, 1e-9)
        recent = max(now - self.last_time, 1e-9)
        record = dict(
            completed_cells=int(cells),
            completed_chunks=int(chunks),
            traits=int(self.traits),
            elapsed_seconds=elapsed,
            cell_traits_per_second=cells * self.traits / elapsed,
        )
        logger.info(
            "%s progress: %s/%s cells × %d traits computed; %d/%d chunks complete%s; "
            "throughput=%.1f cell-traits/s (recent %.1f); elapsed=%.1fs; writes may be queued",
            self.label,
            f"{cells:,}",
            f"{self.total_cells:,}",
            self.traits,
            chunks,
            self.total_chunks,
            "" if dispatched is None else f" ({dispatched} dispatched)",
            record["cell_traits_per_second"],
            (cells - self.last_cells) * self.traits / recent,
            elapsed,
        )
        if self.callback is not None:
            try:
                self.callback(record)
            except OSError as error:
                logger.warning("Could not persist LDSC progress: %s", error)
        self.last_time, self.last_cells = now, cells

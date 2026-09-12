"""Bounded, ordered reads for the shared-LD pipeline."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gsMap.spatial_ldsc.data.chunks import SpatialLDChunkSource


def prefetched_chunks(source: SpatialLDChunkSource, config) -> Iterator[tuple]:
    """Yield ordinary source chunks; close the iterator before closing its source.

    Only NumPy/memmap arrays support concurrent reads here. Shared HDF5 handles
    and other file-reader wrappers use one worker. Outstanding work is bounded
    to twice the worker count and joined on close, including exceptional exit.
    """
    marker = getattr(source, "marker_score_adata", None)
    parallel = marker is not None and isinstance(marker.X, np.ndarray)
    workers = min(4, max(1, getattr(config, "ldsc_read_workers", 4))) if parallel else 1
    depth = 2 * workers
    count = len(source.chunk_starts)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ldsc-shared-reader") as pool:
        futures = deque()
        next_chunk = 0
        try:
            while next_chunk < min(depth, count):
                futures.append(pool.submit(source.fetch_chunk, next_chunk, True))
                next_chunk += 1
            while futures:
                chunk = futures.popleft().result()
                if next_chunk < count:
                    futures.append(pool.submit(source.fetch_chunk, next_chunk, True))
                    next_chunk += 1
                yield chunk
        finally:
            for future in futures:
                future.cancel()

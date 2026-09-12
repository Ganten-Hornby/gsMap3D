"""Marker-score readers shared with report generation.

LDSC data preparation lives in :mod:`gsMap.spatial_ldsc.data.loading`.
This import surface is retained for callers outside the LDSC execution layer.
"""

from .data.loading import (
    acquire_marker_scores_memmap,
    load_marker_scores_memmap_format,
)

__all__ = ["acquire_marker_scores_memmap", "load_marker_scores_memmap_format"]

"""
latent2gene subpackage for gsMap

This package contains all components for converting latent representations to gene marker scores:
- Rank calculation from latent representations
- Connectivity matrix building (spatial → anchor → homogeneous)
- Marker score calculation
- Memory-mapped storage utilities
"""

from .connectivity import ConnectivityMatrixBuilder
from .entry_point import run_latent_to_gene
from .marker_scores import MarkerScoreCalculator
from .rank_calculator import RankCalculator

__all__ = [
    'run_latent_to_gene'
]

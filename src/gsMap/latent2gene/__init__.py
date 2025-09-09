"""
latent2gene subpackage for gsMap

This package contains all components for converting latent representations to gene marker scores:
- Rank calculation from latent representations
- Connectivity matrix building (spatial → anchor → homogeneous)
- Marker score calculation
- Memory-mapped storage utilities
"""

from .rank_calculator import RankCalculator
from .connectivity import ConnectivityMatrixBuilder
from .marker_scores import MarkerScoreCalculator
from .entry_point import run_latent_to_gene

__all__ = [
    'run_latent_to_gene'
]
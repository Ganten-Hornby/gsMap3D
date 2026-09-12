"""Spatial LDSC data, numerical kernels, and execution orchestration.

Subpackages:
    data        input alignment, marker/LD chunks and atomic result output
    backends    sparse LD construction and CPU/GPU regression implementations
    kernels     general JAX regression mathematics
    execution   engines, bounded prefetch and cell-sharded multi-GPU workers

``spatial_ldsc_jax.run_spatial_ldsc_jax`` remains the CLI/Python entrypoint.
``io`` retains the marker-score reader interface used by report generation.

The package initializer intentionally does not import CUDA/compute backends.
"""

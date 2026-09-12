"""Spatial LDSC data, numerical kernels, and execution orchestration.

Module map:
    io / trait_data                 input loading and trait-specific SNP design
    chunk_source / prefetch         marker/LD access and bounded ordered reads
    regression_backend             shared backend policy and result ownership
    moment_* / regression_kernel /
        multitrait_kernel          numerical implementations
    multitrait / multi_gpu*        shared dispatch and cell-sharded workers
    result_writer                  ordered, atomic per-trait output
    pipeline / ldscore_quick_mode  independent-trait execution and engine setup

The package initializer intentionally does not import CUDA/compute backends.
"""

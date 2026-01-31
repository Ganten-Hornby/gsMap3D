# Configuration Reference

The following options are used in the `gsMap` CLI (`gsmap quick-mode`) and Python API.

## Project Setup

- **`--workdir`**:
    The root directory where all output files will be saved.
- **`--project-name`**:
    Name of the project. A subdirectory with this name will be created inside `workdir`.

## Input Data

- **`--dataset-type`**:
    Specifies the data modality.
    - `spatial2D`: Standard 2D spatial transcriptomics (e.g., Visium, Xenium).
    - `spatial3D`: 3D spatial stacks (multiple aligned 2D slices).
    - `scrna`: scRNA-seq (uses KNN on latent space instead of spatial neighbors).

- **`--h5ad-path`**:
    Space-separated list of `.h5ad` file paths.
- **`--h5ad-yaml`**:
    Path to a YAML file mapping sample names to file paths (useful for large batches).
- **`--h5ad-list-file`**:
    Path to a text file containing a list of file paths (one per line).
- **`--sumstats-config-file`** (or `--trait-name`/`--sumstats-file`):
    Configuration for GWAS traits.
- **`--snp-gene-weight-adata-path`**:
    Path to the pre-calculated SNP-to-gene weight matrix (reference resource).

## Algorithm Parameters

- **`--spatial-neighbors`**:
    Number of nearest neighbors to search in physical space (default: 301).
- **`--homogeneous-neighbors`**:
    Number of molecularly similar neighbors to use for GSS calculation (default: 21).

## 3D Specific Options

For `spatial3D` datasets, `gsMap` can integrate information across Z-slices.

- **`--n-adjacent-slices`**:
    Number of slices above and below the focal slice to search for neighbors (e.g., 1 means search focal ± 1 slice).
- **`--cross-slice-marker-score-strategy`**:
    Strategy for aggregating scores across slices:
    - `hierarchical_pool` (Default): Computes scores for each slice independently, then averages them. Robust against batch effects between slices.
    - `global_pool`: Selects the top neighbors globally across all slices. Best when slices are perfectly aligned and normalized.
    - `per_slice_pool`: Forces a fixed number of neighbors from each slice.

## Advanced Options

- **`--latent-representation-cell`**:
    Key in `adata.obsm` for cell identity embedding (default: `emb_cell`).
- **`--latent-representation-niche`**:
    Key in `adata.obsm` for spatial niche embedding (default: `emb_niche`).
- **`--use-gpu` / `--no-gpu`**:
    Enable or disable JAX GPU acceleration.

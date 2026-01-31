# Configuration Reference

The following options are used in the `gsMap` CLI (`gsmap quick-mode`) and Python API.

## Project Setup

- **`--workdir`**:
    The root directory where all output files will be saved.
- **`--project-name`**:
    Name of the project. A subdirectory with this name will be created inside `workdir`.

## Spatial Transcriptomics Input

For details on the required file structure (counts, coordinates, annotations), see [ST Data Format](input_data_format.md#st-data).

- **`--dataset-type`**:
    Specifies the data modality.
    - `spatial2D`: Standard 2D spatial transcriptomics (e.g., Visium, Stereo-seq).
    - `spatial3D`: 3D spatial stacks (multiple aligned 2D slices).
    <!-- - `scrna`: scRNA-seq (uses KNN on latent space instead of spatial neighbors). -->

### H5ad Path

There are three ways to provide input AnnData (.h5ad) files:

1. **Single or Multiple Files** (`--h5ad-path`):
   Specify each file with its own flag. Sample names are derived from filenames (e.g., `sample1.h5ad` -> `sample1`).

   ```bash
   --h5ad-path data/sample1.h5ad \
   --h5ad-path data/sample2.h5ad
   ```

2. **YAML Configuration** (`--h5ad-yaml`):
   A YAML file where keys are sample names and values are file paths.

   ```yaml
   sample1: /path/to/sample1.h5ad
   sample2: /path/to/sample2.h5ad
   ```

3. **List File** (`--h5ad-list-file`):
   A text file where each line is a file path. Sample names are derived from filenames.

   ```text
   /path/to/sample1.h5ad
   /path/to/sample2.h5ad
   ```

```{important}
For `spatial3D` datasets, the order of file paths defines the Z-axis order of slices. Ensure files are listed in the same order as they appear along the Z-axis (e.g., from bottom to top or vice versa).
```

### Data Layers & Coordinates

- **`--data-layer`**: The key in `adata.layers` containing raw counts (e.g., 'counts'). If not provided, defaults to `adata.X`.
- **`--spatial-key`**: The key in `adata.obsm` containing spatial coordinates (default: 'spatial').

```{important}
For `spatial3D` datasets, the spatial coordinates must be **aligned across slices**. Ensure all slices share a common coordinate system before running gsMap.
```

### Annotation

- **`--annotation`**: Key in `adata.obs` containing cell type or spatial domain annotations. Constrains the homogeneous neighbor search within same annotation, improving the accuracy of homogeneous neighbor finding.
- **`--two-stage`**: If provided, this annotation is used to refine the latent embedding in the find latent representation stage.

```{note}
If both cell type and spatial domain are available, using **cell type** is suggested for better homogeneous cell finding.
```

## GWAS Summary Statistics


For details on the required GWAS file format, see [GWAS Data Format](input_data_format.md#gwas-data).


There are two ways to provide GWAS summary statistics:

1. **Single Trait** (`--trait-name` + `--sumstats-file`):
   Provide a single trait name and its corresponding summary statistics file.

   ```bash
   --trait-name "IQ" --sumstats-file "./GWAS/IQ_NG_2018.sumstats.gz"
   ```

2. **Multiple Traits** (`--sumstats-config-file`):
   Provide a YAML configuration file mapping trait names to file paths.

   ```yaml
   # gwas_config.yaml
   IQ: ./GWAS/IQ_NG_2018.sumstats.gz
   Height: ./GWAS/GIANT_EUR_Height_2022_Nature.sumstats.gz
   MCHC: ./GWAS/BCX2_MCHC_EA_GWAMA.sumstats.gz
   ```

   ```bash
   --sumstats-config-file "./GWAS/gwas_config.yaml"
   ```

## SNP to Gene & LD weights

These files define how SNPs are linked to genes and the LD structure used for heritability partitioning. You can use the pre-calculated weights provided by gsMap, or build your own using the `gsmap ldscore-weight-matrix` command with a customized SNP-to-gene BED file and PLINK reference panel. See [Computing Custom LD Score Weight Matrix](../ldscore_weight_matrix.md) for detailed instructions.

- **`--snp-gene-weight-adata-path`**:
    Path to the pre-calculated SNP-to-gene weight matrix (.h5ad format).

- **`--w-ld-dir`**:
    Directory containing LD score weights for heritability partitioning.

## Homogeneous Neighbor & GSS

- **`--spatial-neighbors`**:
    Number of nearest neighbors to search in physical space (default: 301).
- **`--homogeneous-neighbors`**:
    Number of molecularly similar neighbors to use for GSS calculation (default: 21).

### 3D Specific Options

For `spatial3D` datasets, `gsMap` can search for homogeneous neighbors across aligned adjacent slices.

- **`--n-adjacent-slices`**:
    Number of slices above and below the focal slice to include in the neighbor search (e.g., `1` means focal slice ± 1 slice).
- **`--cross-slice-marker-score-strategy`**:
    Strategy for calculating Gene Specificity Scores across slices:
    - `hierarchical_pool` (Default): Select *K* homogeneous neighbors from each adjacent slice independently, compute a per-slice GSS, then average across slices. Robust to batch effects between slices.
    - `global_pool`: Select the top {math}`K \times (2 \times n_\text{adjacent_slices} + 1)` homogeneous neighbors globally across all adjacent slices. Each slice may contribute a variable number of neighbors.
    - `per_slice_pool`: Select *K* homogeneous neighbors from each adjacent slice, then compute a single GSS from the pooled neighbors.  

## Advanced Options

- **`--latent-representation-cell`**:
    Key in `adata.obsm` for cell identity embedding (default: `emb_cell`).
- **`--latent-representation-niche`**:
    Key in `adata.obsm` for spatial niche embedding (default: `emb_niche`).
- **`--use-gpu` / `--no-gpu`**:
    Enable or disable JAX GPU acceleration.

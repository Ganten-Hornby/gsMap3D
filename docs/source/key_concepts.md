(key_concepts)=

# Key Concepts

This guide explains the core data structures, pipeline stages, and configuration options in `gsMap3D`.

```{note}
For usage demonstrations, please refer to the {doc}`/2d_tutorial/index` or {doc}`/3d_tutorial/index`.
```

## Basic Usage

`gsMap3D` can be used as a Python library or via the command line.

````{tab} CLI
```bash
gsmap quick-mode \
    --workdir ./output \
    --project-name my_project \
    --h5ad-path sample1.h5ad \
    --dataset-type spatial2D \
    --trait-name Trait1 \
    --sumstats-file trait1.sumstats.gz \
    --snp-gene-weight-adata-path ./resources/snp_gene_weights.h5ad \
    --w-ld-dir ./resources/w_ld
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="./output",
    project_name="my_project",
    h5ad_path="sample1.h5ad",
    dataset_type="spatial2D",
    trait_name="Trait1",
    sumstats_file="trait1.sumstats.gz",
    snp_gene_weight_adata_path="./resources/snp_gene_weights.h5ad",
    w_ld_dir="./resources/w_ld"
)

run_quick_mode(config)
```
````

## Pipeline Overview

The `gsMap3D` pipeline integrates spatial transcriptomics with GWAS summary statistics through five key stages:

```{mermaid}
graph TD
    %% Define styles
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,rx:10,ry:10;
    classDef output fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;

    %% Nodes
    ST(Spatial Transcriptomics Data):::input
    GWAS(GWAS Summary Statistics):::input

    subgraph Integrated Pipeline
        direction TB
        S1[1. Latent Representation<br/><i>Dual Embedding</i>]:::process
        S2[2. Find Homogeneous Cells]:::process
        S3[3. Gene Specificity Score<br/><i>GSS Calculation</i>]:::process
        S4[4. Spatial LDSC<br/><i>Cell-Trait Association</i>]:::process
        S5[5. Region Identification<br/><i>Cauchy Combination</i>]:::process
    end

    Report(Interactive Report):::output

    %% Edges
    ST --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4
    GWAS --> S4
    S4 --> S5
    S5 --> Report
```

1.  **Dual Embeddings**: gsMap3D constructs batch-corrected dual embeddings to capture complementary aspects of cellular organization. High-dimensional gene expression profiles are projected into a cell-identity embedding, which represents intrinsic cellular states independent of spatial location. In parallel, gene expression is jointly modeled with spatial coordinates to generate a spatial-domain (cell-niche) embedding, which captures local tissue architecture and microenvironmental context.

2. **Identification of Homogeneous cells** Using the dual embeddings, gsMap3D identifies homogeneous cells for each spot by jointly considering transcriptomic similarity and spatial context. For 2D ST data, homogeneous cells are identified within the same section, whereas for 3D ST data, homogeneous cells span adjacent sections, enabling volumetric identification of homogeneous cells across the tissue.

3.  **Gene Specificity Score**: gsMap3D computes a Gene Specificity Score (GSS) for each gene in each cell by aggregating normalized gene expression ranks across its homogeneous cells. The GSS quantifies how highly and specifically a gene is expressed in a given cell.

4.  **Spatial LDSC**: The cell-level GSS annotations are integrated with GWAS summary statistics using S-LDSC to partition trait heritability. This framework assesses trait heritability enrichment of specific cells within specific spatial context.

5.  **Spatial Region or Cell-Type Association** To assess trait associations at the level of spatial regions or cell types, gsMap3D aggregates cell-level association p-values using the Cauchy combination test. This yields robust region- or cell-type–level association statistics while accounting for heterogeneous signals across constituent cells.

## Key Configurations

The following options are used in the `gsMap3D` CLI (`gsmap quick-mode`) and Python API.

### Project Setup

- **`--workdir`**:
    The root directory where all output files will be saved.
- **`--project-name`**:
    Name of the project. A subdirectory with this name will be created inside `workdir`.

### Spatial Transcriptomics Input

The input ST data must be an h5ad file containing at least the gene expression matrix and spatial coordinates.

- **Counts**: Raw UMI counts are preferred (typically in `adata.X` or `adata.layers['counts']`).
- **Coordinates**: Spatial coordinates stored in `adata.obsm['spatial']`.
- **Gene Names**: Must match the species of your data. Use `--homolog-file` to map non-human genes to human orthologs if necessary.
- **Annotations**: Optionally, the h5ad file may include spot (cell) annotations in the `obs` attribute.

```python
import scanpy as sc

adata = sc.read_h5ad("gsMap_example_data/ST/E16.5_E1S1.MOSTA.h5ad")

print(adata.layers["count"].shape)
print(adata.obsm["spatial"].shape)
print(adata.obs["annotation"].value_counts().head())
```

- **`--dataset-type`**:
    Specifies the data modality.
    - `spatial2D`: Standard 2D spatial transcriptomics (e.g., Visium, Stereo-seq).
    - `spatial3D`: 3D spatial stacks (multiple aligned 2D slices).

#### H5ad Path

There are three ways to provide input AnnData (.h5ad) files:

1. **Single or Multiple Files** (`--h5ad-path`):
   Specify each file with its own flag. Sample names are derived from filenames (e.g., `sample1.h5ad` -> `sample1`).

   ```bash
   --h5ad-path data/sample1.h5ad \
   --h5ad-path data/sample2.h5ad
   ```

2. **YAML Configuration (recommended)** (`--h5ad-yaml`):
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

#### Data Layers & Coordinates

- **`--data-layer`**: The key in `adata.layers` containing raw counts (e.g., 'counts'). If not provided, defaults to `adata.X`.
- **`--spatial-key`**: The key in `adata.obsm` containing spatial coordinates (default: 'spatial').

```{important}
For `spatial3D` datasets, the spatial coordinates must be **aligned across slices**. Ensure all slices share a common coordinate system before running gsMap.
```

#### Annotation

- **`--annotation`**: Key in `adata.obs` containing cell type or spatial domain annotations. Constrains the homogeneous neighbor search within same annotation, improving the accuracy of homogeneous neighbor finding.
- **`--two-stage`**: If provided, this annotation is used to refine the latent embedding in the find latent representation stage.

```{note}
If both cell type and spatial domain are available, using **cell type** is suggested for better homogeneous cell finding.
```

### GWAS Summary Statistics

The input GWAS data is a text file containing at least the columns for SNP (rs number), Z (Z-statistics), and N (sample size). Column headers are keywords used by gsMap3D.

```shell
zcat gsMap_example_data/GWAS/IQ_NG_2018.sumstats.gz | head -n 5

SNP  A1 A2 Z N
rs12184267 T C 0.916 225955
rs12184277 G A 0.656 226215
rs12184279 A C 1.050 226224
rs116801199 T G 0.300 226626
```

#### How to Format GWAS Data

You can convert GWAS summary data into the required format using custom code. For convenience, gsMap3D provides a command to do this. Below is an example of how to use the command.

Download the human height GWAS data and decompress it.

```bash
wget https://portals.broadinstitute.org/collaboration/giant/images/4/4e/GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_ALL.gz

gzip -d GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_ALL.gz
```

Convert the summary statistics to the required format.

```bash
gsmap format_sumstats \
--sumstats 'GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_ALL' \
--out 'HEIGHT'
```

You will obtain a file named HEIGHT.sumstats.gz

```bash
zcat HEIGHT.sumstats.gz | head -n 5

SNP  A1 A2 Z N
rs3131969 G A 0.328 1494218.000
rs3131967 C T 0.386 1488150.000
rs12562034 A G 1.714 1554976.000
rs4040617 G A -0.463 1602016.000
```

For more usage options, please refer to:

```bash
gsMap format_sumstats -h
```

#### Providing GWAS Data

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

### SNP to Gene & LD Weights

These files define how SNPs are linked to genes and the LD structure used for heritability partitioning. You can use the pre-calculated weights provided by gsMap, or build your own using the `gsmap ldscore-weight-matrix` command with a customized SNP-to-gene BED file and PLINK reference panel. See [Computing Custom LD Score Weight Matrix](ldscore_weight_matrix.md) for detailed instructions.

- **`--snp-gene-weight-adata-path`**:
    Path to the pre-calculated SNP-to-gene weight matrix (.h5ad format).

- **`--w-ld-dir`**:
    Directory containing LD score weights for heritability partitioning.

### Homogeneous Neighbor & GSS

- **`--spatial-neighbors`**:
    Number of nearest neighbors to search in physical space (default: 301).
- **`--homogeneous-neighbors`**:
    Number of molecularly similar neighbors to use for GSS calculation (default: 21).

#### 3D Specific Options

For `spatial3D` datasets, `gsMap3D` can search for homogeneous neighbors across aligned adjacent slices.

- **`--n-adjacent-slices`**:
    Number of slices above and below the focal slice to include in the neighbor search (e.g., `1` means focal slice ± 1 slice).
- **`--cross-slice-marker-score-strategy`**:
    Strategy for calculating Gene Specificity Scores across slices:
    - `hierarchical_pool` (Default): Select *K* homogeneous neighbors from each adjacent slice independently, compute a per-slice GSS, then average across slices. Robust to batch effects between slices.
    - `global_pool`: Select the top K × (2 × n_adjacent_slices + 1) homogeneous neighbors globally across all adjacent slices. Each slice may contribute a variable number of neighbors.
    - `per_slice_pool`: Select *K* homogeneous neighbors from each adjacent slice, then compute a single GSS from the pooled neighbors.

### Advanced Options

- **`--latent-representation-cell`**:
    Key in `adata.obsm` for cell identity embedding (default: `emb_cell`).
- **`--latent-representation-niche`**:
    Key in `adata.obsm` for spatial niche embedding (default: `emb_niche`).
- **`--use-gpu` / `--no-gpu`**:
    Enable or disable JAX GPU acceleration.

## See Also

- {doc}`Output Files Reference <output_files>`: Detailed documentation of all output files and their formats
- {doc}`2D Tutorial <2d_tutorial/index>`: Step-by-step guide for 2D spatial transcriptomics
- {doc}`3D Tutorial <3d_tutorial/index>`: Step-by-step guide for 3D spatial transcriptomics
- {doc}`CLI Reference <cli_reference>`: Complete command-line reference

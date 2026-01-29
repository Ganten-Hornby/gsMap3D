# CS8 Human Embryo 3D Tutorial

`gsMap3D` extends the gsMap framework to native 3D tissue architectures. By leveraging the continuity of biological domains across adjacent tissue sections, it integrates GWAS data with 3D-reconstructed spatial transcriptomics (ST) data to map trait-associated spots in three-dimensional space.

## Advantages of 3D gsMap

1.  **3D Contextual Mapping**: Captures how genetic risk is embedded within continuous 3D cellular landscapes, bridging the gap between genetic discoveries and spatial cellular manifestations.
2.  **Dual-Embedding Strategy**: Uses both **Cell Embedding** (transcriptomic identity) and **Spatial-Domain Embedding** (local tissue architecture) to identify homogeneous spots across slices, preventing signal leakage and over-smoothing.
3.  **Cross-Slice Integration**: Leverages adjacent slices to model 3D niches, revealing spatial gradients and organizational patterns invisible in 2D snapshots.

## Methodology Overview

The 3D pipeline follows a four-step process:
- **Step 1: Dual Embeddings**: Learns latent representations capturing both spot identity and spatial context.
- **Step 2: 3D Gene Specificity Score (GSS)**: Identifies "homogeneous spots" across adjacent sections and calculates a specificity score for each gene in each 3D niche.
- **Step 3: Genetic Association**: Uses S-LDSC to test for heritability enrichment of GWAS traits in spots with high GSS.
- **Step 4: Regional Aggregation**: Uses the Cauchy Combination Test (CCT) to aggregate spot-level p-values into robust associations for anatomical regions or cell types.

## Case Study: Human Gastrulation (CS8 Human Embryo)

This example demonstrates how to run `gsMap` on a 3D dataset comprising multiple slices of a human embryo.

### Preparation

First, set up environment variables for the resource and data directories:

```bash
WORK_DIR="./gsmap_3d_tutorial"
GSMAP_RESOURCE_DIR="./gsMap_resource_v2"
GASTRULATION_3D_DATA_DIR="./Human_Gastrulation_3D"

mkdir -p $WORK_DIR
```

### Running the Analysis

Run the `quick-mode` command with the `--dataset-type spatial3D` flag:

```bash
gsmap quick-mode \
    --workdir "$WORK_DIR" \
    --project-name "human_gastrulation_3d" \
    --dataset-type "spatial3D" \
    --h5ad-list-file "$GASTRULATION_3D_DATA_DIR/sample_path_list.txt" \
    --w-ld-dir "${GSMAP_RESOURCE_DIR}/quick_mode/weights_hm3_no_hla" \
    --snp-gene-weight-adata-path "${GSMAP_RESOURCE_DIR}/quick_mode/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad" \
    --sumstats-config-file "${GASTRULATION_3D_DATA_DIR}/GWAS/gwas_config.yaml" \
    --annotation "clusters" \
    --spatial-key "spatial_3d" \
    --data-layer "counts" \
    --no-high-quality-cell-qc \
    --memmap-tmp-dir "/data/tmp"
```

### Parameters for 3D Analysis

- `--dataset-type`: Set to `spatial3D` to enable 3D neighborhood search and cross-slice integration.
- `--h5ad-list-file`: A text file where each line is a path to an H5AD file representing one slice.
- `--spatial-key`: The key in `adata.obsm` containing 3D coordinates (e.g., `spatial_3d`).
- `--memmap-tmp-dir`: Directory for memory-mapped files. Using an SSD or high-speed local storage significantly boosts performance for 3D datasets.
- `--no-high-quality-cell-qc`: Skips the high-quality cell filter if your dataset is already curated or you wish to process all spots.

### Interpreting the Results

In the generated 3D report, you can observe how different traits map to specific germ layers:
- **Intelligence (IQ)**: Specifically mapped to the **ectoderm**, the progenitor of the nervous system.
- **MCHC (Mean Corpuscular Hemoglobin Concentration)**: Mapped to the **endoderm**, the progenitor of blood-forming organs like the liver.
- **Height**: Mapped across multiple germ layers, reflecting its highly polygenic and widespread biological influence.

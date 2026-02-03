# CS8 Human Embryo 3D Tutorial

By leveraging the continuity of biological domains across adjacent tissue sections, `gsMap3D` integrates GWAS data with 3D-reconstructed ST data to map trait-associated spots in 3D space.


## Human Gastrulation (CS8 Human Embryo)

This example demonstrates how to run **gsMap3D** on a 3D ST dataset from an early-stage human embryo.

### Preparation

First, download the required resources:

```bash
# Download gsMap3D quick mode resources
wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_quick_mode_resource.tar.gz
tar -xvzf gsMap_quick_mode_resource.tar.gz

# Download 3D example data (Human Gastrulation)
wget https://yanglab.westlake.edu.cn/data/gsMap/Human_Gastrulation_3D.tar.gz
tar -xvzf Human_Gastrulation_3D.tar.gz
```

Set up environment variables for the resource and data directories:

```bash
WORK_DIR="./gsmap_3d_tutorial"
GSMAP_RESOURCE_DIR="./gsMap_quick_mode_resource"
GASTRULATION_3D_DATA_DIR="./Human_Gastrulation_3D"

mkdir -p $WORK_DIR
```

### Running the Analysis

Run the `quick-mode` command with the `--dataset-type spatial3D` flag:

````{tab} CLI
```bash
gsmap quick-mode \
    --workdir "$WORK_DIR" \
    --project-name "human_gastrulation_3d" \
    --dataset-type "spatial3D" \
    --h5ad-list-file "$GASTRULATION_3D_DATA_DIR/sample_path_list.txt" \
    --w-ld-dir "${GSMAP_RESOURCE_DIR}/weights_hm3_no_hla" \
    --snp-gene-weight-adata-path "${GSMAP_RESOURCE_DIR}/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad" \
    --sumstats-config-file "${GASTRULATION_3D_DATA_DIR}/GWAS/gwas_config.yaml" \
    --annotation "clusters" \
    --spatial-key "spatial_3d" \
    --data-layer "counts" \
    --no-high-quality-cell-qc \
    --memmap-tmp-dir "/data/tmp"
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="./gsmap_3d_tutorial",
    project_name="human_gastrulation_3d",
    dataset_type="spatial3D",
    h5ad_list_file="./Human_Gastrulation_3D/sample_path_list.txt",
    w_ld_dir="./gsMap_quick_mode_resource/weights_hm3_no_hla",
    snp_gene_weight_adata_path="./gsMap_quick_mode_resource/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad",
    sumstats_config_file="./Human_Gastrulation_3D/GWAS/gwas_config.yaml",
    annotation="clusters",
    spatial_key="spatial_3d",
    data_layer="counts",
    high_quality_cell_qc=False,
    memmap_tmp_dir="/data/tmp"
)

run_quick_mode(config)
```
````

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

### Output Description

The pipeline generates two main directories:

**`report_data/`** - Data files for downstream analysis:
- `spot_metadata.csv` - Spot-level coordinates and annotations
- `cauchy_results.csv` - Cauchy combination test results
- `umap_data.csv` - UMAP coordinates for visualization
- `gene_list.csv` - List of genes used in analysis
- `gss_stats/` - Gene-trait correlation statistics
- `manhattan_data/` - Manhattan plot data per trait
- `spatial_3d/` - 3D spatial data (for `spatial3D` datasets)

**`gsmap_web_report/`** - Self-contained interactive web report:
- `index.html` - Main report entry point
- `spatial_plots/` - Static LDSC spatial plots
- `gene_diagnostic_plots/` - Gene expression diagnostics
- `annotation_plots/` - Annotation-based plots
- `spatial_3d/` - Interactive 3D HTML visualizations

**Viewing the Report:**

1. **Remote Server**: Start a temporary web server:
   ```bash
   gsmap report-view ./gsmap_web_report --port 8080 --no-browser
   ```

2. **Local PC**: Copy the `gsmap_web_report` folder to your machine and open `index.html` in a browser.

## See Also

- {doc}`Output Files Reference <../output_files>`: Detailed documentation of all output files
- {doc}`2D Tutorial <../2d_tutorial/mouse_embryo>`: For 2D spatial transcriptomics analysis
- {doc}`Customization Guide <../advanced_usage>`: For custom SNP-to-gene weights and embeddings
- {doc}`Scalability <../scalability>`: For performance optimization tips

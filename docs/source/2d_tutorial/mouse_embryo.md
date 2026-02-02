# Mouse Embryo 2D (Quick Mode)

The `Quick Mode` option provides a simplified and efficient way to execute the entire `gsMap` pipeline. It minimizes running time and configuration complexity by utilizing pre-calculated weights based on the 1000G EUR reference panel and protein-coding genes from BED file of Gencode v46. This mode is ideal for users who prefer a streamlined approach. For a more customizable experience, such as using custom BED files, reference panels, and more adjustable parameters, please refer to the {doc}`Customization Guide <../advanced_usage>` guide.

**Note**: This tutorial is designed for 2D ST data. For 3D ST data, please refer to the {doc}`3D Tutorial <../3d_tutorial/human_gastrulation>`.

.. note::
    
    Before starting, ensure you have downloaded the necessary reference resources (SNP-gene weights, LD scores, etc.). 
    `Download gsMap Resources <https://github.com/JianYang-Lab/gsMap>`_

## Preparation

Make sure you have {doc}`installed <../install>` the `gsMap` package before proceeding.

### 1. Download Dependencies

The `gsMap` package in quick mode requires the following resources:

- **LD reference panel weights**, for heritability partitioning.
- **SNP-to-gene weight matrix**, linking SNPs to gene expression specificity.
- **Homologous gene transformations file** (optional), to map genes between species.

To download all the required files:

```bash
wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_quick_mode_resource.tar.gz
tar -xvzf gsMap_quick_mode_resource.tar.gz
```

Directory structure:

```bash
tree -L 2 gsMap_quick_mode_resource

gsMap_quick_mode_resource/
├── 1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad
└── weights_hm3_no_hla
    ├── weights.1.l2.ldscore.gz
    ├── ...
    └── weights.22.l2.ldscore.gz
```

For homolog files (mouse/macaque to human gene mapping), download separately:

```bash
wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_homologs.tar.gz
tar -xvzf gsMap_homologs.tar.gz
```

### 2. Download Example Data

To run the quick mode example, you can download the example data as follows:

```bash
wget https://yanglab.westlake.edu.cn/data/gsMap/mouse_embryo_E16_example_data.tar.gz
tar -xvzf mouse_embryo_E16_example_data.tar.gz
```

Directory structure:

```bash
tree -L 2 mouse_embryo_E16_example_data/

mouse_embryo_E16_example_data/
├── GWAS
│   ├── BCX2_MCHC_EA_GWAMA.sumstats.gz
│   ├── GIANT_EUR_Height_2022_Nature.sumstats.gz
│   ├── gwas_config.yaml
│   └── IQ_NG_2018.sumstats.gz
└── ST
    └── E16.5_E1S1.MOSTA.h5ad
```

## Running `gsMap` in Quick Mode

<span style="color:#31a354"> Required memory: ~80G (for ~120K spots) </span>

First, set up environment variables for resource directories to make commands easier to manage:

```bash
# Define resource and data directories
GSMAP_RESOURCE_DIR="./gsMap_quick_mode_resource"
HOMOLOG_DIR="./gsMap_homologs"
EXAMPLE_DATA_DIR="./mouse_embryo_E16_example_data"
```

Now run the analysis using `gsmap quick-mode`:

````{tab} CLI
```bash
# Create output directory
mkdir -p ./gsmap_2d_tutorial/mouse_embryo

# Run gsMap in quick mode
gsmap quick-mode \
    --workdir "./gsmap_2d_tutorial/mouse_embryo" \
    --project-name "E16.5_E1S1" \
    --dataset-type "spatial2D" \
    --h5ad-path "${EXAMPLE_DATA_DIR}/ST/E16.5_E1S1.MOSTA.h5ad" \
    --homolog-file "${HOMOLOG_DIR}/mouse_human_homologs.txt" \
    --w-ld-dir "${GSMAP_RESOURCE_DIR}/weights_hm3_no_hla" \
    --snp-gene-weight-adata-path "${GSMAP_RESOURCE_DIR}/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad" \
    --annotation "annotation" \
    --spatial-key "spatial" \
    --data-layer "count" \
    --trait-name "IQ" \
    --sumstats-file "${EXAMPLE_DATA_DIR}/GWAS/IQ_NG_2018.sumstats.gz" \
    --plot-origin "lower"
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="./gsmap_2d_tutorial/mouse_embryo",
    project_name="E16.5_E1S1",
    dataset_type="spatial2D",
    h5ad_path="./mouse_embryo_E16_example_data/ST/E16.5_E1S1.MOSTA.h5ad",
    homolog_file="./gsMap_homologs/mouse_human_homologs.txt",
    w_ld_dir="./gsMap_quick_mode_resource/weights_hm3_no_hla",
    snp_gene_weight_adata_path="./gsMap_quick_mode_resource/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad",
    annotation="annotation",
    spatial_key="spatial",
    data_layer="count",
    sumstats_config_dict={"IQ": "./mouse_embryo_E16_example_data/GWAS/IQ_NG_2018.sumstats.gz"},
    plot_origin="lower"
)

run_quick_mode(config)
```
````

### Parameters

- `--workdir`: The working directory where output files will be saved.
- `--project-name`: Identifies the project/sample within the workdir.
- `--dataset-type`: `spatial2D` for standard 2D spatial transcriptomics.
- `--h5ad-path`: Path to the input `.h5ad` file.
- `--homolog-file`: Used to map mouse genes to human homologs.
- `--w-ld-dir`: Directory containing pre-computed LD weights.
- `--snp-gene-weight-adata-path`: Path to the SNP-gene weight matrix.
- `--annotation`: Column in `adata.obs` for cell type or region labels.
- `--spatial-key`: Key in `adata.obsm` for spatial coordinates.
- `--data-layer`: Layer in `adata.layers` to use for raw counts (default is `.X`).
- `--trait-name`: Name of the trait for labeling results.
- `--sumstats-file`: Path to the GWAS summary statistics.
- `--plot-origin`: `lower` or `upper` to match the coordinate system of your data for plotting.

### Analysis of Multiple Traits

You can analyze multiple traits simultaneously by providing a configuration file:

````{tab} CLI
```bash
gsmap quick-mode \
    --workdir "./gsmap_2d_tutorial/mouse_embryo" \
    --project-name "E16.5_E1S1" \
    --dataset-type "spatial2D" \
    --h5ad-path "${EXAMPLE_DATA_DIR}/ST/E16.5_E1S1.MOSTA.h5ad" \
    --homolog-file "${HOMOLOG_DIR}/mouse_human_homologs.txt" \
    --w-ld-dir "${GSMAP_RESOURCE_DIR}/weights_hm3_no_hla" \
    --snp-gene-weight-adata-path "${GSMAP_RESOURCE_DIR}/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad" \
    --annotation "annotation" \
    --sumstats-config-file "${EXAMPLE_DATA_DIR}/GWAS/gwas_config.yaml"
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="./gsmap_2d_tutorial/mouse_embryo",
    project_name="E16.5_E1S1",
    dataset_type="spatial2D",
    h5ad_path="./mouse_embryo_E16_example_data/ST/E16.5_E1S1.MOSTA.h5ad",
    homolog_file="./gsMap_homologs/mouse_human_homologs.txt",
    w_ld_dir="./gsMap_quick_mode_resource/weights_hm3_no_hla",
    snp_gene_weight_adata_path="./gsMap_quick_mode_resource/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad",
    annotation="annotation",
    sumstats_config_dict={
        "IQ": "./mouse_embryo_E16_example_data/GWAS/IQ_NG_2018.sumstats.gz",
        "Height": "./mouse_embryo_E16_example_data/GWAS/GIANT_EUR_Height_2022_Nature.sumstats.gz",
        "MCHC": "./mouse_embryo_E16_example_data/GWAS/BCX2_MCHC_EA_GWAMA.sumstats.gz"
    }
)

run_quick_mode(config)
```
````

### Output Description

- **Web Report**: Generated in the `report` folder. Close-up the HTML file in a browser to explore interactive 2D maps and diagnostic plots.
- **Results**: Spot-level p-values and enrichment results are stored in `spatial_ldsc` and `cauchy_combination`.
- **Intermediate Files**: Latent representations and marker scores are cached to speed up subsequent analyses on the same dataset.

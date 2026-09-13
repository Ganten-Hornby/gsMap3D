# CS8 human embryo

Analyze 62 registered sections with IQ, Height and MCHC GWAS using the default 3D settings.

[Xiao et al. · 2024 · Cell](https://doi.org/10.1016/j.cell.2024.03.041) ·
[gsMap Report preview](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/human-gastrulation/r1/index.html)

## Files needed for this tutorial

Run all commands from `gsmap_tutorials/`. Shared resources are downloaded once
and reused across tutorials.

```text
gsmap_tutorials/
├── resources/1000g-hc-grch38-v50-eur/
│   ├── snp_gene_weights.h5ad
│   └── w_ld/                           # 22 chromosome weight files
└── example_data/
    ├── example-traits/
    │   ├── IQ.sumstats.gz
    │   ├── Height.sumstats.gz
    │   └── MCHC.sumstats.gz
    └── human-gastrulation/
        ├── ST/                         # 62 H5AD sections
        ├── sample_paths.txt
        └── gwas.yaml
```

## 1. Download

First time using gsMap? Follow {doc}`../install` and download the shared
reference and example GWAS from {doc}`../downloads`.
Then download this dataset. **Skip this step if its folder already exists.**

```bash
mkdir -p example_data
wget -O example_data/human-gastrulation.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/3d/human-gastrulation/r1/bundle.tar.gz && \
    tar -xzf example_data/human-gastrulation.tar.gz -C example_data
```

The archive includes `sample_paths.txt` and `gwas.yaml`; no configuration files
need to be created. Download links and checksum instructions are on
{doc}`../downloads`.

## 2. Run gsMap

The first and last entries of `sample_paths.txt` are shown below:

```text
ST/human_gastrulation_slice_1.h5ad
ST/human_gastrulation_slice_2.h5ad
ST/human_gastrulation_slice_3.h5ad
...
ST/human_gastrulation_slice_60.h5ad
ST/human_gastrulation_slice_61.h5ad
ST/human_gastrulation_slice_62.h5ad
```

```{important}
**Path order matters:** list sections in anatomical order along the z axis.
gsMap uses this order for ordinal cross-section matching; it does not sort the
list by coordinates.
```

Choose **CLI or Python** to run the same analysis:

````{tab} CLI
```bash
mkdir -p results
gsmap quick-mode \
    --workdir results \
    --project-name human-gastrulation \
    --dataset-type spatial3D \
    --h5ad-list-file example_data/human-gastrulation/sample_paths.txt \
    --w-ld-dir resources/1000g-hc-grch38-v50-eur/w_ld \
    --snp-gene-weight-adata-path resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad \
    --sumstats-config-file example_data/human-gastrulation/gwas.yaml \
    --annotation clusters \
    --spatial-key spatial_3d \
    --data-layer counts
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig, DatasetType
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="results",
    project_name="human-gastrulation",
    dataset_type=DatasetType.SPATIAL_3D,
    h5ad_list_file="example_data/human-gastrulation/sample_paths.txt",
    w_ld_dir="resources/1000g-hc-grch38-v50-eur/w_ld",
    snp_gene_weight_adata_path="resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad",
    sumstats_config_file="example_data/human-gastrulation/gwas.yaml",
    annotation="clusters",
    spatial_key="spatial_3d",
    data_layer="counts",
)
run_quick_mode(config)
```
````

This example uses `counts` for expression, `clusters` for annotation and
`spatial_3d` for coordinates. The default ordinal search uses two sections on
each side, with boundary compensation. Human genes need no homolog conversion.
See {doc}`../advanced_usage` to change these settings.

## 3. View results

[gsMap Report preview](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/human-gastrulation/r1/index.html)

```bash
gsmap report-view results/human-gastrulation/gsmap_web_report --port 8080 --no-browser
```

Open the printed HTTP address. Select a trait to explore its spatial association
map and annotation-level enrichment. Outputs and logs are in
`results/human-gastrulation/`; see {doc}`../output_files` for file details.

### Open in SpatialVista

The report automatically creates a SpatialVista H5AD with all result spots,
annotations and trait −log10(P) values. With the
[gsMap-enabled SpatialVista viewer](https://github.com/Ganten-Hornby/spatial-vista-py/tree/feature/gsmap-trait-discovery)
installed, open it directly:

```bash
spatialvista --input results/human-gastrulation/human-gastrulation.spatialvista.h5ad
```

Select **Query Traits** to search IQ, Height or MCHC. No coordinate or
annotation flags are needed. For remote use, add `--no-browser --port 8765`
and forward the port to your computer.

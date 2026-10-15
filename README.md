# gsMap3D

**gsMap3D** integrates 3D spatial transcriptomics (ST) data with genome-wide association study (GWAS) summary statistics to map cells associated with human complex traits and diseases.

______________________________________________________________________

## ✨ What's New

```text
- Support for 3D ST data mapping
- Dual-embedding strategy for more accurate gene specificity scoring
- GPU-accelerated embedding, marker-score and LDSC computation
- Symmetric graphs with an exact external-neighbor count and radius alternatives
- Ordinal or calibrated physical-z cross-section matching
- Paired ancestry-specific V50 LD resources and executable tutorials
```

______________________________________________________________________

## 🚀 Features

- **Spatially-aware High-Resolution Trait Mapping**\
    Maps trait-associated cells at single-cell resolution, offering insights into their spatial distributions.

- **Spatial Region Identification**\
    Aggregates trait–cell association p-values into trait–tissue region association p-values, prioritizing tissue regions relevant to traits of interest.

- **Putative Causal Genes Identification**\
    Prioritizes putative causal genes by associating gene expression levels with cell–trait relevance.

- **Scalability**\
    Employs [JAX](https://github.com/google/jax) JIT and GPU/TPU acceleration to scale to million-scale cells (spots) spatial omics datasets.

______________________________________________________________________

## 🧠 Overview of `gsMap3D`

`gsMap3D` operates on a four-step process:

### 1. Gene Specificity Assessment in 3D Spatial Contexts

To address technical noise and capture spatial correlations of gene expression across consecutive ST sections,\
`gsMap3D` constructs batch-corrected dual embeddings of molecular and spatial features. These embeddings jointly model transcriptomic similarity and local spatial context to identify homogeneous cells in 3D space. Gene specificity scores (GSS) are then computed by aggregating normalized gene expression ranks across these 3D homogeneous cells, enabling robust identification of genes that are both highly and specifically expressed in focal cells.

### 2. Linking Gene Specificity to Genetic Variants

`gsMap3D` links gene specificity scores to single nucleotide polymorphisms (SNPs) by assigning GSS to SNPs based on their proximity to gene transcription start sites (TSS) and SNP-to-gene epigenetic linking maps, thereby connecting spatially resolved gene expression patterns with trait-associated genetic variation.

### 3. 3D Spatial S-LDSC for Cell–Trait Association

To quantify cell–trait associations in 3D space, `gsMap3D` integrates the 3D gene specificity scores with GWAS summary statistics using stratified LD score regression (S-LDSC). This framework enables the estimation of trait relevance for individual cells by associating their stratified LD scores with GWAS signals.

### 4. Spatial Region–Trait Association Analysis

To evaluate associations between traits and spatial regions, `gsMap3D` aggregates p-values from cells within a given 3D spatial region using the Cauchy combination test, yielding region-level association statistics that reflect coordinated genetic effects across spatially organized cell populations.

![gsMap3D Method Schematic](docs/source/_static/gsMap3D_Method_schematic.png)

______________________________________________________________________

## 📚 Documentation

Please see our [Documentation](https://yanglab.westlake.edu.cn/gsmap3d-portal/docs) and [Website](https://yanglab.westlake.edu.cn/gsmap3d-portal).

Start with [Downloads](https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1/docs/r1/downloads.html) to install shared resources once,
then follow the direct CLI/Python [tutorials](https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1/docs/r1/3d_tutorial/human_gastrulation.html).
Each tutorial shows its required input files. The
[Customization Guide](https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1/docs/r1/advanced_usage.html) explains how to select your
own data, reference weights, embeddings and analysis settings. Download/report
links marked staged are not yet public.

______________________________________________________________________

## 📝 How to Cite

If you use `gsMap3D` in your studies, please cite:

- **gsMap3D**: *to be updated.*

- **gsMap**:\
    Song, L., Chen, W., Hou, J., Guo, M. & Yang, J.\
    *Spatially resolved mapping of cells associated with human complex traits.*\
    **Nature** (2025).\
    [https://doi.org/10.1038/s41586-025-08757-x](https://doi.org/10.1038/s41586-025-08757-x)

______________________________________________________________________

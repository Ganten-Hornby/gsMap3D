.. _key_concepts:

Key Concepts
============

This guide explains the core data structures, pipeline stages, and configuration options in ``gsMap``.

.. note::
   For step-by-step instructions, please refer to the :doc:`/2d_tutorial/index` or :doc:`/3d_tutorial/index`.

Basic Usage
-----------
``gsMap`` can be used as a Python library or via the command line.

.. tab:: CLI

    .. code-block:: bash

        gsmap quick-mode \
            --workdir ./output \
            --project-name my_project \
            --h5ad-path sample1.h5ad \
            --dataset-type spatial2D \
            --trait-name Trait1 \
            --sumstats-file trait1.sumstats.gz \
            --snp-gene-weight-adata-path ./resources/snp_gene_weights.h5ad \
            --w-ld-dir ./resources/w_ld

.. tab:: Python

    .. code-block:: python

        from gsMap.config import QuickModeConfig
        from gsMap.pipeline import run_quick_mode

        config = QuickModeConfig(
            workdir="./output",
            project_name="my_project",
            h5ad_path="sample1.h5ad",
            dataset_type="spatial2D",
            sumstats_config_dict={"Trait1": "trait1.sumstats.gz"},
            snp_gene_weight_adata_path="./resources/snp_gene_weights.h5ad",
            w_ld_dir="./resources/w_ld"
        )

        run_quick_mode(config)

Pipeline Overview
-----------------

The ``gsMap`` pipeline integrates spatial transcriptomics with GWAS summary statistics through five key stages:

.. mermaid::

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
            S2[2. Find Homogeneous Neighbors]:::process
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

1.  **Latent Representation**: By capture the underlying cell states and it's cell niche information gsMap use a dual embedding module compress the high-dimensional gene expression into the cell identity embedding and compress the cell niche information into the cell niche embedding.
2.  **Find Homogeneous Neighbors**: For each cell or spot, we identify "homogeneous neighbors"—other spots with similar molecular profiles and spatial niche—either within the same tissue slice (for 2D data) or across adjacent slices (for 3D data).
3.  **Gene Specificity Score (GSS)**: We compute a Gene Specificity Score for every gene in every cell by aggregating expression information from its homogeneous neighbors. This robustly quantifies how specific a gene's expression is to that cell and niche context.
4.  **Spatial LDSC**: These GSS scores are integrated with GWAS summary statistics using Stratified LD Score Regression (S-LDSC) to map trait heritability to specific cells and spatial regions.
5.  **Spatial Region / Cell Type Identification**: To evaluate the association of a specific spatial region or cell type with traits, gsMap employs the *Cauchy combination test* to aggregate p-values from individual spots within that spatial region or cell type.



.. toctree::
   :maxdepth: 2
   :caption: Detailed Guides

   input_data_format
   configuration

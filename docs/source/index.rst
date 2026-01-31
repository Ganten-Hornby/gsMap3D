Welcome to gsMap's documentation!
===================================

**gsMap**: **g**\ enetically informed **s**\ patial **Map**\ ping of cells for complex traits.

**gsMap** integrates spatial transcriptomics (ST) data with genome-wide association study (GWAS) summary statistics to map cells to human complex traits and diseases.



Features
--------

- **Spatially-aware High-Resolution Trait Mapping**: Maps trait-associated cells at single-cell resolution, offering insights into their spatial distributions.
- **Spatial Region Identification**: Aggregates trait-cell association p-values into trait-tissue region association p-values, prioritizing tissue regions relevant to traits of interest.
- **Putative Causal Genes Identification**: Prioritizes putative causal genes by associating gene expression levels with cell-trait relevance.
- **Scalability**: Employs `JAX <https://github.com/google/jax>`_ JIT and GPU/TPU acceleration to scale to million-scale cells/spots spatial omics datasets.


Explore gsMap
-------------

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - **Installation**
       
       Get gsMap running on your system using uv, pip, or conda.
       
       :doc:`Go to Installation <install>`
     - **Key Concepts**
       
       Understand how to prepare data and configure gsMap.
       
       :doc:`Go to Key Concepts <key_concepts/index>`

   * - **Tutorials**
       
       Step-by-step guides for 2D and 3D spatial data.
       
       :doc:`2D Tutorial <2d_tutorial/index>` | :doc:`3D Tutorial <3d_tutorial/index>`
     - **Advanced Usage**
       
       Customize SNP-to-gene linking, use custom embeddings, and optimize performance.
       
       :doc:`Go to Advanced Usage <advanced_usage>`

Overview of ``gsMap`` Method
-----------------------------

``gsMap`` operates on a four-step process:

1. **Gene Specificity Assessment in Spatial Contexts**: To address technical noise and capture spatial correlations of gene expression profiles in ST data, ``gsMap`` leverages GNNs to identify homogeneous spots for each spot and estimates gene specificity scores by aggregating information from those homogeneous spots.
2. **Linking Gene Specificity to SNPs**: ``gsMap`` assigns gene specificity scores to single nucleotide polymorphisms (SNPs) based on their proximity to gene transcription start sites (TSS) and SNP-to-gene epigenetic linking maps.
3. **Spatial S-LDSC**: To estimate the relevance of spots to traits, ``gsMap`` associates stratified LD scores of individual spots with GWAS summary statistics using the S-LDSC framework.
4. **Spatial Region Identification**: To evaluate the association of a specific spatial region with traits, ``gsMap`` employs the Cauchy combination test to aggregate p-values from individual spots within that spatial region.

.. image:: _static/schematic.svg
   :width: 600
   :alt: Model architecture

Schematics of ``gsMap`` method. For more details about ``gsMap``, please check out our `publication <https://doi.org/10.1038/s41586-025-08757-x>`_.


How to Cite
------------
If you use ``gsMap`` in your studies, please cite:

   Song, L., Chen, W., Hou, J., Guo, M. & Yang, J. "**Spatially resolved mapping of cells associated with human complex traits.**" *Nature* (2025). `doi:10.1038/s41586-025-08757-x <https://doi.org/10.1038/s41586-025-08757-x>`_


.. toctree::
    :maxdepth: 2
    :hidden:

    install
    key_concepts/index
    gsMap2D <2d_tutorial/index>
    gsMap3D <3d_tutorial/index>
    advanced_usage
    ldscore_weight_matrix
    notes
    release



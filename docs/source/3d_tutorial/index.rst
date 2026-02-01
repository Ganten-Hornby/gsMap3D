gsMap3D
========

**Why 3D Mapping Matters**

Disease genetic risk is fundamentally encoded within the native three-dimensional cellular architecture of tissues. Traditional 2D tissue sections, while informative, can only capture a partial view of the spatial organization and may miss critical cross-section biological signals. This limitation often leads to incomplete or misleading interpretations of where disease-associated genes actually function.

**gsMap3D** is a comprehensive computational framework that bridges this gap by integrating genome-wide association studies (GWAS) with 3D-reconstructed spatial transcriptomics data to reveal the true spatial organization of disease risk.

Methodology Overview
--------------------

The gsMap3D framework utilizes a **Dual-Embedding Strategy**:

1. **Cell Embedding**: Captures each spot's transcriptomic identity.
2. **Spatial-Domain/Niche Embedding**: Encodes the local 3D spatial context across tissue sections.

By leveraging these dual embeddings, gsMap3D identifies **homogeneous cell neighborhoods** in 3D space—cells that share both similar transcriptomic identity and spatial niche context. These neighborhoods are then used to compute a **3D Gene Specificity Score (GSS)**, which quantifies the spatial specificity of gene expression in three dimensions. Finally, the 3D GSS is linked to GWAS summary statistics using the S-LDSC framework to pinpoint trait-associated cellular niches within the reconstructed 3D tissue architecture.

Advantages of gsMap3D
---------------------

* **Comprehensive Cell-Trait Associations**: A joint 3D analysis provides a more complete view of tissue biology, enabling the discovery of novel tissue associations. For example, in the mouse embryo, 3D mapping pinpointed the thymus as the most strongly associated tissue for allergic diseases—a signal that was missed in 2D analysis.
* **Improved Accuracy**: The dual-embedding strategy reduces the over-smoothing issues that arise when using spatial-domain embedding alone, while also preventing signal leakage across tissue boundaries. By integrating both cell identity and spatial context, gsMap3D correctly prioritizes relevant organs within and across sections.
* **Scalability**: Engineered to handle millions of spots with full GPU acceleration and efficient dual-embedding strategies.

Tutorials
---------

.. toctree::
    :maxdepth: 1

    CS8 Human Embryo <human_gastrulation.md>

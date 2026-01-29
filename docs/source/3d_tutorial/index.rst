gsMap3D
========

gsMap3D addresses the fundamental knowledge gap of how disease genetic risk is embedded within the native 3D cellular architecture of tissues. It is a comprehensive computational framework designed to integrate genome-wide association studies (GWAS) with 3D-reconstructed spatial transcriptomics data.

Advantages of gsMap3D
---------------------

* **Discovery of Novel Tissue Associations**: A joint 3D analysis provides a more complete view of tissue biology. For example, in the mouse embryo, 3D mapping pinpointed the thymus as the most strongly associated tissue for allergic diseases, which was missed in 2D analysis.
* **Improved Accuracy**: 3D context-aware mapping prevents signal leakage and correctly prioritizes relevant organs by integrating information within and across sections.
* **Scalability**: Engineered to handle millions of spots with full GPU acceleration and efficient dual-embedding strategies.

Methodology Overview
--------------------

The gsMap3D framework utilizes a **Dual-Embedding Strategy** to capture both a spot's transcriptomic identity (Cell Embedding) and its local 3D spatial context (Spatial-Domain Embedding). This approach enables the calculation of a **3D Gene Specificity Score (GSS)**, which is then linked to GWAS summary statistics using the S-LDSC framework to identify trait-associated niches in three-dimensional space.

Tutorials
---------

.. toctree::
    :maxdepth: 1

    CS8 Human Embryo <human_gastrulation.md>

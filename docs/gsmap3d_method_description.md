gsMap3D: A Tutorial for Mapping Complex Traits in 3D Tissue Architecture

This document transforms the complex methodology of the gsMap3D framework, as detailed in the original research paper, into a practical guide for researchers. The objective of this tutorial is to empower users to leverage gsMap3D for mapping the genetic basis of complex traits within the native 3D context of tissues, bridging the gap between genetic discoveries and their spatial cellular manifestations.


--------------------------------------------------------------------------------


1.0 Introduction to gsMap3D

The complex functions of human organs rely on intricate 3D architectures where specialized cells are organized into spatially defined circuits and functional domains. Understanding how genetic risk translates into disease therefore requires projecting this risk into the native 3D cellular landscape. Traditional spatial transcriptomics (ST) analyses have been largely restricted to individual 2D sections, offering only a fragmented view of tissue organization. This 2D approach can obscure the continuous spatial gradients and cellular interactions that span across sectioning planes, leaving a fundamental gap in our understanding of the spatial organization of genetic effects.

1.1 What Problem Does gsMap3D Solve?

gsMap3D addresses the fundamental knowledge gap of how disease genetic risk is embedded within the native 3D cellular architecture of tissues. It is a comprehensive computational framework designed to integrate genome-wide association studies (GWAS) with 3D-reconstructed spatial transcriptomics data. Its primary purpose is to map trait-associated spots in three-dimensional space, moving beyond the limitations of 2D analysis to reveal how genetic effects are structured by complex tissue organization.

1.2 Who is This Tutorial For?

This tutorial is intended for computational biologists, geneticists, and neuroscientists who have experience analyzing genomics and spatial transcriptomics data. It is designed for researchers who wish to apply the gsMap3D framework to their own research questions, seeking to understand the spatial etiology of complex traits in tissues of interest.

1.3 Key Capabilities

The gsMap3D framework offers a powerful suite of capabilities for spatial genetic analysis:

* GWAS and 3D ST Integration: Systematically links GWAS summary statistics with high-resolution, 3D-reconstructed ST data to identify spots associated with complex human traits.
* High-Accuracy Dual-Embedding Strategy: Employs a novel dual-embedding approach that captures both the transcriptomic identity of spots and their local spatial context, preventing signal leakage and improving the accuracy of association mapping.
* Scalability and Performance: Utilizes full GPU acceleration and efficient training strategies to handle modern, large-scale ST datasets containing millions of spots.
* Interactive 3D Visualization: Generates a comprehensive, interactive web report and standalone visualization tools to explore trait-association maps, gene specificity scores, and learned embeddings in 3D.

This tutorial will now delve into the core principles that underpin the framework's ability to perform these advanced analyses.

2.0 Core Concepts of the gsMap3D Framework

To effectively apply gsMap3D, it is essential to understand its theoretical underpinnings. This section demystifies the core algorithms and data representations that enable the framework to link genetic variation to the intricate cellular organization of tissues in three-dimensional space.

2.1 The Guiding Principle: From 2D Slices to 3D Niches

The fundamental principle of gsMap3D is that adjacent ST sections sampled from the same tissue capture continuous biological domains. While a single 2D section provides a snapshot, a series of consecutive sections contains the information needed to reconstruct a more complete, volumetric view. By leveraging this continuity, gsMap3D integrates information both within and across sections to model the 3D context, revealing spatial gradients and organizational patterns of genetic effects that are invisible to single-section analysis.

2.2 The Dual-Embedding Strategy: Capturing Identity and Context

A core innovation of gsMap3D is its dual-embedding representation, which models both a spot's intrinsic identity and its spatial environment. This approach is strategically designed to prevent the over-smoothing and "signal leakage" that can occur when relying solely on spatial context. A single-embedding strategy suffers from this issue, as spatially adjacent but transcriptionally distinct spots can be incorrectly grouped, blurring biological signals. The dual-embedding strategy avoids this by treating identity and context as separate but complementary features. In a simulation using hepatocyte-specific genes, the dual-embedding strategy correctly enriched for hepatocytes (Odds Ratio = 3.3), whereas the single-embedding approach showed weak, non-specific enrichment (OR = 1.2) due to signal leakage to adjacent non-hepatocyte cells.

Cell Embedding	Spatial-Domain Embedding
Captures the spot identity based solely on gene expression.	Captures the local tissue architecture by integrating gene expression with spatial coordinates.
Distinguishes transcriptionally distinct spots, even if they are located in the same spatial neighborhood.	Identifies spots that share a similar local spatial context, revealing spatially coherent domains across adjacent sections.

2.3 The 3D Gene Specificity Score (GSS)

The 3D Gene Specificity Score (GSS) is a key metric calculated by gsMap3D. It quantifies how highly and specifically a gene is expressed within a local 3D neighborhood of "homogeneous spots." These homogeneous spots are defined as spots that are similar in both their gene expression profiles (from the cell embedding) and their local spatial context (from the spatial-domain embedding) across adjacent tissue sections. A high GSS indicates that a gene's expression is ranked significantly higher within this specific 3D niche compared to its baseline rank across the entire dataset, marking it as a key identifier for that niche.

2.4 Linking Genetics to Space: Heritability Enrichment

The framework connects the biological information captured by the 3D GSS to genetic data from GWAS. gsMap3D uses the stratified LD score regression (S-LDSC) method to perform this integration. At a high level, S-LDSC tests whether single nucleotide polymorphisms (SNPs) linked to genes with a high GSS in a specific spot are statistically enriched for the heritability of a given GWAS trait. A significant enrichment suggests that the niche defined by that spot is biologically relevant to the trait.

Having covered the conceptual foundation, we will now proceed to the practical requirements for setting up and running a gsMap3D analysis.

3.0 System Requirements and Installation

Setting up the gsMap3D environment requires attention to its computational demands. The framework is engineered for performance on large-scale datasets, and understanding its technical prerequisites is the first step toward a successful analysis.

3.1 Technical Prerequisites

gsMap3D is a computationally intensive framework designed to process and analyze modern 3D spatial transcriptomics datasets that can contain millions of spots.

* GPU Acceleration: All major steps of the pipeline are GPU-accelerated to ensure computational efficiency. A system with a compatible NVIDIA GPU (e.g., NVIDIA A100) is strongly recommended for optimal performance.
* High-Performance Computing: Due to the scale of the data and the computational complexity, running gsMap3D within a high-performance computing (HPC) environment is beneficial.

3.2 Installation Guide

This tutorial focuses on the methodology and practical application of gsMap3D. For specific, up-to-date installation commands, dependency lists, and environment setup instructions, refer to the official source code repository.

Official Repository: [Link to GitHub Repository]

With the software environment configured, the next critical step is preparing the necessary input data for the pipeline.

4.0 Data Preparation and Input Formatting

The quality and correct formatting of input data are critical for a successful gsMap3D analysis. Inaccurate or misaligned data can lead to spurious results. This section provides a detailed breakdown of each required input file, ensuring your data is properly structured for the pipeline.

1. Provide a series of consecutive ST sections, each with transcriptome-wide gene expression profiles and their corresponding spatial coordinates. Ensure that the sections are properly aligned to form a coherent 3D reconstruction, as the framework's core principle relies on the continuity between adjacent slices.
2. Prepare standard GWAS summary statistics for the trait or disease of interest. Use traits with a heritability estimate yielding a χ² > 25. (This threshold ensures the GWAS has sufficient polygenicity and statistical power for the S-LDSC heritability partitioning analysis to yield robust results).
3. Provide an LD reference panel that is ancestrally matched to the GWAS summary statistics for the S-LDSC analysis. A common choice is the reference panel from the 1000 Genomes Project Phase 3.
4. By default, gsMap3D maps SNPs to genes within a ±50 kb window. You can optionally provide external epigenomic maps, such as enhancer-gene links from projects like ENCODE or Roadmap Epigenomics. This allows the framework to capture potential long-range regulatory interactions, extending the SNP-to-gene mapping beyond the default genomic window.

Once these inputs are prepared, you are ready to execute the main analysis pipeline.

5.0 Step-by-Step Analysis Workflow

This section provides a step-by-step walkthrough of the analysis pipeline, from preparing your input data to generating the final association maps. The gsMap3D pipeline is a multi-stage computational process that transforms raw spatial and genetic data into a 3D map of trait-spot associations.

5.1 Step 1: Generating Dual Embeddings

The analysis begins by processing the raw ST data to create the core data representations. First, select a set of highly variable genes (HVGs) across all sections to serve as informative features. Next, perform a graph-convolutional propagation step on each section to generate a spatially informed expression matrix. Finally, train a batch-aware variational autoencoder (VAE) model on this data. This VAE approach is critical for disentangling true biological variation from technical batch effects across different tissue sections, ensuring a coherent 3D representation. The model produces two key outputs for each spot across all sections: the cell embedding, capturing transcriptomic identity, and the spatial-domain embedding, capturing local 3D spatial context.

5.2 Step 2: Calculating the 3D Gene Specificity Score (GSS)

Using the learned embeddings, the framework calculates the 3D GSS for every gene in every spot. This is achieved through a two-stage process to identify a set of "homogeneous spots" for each focal spot:

1. Identify Spatially Coherent Spots: Using cosine similarity on the spatial-domain embeddings, the algorithm first selects a set of spots from adjacent sections that share a similar local tissue architecture.
2. Refine by Transcriptional Similarity: Within this spatially coherent set, the algorithm then uses cosine similarity on the cell embeddings to select the subset of spots that are most transcriptionally similar to the focal spot.

The 3D GSS is then calculated by comparing each gene's expression rank within this refined 3D neighborhood of homogeneous spots to its baseline rank across all spots.

5.3 Step 3: Associating GSS with GWAS Traits

This is the core genetic association step. The framework first annotates SNPs from the GWAS summary statistics file with the 3D GSS values calculated in the previous step. It then employs the S-LDSC model to test for heritability enrichment. This process yields a p-value for each spot, which quantifies the statistical significance of the association between that spot's unique 3D niche and the genetic architecture of the trait.

5.4 Step 4: Deriving Region-Level Associations

In the final step, gsMap3D aggregates the individual spot-level association p-values to derive a single, robust association p-value for predefined anatomical regions or cell types. This is accomplished using the Cauchy Combination Test (CCT). To mitigate the risk of inflation from a few outlier spots with extremely small p-values, the framework applies a conservative filtering strategy before aggregation, removing spots whose -log10 p-values exceeded the median plus two interquartile ranges.

After completing the pipeline, the next step is to explore and interpret the rich outputs generated by the analysis.

6.0 Interpreting and Visualizing Outputs

The final output of gsMap3D is a rich, multi-layered dataset that provides deep insights into the spatial organization of genetic risk. This section will guide you on how to navigate and interpret the primary results, with a focus on the powerful interactive visualization tools provided by the framework.

6.1 The Interactive Web Report

Upon completion, gsMap3D generates a comprehensive and user-friendly web report that allows for interactive exploration of the results. Key features of this report include:

* 3D Interactive Spatial View: Explore trait-association p-values across the entire reconstructed tissue, allowing you to rotate, zoom, and inspect signals from any angle.
* 2D Gallery View: Systematically inspect the association maps for individual 2D sections.
* Gene Expression and GSS Plots: Investigate the specific genes driving an association signal by visualizing their expression levels and 3D GSS values in space.
* UMAP Visualizations: Assess the quality and structure of the learned cell and spatial-domain embeddings.
* GWAS Manhattan Plot: For reviewing the input genetic data.
* Region-Trait Association Summary: Get a high-level overview of the results with a summary of the aggregated p-values for predefined anatomical regions.

6.2 Standalone Visualization Tools

For users requiring greater flexibility for custom analyses and publication-quality figure generation, gsMap3D also provides a standalone local application and Jupyter widgets. These tools accept the standard gsMap3D outputs and support multi-section rendering, 3D GSS visualization, and interactive inspection of trait-spot associations, empowering deeper, user-driven exploration of the data.

Beyond running the software and visualizing results, achieving robust scientific conclusions requires thoughtful experimental design and careful interpretation.

7.0 Advanced Considerations and Best Practices

Robust scientific conclusions depend not only on running the software correctly but also on thoughtful experimental design and interpretation. This section covers key insights derived from the original study to help you maximize the value and validity of your gsMap3D analyses.

7.1 The Power of Joint 3D Analysis

Relying on a single representative 2D section can lead to false negatives and misinterpretation of organ-level associations—a limitation gsMap3D is designed to overcome. A joint 3D analysis provides a more complete and powerful view of tissue biology, as clearly demonstrated by several key findings in the original study that would have been missed in a 2D-only analysis:

* Discovery of Novel Tissue Associations: In the mouse embryo, 3D mapping pinpointed the thymus as the most strongly associated tissue for allergic diseases and rheumatoid arthritis. This association was not detected when analyzing only the representative middle section.
* Improved Accuracy: For atrial fibrillation, a single-section analysis failed to prioritize the heart as the most relevant organ. In contrast, the more complete tissue coverage provided by the 3D analysis correctly and robustly identified the heart as the primary site of genetic association.

These examples underscore that the 3D context is indispensable for accurately detecting and interpreting tissue-trait associations.

7.2 Best Practice: Using Normal vs. Diseased Tissues

For mapping inherited genetic risk, use normal (non-diseased) tissue. Using diseased tissue introduces confounding transcriptional changes that are a consequence of the disease, which will obscure the underlying genetic etiology you aim to uncover. gsMap3D is designed to identify the cellular mediators of inherited genetic risk, and transcriptional changes in diseased tissue can confound the analysis and mask the causal cell populations responsible for genetic susceptibility.

7.3 A Note on Scalability and Performance

gsMap3D is engineered to handle the scale of modern spatial omics. The framework employs several strategies for computational efficiency:

* Down-Sampled Training: The model is trained efficiently on a representative subset of spots, allowing it to learn robust embeddings without processing the entire dataset during the training phase.
* Full GPU Acceleration: All major steps, including the computationally demanding S-LDSC analysis, are fully GPU-accelerated.

These optimizations ensure that gsMap3D can efficiently analyze large-scale ST datasets containing millions of spots, making it a practical tool for current and future research.

This tutorial has provided a comprehensive guide to understanding and applying gsMap3D to your research.

8.0 Conclusion and Further Resources

gsMap3D provides a foundational framework for uncovering how human genetic effects are embedded within the 3D organization of tissues. By integrating GWAS with 3D-reconstructed spatial transcriptomics, it enables a paradigm shift in complex trait genetics—moving from the identification of "disease-relevant cells" to the pinpointing of "disease-relevant 3D spatial niches." This approach reveals that genetic effects are not uniform but are structured by tissue regionalization and spatially organized cell-cell interactions, underscoring that 3D context is indispensable for understanding disease biology.

Key Resources

* Original Publication: "Mapping the cellular etiology of complex traits in 3D tissue architecture"
* Source Code on GitHub: The source code for gsMap3D is available on GitHub. [Link to GitHub Repository]
* Online 3D Results Portal: An interactive portal to explore all 3D mapping results from the study is available at https://yanglab.westlake.edu.cn/gsmap/home.
* Data Availability: The study utilized several key public ST datasets, including the mouse embryo (E11.5, E16.5) and adult mouse brain (MERFISH). For specific accession numbers and links, please refer to the "Data availability" section of the original publication.

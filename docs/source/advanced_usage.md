# Customization Guide

This section covers advanced configurations for `gsMap3D`, allowing users to customize key steps of the pipeline and optimize performance for large-scale datasets.

## Customize SNP-to-Gene Linking

By default, `gsMap3D` utilizes pre-calculated SNP-to-gene weights based on the **1000 Genomes EUR reference panel** and **protein-coding genes from Gencode v46**. This streamlined mode is ideal for most users.

For more customizable analyses—such as using custom gene BED files, alternative LD reference panels, or population-specific data—you can generate your own SNP-to-gene weight matrix. See the [Computing Custom LD Score Weight Matrix](ldscore_weight_matrix.md) guide for detailed instructions.

### Using a Custom Weight Matrix

Once you have generated a custom SNP-to-gene weight matrix, provide it to `gsMap3D` using:

```bash
gsmap quick-mode \
    --snp-gene-weight-adata-path "/path/to/custom_snp_gene_weights.h5ad" \
    ...
```

## Use your own embedding

While `gsMap3D` provides GNN-based embeddings, you can utilize embeddings from other tools (e.g., Scanpy, Seurat, or custom VAEs) by specifying the corresponding keys in `adata.obsm`. In this case, you should skip the `find_latent` step by setting `--start-step latent2gene`.

```bash
gsmap quick-mode \
    --start-step latent2gene \
    --latent-representation-cell "X_pca" \
    --latent-representation-niche "X_spatial_pca" \
    ...
```

```{note}
When using custom embeddings, ensure that your gene expression data is stored in `adata.X` (the main data matrix). The `latent2gene` step computes Gene Specificity Scores directly from the expression values in `X`.
```

## Use your customized GSS

If you have pre-calculated Gene Specificity Scores (GSS) or marker scores from other analytical pipelines, you can provide them directly to the `spatial-ldsc` step to skip the earlier computation steps.

```bash
gsmap spatial-ldsc \
    --marker-score-format "h5ad" \
    --marker-score-h5ad-path "/path/to/custom_marker_scores.h5ad" \
    --snp-gene-weight-adata-path "/path/to/snp_gene_weights.h5ad" \
    ...
```

### Format Requirements

The custom GSS file must be in **h5ad format** with:

- **Rows**: Cells/spots (observations)
- **Columns**: Genes (variables)

```{note}
The gene names in your custom GSS (`.var_names`) must overlap with the gene names in the SNP-to-gene weight matrix (`--snp-gene-weight-adata-path`). Only overlapping genes will be used for the spatial-LDSC analysis.
```

## Scalability & Performance

For tips on optimizing performance for large-scale datasets—including speed optimization, GPU/TPU acceleration, JAX memory management, and handling million-scale spatial omics data—see the [Scalability](scalability.md) guide.

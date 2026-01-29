# Advanced Usage

This section covers advanced configurations for `gsMap`, allowing users to customize key steps of the pipeline.

## Customize SNP to gene linking

By default, `gsMap` uses pre-calculated SNP-to-gene weights based on TSS and snp-to-gene maps. Users can construct their own SNP-by-gene matrix to incorporate custom genomic windows or multi-omics evidence.

To use a custom SNP-to-gene weight matrix:

```bash
gsmap quick-mode \
    --snp-gene-weight-adata-path "/path/to/custom_snp_gene_weights.h5ad" \
    ...
```

## Use your customized GSS

If you have pre-calculated Gene Specificity Scores (GSS) or marker scores from other analytical pipelines, you can provide them directly in the `spatial-ldsc` step.

```bash
gsmap spatial-ldsc \
    --marker-score-format "h5ad" \
    --marker-score-h5ad-path "/path/to/custom_marker_scores.h5ad" \
    ...
```

The supported formats for GSS include `memmap`, `feather`, and `h5ad`.

## Use your own embedding

While `gsMap` provides GNN-based embeddings, you can utilize embeddings from other tools (e.g., Scanpy, Seurat, or custom VAEs) by specifying the corresponding keys in `adata.obsm`.

```bash
gsmap quick-mode \
    --latent-representation-cell "my_cell_embedding_key" \
    --latent-representation-niche "my_niche_embedding_key" \
    ...
```

This bypasses the internal latent representation finding step and uses your provided embeddings for all downstream calculations.

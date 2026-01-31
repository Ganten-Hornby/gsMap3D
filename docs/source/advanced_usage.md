# Advanced Usage

This section covers advanced configurations for `gsMap`, allowing users to customize key steps of the pipeline and optimize performance for large-scale datasets.

## Customize SNP to gene linking

By default, `gsMap` uses pre-calculated SNP-to-gene weights based on TSS and snp-to-gene maps. Users can construct their own SNP-by-gene matrix to incorporate custom genomic windows or multi-omics evidence.

To use a custom SNP-to-gene weight matrix:

```bash
gsmap quick-mode \
    --snp-gene-weight-adata-path "/path/to/custom_snp_gene_weights.h5ad" \
    ...
```

For detailed instructions on generating this matrix using your own SNP-to-gene BED file and PLINK reference panel, see [Computing Custom LD Score Weight Matrix](ldscore_weight_matrix.md).

## Use your customized GSS

If you have pre-calculated Gene Specificity Scores (GSS) or marker scores from other analytical pipelines, you can provide them directly in the `spatial-ldsc` step to skip the earlier computation steps.

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
    --latent-representation-cell "X_pca" \
    --latent-representation-niche "X_spatial_pca" \
    ...
```

This bypasses the internal latent representation finding step and uses your provided embeddings for all downstream calculations.

## Running Scalability

`gsMap` is designed to handle large-scale spatial omics data. Here are some tips to optimize performance.

### Running Speed Optimization

The **latent-to-gene** step involves calculating the Gene Specificity Score (GSS) using a rank-based approach. This process requires frequent random access to the gene rank matrix, which can be a bottleneck on slower storage systems (like HDDs or network drives).

To boost performance, use the `--memmap-tmp-dir` option to specify a directory on a high-speed **SSD** (Solid State Drive). `gsMap` will copy the memory-mapped rank matrix to this location for faster random reads.

```bash
gsmap quick-mode \
    --memmap-tmp-dir /mnt/fast_ssd/tmp \
    ...
```

### GPU/TPU Acceleration

`gsMap` leverages **JAX** for accelerated computation in the `spatial-ldsc` step.

*   **Enable GPU**: Use the `--use-gpu` flag (enabled by default).
*   **Disable GPU**: Use `--no-gpu` if you encounter memory issues or lack GPU hardware.

#### JAX Memory Management
JAX attempts to preallocate a significant portion of GPU memory by default. If this causes OOM (Out Of Memory) errors or conflicts with other processes, you can control it via environment variables before running `gsMap`:

```bash
# Prevent JAX from preallocating all memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Limit JAX to a specific fraction of memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=.75
```

#### Device Selection
To restrict `gsMap` to specific GPU devices, use `CUDA_VISIBLE_DEVICES`:

```bash
# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0
gsmap quick-mode ...
```
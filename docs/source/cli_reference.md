# `gsmap`

gsMap: genetically informed spatial mapping of cells for complex traits

**Usage**:

```console
$ gsmap [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `-v, --version`: Show version and exit
* `--help`: Show this message and exit.

**Commands**:

* `quick-mode`: Run the complete gsMap pipeline with all...
* `find-latent`: Find latent representations of each spot...
* `latent-to-gene`: Estimate gene marker scores for each spot...
* `spatial-ldsc`: Run spatial LDSC analysis for genetic...
* `cauchy-combination`: Run Cauchy combination test to combine...
* `ldscore-weight-matrix`: Compute LD score weight matrices for...
* `format-sumstats`: Format GWAS summary statistics for gsMap...
* `report-view`: Launch a local web server to view the...

## `gsmap quick-mode`

Run the complete gsMap pipeline with all steps.

This command runs the gsMap analysis pipeline including:
- Data loading and preprocessing (Find Latent)
- Gene expression analysis (Latent to Gene)
- GWAS integration (Spatial LDSC)
- Cauchy Combination Test
- Result generation (Report)

Requires pre-generated SNP-gene weight matrix and LD weights.

**Usage**:

```console
$ gsmap quick-mode [OPTIONS]
```

**Options**:

* `--workdir DIRECTORY`: Path to the working directory  [required]
* `--project-name TEXT`: Name of the project  [required]
* `--h5ad-path PATH`: Space-separated list of h5ad file paths. Sample names are derived from file names without suffix.
* `--h5ad-yaml FILE`: YAML file with sample names and h5ad paths
* `--h5ad-list-file FILE`: Each row is a h5ad file path, sample name is the file name without suffix
* `--data-layer TEXT`: Gene expression raw counts data layer in h5ad layers, e.g., &#x27;count&#x27;, &#x27;counts&#x27;. Other wise use &#x27;X&#x27; for adata.X  [default: X]
* `--spatial-key TEXT`: Spatial key in adata.obsm  [default: spatial]
* `--annotation TEXT`: Name of the annotation in adata.obs to use
* `--homolog-file FILE`: Path to homologous gene conversion file
* `--latent-representation-niche TEXT`: Key for spatial niche embedding in obsm
* `--latent-representation-cell TEXT`: Key for cell identity embedding in obsm  [default: emb_cell]
* `--high-quality-cell-qc / --no-high-quality-cell-qc`: Enable/disable high quality cell QC based on module scores. If enabled, it will compute DEG and module scores.  [default: high-quality-cell-qc]
* `--two-stage / --single-stage`: Tune the cell embeddings based on the provided annotation  [default: single-stage]
* `--n-cell-training INTEGER RANGE`: Number of cells used for training  [default: 100000; 1000&lt;=x&lt;=1000000]
* `--dataset-type [scrna|spatial2d|spatial3d]`: Type of dataset: scRNA (uses KNN on latent space), spatial2D (2D spatial), or spatial3D (multi-slice)  [default: spatial2D]
* `--spatial-neighbors INTEGER RANGE`: k1: Number of spatial neighbors in it&#x27;s own slice for spatial dataset  [default: 301; 10&lt;=x&lt;=5000]
* `--homogeneous-neighbors INTEGER RANGE`: k3: Number of homogeneous neighbors per cell (for spatial) or KNN neighbors (for scRNA-seq)  [default: 21; 1&lt;=x&lt;=200]
* `--cell-embedding-similarity-threshold FLOAT RANGE`: Minimum similarity threshold for cell embedding.  [default: 0.0; 0.0&lt;=x&lt;=1.0]
* `--spatial-domain-similarity-threshold FLOAT RANGE`: Minimum similarity threshold for spatial domain embedding.  [default: 0.6; 0.0&lt;=x&lt;=1.0]
* `--no-expression-fraction`: Skip expression fraction filtering
* `--adjacent-slice-spatial-neighbors INTEGER RANGE`: Number of spatial neighbors to find on each adjacent slice for 3D data  [default: 200; 10&lt;=x&lt;=2000]
* `--n-adjacent-slices INTEGER RANGE`: Number of adjacent slices to search above and below (± n_adjacent_slices) in 3D space for each focal spot. Padding will be applied automatically.  [default: 1; 0&lt;=x&lt;=5]
* `--cross-slice-marker-score-strategy [global_pool|per_slice_pool|hierarchical_pool]`: Strategy for computing marker scores across slices in spatial3D datasets. &#x27;global_pool&#x27;: Select the top K most similar neighbors globally across all slices combined. &#x27;per_slice_pool&#x27;: Select a fixed number of neighbors (K) from each slice independently, then compute a single weighted average score from all selected neighbors. &#x27;hierarchical_pool&#x27;: Compute an independent marker score for each slice using its top K neighbors, then take the average of these per-slice scores.  [default: hierarchical_pool]
* `--high-quality-neighbor-filter / --no-high-quality-filter`: Only find neighbors within high quality cells (requires High_quality column in obs)  [default: no-high-quality-filter]
* `--use-gpu / --no-gpu`: Use GPU for JAX-accelerated spatial LDSC implementation  [default: use-gpu]
* `--memmap-tmp-dir DIRECTORY`: Temporary directory for memory-mapped files to improve I/O performance on slow filesystems. If provided, memory maps will be copied to this directory for faster random access during computation.
* `--mkscore-batch-size INTEGER RANGE`: Number of cells per batch for marker score calculation. Reduce this value (e.g., 50) if encountering GPU OOM errors.  [default: 500; 10&lt;=x&lt;=1000]
* `--rank-read-workers INTEGER RANGE`: Number of parallel reader threads for rank memory map  [default: 16; 1&lt;=x&lt;=50]
* `--mkscore-compute-workers INTEGER RANGE`: Number of parallel compute threads for marker score calculation  [default: 4; 1&lt;=x&lt;=16]
* `--mkscore-write-workers INTEGER RANGE`: Number of parallel writer threads for marker scores  [default: 4; 1&lt;=x&lt;=50]
* `--compute-input-queue-size INTEGER RANGE`: Maximum size of compute input queue (multiplier of mkscore_compute_workers)  [default: 5; 1&lt;=x&lt;=10]
* `--writer-queue-size INTEGER RANGE`: Maximum size of writer input queue  [default: 100; 10&lt;=x&lt;=500]
* `--ldsc-read-workers INTEGER RANGE`: Number of read workers  [default: 10; x&gt;=1]
* `--ldsc-compute-workers INTEGER RANGE`: Number of compute workers for LDSC regression  [default: 10; x&gt;=1]
* `--spots-per-chunk-quick-mode INTEGER RANGE`: Number of spots per chunk in quick mode  [default: 50; x&gt;=1]
* `--trait-name TEXT`: Name of the trait for GWAS analysis
* `--sumstats-file FILE`: Path to GWAS summary statistics file
* `--sumstats-config-file FILE`: Path to sumstats config file
* `--w-ld-dir DIRECTORY`: Directory containing the weights files (w_ld)
* `--additional-baseline-h5ad-path-list FILE`: List of additional baseline h5ad paths
* `--chisq-max INTEGER`: Maximum chi-square value
* `--cell-indices-range <INTEGER INTEGER>...`: 0-based range [start, end) of cell indices to process
* `--sample-filter TEXT`: Filter processing to a specific sample
* `--n-blocks INTEGER RANGE`: Number of jackknife blocks  [default: 200; x&gt;=1]
* `--snp-gene-weight-adata-path FILE`: Path to the SNP-gene weight matrix (H5AD format)
* `--marker-score-feather-path FILE`: Path to marker score feather file
* `--marker-score-h5ad-path FILE`: Path to marker score h5ad file
* `--marker-score-format [memmap|feather|h5ad]`: Format of marker scores
* `--cauchy-annotations TEXT`: List of annotations in adata.obs to use
* `--downsampling-n-spots-pcc INTEGER RANGE`: Number of spots to downsample for PCC calculation if n_spots &gt; this value  [default: 20000; 1000&lt;=x&lt;=100000]
* `--downsampling-n-spots-3d INTEGER RANGE`: Number of spots to downsample for 3D visualization if n_spots &gt; this value  [default: 1000000; 1000&lt;=x&lt;=2000000]
* `--downsampling-n-spots-2d INTEGER RANGE`: Max spots per sample for 2D distribution plots. Samples with more spots will be randomly downsampled.  [default: 250000; 10000&lt;=x&lt;=500000]
* `--top-corr-genes INTEGER RANGE`: Number of top correlated genes to display  [default: 50; 1&lt;=x&lt;=500]
* `--plot-origin TEXT`: Plot origin for spatial plots (&#x27;upper&#x27; or &#x27;lower&#x27;). &#x27;upper&#x27; will flip the y-axis (standard for images).  [default: upper]
* `--start-step TEXT`: Step to start execution from (find_latent, latent2gene, spatial_ldsc, cauchy, report)  [default: find_latent]
* `--stop-step TEXT`: Step to stop execution at (inclusive)
* `--help`: Show this message and exit.

## `gsmap find-latent`

Find latent representations of each spot using Graph Neural Networks.

This step:
- Loads spatial transcriptomics data
- Builds neighborhood graphs
- Learns latent representations using GNN
- Saves the model and embeddings

**Usage**:

```console
$ gsmap find-latent [OPTIONS]
```

**Options**:

* `--workdir DIRECTORY`: Path to the working directory  [required]
* `--project-name TEXT`: Name of the project  [required]
* `--h5ad-path PATH`: Space-separated list of h5ad file paths. Sample names are derived from file names without suffix.
* `--h5ad-yaml FILE`: YAML file with sample names and h5ad paths
* `--h5ad-list-file FILE`: Each row is a h5ad file path, sample name is the file name without suffix
* `--data-layer TEXT`: Gene expression raw counts data layer in h5ad layers, e.g., &#x27;count&#x27;, &#x27;counts&#x27;. Other wise use &#x27;X&#x27; for adata.X  [default: X]
* `--spatial-key TEXT`: Spatial key in adata.obsm storing spatial coordinates  [default: spatial]
* `--annotation TEXT`: Annotation of cell type in adata.obs to use
* `--homolog-file FILE`: Path to homologous gene conversion file
* `--latent-representation-niche TEXT`: Key for spatial niche embedding in obsm  [default: emb_niche]
* `--latent-representation-cell TEXT`: Key for cell identity embedding in obsm  [default: emb_cell]
* `--high-quality-cell-qc / --no-high-quality-cell-qc`: Enable/disable high quality cell QC based on module scores. If enabled, it will compute DEG and module scores.  [default: high-quality-cell-qc]
* `--feat-cell INTEGER RANGE`: Number of top variable features to retain  [default: 2000; 100&lt;=x&lt;=10000]
* `--n-neighbors INTEGER RANGE`: Number of neighbors for LGCN  [default: 10; 1&lt;=x&lt;=50]
* `--k INTEGER RANGE`: Graph convolution depth for LGCN  [default: 3; 1&lt;=x&lt;=10]
* `--hidden-size INTEGER RANGE`: Units in the first hidden layer  [default: 128; 32&lt;=x&lt;=512]
* `--embedding-size INTEGER RANGE`: Size of the latent embedding layer  [default: 32; 8&lt;=x&lt;=128]
* `--use-tf`: Enable transformer module
* `--module-dim INTEGER RANGE`: Dimensionality of transformer modules  [default: 30; 10&lt;=x&lt;=100]
* `--hidden-gmf INTEGER RANGE`: Hidden units for global mean feature extractor  [default: 128; 32&lt;=x&lt;=512]
* `--n-modules INTEGER RANGE`: Number of transformer modules  [default: 16; 4&lt;=x&lt;=64]
* `--nhead INTEGER RANGE`: Number of attention heads in transformer  [default: 4; 1&lt;=x&lt;=16]
* `--n-enc-layer INTEGER RANGE`: Number of transformer encoder layers  [default: 2; 1&lt;=x&lt;=8]
* `--distribution TEXT`: Distribution type for loss calculation  [default: nb]
* `--batch-size INTEGER RANGE`: Batch size for training  [default: 1024; 32&lt;=x&lt;=4096]
* `--itermax INTEGER RANGE`: Maximum number of training iterations  [default: 100; 10&lt;=x&lt;=1000]
* `--patience INTEGER RANGE`: Early stopping patience  [default: 10; 1&lt;=x&lt;=50]
* `--two-stage / --single-stage`: Tune the cell embeddings based on the provided annotation  [default: single-stage]
* `--do-sampling / --no-sampling`: Down-sampling cells in training  [default: do-sampling]
* `--n-cell-training INTEGER RANGE`: Number of cells used for training  [default: 100000; 1000&lt;=x&lt;=1000000]
* `--help`: Show this message and exit.

## `gsmap latent-to-gene`

Estimate gene marker scores for each spot using latent representations.

This step:
- Loads latent representations
- Estimates gene marker scores
- Performs spatial smoothing
- Saves marker scores for LDSC

**Usage**:

```console
$ gsmap latent-to-gene [OPTIONS]
```

**Options**:

* `--workdir DIRECTORY`: Path to the working directory  [required]
* `--project-name TEXT`: Name of the project  [required]
* `--dataset-type [scrna|spatial2d|spatial3d]`: Type of dataset: scRNA (uses KNN on latent space), spatial2D (2D spatial), or spatial3D (multi-slice)  [default: spatial2D]
* `--h5ad-path PATH`: Space-separated list of h5ad file paths. Sample names are derived from file names without suffix.
* `--h5ad-yaml FILE`: YAML file with sample names and h5ad paths
* `--h5ad-list-file FILE`: Each row is a h5ad file path, sample name is the file name without suffix
* `--annotation TEXT`: Cell type annotation in adata.obs to use. This would constrain finding homogeneous spots within each cell type
* `--data-layer TEXT`: Gene expression raw counts data layer in h5ad layers, e.g., &#x27;count&#x27;, &#x27;counts&#x27;. Other wise use &#x27;X&#x27; for adata.X  [default: X]
* `--latent-representation-niche TEXT`: Key for spatial niche embedding in obsm
* `--latent-representation-cell TEXT`: Key for cell identity embedding in obsm  [default: emb_cell]
* `--spatial-key TEXT`: Spatial key in adata.obsm  [default: spatial]
* `--spatial-neighbors INTEGER RANGE`: k1: Number of spatial neighbors in it&#x27;s own slice for spatial dataset  [default: 301; 10&lt;=x&lt;=5000]
* `--homogeneous-neighbors INTEGER RANGE`: k3: Number of homogeneous neighbors per cell (for spatial) or KNN neighbors (for scRNA-seq)  [default: 21; 1&lt;=x&lt;=200]
* `--cell-embedding-similarity-threshold FLOAT RANGE`: Minimum similarity threshold for cell embedding.  [default: 0.0; 0.0&lt;=x&lt;=1.0]
* `--spatial-domain-similarity-threshold FLOAT RANGE`: Minimum similarity threshold for spatial domain embedding.  [default: 0.6; 0.0&lt;=x&lt;=1.0]
* `--no-expression-fraction`: Skip expression fraction filtering
* `--adjacent-slice-spatial-neighbors INTEGER RANGE`: Number of spatial neighbors to find on each adjacent slice for 3D data  [default: 200; 10&lt;=x&lt;=2000]
* `--n-adjacent-slices INTEGER RANGE`: Number of adjacent slices to search above and below (± n_adjacent_slices) in 3D space for each focal spot. Padding will be applied automatically.  [default: 1; 0&lt;=x&lt;=5]
* `--cross-slice-marker-score-strategy [global_pool|per_slice_pool|hierarchical_pool]`: Strategy for computing marker scores across slices in spatial3D datasets. &#x27;global_pool&#x27;: Select the top K most similar neighbors globally across all slices combined. &#x27;per_slice_pool&#x27;: Select a fixed number of neighbors (K) from each slice independently, then compute a single weighted average score from all selected neighbors. &#x27;hierarchical_pool&#x27;: Compute an independent marker score for each slice using its top K neighbors, then take the average of these per-slice scores.  [default: hierarchical_pool]
* `--high-quality-neighbor-filter / --no-high-quality-filter`: Only find neighbors within high quality cells (requires High_quality column in obs)  [default: no-high-quality-filter]
* `--use-gpu / --no-gpu`: Use GPU for JAX computations (requires sufficient GPU memory)  [default: use-gpu]
* `--memmap-tmp-dir DIRECTORY`: Temporary directory for memory-mapped files to improve I/O performance on slow filesystems. If provided, memory maps will be copied to this directory for faster random access during computation.
* `--mkscore-batch-size INTEGER RANGE`: Number of cells per batch for marker score calculation. Reduce this value (e.g., 50) if encountering GPU OOM errors.  [default: 500; 10&lt;=x&lt;=1000]
* `--rank-read-workers INTEGER RANGE`: Number of parallel reader threads for rank memory map  [default: 16; 1&lt;=x&lt;=50]
* `--mkscore-compute-workers INTEGER RANGE`: Number of parallel compute threads for marker score calculation  [default: 4; 1&lt;=x&lt;=16]
* `--mkscore-write-workers INTEGER RANGE`: Number of parallel writer threads for marker scores  [default: 4; 1&lt;=x&lt;=50]
* `--compute-input-queue-size INTEGER RANGE`: Maximum size of compute input queue (multiplier of mkscore_compute_workers)  [default: 5; 1&lt;=x&lt;=10]
* `--writer-queue-size INTEGER RANGE`: Maximum size of writer input queue  [default: 100; 10&lt;=x&lt;=500]
* `--help`: Show this message and exit.

## `gsmap spatial-ldsc`

Run spatial LDSC analysis for genetic association.

This step:
- Loads LD scores and GWAS summary statistics
- Performs spatial LDSC regression
- Computes enrichment statistics
- Saves results for downstream analysis

**Usage**:

```console
$ gsmap spatial-ldsc [OPTIONS]
```

**Options**:

* `--workdir DIRECTORY`: Path to the working directory  [required]
* `--project-name TEXT`: Name of the project  [required]
* `--use-gpu / --no-gpu`: Use GPU for JAX-accelerated spatial LDSC implementation  [default: use-gpu]
* `--memmap-tmp-dir DIRECTORY`: Temporary directory for memory-mapped files to improve I/O performance on slow filesystems. If provided, memory maps will be copied to this directory for faster random access during computation.
* `--ldsc-read-workers INTEGER RANGE`: Number of read workers  [default: 10; x&gt;=1]
* `--ldsc-compute-workers INTEGER RANGE`: Number of compute workers for LDSC regression  [default: 10; x&gt;=1]
* `--spots-per-chunk-quick-mode INTEGER RANGE`: Number of spots per chunk in quick mode  [default: 50; x&gt;=1]
* `--trait-name TEXT`: Name of the trait for GWAS analysis
* `--sumstats-file FILE`: Path to GWAS summary statistics file
* `--sumstats-config-file FILE`: Path to sumstats config file
* `--w-ld-dir DIRECTORY`: Directory containing the weights files (w_ld)
* `--additional-baseline-h5ad-path-list FILE`: List of additional baseline h5ad paths
* `--chisq-max INTEGER`: Maximum chi-square value
* `--cell-indices-range <INTEGER INTEGER>...`: 0-based range [start, end) of cell indices to process
* `--sample-filter TEXT`: Filter processing to a specific sample
* `--n-blocks INTEGER RANGE`: Number of jackknife blocks  [default: 200; x&gt;=1]
* `--snp-gene-weight-adata-path FILE`: Path to the SNP-gene weight matrix (H5AD format)
* `--marker-score-feather-path FILE`: Path to marker score feather file
* `--marker-score-h5ad-path FILE`: Path to marker score h5ad file
* `--marker-score-format [memmap|feather|h5ad]`: Format of marker scores
* `--help`: Show this message and exit.

## `gsmap cauchy-combination`

Run Cauchy combination test to combine spatial LDSC results across spots.

This step:
- Loads spatial LDSC results for a trait
- Removes outliers
- Performs Cauchy combination test for each annotation
- Computes Fisher&#x27;s exact test for enrichment

**Usage**:

```console
$ gsmap cauchy-combination [OPTIONS]
```

**Options**:

* `--workdir DIRECTORY`: Path to the working directory  [required]
* `--project-name TEXT`: Name of the project  [required]
* `--trait-name TEXT`: Name of the trait for GWAS analysis
* `--sumstats-file FILE`: Path to GWAS summary statistics file
* `--sumstats-config-file FILE`: Path to sumstats config file
* `--annotation TEXT`: Name of the annotation in adata.obs to use
* `--cauchy-annotations TEXT`: List of annotations in adata.obs to use
* `--help`: Show this message and exit.

## `gsmap ldscore-weight-matrix`

Compute LD score weight matrices for features.

This command runs the LDScorePipeline to:
- Load genotypes (PLINK)
- Load feature mappings (BED/Dict) or Annotations
- Compute LD-based weights between SNPs and Features
- Save results as AnnData (.h5ad)

**Usage**:

```console
$ gsmap ldscore-weight-matrix [OPTIONS]
```

**Options**:

* `--bfile-root TEXT`: Reference panel prefix template (e.g., &#x27;data/1000G.{chr}&#x27;)  [required]
* `--hm3-snp-path FILE`: Path to HM3 SNP list  [required]
* `--output-dir PATH`: Output directory. If None, uses {workdir}/{project_name}/generate_ldscore
* `--output-filename TEXT`: Prefix for output files  [default: ld_score_weights]
* `--omics-h5ad-path FILE`: Path to omics H5AD file
* `--mapping-type TEXT`: Mapping type: &#x27;bed&#x27; or &#x27;dict&#x27;  [default: bed]
* `--mapping-file FILE`: Path to mapping file
* `--annot-file TEXT`: Template for annotation files (e.g., &#x27;baseline.{chr}.annot.gz&#x27;)
* `--feature-window-size INTEGER`: bp window for mapping (e.g. TSS window)  [default: 0]
* `--strategy TEXT`: Strategy: &#x27;score&#x27;, &#x27;tss&#x27;, &#x27;center&#x27;, &#x27;allow_repeat&#x27;  [default: score]
* `--ld-wind FLOAT`: LD window size  [default: 1.0]
* `--ld-unit TEXT`: LD unit: &#x27;SNP&#x27;, &#x27;KB&#x27;, &#x27;CM&#x27;  [default: CM]
* `--maf-min FLOAT`: Minimum MAF filter  [default: 0.01]
* `--chromosomes TEXT`: Chromosomes to process. &#x27;all&#x27; uses 1-22 autosomes, or provide a comma-separated list of chromosomes (e.g., &#x27;1,2,3&#x27;)  [default: all]
* `--batch-size-hm3 INTEGER`: Batch size for HM3 SNPs  [default: 50]
* `--calculate-w-ld / --no-calculate-w-ld`: Whether to calculate w_ld  [default: no-calculate-w-ld]
* `--w-ld-dir PATH`: Directory for w_ld outputs
* `--help`: Show this message and exit.

## `gsmap format-sumstats`

Format GWAS summary statistics for gsMap or COJO.

This command:
- Filters SNPs based on INFO, MAF, and P-value
- Converts SNP positions to RSID (if dbsnp is provided)
- Saves formatted summary statistics

**Usage**:

```console
$ gsmap format-sumstats [OPTIONS]
```

**Options**:

* `--sumstats TEXT`: Path to gwas summary data  [required]
* `--out TEXT`: Path to save the formatted gwas data  [required]
* `--snp TEXT`: Name of snp column
* `--a1 TEXT`: Name of effect allele column
* `--a2 TEXT`: Name of none-effect allele column
* `--info TEXT`: Name of info column
* `--beta TEXT`: Name of gwas beta column.
* `--se TEXT`: Name of gwas standar error of beta column
* `--p TEXT`: Name of p-value column
* `--frq TEXT`: Name of A1 ferquency column
* `--n TEXT`: Name of sample size column
* `--z TEXT`: Name of gwas Z-statistics column
* `--or TEXT`: Name of gwas OR column
* `--se-or TEXT`: Name of standar error of OR column
* `--chr TEXT`: Name of SNP chromosome column  [default: Chr]
* `--pos TEXT`: Name of SNP positions column  [default: Pos]
* `--dbsnp TEXT`: Path to reference dnsnp file
* `--chunksize INTEGER`: Chunk size for loading dbsnp file  [default: 1000000]
* `--format [gsmap|cojo]`: Format of output data  [default: gsMap]
* `--info-min FLOAT`: Minimum INFO score.  [default: 0.9]
* `--maf-min FLOAT`: Minimum MAF.  [default: 0.01]
* `--keep-chr-pos / --no-keep-chr-pos`: Keep SNP chromosome and position columns in the output data  [default: no-keep-chr-pos]
* `--help`: Show this message and exit.

## `gsmap report-view`

Launch a local web server to view the gsMap report.

This command starts a simple HTTP server to serve the report files,
which is necessary for proper loading of JavaScript modules.

Example:
    gsmap report-view /path/to/project/gsmap_web_report
    gsmap report-view /path/to/project/gsmap_web_report --port 9000

**Usage**:

```console
$ gsmap report-view [OPTIONS] REPORT_PATH
```

**Arguments**:

* `REPORT_PATH`: Path to gsmap_web_report directory containing index.html  [required]

**Options**:

* `--port INTEGER`: Port to serve the report on  [default: 8080]
* `--no-browser`: Don&#x27;t automatically open browser
* `--help`: Show this message and exit.

**Demonstration**:  
A demonstration report can be found [here](https://yanglab.westlake.edu.cn/gsmap3d-report).
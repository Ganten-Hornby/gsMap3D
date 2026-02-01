# Scalability and Performance

`gsMap` is designed to efficiently process large-scale spatial transcriptomics datasets with millions of cells/spots. This guide covers the key scalability features and how to optimize performance for your hardware.

## Overview

gsMap employs multiple strategies to achieve high performance:

- **GPU/TPU Acceleration**: JAX-based computation with JIT compilation
- **Multi-threaded Pipeline**: Parallel reader, compute, and writer workers
- **Memory Mapping**: Efficient disk-based storage with SSD optimization
- **Vectorized Operations**: Batched matrix operations using einsum
- **Sparse Matrix Support**: Memory-efficient handling of sparse data

## GPU/TPU Acceleration

gsMap uses [JAX](https://github.com/google/jax) for GPU/TPU acceleration in the Spatial LDSC computation stage. JAX provides Just-In-Time (JIT) compilation and automatic differentiation, enabling efficient execution on accelerators.

### Enabling/Disabling GPU

````{tab} CLI
```bash
# Enable GPU (default)
gsmap quick-mode --use-gpu ...

# Disable GPU (CPU only)
gsmap quick-mode --no-gpu ...
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig

# Enable GPU (default)
config = QuickModeConfig(
    use_gpu=True,
    ...
)

# Disable GPU
config = QuickModeConfig(
    use_gpu=False,
    ...
)
```
````

### JAX Memory Management

By default, JAX pre-allocates approximately 75% of the total GPU memory upon initialization to minimize memory fragmentation and allocation overhead. However, if you want to allocate memory as needed or constrain JAX to use a specific proportion of the memory, you can set the following environment variables:

```bash
# Disable pre-allocation (allocate memory as needed)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Limit GPU memory to a specific fraction (e.g., 0.5 for 50%)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

### Device Selection

To select a specific GPU device:

```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Use GPU 1
export CUDA_VISIBLE_DEVICES=1

```

### Performance Characteristics

The JAX-accelerated Spatial LDSC computation uses:

- **Batched Matrix Operations**: Processes multiple spots simultaneously using efficient einsum operations
- **JIT Compilation**: Functions are compiled once and reused, reducing overhead
- **Float32 Precision**: Uses 32-bit floats for speed and memory efficiency (configurable)

```{note}
The first run may be slower due to JIT compilation. Subsequent runs will be faster as compiled functions are cached.
```

## GSS calculation

gsMap implements a three-stage parallel message queue pipeline for the Gene Specificity Score (GSS) calculation:

```{mermaid}
graph LR
    subgraph "Stage 1: Gene Rank Reading"
        R1[Reader 1]
        R2[Reader 2]
        R3[Reader ...]
    end

    subgraph "Stage 2: GSS Calculation"
        C1[Compute 1]
        C2[Compute 2]
        C3[Compute ...]
    end

    subgraph "Stage 3: Write GSS"
        W1[Writer 1]
        W2[Writer 2]
        W3[Writer ...]
    end

    R1 --> Q1[Rank→Compute Queue]
    R2 --> Q1
    R3 --> Q1
    Q1 --> C1
    Q1 --> C2
    Q1 --> C3
    C1 --> Q2[Compute→Write Queue]
    C2 --> Q2
    C3 --> Q2
    Q2 --> W1
    Q2 --> W2
    Q2 --> W3
```

### Pipeline Stages

1. **Stage 1 - Gene Rank Reading**: Reader workers load pre-computed gene rank data from memory-mapped files in parallel. Each reader fetches batch data containing neighbor indices for homogeneous spots.

2. **Stage 2 - GSS Calculation**: Compute workers receive rank data via the **Rank→Compute Queue** and calculate marker scores using JAX-accelerated weighted geometric mean operations. The computation transforms gene ranks into Gene Specificity Scores (GSS).

3. **Stage 3 - Write GSS**: Writer workers receive computed GSS results via the **Compute→Write Queue** and write them back to memory-mapped files for downstream Spatial LDSC analysis.

### Worker Configuration

The number of workers for each stage can be configured:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `rank_read_workers` | Parallel reader threads for Stage 1 | 100 | 1-16 |
| `mkscore_compute_workers` | Parallel compute threads for Stage 2 | 4 | 1-16 |
| `mkscore_write_workers` | Parallel writer threads for Stage 3 | 4 | 1-16 |

### Queue Configuration

The message queues between stages control data flow and backpressure:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `compute_input_queue_size` | Maximum size of **Rank→Compute Queue** (multiplier of `mkscore_compute_workers`). Controls how many batches can be buffered between reading and computing stages. | 5 | 1-10 |
| `writer_queue_size` | Maximum size of **Compute→Write Queue**. Controls how many computed batches can be buffered before being written to disk. | 100 | 10-500 |

```{tip}
Queue sizes balance memory usage and throughput. Larger queues allow better buffering when stage speeds vary, but consume more memory. The default values work well for most systems.
```

### Performance Monitoring

After processing each cell type, gsMap displays a pipeline summary table showing throughput metrics for each stage:

```text
╭─────────────────────────────────── Pipeline Summary ───────────────────────────────────╮
│                   ✓ Completed Brain                                                    │
│  Property                                       Value                                  │
│  Total Batches                                     59                                  │
│  Time Elapsed                                   2.66s                                  │
│  Pipeline Throughput  22.15 batches/s (11076 cells/s)                                  │
│           Component Performance (per worker)                                           │
│ ┏━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓                                │
│ ┃ Component ┃ Throughput ┃ Workers ┃ Total Throughput ┃                                │
│ ┡━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩                                │
│ │ Reader    │   8.99 b/s │      16 │        71942 c/s │                                │
│ │ Computer  │   7.24 b/s │       4 │        14472 c/s │                                │
│ │ Writer    │ 101.91 b/s │       4 │       203821 c/s │                                │
│ └───────────┴────────────┴─────────┴──────────────────┘                                │
╰────────────────────────────────────────────────────────────────────────────────────────╯
```

Use this table to identify pipeline bottlenecks:

- **Throughput**: Per-worker processing speed in batches/second (b/s)
- **Total Throughput**: Combined throughput across all workers in cells/second (c/s)
- **Pipeline Throughput**: The effective end-to-end throughput, limited by the slowest stage

```{tip}
**Identifying Bottlenecks**: The stage with the lowest **Total Throughput** is the bottleneck limiting overall performance.

- **Reader is the bottleneck** (Reader throughput < Computer throughput): 
  - Increase `rank_read_workers` to add more parallel readers
  - Use [`--memmap-tmp-dir`](#memory-mapping-with-ssd-optimization) to copy memory-mapped files to a fast SSD
  
- **Computer is the bottleneck** (Computer throughput < Reader/Writer throughput):
  - Ensure GPU acceleration is enabled (`--use-gpu`)
  - Increase available CPU cores or `mkscore_compute_workers` if using CPU-only mode
- **Writer is the bottleneck** (Writer throughput < Computer throughput):
  - Increase `mkscore_write_workers`
  - Use a faster storage device for output
```

## Memory Mapping

`gsMap` uses memory-mapped files to handle data that exceeds available RAM. The `--memmap-tmp-dir` option allows you to specify a fast SSD for temporary files, significantly improving I/O performance.

### Why Use memmap-tmp-dir?

When your working directory is on a slow filesystem (e.g., network storage, HDD), random access to memory-mapped files can become a bottleneck. By specifying a fast local SSD as the temporary directory, gsMap:

1. Copies memory-mapped files to the fast storage
2. Performs all random access operations on the fast storage
3. Syncs results back to the original location when complete

### Usage

````{tab} CLI
```bash
gsmap quick-mode \
    --memmap-tmp-dir /path/to/fast/ssd/tmp \
    ...
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig

config = QuickModeConfig(
    memmap_tmp_dir="/path/to/fast/ssd/tmp",
    ...
)
```
````

### Recommended Setup

```bash
# Create a dedicated temp directory on your fastest SSD
mkdir -p /mnt/nvme_ssd/gsmap_tmp

# Run gsMap with the temp directory
gsmap quick-mode \
    --workdir /network/storage/results \
    --memmap-tmp-dir /mnt/nvme_ssd/gsmap_tmp \
    ...
```

### Storage Requirements

The temporary directory needs sufficient space for intermediate memory-mapped files. Storage is estimated as:

```{math}
\text{Temp Storage Space} = \text{spots} \times \text{genes} \times 4\ \text{bytes} \times 2\ \text{(rank + marker score)}
```

Assuming ~20,000 genes:

| Dataset Size | Calculation | Approximate Temp Space |
|--------------|-------------|------------------------|
| 100K spots | 100K × 20K × 4 × 2 | ~16 GB |
| 1M spots | 1M × 20K × 4 × 2 | ~160 GB |
| 10M spots | 10M × 20K × 4 × 2 | ~1.6 TB |

```{important}
Ensure your temporary directory has enough free space before running. gsMap will fail if the disk runs out of space during computation.
```

## Batch Processing

gsMap processes data in batches to balance memory usage and computational efficiency:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rank_batch_size` | Spots per batch for rank calculation | 500 |
| `mkscore_batch_size` | Spots per batch for marker score | 500 |
| `spots_per_chunk_quick_mode` | Spots per chunk for Spatial LDSC | 50 |

Larger batch sizes can improve throughput but require more memory. Adjust based on your available RAM and GPU memory.


## Performance Recommendations


### For Very Large Datasets (>1M spots)

1. **Use high-end GPU**: NVIDIA A100 or similar recommended
2. **Use NVMe SSD**: Fast local storage is critical. Use `--memmap-tmp-dir` to specify a fast SSD for temporary files to avoid reading becoming a bottleneck. If SSD is not available, increase `rank_read_workers` to compensate. 
3. **Increase batch sizes**: If memory allows, larger batches improve throughput


### For CPU-Only Systems

1. **Disable GPU**: Use `--no-gpu` to avoid JAX GPU initialization overhead
2. **Increase CPU cores and compute workers**: The Spatial LDSC step scales near-linearly with the number of available CPU cores. Increase `ldsc_compute_workers` to match available cores.
3. **Use SSD**: Fast storage becomes even more important without GPU acceleration

### For Cluster Users

When running on HPC clusters, you can parallelize gsMap across multiple nodes:

#### 1. Submit Each Trait as a Separate Job

For multiple GWAS traits, submit each as an independent job:

```bash
# Job 1: Process trait1
gsmap quick-mode \
    --start-step spatial_ldsc --stop-step spatial_ldsc \
    --trait-name trait1 --sumstats-file /path/to/trait1.sumstats.gz \
    ...

# Job 2: Process trait2  
gsmap quick-mode \
    --start-step spatial_ldsc --stop-step spatial_ldsc \
    --trait-name trait2 --sumstats-file /path/to/trait2.sumstats.gz \
    ...
```

#### 2. Split Large Datasets by Cell Indices

For very large datasets, use `--cell-indices-range` to process subsets of cells in parallel:

````{tab} CLI
```bash
# Job 1: Process cells 0-1000000
gsmap quick-mode \
    --start-step spatial_ldsc --stop-step spatial_ldsc \
    --cell-indices-range 0 1000000 \
    --trait-name trait1 --sumstats-file /path/to/trait1.sumstats.gz \
    ...

# Job 2: Process cells 1000000-2000000
gsmap quick-mode \
    --start-step spatial_ldsc --stop-step spatial_ldsc \
    --cell-indices-range 1000000 2000000 \
    --trait-name trait1 --sumstats-file /path/to/trait1.sumstats.gz \
    ...
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig
from gsMap.run_all_mode import run_quick_mode

# Job 1: Process cells 0-1000000
config = QuickModeConfig(
    start_step="spatial_ldsc",
    stop_step="spatial_ldsc",
    cell_indices_range=(0, 1000000),
    trait_name="trait1",
    sumstats_file="/path/to/trait1.sumstats.gz",
    ...
)
run_quick_mode(config)

# Job 2: Process cells 1000000-2000000
config = QuickModeConfig(
    start_step="spatial_ldsc",
    stop_step="spatial_ldsc",
    cell_indices_range=(1000000, 2000000),
    trait_name="trait1",
    sumstats_file="/path/to/trait1.sumstats.gz",
    ...
)
run_quick_mode(config)
```
````

```{note}
The `--cell-indices-range` option uses 0-based indexing with the format `[start, end)` (start inclusive, end exclusive).
```

#### 3. Merge Results

After all jobs complete, merge the partial result files. Each partial file is named with the cell range:

```text
# File naming pattern for partial results:
{workdir}/{project_name}/spatial_ldsc/{project_name}_{trait_name}_cells_{start}_{end}.csv.gz

# Example files:
my_project_trait1_cells_0_1000000.csv.gz
my_project_trait1_cells_1000000_2000000.csv.gz
```

Use pandas to concatenate the partial results:

```python
import pandas as pd
from pathlib import Path

workdir = Path('/path/to/workdir')
project = 'my_project'
trait = 'trait1'

# Find all partial result files
result_dir = workdir / project / 'spatial_ldsc'
partial_files = sorted(result_dir.glob(f'{project}_{trait}_cells_*.csv.gz'))

print(f"Found {len(partial_files)} partial result files")

# Concatenate and save
results = pd.concat([pd.read_csv(f) for f in partial_files], ignore_index=True)
results.to_csv(result_dir / f'{project}_{trait}.csv.gz', index=False, compression='gzip')

print(f"Merged {len(results)} spots to {result_dir / f'{project}_{trait}.csv.gz'}")
```



## Troubleshooting

### GPU Out of Memory (OOM) Errors

If you encounter GPU out of memory errors in latent to gene step, try the following solutions in order:

**1. Reduce batch size** (recommended first step):

```bash
# Reduce the marker score batch size (default: 500)
gsmap quick-mode --mkscore-batch-size 50 ...
```

**2. Limit JAX memory allocation**:

```bash
# Remove JAX memory pre-allocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

**3. Disable GPU entirely**:

```bash
# Fall back to CPU computation
gsmap quick-mode --no-gpu ...
```

### JAX/CUDA Issues

```bash
# Check JAX device availability
python -c "import jax; print(jax.devices())"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

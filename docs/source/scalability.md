# Scalability and Performance

`gsMap3D` is designed to efficiently process large-scale 3D spatial transcriptomics datasets with millions of cells/spots. This guide covers the key scalability features and how to optimize performance for your hardware.

## Overview

gsMap3D employs multiple strategies to achieve high performance:

- **GPU/TPU Acceleration**: JAX-based computation with JIT compilation
- **Bounded Rank Cache**: One compact cache or capacity-limited windows per cell type
- **Memory Mapping**: Efficient disk-based storage with SSD optimization
- **Vectorized Operations**: Batched matrix operations using einsum
- **Sparse Matrix Support**: Memory-efficient handling of sparse data

## GPU/TPU Acceleration

gsMap3D uses [JAX](https://github.com/google/jax) for GPU/TPU acceleration in the latent-to-gene and Spatial LDSC computation stages. JAX provides Just-In-Time (JIT) compilation, enabling efficient execution on accelerators.

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
config = QuickModeConfig(use_gpu=True, ...)

# Disable GPU
config = QuickModeConfig(use_gpu=False, ...)
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

```{note}
The first run may be slower due to JIT compilation. Subsequent runs will be faster as compiled functions are cached.
```

## GSS calculation

The latent-to-gene stage calculates the Gene Specificity Score (GSS) with one bounded rank-cache engine. Homogeneous cells are placed in a locality-preserving order before rank access:

- Spatial2D data are ordered separately within each section.
- Spatial3D data use measured physical x/y/z coordinates when a varying z coordinate is available.
- Spatial3D data without measured z use an explicit ordinal section coordinate.
- Non-spatial data use homogeneous-neighbor overlap.

For a cell type whose distinct rank rows fit under the 2 GiB cache limit, gsMap3D reads each distinct row once and transfers one right-sized cache to the JAX device. Larger cell types are divided into maximal contiguous windows under the same limit. On GPU, the next host window is prefetched and rank rows shared by adjacent windows remain in stable device-cache slots. This design bounds peak cache memory without reverting to random rank reads for every marker-score batch.

### Execution controls

| Parameter                  | Description                                                      | Default | Range  |
| -------------------------- | ---------------------------------------------------------------- | ------- | ------ |
| `rank_read_workers`        | Threads that materialize distinct rows for a rank-cache window    | 120     | 1-128  |
| `mkscore_compute_workers`  | Concurrent fixed-shape JAX marker-score submissions               | 4       | 1-16   |
| `mkscore_write_workers`    | Threads that write completed GSS batches                          | 2       | 1-50   |
| `compute_input_queue_size` | Maximum pending batches per compute worker                        | 1       | 1-10   |
| `writer_queue_size`        | Capacity of the bounded completed-batch writer queue              | 20      | 10-500 |

`mkscore_batch_size=0` selects a deterministic workload- and device-aware batch size. An explicit positive value overrides that selection.

After each cell type, the log reports end-to-end cells per second and separates time spent remapping rows, reading rank windows, and updating the device cache. The same record includes the numbers of distinct, newly read, and adjacent-window-retained rows.

```{tip}
If rank-window read time dominates, place the rank memory map on local NVMe storage or use `--memmap-tmp-dir` when the complete rank and output files fit there. If compute dominates, use GPU acceleration. Increasing pending batches cannot compensate for slow storage and raises transient memory use.
```
## Memory Mapping

`gsMap3D` uses memory-mapped files to handle data that exceeds available RAM. The `--memmap-tmp-dir` option allows you to specify a fast SSD for temporary files, significantly improving I/O performance.

### Why Use memmap-tmp-dir?

When your working directory is on a slow filesystem (e.g., network storage, HDD), random access to memory-mapped files can become a bottleneck. By specifying a fast local SSD as the temporary directory, gsMap3D:

1. Copies memory-mapped files to the fast storage
1. Performs all random access operations on the fast storage
1. Syncs results back to the original location when complete

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

config = QuickModeConfig(memmap_tmp_dir="/path/to/fast/ssd/tmp", ...)
```
````

### Recommended Setup

```bash
# Create a dedicated temp directory on your fastest SSD
mkdir -p /mnt/nvme_ssd/gsmap_tmp

# Run gsMap3D with the temp directory
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

| Dataset Size | Calculation        | Approximate Temp Space |
| ------------ | ------------------ | ---------------------- |
| 100K spots   | 100K × 20K × 4 × 2 | ~16 GB                 |
| 1M spots     | 1M × 20K × 4 × 2   | ~160 GB                |
| 10M spots    | 10M × 20K × 4 × 2  | ~1.6 TB                |

```{important}
Ensure your temporary directory has enough free space before running. gsMap3D will fail if the disk runs out of space during computation.
```

## Batch Processing

gsMap3D processes data in batches to balance memory usage and computational efficiency:

| Parameter                    | Description                          | Default |
| ---------------------------- | ------------------------------------ | ------- |
| `rank_batch_size`            | Spots per batch for rank calculation | 500     |
| `mkscore_batch_size`         | Spots per batch for marker score     | 500     |
| `spots_per_chunk_quick_mode` | Spots per chunk for Spatial LDSC     | 50      |

Larger batch sizes can improve throughput but require more memory. Adjust based on your available RAM and GPU memory.

## Performance Recommendations

### For Very Large Datasets (>1M spots)

1. **Use GPU acceleration**: GPU significantly speeds up both latent-to-gene and Spatial LDSC computation. Enabled by default with `--use-gpu`.
1. **Use NVMe SSD**: Fast local storage is critical. Use `--memmap-tmp-dir` to specify a fast SSD for temporary files to avoid reading becoming a bottleneck. If SSD is not available, increase `rank_read_workers` to compensate.
1. **Increase batch sizes**: If memory allows, larger batches improve throughput

### For CPU-Only Systems

1. **Disable GPU**: Use `--no-gpu` to avoid JAX GPU initialization overhead
1. **Increase CPU cores and compute workers**: The Spatial LDSC step scales near-linearly with the number of available CPU cores. Increase `ldsc_compute_workers` to match available cores.
1. **Use NVMe SSD**: Fast storage significantly speeds up latent-to-gene computation. Use `--memmap-tmp-dir` to specify a fast SSD.

### For Cluster Users

When running on HPC clusters, you can parallelize gsMap3D across multiple nodes:

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
    ...,
)
run_quick_mode(config)

# Job 2: Process cells 1000000-2000000
config = QuickModeConfig(
    start_step="spatial_ldsc",
    stop_step="spatial_ldsc",
    cell_indices_range=(1000000, 2000000),
    trait_name="trait1",
    sumstats_file="/path/to/trait1.sumstats.gz",
    ...,
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

workdir = Path("/path/to/workdir")
project = "my_project"
trait = "trait1"

# Find all partial result files
result_dir = workdir / project / "spatial_ldsc"
partial_files = sorted(result_dir.glob(f"{project}_{trait}_cells_*.csv.gz"))

print(f"Found {len(partial_files)} partial result files")

# Concatenate and save
results = pd.concat([pd.read_csv(f) for f in partial_files], ignore_index=True)
results.to_csv(
    result_dir / f"{project}_{trait}.csv.gz", index=False, compression="gzip"
)

print(f"Merged {len(results)} spots to {result_dir / f'{project}_{trait}.csv.gz'}")
```

## Troubleshooting

### GPU Out of Memory (OOM) Errors

The rank and marker-score batch sizes are selected deterministically from the
workload. Worker and queue counts use stable defaults and are never inferred
from host or scheduler resources. If you encounter a GPU out-of-memory error,
try the following solutions in order:

**1. Override the automatic marker-score batch size**:

```bash
gsmap quick-mode --mkscore-batch-size 25 ...
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

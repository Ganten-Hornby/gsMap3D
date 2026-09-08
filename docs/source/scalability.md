# Performance and scalability

Use this guide to choose resources, reduce memory use and reuse completed work.
For the biological workflow and the meaning of GSS, see {doc}`key_concepts`.
For graph, matching or input changes, see {doc}`advanced_usage`.

## Example used in this guide

All examples use **mouse embryo E16.5 E1S1 in Spatial2D mode**. Download its files
using the {doc}`2d_tutorial/mouse_embryo`, then run from `gsmap_tutorials/`.
There is no cross-section matching in this example. Keep the tutorial's graph
and homogeneous-neighbor defaults unchanged when comparing performance.

```text
gsmap_tutorials/
├── resources/
│   ├── 1000g-hc-grch38-v50-eur/
│   │   ├── snp_gene_weights.h5ad
│   │   └── w_ld/
│   └── homologs/mouse_human_homologs.txt
├── example_data/
│   ├── mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad
│   └── example-traits/IQ.sumstats.gz
└── results/mouse-e16-mosta/           # Created by the analysis
```

Complete commands below have **CLI/Python tabs**. Choose one interface.
Short CLI flag excerpts are marked as text: add or replace them in a complete
command. Python tuning snippets modify its `config` **before**
`run_quick_mode(config)`. They are not separate analyses.

## Performance recommendations

- **Start with the defaults.** Measure the whole step, including loading and saving.
- **Use a GPU when helpful.** Try one GPU for the compute-heavy stages; extra
  workers or devices do not guarantee a faster run.
- **Use an SSD if I/O is slow.** It is optional; project storage is supported.
- **Reuse GSS for additional traits.** Do not recalculate embeddings or ranks
  when only the GWAS changes.
- **Leave room for memory and output.** More workers and larger batches can
  increase memory use. Download size is not working disk space.

## Choose CPU or GPU

Embedding uses PyTorch and a visible CUDA GPU when available. Marker scoring
and spatial LDSC use JAX; `--use-gpu` / `--no-gpu` controls these latter stages.

### GPU run

Use the defaults or set GPU use explicitly:

````{tab} CLI
```text
--use-gpu
```
````

````{tab} Python
```python
config.use_gpu = True
```
````

### CPU-only run

Before starting the CLI or Python process, hide CUDA devices:

```bash
export CUDA_VISIBLE_DEVICES=""
```

Then choose CPU execution. Four regression workers below are an example, not
a requirement; use a count suitable for your available cores and memory.

````{tab} CLI
```text
--no-gpu
--ldsc-compute-workers 4
```
````

````{tab} Python
```python
config.use_gpu = False
config.ldsc_compute_workers = 4
```
````

```{note}
`--no-gpu` alone does not disable PyTorch embedding on a visible GPU.
Set the environment before imports for a fully CPU-only run. For a GPU job,
use a fresh shell without the CPU-only override and respect scheduler-assigned devices.
```

## I/O and fast storage

Ranks and marker scores are stored on disk and read as needed. Use
`--memmap-tmp-dir` to place eligible temporary arrays on a fast **SSD or NVMe**
disk. Without it, gsMap uses project storage.

1. Choose an existing, writable directory on the fast disk.
2. Check that it has room for uncompressed arrays and temporary copies.
3. Keep the project on persistent storage and let synchronization finish.

For example, if `/fast-ssd/gsmap` is a directory on your SSD:

````{tab} CLI
```text
--memmap-tmp-dir /fast-ssd/gsmap
```
````

````{tab} Python
```python
from pathlib import Path

config.memmap_tmp_dir = Path("/fast-ssd/gsmap")
```
````

New ranks can be written to scratch first and synchronized back to the project.
If staging cannot fit safely, gsMap falls back to project storage and logs the
decision. Compare runtime including those copies.

```{tip}
This setting selects disk storage, not RAM or GPU memory. A directory named
`scratch` on the same slow filesystem is not faster.
```

A float16 matrix needs `2 × cells × genes` bytes: one million cells × 20,000
genes is about **40 GB per matrix**. Ranks, marker scores and temporary copies
can coexist. This is a disk-size estimate, not a peak-RAM estimate.

## Tune memory and throughput

1. **Identify the limiting step** in its log.
2. **Change one setting** on the same input.
3. **Compare total time and memory**, not just a batch's cells/s.

| Setting | Default | When to adjust |
|---|---|---|
| `--mkscore-batch-size` | `0` (automatic) | Set a smaller explicit batch if marker-score computation runs out of memory. |
| `--spots-per-chunk-quick-mode` | `100` | Reduce for lower per-chunk spatial-LDSC memory use. |
| `--ldsc-read-workers` | `10` | CPU sparse-transform threads, or GPU marker-preparation workers; disk reads remain sequential. |
| `--ldsc-compute-workers` | `10` | Tune CPU regression concurrency within your CPU/memory budget; the effective count may be lower. |

For example, test these lower-memory settings. A marker batch of 50 helps only
if it is smaller than the effective batch reported in your log.

````{tab} CLI
```text
--mkscore-batch-size 50
--spots-per-chunk-quick-mode 25
```
````

````{tab} Python
```python
config.mkscore_batch_size = 50
config.spots_per_chunk_quick_mode = 25
```
````

```{note}
Batching does not remove a cell's selected neighbors. gsMap loads its complete
selected neighborhood before scoring, even when the rank rows are far apart.
These settings change how work is scheduled, not which cells enter the analysis.
```

More workers are not always faster. Keep reader, writer and queue settings at
their defaults unless the logs identify a specific bottleneck. GPU LDSC normally
uses one compute worker; increasing the CPU worker setting does not create more
GPU workers. See {doc}`cli_reference` for all controls.

CPU sparse transforms cooperate on one batch. Regression workers submit batches
to JAX, which also parallelizes internally; both stages share cores and memory
bandwidth. CPU worker caps respect affinity and `SLURM_CPUS_PER_TASK` when present,
but are not a global thread limit for every library.

GPU LDSC keeps the SNP × gene matrix on the device, uploads marker batches and
passes the resulting LD scores directly to JAX. Reported overall throughput
includes final output writing; component throughput measures only that component.

## Cluster users

The simplest way to parallelize multiple traits is **one spatial-LDSC job per
trait**, after preprocessing has completed.

1. **Preprocess once.** Calculate embeddings and GSS.
2. **Allocate resources.** Run on a compute node, not a login node. Set worker
   counts for the CPUs assigned to the job, not the whole node.
3. **Run independent trait jobs.** Keep shared inputs unchanged and write distinct
   trait result files. Use only the GPUs assigned by your scheduler.
4. **Check completion, then report.** Inspect job status and logs before combining
   results. Do not let independent jobs overwrite the same report.

### Example: calculate embeddings and GSS

Skip this example if the E16.5 tutorial has already completed these stages.
It uses `count`, `annotation`, `spatial` and the mouse–human homolog file,
just like the tutorial. It stops before spatial LDSC.

````{tab} CLI
```bash
mkdir -p results
gsmap quick-mode \
    --workdir results \
    --project-name mouse-e16-mosta \
    --dataset-type spatial2D \
    --h5ad-path example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad \
    --homolog-file resources/homologs/mouse_human_homologs.txt \
    --species MOUSE_GENE_SYM \
    --annotation annotation \
    --spatial-key spatial \
    --data-layer count \
    --trait-name IQ \
    --sumstats-file example_data/example-traits/IQ.sumstats.gz \
    --plot-origin lower \
    --stop-step latent2gene
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="results",
    project_name="mouse-e16-mosta",
    dataset_type="spatial2D",
    h5ad_path=["example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad"],
    homolog_file="resources/homologs/mouse_human_homologs.txt",
    species="MOUSE_GENE_SYM",
    annotation="annotation",
    spatial_key="spatial",
    data_layer="count",
    trait_name="IQ",
    sumstats_file="example_data/example-traits/IQ.sumstats.gz",
    plot_origin="lower",
    stop_step="latent2gene",
)
run_quick_mode(config)
```
````

### Example: one trait per job

After preprocessing finishes, submit the following command in a compute job.
`--start-step spatial_ldsc --stop-step spatial_ldsc` runs only association
testing; it reuses the saved ranks and GSS.

````{tab} CLI
```bash
gsmap quick-mode \
    --workdir results \
    --project-name mouse-e16-mosta \
    --dataset-type spatial2D \
    --annotation annotation \
    --start-step spatial_ldsc \
    --stop-step spatial_ldsc \
    --w-ld-dir resources/1000g-hc-grch38-v50-eur/w_ld \
    --snp-gene-weight-adata-path resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad \
    --trait-name IQ \
    --sumstats-file example_data/example-traits/IQ.sumstats.gz
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="results",
    project_name="mouse-e16-mosta",
    dataset_type="spatial2D",
    annotation="annotation",
    start_step="spatial_ldsc",
    stop_step="spatial_ldsc",
    w_ld_dir="resources/1000g-hc-grch38-v50-eur/w_ld",
    snp_gene_weight_adata_path="resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad",
    trait_name="IQ",
    sumstats_file="example_data/example-traits/IQ.sumstats.gz",
)
run_quick_mode(config)
```
````

To analyze another downloaded trait, change **both** `IQ` and its GWAS path,
for example to `Height` or `MCHC`. The original E16.5 example uses IQ only.
For a CPU job, apply the CPU settings above. Limit concurrency if many jobs
saturate the shared filesystem.

```{important}
Keep the complete `latent_to_gene/` directory, including metadata, the
observation-index mapping and `concatenated_latent_adata.h5ad`. Use persistent
project storage; node-local SSD may disappear after a job ends.
See {doc}`output_files` for the files needed to reuse scores.
```

For an exceptionally large single trait, `--cell-indices-range` can split LDSC
into nonoverlapping ranges. This is an advanced option: merge partial results
and verify complete coverage with no duplicate cells before enrichment/reporting.
Prefer trait-level parallelism when possible.

## Keep large reports responsive

Report limits control browser views and diagnostic plots, **not which cells**
enter spatial LDSC.

| CLI flag | Default | Scope |
|---|---|---|
| `--downsampling-n-spots-2d` | `100000` | Displayed spots per section. |
| `--downsampling-n-spots-3d` | `100000` | Displayed spots in the combined 3D view. |
| `--downsampling-n-spots-umap` | `50000` | Spots used for each report UMAP. |
| `--downsampling-n-spots-pcc` | `20000` | Sampled spots for GSS correlations. |

To display all analyzed E16.5 E1S1 spots, use a 2D cap above its 121,767 input
spots. Leave UMAP and correlation limits unchanged unless you need more detail.

````{tab} CLI
```text
--downsampling-n-spots-2d 150000
```
````

````{tab} Python
```python
config.downsampling_n_spots_2d = 150000
```
````

### Example: finish enrichment and the report

After the **IQ** LDSC result is complete, this runs Cauchy enrichment and builds
the IQ report. It also applies the larger 2D display limit without recalculating
embeddings, GSS or spatial LDSC.

````{tab} CLI
```bash
gsmap quick-mode \
    --workdir results \
    --project-name mouse-e16-mosta \
    --dataset-type spatial2D \
    --annotation annotation \
    --spatial-key spatial \
    --plot-origin lower \
    --start-step cauchy \
    --trait-name IQ \
    --sumstats-file example_data/example-traits/IQ.sumstats.gz \
    --downsampling-n-spots-2d 150000
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="results",
    project_name="mouse-e16-mosta",
    dataset_type="spatial2D",
    annotation="annotation",
    spatial_key="spatial",
    plot_origin="lower",
    start_step="cauchy",
    trait_name="IQ",
    sumstats_file="example_data/example-traits/IQ.sumstats.gz",
    downsampling_n_spots_2d=150000,
)
run_quick_mode(config)
```
````

If you ran additional trait jobs, use a multi-trait GWAS configuration containing
all completed traits instead of the single IQ options. The
{doc}`advanced_usage` shows how to specify that configuration. Generate the
combined report once, after every requested trait is complete.

## Where the work happens

The biological workflow diagram is in
{ref}`Key Concepts <pipeline-overview>`. For performance diagnosis,
start with the log for the slow step:

| Step | What to check | Log under `logs/latest/` |
|---|---|---|
| Embeddings | Expression preparation, graph size, training and device memory | `01_find_latent.log` |
| Ranks and marker scores | Expression/rank reading, score computation and output I/O | `02_latent_to_gene.log` |
| Spatial LDSC | GSS-to-LD transformation and regression time | `03_spatial_ldsc.log` and per-trait logs |
| Report | Plots, UMAP, sampled GSS reads and browser payload size | `05_report.log` |

For example, inspect the end of the E16.5 marker-score log:

```bash
tail -n 40 results/mouse-e16-mosta/logs/latest/02_latent_to_gene.log
```

Look at sustained timings across the step. Brief pauses or low GPU utilization
do not, by themselves, identify the bottleneck. See {doc}`output_files` for
the full log layout and {doc}`advanced_usage` for safe restart settings.

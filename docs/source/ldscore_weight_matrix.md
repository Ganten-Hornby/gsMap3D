# Computing Custom LD Score Weight Matrix

This guide explains how to compute a custom SNP-to-gene LD score weight matrix using the `gsmap ldscore-weight-matrix` command. This is useful when you want to use your own SNP-to-gene mapping (e.g., based on eQTL data, enhancer-gene links, or custom genomic windows) instead of the default pre-calculated weights.

## Overview

The `ldscore-weight-matrix` command computes LD-based weights between SNPs and genomic features (e.g., genes). It:

1. Loads genotype data from PLINK binary files (reference panel)
1. Loads SNP-to-feature mappings from a BED file or dictionary
1. Computes LD scores between HapMap3 (HM3) SNPs and reference SNPs
1. Generates a weight matrix that accounts for LD structure
1. Outputs an AnnData (.h5ad) file for use in `gsmap quick-mode`

## Prerequisites

Before running this command, you need:

1. **PLINK Reference Panel**: Genotype data in PLINK binary format (.bed/.bim/.fam) for each chromosome. We recommend using the 1000 Genomes Project European samples.

1. **HapMap3 SNP Directory**: A directory containing per-chromosome HapMap3 SNP lists (e.g., `hm.{chr}.snp`, one SNP ID per line). These are the SNPs used in LD score regression.

1. **SNP-to-Gene Mapping File**: A BED file defining which SNPs map to which genes/features.

## Input File Formats

### SNP-to-Gene BED File

The BED file should have the following columns (tab-separated):

| Column | Name       | Description                                                                                                                                |
| ------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 1      | Chromosome | Chromosome (e.g., `chr1` or `1`)                                                                                                           |
| 2      | Start      | Start position (0-based)                                                                                                                   |
| 3      | End        | End position                                                                                                                               |
| 4      | Name       | Feature/gene name                                                                                                                          |
| 5      | Score      | SNP-to-gene Mapping score (optional, used when `--strategy score`). When provided, the SNP is assigned to the gene with the highest score. |
| 6      | Strand     | Strand (`+` or `-`, optional)                                                                                                              |

**Gencode BED example** (gene body regions):

```text
chr1    65418    71585    OR4F5    1    +
chr1    367639   368634   OR4F29   1    +
chr1    859302   879955   SAMD11   1    +
```

**Enhancer BED example** (enhancer-gene links):

```text
chr1    33454352    33455149    A3GALT2
chr1    33564939    33565256    A3GALT2
chr1    33592784    33593190    A3GALT2
```

### HM3 SNP Directory

A directory containing per-chromosome HM3 SNP files. Each file is a plain text file with one SNP ID per line. The following naming conventions are supported (checked in order):

- `hm.{chr}.snp`
- `hm3_snps.chr{chr}.txt`
- `hapmap3_snps.chr{chr}.txt`
- `chr{chr}.snplist`
- `w_hm3.snplist.chr{chr}`

Example file content:

```text
rs12345
rs67890
rs11111
...
```

### PLINK Reference Panel

Standard PLINK binary files with chromosome-specific naming:

- `1000G.EUR.QC.{chr}.bed`
- `1000G.EUR.QC.{chr}.bim`
- `1000G.EUR.QC.{chr}.fam`

Where `{chr}` is replaced by chromosome number (1-22).

## Parameters

### Required Parameters

| Parameter       | Description                                                                                               |
| --------------- | --------------------------------------------------------------------------------------------------------- |
| `--bfile-root`  | Path template for PLINK binary files. Must contain `{chr}` placeholder (e.g., `data/1000G.EUR.QC.{chr}`). |
| `--hm3-snp-dir` | Directory containing per-chromosome HapMap3 SNP lists (e.g., `hapmap3_snps/`).                            |
| `--output-dir`  | Directory where output files will be saved.                                                               |

### Snp-to-Gene Mapping Parameters

| Parameter               | Default   | Description                                                                                                                                                                                                 |
| ----------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--mapping-file`        | None      | Path to the SNP-to-gene mapping file (BED or dictionary format).                                                                                                                                            |
| `--mapping-type`        | `"bed"`   | Type of mapping file: `"bed"` for BED format or `"dict"` for dictionary format.                                                                                                                             |
| `--feature-window-size` | `0`       | Window size (in bp) to extend around features. For example, `50000` extends 50kb around each feature.                                                                                                       |
| `--strategy`            | `"score"` | Strategy for resolving SNPs mapping to multiple features. Options: `"score"` (highest score), `"tss"` (closest to TSS), `"center"` (closest to feature center), `"allow_repeat"` (allow multiple mappings). |

### LD Calculation Parameters

| Parameter   | Default | Description                                                                                 |
| ----------- | ------- | ------------------------------------------------------------------------------------------- |
| `--ld-wind` | `1.0`   | LD window size for computing LD scores.                                                     |
| `--ld-unit` | `"CM"`  | Unit for LD window: `"CM"` (centiMorgans), `"KB"` (kilobases), or `"SNP"` (number of SNPs). |
| `--maf-min` | `0.01`  | Minimum minor allele frequency filter. SNPs with MAF below this threshold are excluded.     |

### Computation Parameters

| Parameter           | Default              | Description                                                                                                  |
| ------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--chromosomes`     | `"all"`              | Chromosomes to process. Use `"all"` for autosomes 1-22, or specify a comma-separated list (e.g., `"1,2,3"`). |
| `--batch-size-hm3`  | `50`                 | Number of HM3 SNPs to process per batch. Larger values use more memory but may be faster.                    |
| `--output-filename` | `"ld_score_weights"` | Prefix for output files.                                                                                     |

### Optional w_ld Calculation

| Parameter          | Default | Description                                                                                                 |
| ------------------ | ------- | ----------------------------------------------------------------------------------------------------------- |
| `--calculate-w-ld` | `False` | Whether to also calculate LD score weights (w_ld), which is used to account for heteroscedasticity in LDSC. |
| `--w-ld-dir`       | None    | Directory to save w_ld output files. Required if `--calculate-w-ld` is enabled.                             |

## Mapping Strategies

When a SNP falls within multiple features, the `--strategy` parameter determines how to resolve the conflict:

| Strategy       | Description                                                          | Use Case                                                   |
| -------------- | -------------------------------------------------------------------- | ---------------------------------------------------------- |
| `score`        | Keep the mapping with the highest score (from column 5 of BED file). | When you have confidence scores (e.g., eQTL effect sizes). |
| `tss`          | Keep the mapping closest to the transcription start site.            | Gene-based analysis using Gencode/RefSeq annotations.      |
| `center`       | Keep the mapping closest to the feature center.                      | Enhancer-based analysis where the center is most relevant. |
| `allow_repeat` | Allow the SNP to map to multiple features (no filtering).            | When you want to preserve all possible mappings.           |

## Output

The command produces an AnnData file (`{output_filename}.h5ad`) with the following structure:

- **X**: Sparse weight matrix (rows = HM3 SNPs, columns = features/genes)
- **obs**: SNP metadata
    - Index: SNP names
    - `CHR`: Chromosome number
    - `BP`: Base pair position
- **var**: Feature metadata
    - Index: Feature/gene names

## Examples

### Download Example Data

To follow the examples below, first download the example data:

```bash
wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_ldscore_weight_matrix_example_data.tar.gz
tar -xvzf gsMap_ldscore_weight_matrix_example_data.tar.gz
```

Directory structure:

```bash
tree -L 2 gsMap_ldscore_weight_matrix_example_data/

gsMap_ldscore_weight_matrix_example_data/
├── genome_annotation
│   ├── enhancer
│   └── gencode
├── hapmap3_snps
│   ├── hm.1.snp
│   ├── hm.2.snp
│   ├── ...
│   └── hm.22.snp
└── LD_Reference_Panel
    └── 1000G_EUR_Phase3_plink
```

### Example 1: TSS-based Mapping with Gencode Annotation

This example creates weights based on proximity to transcription start sites (TSS) using Gencode protein-coding genes with a 50kb window.

````{tab} CLI

```bash
# Set up environment variables
LDSCORE_DATA_DIR="./gsMap_ldscore_weight_matrix_example_data"
PLINK_REF="${LDSCORE_DATA_DIR}/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC"
HM3_DIR="${LDSCORE_DATA_DIR}/hapmap3_snps"
GENCODE_BED="${LDSCORE_DATA_DIR}/genome_annotation/gencode/gencode_v46lift37_protein_coding.bed"

# Create output directory
mkdir -p ./output/gencode_tss

# Run ldscore-weight-matrix with TSS strategy
gsmap ldscore-weight-matrix \
    --bfile-root "${PLINK_REF}.{chr}" \
    --hm3-snp-dir "${HM3_DIR}" \
    --mapping-file "${GENCODE_BED}" \
    --mapping-type "bed" \
    --output-dir "./output/gencode_tss" \
    --output-filename "gencode_v46_tss_50kb_weights" \
    --feature-window-size 50000 \
    --strategy "tss" \
    --ld-wind 1.0 \
    --ld-unit "CM" \
    --chromosomes "all"
```

````

````{tab} Python

```python
from gsMap.config import LDScoreConfig
from gsMap.ldscore.pipeline import LDScorePipeline
from pathlib import Path

# Set up paths
ldscore_data_dir = Path("./gsMap_ldscore_weight_matrix_example_data")
plink_ref = ldscore_data_dir / "LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC"
hm3_dir = ldscore_data_dir / "hapmap3_snps"
gencode_bed = (
    ldscore_data_dir / "genome_annotation/gencode/gencode_v46lift37_protein_coding.bed"
)

# Create output directory
output_dir = Path("./output/gencode_tss")
output_dir.mkdir(parents=True, exist_ok=True)

# Configure and run pipeline
config = LDScoreConfig(
    bfile_root=f"{plink_ref}.{{chr}}",
    hm3_snp_dir=hm3_dir,
    mapping_file=gencode_bed,
    mapping_type="bed",
    output_dir=output_dir,
    output_filename="gencode_v46_tss_50kb_weights",
    feature_window_size=50000,
    strategy="tss",
    ld_wind=1.0,
    ld_unit="CM",
    chromosomes="all",
)

pipeline = LDScorePipeline(config)
pipeline.run()
```

````

**Use in gsMap3D:**

```bash
gsmap quick-mode \
    --snp-gene-weight-adata-path "./output/gencode_tss/gencode_v46_tss_50kb_weights.h5ad" \
    --w-ld-dir "/path/to/w_ld" \
    ...
```

### Example 2: Enhancer-based Mapping with Center Strategy

This example creates weights using brain-specific enhancer-gene links from the ABC model, mapping SNPs to the closest enhancer center.

````{tab} CLI

```bash
# Set up environment variables
LDSCORE_DATA_DIR="./gsMap_ldscore_weight_matrix_example_data"
PLINK_REF="${LDSCORE_DATA_DIR}/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC"
HM3_DIR="${LDSCORE_DATA_DIR}/hapmap3_snps"
ENHANCER_BED="${LDSCORE_DATA_DIR}/genome_annotation/enhancer/by_tissue/BRN/ABC_roadmap_merged.bed"

# Create output directory
mkdir -p ./output/enhancer_brain

# Run ldscore-weight-matrix with center strategy
gsmap ldscore-weight-matrix \
    --bfile-root "${PLINK_REF}.{chr}" \
    --hm3-snp-dir "${HM3_DIR}" \
    --mapping-file "${ENHANCER_BED}" \
    --mapping-type "bed" \
    --output-dir "./output/enhancer_brain" \
    --output-filename "brain_enhancer_abc_weights" \
    --feature-window-size 0 \
    --strategy "center" \
    --ld-wind 1.0 \
    --ld-unit "CM" \
    --chromosomes "all"
```

````

````{tab} Python

```python
from gsMap.config import LDScoreConfig
from gsMap.ldscore.pipeline import LDScorePipeline
from pathlib import Path

# Set up paths
ldscore_data_dir = Path("./gsMap_ldscore_weight_matrix_example_data")
plink_ref = ldscore_data_dir / "LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC"
hm3_dir = ldscore_data_dir / "hapmap3_snps"
enhancer_bed = (
    ldscore_data_dir / "genome_annotation/enhancer/by_tissue/BRN/ABC_roadmap_merged.bed"
)

# Create output directory
output_dir = Path("./output/enhancer_brain")
output_dir.mkdir(parents=True, exist_ok=True)

# Configure and run pipeline
config = LDScoreConfig(
    bfile_root=f"{plink_ref}.{{chr}}",
    hm3_snp_dir=hm3_dir,
    mapping_file=enhancer_bed,
    mapping_type="bed",
    output_dir=output_dir,
    output_filename="brain_enhancer_abc_weights",
    feature_window_size=0,
    strategy="center",
    ld_wind=1.0,
    ld_unit="CM",
    chromosomes="all",
)

pipeline = LDScorePipeline(config)
pipeline.run()
```

````

**Use in gsMap3D:**

```bash
gsmap quick-mode \
    --snp-gene-weight-adata-path "./output/enhancer_brain/brain_enhancer_abc_weights.h5ad" \
    --w-ld-dir "/path/to/w_ld" \
    ...
```

## Available Enhancer Annotations

The example data includes enhancer-gene links for multiple tissues:

| Tissue Code | Description          |
| ----------- | -------------------- |
| `ALL`       | All tissues combined |
| `BLD`       | Blood                |
| `BRN`       | Brain                |
| `FAT`       | Adipose tissue       |
| `GI`        | Gastrointestinal     |
| `HRT`       | Heart                |
| `KID`       | Kidney               |
| `LIV`       | Liver                |
| `LNG`       | Lung                 |
| `PANC`      | Pancreas             |
| `SKIN`      | Skin                 |

Each tissue folder contains:

- `ABC.bed`: Activity-by-Contact model predictions
- `roadmap.bed`: Roadmap Epigenomics enhancers
- `ABC_roadmap_merged.bed`: Combined ABC and Roadmap annotations

## Troubleshooting

### Missing PLINK files

If you see an error about missing PLINK files, ensure:

1. The `--bfile-root` path template is correct
1. The `{chr}` placeholder is included in the path
1. All chromosome files exist for the specified chromosomes

The `--chromosomes` parameter controls which chromosomes to process:

- `--chromosomes "all"` (default): Processes autosomes 1-22
- `--chromosomes "1,2,3"`: Processes only the specified chromosomes (comma-separated list)

### Memory errors

If you encounter out-of-memory errors:

1. Reduce `--batch-size-hm3` (e.g., from 50 to 25)
1. Use a machine with more RAM

- **Memory**: Memory usage scales linearly with the number of SNPs in the reference panel and the number of features. For example, using 1000 Genomes Phase 3 as the reference panel with Gencode v46 protein-coding genes (~20,000 genes) costs approximately 25 GB memory.

### No SNPs mapped

If the output has very few or no SNPs:

1. Check that chromosome naming is consistent (e.g., `chr1` vs `1`)
1. Verify the BED file coordinates are correct (0-based start)
1. Increase `--feature-window-size` if using TSS-based mapping

### BED file has header

If your BED file has a header line (e.g., `Chromosome Start End symbol`), the pipeline will automatically skip it. Ensure the header doesn't match the expected data format.

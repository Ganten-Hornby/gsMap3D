# Downloads

[Start a tutorial](#start-a-tutorial) · [Browse all resources](#browse-all-resources)

Download a dataset, reuse the shared reference, and explore its report preview.

## Start a tutorial

After {doc}`install`, download the shared resources once. Every tutorial reuses
them; only the spatial dataset changes.

```text
gsmap_tutorials/
├── resources/
│   ├── 1000g-hc-grch38-v50-eur/
│   │   ├── snp_gene_weights.h5ad
│   │   └── w_ld/
│   └── homologs/
└── example_data/
    ├── example-traits/            # shared GWAS
    └── <dataset>/                # download from your chosen tutorial
```

Create a workspace, or enter the one you already use:

```bash
mkdir -p gsmap_tutorials
cd gsmap_tutorials
```

### 1. Download shared resources

```{tip}
**Skip a download if its extracted folder already exists.** These commands use
the EUR reference required by the tutorials; other ancestries are in the catalog.
```

**LD reference** — the SNP × gene matrix and matching regression weights:

```bash
mkdir -p resources
wget -O resources/1000g-hc-grch38-v50-eur.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-eur/r1/bundle.tar.gz && \
    tar -xzf resources/1000g-hc-grch38-v50-eur.tar.gz -C resources
```

**Homolog tables** — needed for mouse/macaque data; skip for human data:

```bash
mkdir -p resources
wget -O resources/homologs.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/homologs/homologs/r1/bundle.tar.gz && \
    tar -xzf resources/homologs.tar.gz -C resources
```

**Example GWAS** — IQ, AF, LDL, Height, MCHC and serum creatinine:

```bash
mkdir -p example_data
wget -O example_data/example-traits.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/gwas/example-traits/r1/bundle.tar.gz && \
    tar -xzf example_data/example-traits.tar.gz -C example_data
```

### 2. Choose a tutorial

Each tutorial downloads only its own dataset, then shows how to run gsMap and
open the report. Continue using this workspace.

| Dataset | Tutorial |
|---|---|
| Mouse brain and kidney, Visium | {doc}`2D Visium <2d_tutorial/10x>` |
| Mouse E16.5 embryo, MOSTA | {doc}`2D embryo <2d_tutorial/mouse_embryo>` |
| Human CS8 embryo, 62 sections | {doc}`3D gastrulation <3d_tutorial/human_gastrulation>` |
| Mouse E11.5 embryo, 84 sections | {doc}`E11.5 tutorial <3d_tutorial/mouse_e11>` |

<details>
<summary>Verify your download</summary>

Every archive has a **SHA-256** link in the catalog. Print your archive's checksum
and compare the 64-character value with that link:

```bash
sha256sum resources/1000g-hc-grch38-v50-eur.tar.gz
```

Local archive filenames may differ; the checksum value must match.
After extraction, check the files inside the bundle:

```bash
(cd resources/1000g-hc-grch38-v50-eur && sha256sum --check SHA256SUMS)
```

All entries should report `OK`. Use the same check from any extracted bundle
folder. Do not use files that fail verification.

</details>

Using your own data or storing resources elsewhere? See {doc}`advanced_usage`
for paths and CLI flags.

## Browse all resources

Click a dataset name to open its tutorial. Publication links open the original
study or provider page; report previews open the interactive gsMap results.

### 2D example datasets

| Dataset | Spots / cells | Publication / source | Report preview | Download |
|---|---:|---|---|---|
| [Mouse E16.5 embryo (MOSTA)](2d_tutorial/mouse_embryo.md)<br>Single E1S1 section with anatomical annotations. | 121,767 | [Chen et al. · 2022 · Cell](https://doi.org/10.1016/j.cell.2022.04.003) | [View report](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/mouse-e16-mosta/r1/index.html) | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/2d/mouse-e16-mosta/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/2d/mouse-e16-mosta/r1/bundle.tar.gz.sha256)<br>1.3 GiB |
| [Mouse brain and kidney (Visium)](2d_tutorial/10x.md)<br>Adult brain (coronal and sagittal) and kidney; three Visium sections. | 7,628<br>Coronal: 2,902<br>Sagittal: 3,289<br>Kidney: 1,437 | [10x Genomics · 2020 · Coronal](https://www.10xgenomics.com/datasets/adult-mouse-brain-section-1-coronal-stains-dapi-anti-neu-n-1-standard-1-1-0)<br>[10x Genomics · 2020 · Sagittal](https://www.10xgenomics.com/datasets/mouse-brain-serial-section-2-sagittal-posterior-1-standard-1-1-0)<br>[10x Genomics · 2020 · Kidney](https://www.10xgenomics.com/datasets/mouse-kidney-section-coronal-1-standard-1-1-0) | [Coronal](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/visium-coronal/r1/index.html)<br>[Sagittal](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/visium-sagittal/r1/index.html)<br>[Kidney](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/visium-kidney/r1/index.html) | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/2d/mouse-visium/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/2d/mouse-visium/r1/bundle.tar.gz.sha256)<br>197.6 MiB |

Counts are the spots or cells in the downloadable inputs, before gsMap QC.

### 3D example datasets

| Dataset | Spots / cells | Publication / source | Report preview | Download |
|---|---:|---|---|---|
| [Human CS8 embryo](3d_tutorial/human_gastrulation.md)<br>62 registered sections with spatial annotations. | 38,562 | [Xiao et al. · 2024 · Cell](https://doi.org/10.1016/j.cell.2024.03.041) | [View report](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/human-gastrulation/r1/index.html) | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/3d/human-gastrulation/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/3d/human-gastrulation/r1/bundle.tar.gz.sha256)<br>154.9 MiB |
| [Mouse E11.5 embryo](3d_tutorial/mouse_e11.md)<br>84 serial sections; raw counts in X, spatial domains and manual cell-type annotations. | 6,993,667 | [Qiu et al. · 2024 · Cell](https://doi.org/10.1016/j.cell.2024.10.011) | [View report](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/e11-ordinal/r1/index.html) | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/3d/mouse-e11/r2/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/3d/mouse-e11/r2/bundle.tar.gz.sha256)<br>5.3 GiB |

Counts are the spots or cells in the downloadable inputs, before gsMap QC.

### Example GWAS

| Resource | Description | Size | Download |
|---|---|---:|---|
| Six example GWAS | IQ (2018), Kazuo AF, LDL (2021), Height (2022), MCHC and serum creatinine. Formatted SNP/A1/A2/Z/N; ancestry selection must follow each source study. | 232.8 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/gwas/example-traits/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/gwas/example-traits/r1/bundle.tar.gz.sha256) |

### SNP × gene LD-weight matrices

```{important}
Choose ancestry from the **GWAS population**, not the tissue species.
Each archive contains the SNP × gene matrix and its matching `w_ld/` files; use them together.
```

#### GENCODE v50 · GRCh38

| Ancestry | Reference N | Size | Download |
|---|---:|---:|---|
| AFR | 661 | 111.5 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-afr/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-afr/r1/bundle.tar.gz.sha256) |
| AMR | 347 | 102.0 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-amr/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-amr/r1/bundle.tar.gz.sha256) |
| EAS | 504 | 84.0 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-eas/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-eas/r1/bundle.tar.gz.sha256) |
| EUR | 503 | 92.1 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-eur/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-eur/r1/bundle.tar.gz.sha256) |
| SAS | 489 | 92.9 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-sas/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-sas/r1/bundle.tar.gz.sha256) |

#### GENCODE v46 · GRCh37

| Ancestry | Reference N | Size | Download |
|---|---:|---:|---|
| EUR | 489 | 91.2 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-grch37-v46-eur/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-grch37-v46-eur/r1/bundle.tar.gz.sha256) |

V50 uses unrelated 1000G high-coverage panels, exact ±1 cM windows and MAC > 5.
Broad-reference SNPs retain MHC; regression SNPs and regression weights exclude it.
V46 uses the GRCh37 1000 Genomes Phase 3 low-coverage EUR reference panel, paired with its original `weights_hm3_no_hla`. Do not mix the pairs.
[1000 Genomes reference releases](https://www.internationalgenome.org/announcements/3202-samples-at-high-coverage-from-NYGC/).

### Homologs and additional annotations

| Resource | Description | Size | Download |
|---|---|---:|---|
| Mouse/macaque–human homologs | Mouse/macaque to human gene-symbol mappings. | 0.2 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/homologs/homologs/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/homologs/homologs/r1/bundle.tar.gz.sha256) |
| Additional annotations | Optional additional annotation resources; inspect build and format before use. | 143.2 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/annotations/additional-annotations/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/annotations/additional-annotations/r1/bundle.tar.gz.sha256) |

### Other resources

| Resource | Description | Size | Download |
|---|---|---:|---|
| LD-matrix tutorial inputs | GRCh37 1000 Genomes Phase 3 low-coverage EUR PLINK panel (489 individuals; chromosomes 1–22), HM3 SNP lists and GENCODE v46 gene BED; tissue-specific ABC/Roadmap enhancer–gene BED files. | 551.4 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_panels/ldscore-tutorial/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_panels/ldscore-tutorial/r1/bundle.tar.gz.sha256) |
| Integration-test data | Integration-test fixtures; not a biological production dataset. | 559.6 MiB | [Archive](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/tutorials/ci-test-data/r1/bundle.tar.gz) · [SHA-256](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/tutorials/ci-test-data/r1/bundle.tar.gz.sha256) |

Each bundle includes a README, metadata and `SHA256SUMS`. Tutorials include links to their report previews.
See [Customization](advanced_usage.md) for using other paths or reference pairs.

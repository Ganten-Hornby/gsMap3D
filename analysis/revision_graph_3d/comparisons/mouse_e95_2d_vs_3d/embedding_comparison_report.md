# Mouse E9.5 section-wise 2D versus section-aware 3D embeddings

## Scope and comparability

All **646,893 spots** were matched by globally unique observation ID. Section labels, `heart_region`, `transf_anno_E95`, and 3D coordinates agree for 100.0%, 100.0%, 100.0%, and 100.0% of matched spots, respectively.

This is a descriptive comparison, **not an isolated graph-only ablation**. The uploaded baseline and section-aware run differ in additional settings:

| Setting | Uploaded section-wise 2D | Section-aware 3D |
|---|---:|---:|
| Graph mode | per-section 2D kNN | section_aware_3d |
| Training annotation | mapped_celltype | transf_anno_E95 |
| Model class size | 54 | 32 |
| Genes used/common | 13581 | 17634 |
| Two-stage training | True | False |

## Heart-region annotation scope

`heart_region` is defined for **44,644 / 646,893 spots (6.90%)**. The two defined regions contain 28,931 LV/atrial observations and 15,713 right/outflow-tract observations. All heart-region purity and UMAP results below exclude `NA`.

All 44,644 observations with a defined `heart_region` are used for both the purity calculation and UMAP; there is no subsampling. This preserves the observed region counts of 28,931 LV/atrial and 15,713 right/outflow-tract observations.

The cell-type UMAP likewise uses all 646,893 matched observations colored by `transf_anno_E95`, with no subsampling.

This annotation is not restricted to cardiomyocytes: only 16,082 / 44,644 defined observations (36.0%) have a `mapped_celltype` containing `cardiomyocytes`. The remainder includes cardiopharyngeal mesoderm, primitive erythroid cells, endocardial cells, proepicardium, and arterial endothelial cells.

| heart_region | mapped_celltype | n | Fraction within region |
|---|---|---:|---:|
| LV/atrial region | LV/atrioventricular canal/common atrium cardiomyocytes | 11,491 | 39.7% |
| LV/atrial region | Cardiopharyngeal mesoderm | 4,372 | 15.1% |
| LV/atrial region | Primitive erythroid cells | 4,167 | 14.4% |
| LV/atrial region | Endocardial cells | 3,393 | 11.7% |
| LV/atrial region | Proepicardium | 3,025 | 10.5% |
| LV/atrial region | Arterial endothelial cells | 2,466 | 8.5% |
| LV/atrial region | Second heart field-derived cardiomyocytes | 17 | 0.1% |
| Right/outflow tract region | Second heart field-derived cardiomyocytes | 4,560 | 29.0% |
| Right/outflow tract region | Primitive erythroid cells | 3,897 | 24.8% |
| Right/outflow tract region | Cardiopharyngeal mesoderm | 3,296 | 21.0% |
| Right/outflow tract region | Arterial endothelial cells | 1,450 | 9.2% |
| Right/outflow tract region | Endocardial cells | 1,429 | 9.1% |
| Right/outflow tract region | Proepicardium | 1,067 | 6.8% |
| Right/outflow tract region | LV/atrioventricular canal/common atrium cardiomyocytes | 14 | 0.1% |

## Metrics

| Representation | Linear CKA | Procrustes similarity | kNN overlap@15 | 2D heart-region purity | 3D heart-region purity | 2D different-section kNN | 3D different-section kNN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `emb_cell` | 0.1077 | 0.2957 | 0.0024 | 0.5818 | 0.5841 | 0.9405 | 0.9019 |
| `emb_niche` | 0.1298 | 0.3160 | 0.0050 | 0.6084 | 0.7762 | 0.9375 | 0.6579 |

`Heart-region kNN purity@15` is the mean, across focal spots, of the fraction of the 15 nearest latent neighbors carrying the same non-`NA` `heart_region` label. Both focal spots and the latent-neighbor candidate pool are restricted to observations with a defined heart region. It is a neighborhood-structure statistic, not prediction accuracy.

For `emb_niche`, heart-region purity changed from 60.84% to 77.62%: an absolute increase of 16.77 percentage points and a relative increase of 27.6%. This does not mean that purity is 27.6%, and the descriptive comparison does not isolate a causal effect of the 3D graph because other training settings also differ.

`kNN overlap@15` is the mean shared-neighbor fraction between the two fixed-size neighbor sets on the same stratified sample. It is symmetric and does not define either run as a positive/reference class. Linear CKA and Procrustes similarity are invariant to arbitrary orthogonal rotations of the latent axes.

## Physical-neighborhood continuity

The evaluation uses exactly 300,000 directed within-section edges and 304,784 directed cross-section edges. Cross-section candidates are accepted only within the radius inferred from each focal spot's 10th external within-section neighbor, then capped at the 10 nearest qualifying spots per adjacent section. The radius matches the cross-section sampling scale to the within-section neighborhood; the cap only removes excess qualifying spots and never creates an edge. No edge is forced across a gap. The zero-cross-section-neighbor rate is 3.62%.

| Representation | Edge type | 2D mean cosine | 3D mean cosine |
|---|---|---:|---:|
| `emb_cell` | Within-section 10-NN | 0.4846 | 0.5665 |
| `emb_cell` | Adjacent-section, within radius and per-section cap | 0.4105 | 0.4921 |
| `emb_niche` | Within-section 10-NN | 0.6278 | 0.9440 |
| `emb_niche` | Adjacent-section, within radius and per-section cap | 0.4947 | 0.8491 |

Higher spatial-edge cosine means a smoother latent representation along those physical edges. It is not, by itself, proof of better biological accuracy because excessive smoothing can also raise this score.

## Outputs

- `embedding_comparison_shared_umap.png` / `.pdf`: all matched observations in a shared UMAP colored by `transf_anno_E95`, with no subsampling.
- `embedding_comparison_heart_region_umap.png` / `.pdf`: all non-`NA` heart-region spots in a shared UMAP for each representation after orthogonal Procrustes alignment, colored by `heart_region` and plotted with `scanpy.pl.umap`.
- `embedding_comparison_metrics.png` / `.pdf`: compact metric summary.
- `embedding_comparison_metrics.tsv`: tidy metric values.
- `heart_region_celltype_composition.tsv`: mapped cell-type composition within each defined heart region.
- `embedding_comparison_summary.json`: matching, sampling, and spatial-evaluation audit data.

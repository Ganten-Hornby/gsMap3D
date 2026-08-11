# Mouse E9.5 section-wise 2D versus section-aware 3D embeddings

## Scope and comparability

All **646,893 spots** were matched by globally unique observation ID. Section labels, `transf_anno_E95`, and 3D coordinates agree for 100.0%, 100.0%, and 100.0% of matched spots, respectively.

This is a descriptive comparison, **not an isolated graph-only ablation**. The uploaded baseline and section-aware run differ in additional settings:

| Setting | Uploaded section-wise 2D | Section-aware 3D |
|---|---:|---:|
| Graph mode | per-section 2D kNN | section_aware_3d |
| Training annotation | mapped_celltype | transf_anno_E95 |
| Model class size | 54 | 32 |
| Genes used/common | 13581 | 17634 |
| Two-stage training | True | False |

## Metrics

| Representation | Linear CKA | Procrustes similarity | kNN recall@15 | 2D label purity | 3D label purity | 2D different-section kNN | 3D different-section kNN |
|---|---:|---:|---:|---:|---:|---:|---:|
| `emb_cell` | 0.1077 | 0.2957 | 0.0024 | 0.1026 | 0.1108 | 0.9405 | 0.9019 |
| `emb_niche` | 0.1298 | 0.3160 | 0.0050 | 0.1018 | 0.2214 | 0.9375 | 0.6579 |

`kNN recall@15` is the mean fraction of latent neighbors shared by the two runs on the same stratified sample. Linear CKA and Procrustes similarity are invariant to arbitrary orthogonal rotations of the latent axes.

## Physical-neighborhood continuity

The evaluation uses exactly 300,000 directed within-section edges and 304,784 directed cross-section edges. Cross-section candidates are accepted only within each focal spot's 10th-within-section-neighbor radius; no edge is forced across a gap. The zero-cross-section-neighbor rate is 3.62%.

| Representation | Edge type | 2D mean cosine | 3D mean cosine |
|---|---|---:|---:|
| `emb_cell` | Within-section 10-NN | 0.4846 | 0.5665 |
| `emb_cell` | Adjacent-section, within radius | 0.4105 | 0.4921 |
| `emb_niche` | Within-section 10-NN | 0.6278 | 0.9440 |
| `emb_niche` | Adjacent-section, within radius | 0.4947 | 0.8491 |

Higher spatial-edge cosine means a smoother latent representation along those physical edges. It is not, by itself, proof of better biological accuracy because excessive smoothing can also raise this score.

## Outputs

- `embedding_comparison_shared_umap.png` / `.pdf`: matched spots in a shared UMAP for each representation after orthogonal Procrustes alignment, plotted with `scanpy.pl.umap`.
- `embedding_comparison_metrics.png` / `.pdf`: compact metric summary.
- `embedding_comparison_metrics.tsv`: tidy metric values.
- `embedding_comparison_summary.json`: matching, sampling, and spatial-evaluation audit data.

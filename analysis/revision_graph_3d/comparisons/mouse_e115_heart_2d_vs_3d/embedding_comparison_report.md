# Mouse E11.5 heart: matched sectionwise-2D versus section-aware-3D embeddings

## Scope

Both arms contain the same **98,966 observations** and were trained with identical data, `heart_regions` annotation, two-stage objective, model settings, and random seed. The sectionwise-2D graph contains 1,141,174 directed edges; the section-aware-3D graph contains 2,468,990.
The identical seed and observation order make the train/validation split expected to match, but the split membership was not persisted, so it was not independently audited.

All metrics and UMAP panels use every matched observation. No visualization or metric is downsampled.

`heart_regions` was explicitly used as the training annotation. Consequently, heart-region purity and UMAP separation quantify training-label fit and are not independent validation.

## Metrics

| Representation | Linear CKA | Procrustes similarity | kNN overlap@15 | 2D training-region purity | 3D training-region purity | 2D ctype purity | 3D ctype purity | 2D adjacent-section kNN | 3D adjacent-section kNN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `emb_cell` | 0.7006 | 0.7861 | 0.0338 | 0.5245 | 0.4873 | 0.9050 | 0.9017 | 0.0781 | 0.0741 |
| `emb_niche` | 0.9573 | 0.9770 | 0.0503 | 0.9949 | 0.9975 | 0.9154 | 0.9118 | 0.1304 | 0.2198 |

`kNN overlap@15` is the mean shared-neighbor fraction between the two fixed-size neighbor sets. It is symmetric and does not define either arm as a positive/reference class.

The independent frequency-matched same-label baseline is 0.2535 for `heart_regions` and 0.8989 for `ctype_user`. The latter purity is dominated by the `Cardiac muscle lineages` majority class (94.79% of observations), so values near 0.90 provide little evidence of cell-type structure beyond label frequency.

## Physical-edge cosine similarity

An embedding-independent physical graph uses each spot's 10th external within-section neighbor distance as its focal radius, then accepts every spot in the immediately adjacent sections within that radius. It contains 989,660 directed within-section edges and 801,494 directed cross-section edges. No fallback edge is forced; 19.36% of focal spots have no qualifying cross-section neighbor.

| Representation | Physical edge type | 2D mean cosine | 3D mean cosine |
|---|---|---:|---:|
| `emb_cell` | Within-section | 0.5405 | 0.5161 |
| `emb_cell` | Cross-section radius | 0.4893 | 0.4823 |
| `emb_niche` | Within-section | 0.9917 | 0.9923 |
| `emb_niche` | Cross-section radius | 0.9684 | 0.9819 |

Higher cosine means the embedding is smoother along these independently defined physical edges. It can also reflect oversmoothing and is not evidence of biological accuracy by itself.

## Main observations

- `emb_niche` retains similar global geometry (CKA 0.9573; Procrustes 0.9770) while its exact kNN overlap is 5.03%.
- The adjacent-section fraction among `emb_niche` latent neighbors changes from 13.04% to 21.98% (+8.94 percentage points; +68.6% relative).
- On independently defined cross-section radius edges, `emb_niche` mean cosine changes from 0.9684 to 0.9819; `emb_cell` changes from 0.4893 to 0.4823.
- Training-region purity is almost saturated for `emb_niche` in both arms and is not independent validation because `heart_regions` was the training target.

## Outputs

- `embedding_comparison_metrics.png` / `.pdf`: metric summary.
- `embedding_comparison_heart_regions_umap.png` / `.pdf`: full-data shared UMAP colored by the training annotation.
- `embedding_comparison_ctype_user_umap.png` / `.pdf`: full-data shared UMAP colored by cell-type annotation.
- `embedding_comparison_shared_umap_coordinates.npz`: all shared UMAP coordinates.
- `embedding_comparison_metrics.tsv`: tidy metric values.
- `embedding_comparison_summary.json`: input matching, parameters, and no-subsampling audit.
